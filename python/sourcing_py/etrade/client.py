"""E*TRADE wsod REST client (Phase 2).

Reuses common/http.py for retry/backoff and common/ratelimit.py for cross-process
pacing. Authentication comes from state/etrade/session.json — this client is a pure
consumer; it never logs in itself (see login.py for the two ways to produce the file).

Sampling the live wsod API (2026-08) showed it authenticates with a **bearer token**
(`authorization: Bearer <token>`) plus browser `origin`/`referer` headers. Two session
shapes are supported, in priority order:
  • bearer token (PRIMARY / verified — the full run used this): seeded manually as
    {"accessToken": "<tok>"}, or captured by the puppeteer service into
    requestHeaders.authorization.
  • cookies + `stk1`/`stk2` (what the Node puppeteer login persists by default): passed
    through by `_headers()`. Whether this alone authenticates the fundamentals API is
    UNVERIFIED — on 401/403, fall back to a fresh bearer token.
`_bearer_token()` therefore checks the common token keys AND an `authorization` header,
and `_headers()` always forwards any `cookies` / `requestHeaders` the session carries.

GATED: `EtradeClient.require_session()` raises a clear ConfigError if session.json is
absent or carries neither a token nor cookies, so the fetch phase fails fast instead of
making unauthenticated calls. Produce the file with `sourcing-py etrade login` (puppeteer)
or by seeding a bearer token.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common import config, http, ratelimit
from ..common.errors import ConfigError, FetchError

# All E*TRADE calls share one throttle bucket (state/rate/etrade.lock).
_PROVIDER = "etrade"
# ~7 calls per stock (1 lookup + 4 statements + 2 filings). Pace each request so the
# effective throughput stays near `stocks_per_sec` stocks/second.
_CALLS_PER_STOCK = 7
# Browser headers the wsod API expects alongside the bearer token.
_ORIGIN = "https://www.etrade.wallst.com"


def _bearer_token(session: dict) -> str | None:
    """Pull the wsod bearer token out of the session, tolerant of where the login
    writes it. Checks common top-level keys and an explicit `authorization` header
    (stripping a leading 'Bearer ')."""
    for key in ("accessToken", "access_token", "bearer", "token"):
        val = session.get(key)
        if val:
            return str(val)
    auth = (session.get("requestHeaders") or {}).get("authorization") or session.get("authorization")
    if auth:
        return str(auth).removeprefix("Bearer ").strip()
    return None


class EtradeClient:
    def __init__(self) -> None:
        self.cfg = config.etrade_config()
        self.base_url = self.cfg["base_url"]
        self._session: dict | None = None
        # min seconds between requests to hold ~stocks_per_sec stocks/sec overall
        sps = self.cfg["stocks_per_sec"]
        self._min_interval = (1.0 / (sps * _CALLS_PER_STOCK)) if sps > 0 else 0.0

    # -- session / auth ---------------------------------------------------------
    def session_path(self) -> Path:
        return self.cfg["session_path"]

    def require_session(self) -> dict:
        """Load session.json or raise ConfigError. Cached after first read."""
        if self._session is not None:
            return self._session
        path = self.session_path()
        if not path.exists():
            raise ConfigError(
                f"E*TRADE session not found at {path}. The fundamentals fetch needs the "
                "browser session produced by the Node E*TRADE login (see "
                "node/src/services/etrade/README.md). Start it with `serve etrade`, complete "
                "the OTP flow, then retry."
            )
        data = json.loads(path.read_text())
        if not (_bearer_token(data) or data.get("cookies")):
            raise ConfigError(
                f"E*TRADE session at {path} has no bearer token or cookies — re-run the login."
            )
        self._session = data
        return data

    def _headers(self) -> dict[str, str]:
        s = self.require_session()
        headers: dict[str, str] = {"accept": "application/json", "origin": _ORIGIN, "referer": f"{_ORIGIN}/"}
        token = _bearer_token(s)
        if token:
            headers["authorization"] = f"Bearer {token}"
        if s.get("cookies"):
            headers["cookie"] = s["cookies"]
        for k, v in (s.get("requestHeaders") or {}).items():
            if v and k.lower() != "authorization":  # already set above from the token
                headers[k] = v  # e.g. stk1 / stk2
        return headers

    # -- requests ---------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        ratelimit.throttle(_PROVIDER, self._min_interval)
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            with http.get_client(headers=self._headers()) as client:
                return http.get_json(client, url, params=params or {})
        except FetchError as exc:
            # 401/403 = the token/session expired or is invalid. This is fatal for the
            # whole run (every remaining symbol would fail the same way), not a per-symbol
            # problem — raise ConfigError so fetch() aborts. Already-stored symbols persist;
            # refresh the token and re-run with --resume to continue where it stopped.
            if exc.status_code in (401, 403):
                raise ConfigError(
                    f"E*TRADE API returned HTTP {exc.status_code} — the session token "
                    f"(state/etrade/session.json) has expired or is invalid. Refresh it and "
                    f"re-run with --resume to continue from where the run stopped."
                ) from exc
            raise

    def resolve_xid(self, symbol: str) -> dict:
        """symbol-lookup/<SYMBOL> → the `symbolInfo` dict (contains XID)."""
        raw = self._get(f"symbol-lookup/{symbol}")
        return (raw.get("data") or {}).get("symbolInfo") or {}

    def balance_sheet(self, xid: int, period: str) -> dict:
        return self._get(f"balance-sheet/{xid}", {"period": period})  # period: 'q' | 'a'

    def income_statement(self, xid: int, period: str) -> dict:
        return self._get(f"income-statement/{xid}", {"period": period})

    def sec_filings(self, xid: int, form_type: str) -> dict:
        return self._get(
            f"sec-filings/{xid}",
            {"sortBy": "null", "sortDir": "desc", "first": 0, "formType": form_type},
        )
