"""Quarter-end point-in-time share counts and market caps (CS + ADRC tickers).

Endpoint, one call per ticker per snapshot date:
    GET {base_url}/v3/reference/tickers/{T}?date=YYYY-MM-DD&apiKey=...

Spike S2 (2026-10-27): NVDA at 2024-05-01 returns the pre-split weighted shares (2.5B)
and that day's market_cap; delisted names answer for dates they were listed (404 after,
or before listing). Not every ticker carries market_cap, and `close_at_snap` is not
returned by this endpoint (kept in the schema, null).

The universe is the newest `tickers_cs` file (rows with `type` CS or ADRC when the file
has it), including delisted names whose `delisted_utc` is after the snapshot date. One
file per snapshot date (`out/shares/shares-<snap_date>.jsonl`); a date with a file is
skipped, so the backfill is resumable and idempotent. Snapshot dates are the last session
on or before each quarter end from 2021-09-30.

Run:
    uv run sourcing-py run massive_shares                    # every quarter end still missing (no calls when none is)
    uv run sourcing-py run massive_shares --date 2024-03-28  # one snapshot date
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

from sourcing_py.common import config, http, ratelimit
from sourcing_py.common.errors import FetchError
from sourcing_py.common.source import Source

_PROVIDER = "massive"
FIRST_QUARTER_END = date(2021, 9, 30)
_TYPES = {"CS", "ADRC"}
#: more than this share of failed (non-404) calls aborts the date, so no half file is kept
_MAX_FAIL_FRACTION = 0.005


def quarter_ends(start: date, end: date) -> list[date]:
    out = []
    for y in range(start.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if start <= q <= end:
                out.append(q)
    return out


def snapshot_dates(today: date) -> list[date]:
    """Last session on/before each quarter end from 2021-09-30, strictly before today."""
    from sourcing_py.utils.trading_calendar import sessions

    out = []
    for q in quarter_ends(FIRST_QUARTER_END, today - timedelta(days=1)):
        s = sessions(q - timedelta(days=10), q)
        if s:
            out.append(s[-1])
    return out


def shares_dir() -> Path:
    return config.out_root() / "shares"


def missing_dates(today: date) -> list[date]:
    return [d for d in snapshot_dates(today) if not (shares_dir() / f"shares-{d.isoformat()}.jsonl").exists()]


def universe(snap: date, tickers_dir: Path | None = None) -> list[str]:
    """CS+ADRC tickers from the newest tickers_cs file that were active on `snap`."""
    directory = tickers_dir or config.out_root() / "tickers_cs"
    files = sorted(directory.glob("tickers_cs-*.jsonl"))
    if not files:
        raise FetchError(f"no tickers_cs file under {directory}; run `sourcing-py run massive_tickers_cs`")
    symbols = []
    for line in files[-1].read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ticker") or (r.get("type") and r["type"] not in _TYPES):
            continue
        gone = (r.get("delisted_utc") or "")[:10]
        if not r.get("active") and gone and gone <= snap.isoformat():
            continue  # delisted on/before the snapshot
        symbols.append(r["ticker"])
    return sorted(set(symbols))


class MassiveSharesSource(Source):
    name = "massive_shares"

    # -- one ticker, one date -----------------------------------------------------
    def _one(self, client: httpx.Client, cfg: dict, symbol: str, snap: str) -> dict | None:
        rpm = cfg["rate_limit_per_min"]
        ratelimit.throttle(_PROVIDER, 60.0 / rpm if rpm else 0.0)
        try:
            j = http.get_json(client, f"{cfg['base_url']}/v3/reference/tickers/{symbol}",
                              params={"date": snap, "apiKey": cfg["api_key"]})
        except FetchError as exc:
            if exc.status_code == 404:
                return None  # not listed on that date
            raise
        r = j.get("results") or {}
        return {
            "symbol": symbol, "snap_date": snap,
            "market_cap": r.get("market_cap"),
            "weighted_shares_outstanding": r.get("weighted_shares_outstanding"),
            "share_class_shares_outstanding": r.get("share_class_shares_outstanding"),
            "close_at_snap": r.get("close_at_snap"),
        }

    def fetch(self, params: dict[str, Any]) -> list[dict]:
        snap = params.get("date")
        if not snap:
            raise ValueError("massive_shares fetch needs a snapshot date")
        cfg = config.massive_config()
        symbols = universe(date.fromisoformat(snap))
        out: list[dict] = []
        failed = 0

        with http.get_client() as client:
            def job(sym: str):
                try:
                    return self._one(client, cfg, sym, snap)
                except FetchError:
                    return False

            with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
                for res in pool.map(job, symbols):
                    if res is False:
                        failed += 1
                    elif res is not None:
                        out.append(res)
        if failed > max(3, _MAX_FAIL_FRACTION * len(symbols)):
            raise FetchError(f"{snap}: {failed}/{len(symbols)} overview calls failed; not writing a partial file")
        return out

    def extract(self, raw: list[dict]) -> list[dict]:
        return raw

    def partition_date(self, params: dict[str, Any]) -> date | None:
        d = params.get("date")
        return date.fromisoformat(d) if d else None

    def preflight(self) -> None:
        config.massive_config()

    def run(self, *, params: dict[str, Any] | None = None, validate: bool = True,
            skip_empty: bool = True) -> Path | None:
        """With --date: that snapshot (skipped if its file exists unless --force). Without:
        every quarter-end snapshot still missing. With none missing it makes no API call (the daily job)."""
        params = params or {}
        if params.get("date"):
            if (shares_dir() / f"shares-{params['date']}.jsonl").exists() and not params.get("force"):
                return None
            return super().run(params=params, validate=validate, skip_empty=skip_empty)
        last = None
        for d in missing_dates(date.today()):
            print(f"massive_shares: snapshot {d}")
            last = super().run(params={**params, "date": d.isoformat()}, validate=validate, skip_empty=skip_empty) or last
        return last
