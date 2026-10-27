"""Massive.com common-stock tickers, listed and delisted (the massive-ext universe).

Endpoint (paged by `next_url`, 1000 per page):
    GET {base_url}/v3/reference/tickers?market=stocks&type=CS|ADRC&active=true|false&apiKey=...

Delisted names are included so a multi-year backfill covers stocks that have since gone.
One file per NY date (re-running the same day overwrites it).

Run:
    uv run sourcing-py run massive_tickers_cs
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from sourcing_py.common import config, http, ratelimit
from sourcing_py.common.source import Source

_PROVIDER = "massive"
# CS = common stock, ADRC = ADR common; older files have no `type` and are all CS.
_TYPES = ("CS", "ADRC")
_FIELDS = ("ticker", "name", "primary_exchange", "active", "cik", "composite_figi",
           "share_class_figi", "delisted_utc", "last_updated_utc", "type")


def pages(client: httpx.Client, cfg: dict, url: str, params: dict):
    """Yield each page's results, following `next_url` (which omits the apiKey)."""
    rpm = cfg["rate_limit_per_min"]
    while url:
        ratelimit.throttle(_PROVIDER, 60.0 / rpm if rpm else 0.0)
        page = http.get_json(client, url, params=params)
        yield page.get("results") or []
        nxt = page.get("next_url")
        # passing params= would replace next_url's cursor query, so merge the key in instead
        url, params = (str(httpx.URL(nxt).copy_merge_params({"apiKey": cfg["api_key"]})), None) if nxt else (None, None)


class MassiveTickersCsSource(Source):
    name = "massive_tickers_cs"

    def fetch(self, params: dict[str, Any]) -> list[dict]:
        cfg = config.massive_config()
        url = f"{cfg['base_url']}/v3/reference/tickers"
        out: list[dict] = []
        with http.get_client() as client:
            for typ in _TYPES:
                for active in ("true", "false"):
                    query = {"market": "stocks", "type": typ, "active": active, "limit": 1000,
                             "apiKey": cfg["api_key"]}
                    for results in pages(client, cfg, url, query):
                        out += [{**r, "type": r.get("type") or typ} for r in results]
        return out

    def partition_date(self, params: dict[str, Any]) -> date | None:
        return datetime.now(ZoneInfo("America/New_York")).date()

    def preflight(self) -> None:
        config.massive_config()

    def extract(self, raw: list[dict]) -> list[dict]:
        return [{k: r.get(k) for k in _FIELDS} for r in raw if r.get("ticker")]
