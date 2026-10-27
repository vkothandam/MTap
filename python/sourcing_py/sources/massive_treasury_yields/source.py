"""Massive.com US Treasury yields (daily).

Endpoint (paged by `next_url`):
    GET {base_url}/fed/v1/treasury-yields?date.gte=..&date.lte=..&limit=50000&apiKey=...

Each run writes one vintage file (`out/treasury_yields/treasury_yields-<run stamp>.jsonl`);
every file is kept, the loader takes the latest vintage per date.

Run:
    uv run sourcing-py run massive_treasury_yields --days 10      # the daily window
    uv run sourcing-py run massive_treasury_yields                # the full history (backfill)
    uv run sourcing-py run massive_treasury_yields --since 2021-01-01
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sourcing_py.common import config, http
from sourcing_py.common.source import Source

from ..massive_tickers_cs.source import pages

FIELDS = (
    "yield_1_month", "yield_3_month", "yield_6_month", "yield_1_year", "yield_2_year",
    "yield_3_year", "yield_5_year", "yield_7_year", "yield_10_year", "yield_20_year",
    "yield_30_year",
)


class MassiveTreasuryYieldsSource(Source):
    name = "massive_treasury_yields"

    def fetch(self, params: dict[str, Any]) -> list[dict]:
        cfg = config.massive_config()
        query: dict[str, Any] = {"limit": 50000, "sort": "date.asc", "apiKey": cfg["api_key"]}
        since = params.get("since")
        if params.get("days") not in (None, True):
            since = (date.today() - timedelta(days=int(params["days"]))).isoformat()
        if since and since is not True:
            query["date.gte"] = since
        if params.get("until") not in (None, True):
            query["date.lte"] = params["until"]
        out: list[dict] = []
        with http.get_client() as client:
            for results in pages(client, cfg, f"{cfg['base_url']}/fed/v1/treasury-yields", query):
                out += results
        return out

    def preflight(self) -> None:
        config.massive_config()

    def extract(self, raw: list[dict]) -> list[dict]:
        return [{"date": r["date"], **{f: r.get(f) for f in FIELDS}} for r in raw if r.get("date")]
