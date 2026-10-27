"""Massive.com US inflation indicators (monthly).

Endpoint (paged by `next_url`):
    GET {base_url}/fed/v1/inflation?date.gte=..&limit=50000&apiKey=...

Each run writes one vintage file (`out/inflation/inflation-<run stamp>.jsonl`) and every
file is kept: values are revised, and the loader keeps the FIRST vintage of each month
(its first release) so a model never sees a revised value it could not have had.

Run:
    uv run sourcing-py run massive_inflation --if-due     # the daily check: fetch only when a month is due
    uv run sourcing-py run massive_inflation              # everything (backfill)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from sourcing_py.common import config, http
from sourcing_py.common.source import Source

from ..massive_tickers_cs.source import pages

FIELDS = ("cpi", "cpi_core", "cpi_year_over_year", "pce", "pce_core", "pce_spending")


def expected_month(today: date) -> date:
    """The reference month that should be stored: the month before last (CPI for month M
    is released mid-M+1, so the previous month is usually still unpublished)."""
    y, m = today.year, today.month - 2
    if m < 1:
        y, m = y - 1, m + 12
    return date(y, m, 1)


def stored_latest_month(directory: Path) -> date | None:
    """Newest reference month with a CPI value in any stored vintage file (cheap: reads the
    newest file only, since each fetch holds the whole series)."""
    files = sorted(directory.glob("inflation-*.jsonl"))
    latest: date | None = None
    for path in files[-1:]:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("cpi") is not None and rec.get("date"):
                d = date.fromisoformat(rec["date"])
                latest = d if latest is None or d > latest else latest
    return latest


def is_due(directory: Path, today: date) -> bool:
    have = stored_latest_month(directory)
    return have is None or have < expected_month(today)


class MassiveInflationSource(Source):
    name = "massive_inflation"

    def fetch(self, params: dict[str, Any]) -> list[dict]:
        if params.get("if-due") and not is_due(config.out_root() / self.config.dataset, date.today()):
            return []
        cfg = config.massive_config()
        query: dict[str, Any] = {"limit": 50000, "sort": "date.asc", "apiKey": cfg["api_key"]}
        if params.get("since") not in (None, True):
            query["date.gte"] = params["since"]
        out: list[dict] = []
        with http.get_client() as client:
            for results in pages(client, cfg, f"{cfg['base_url']}/fed/v1/inflation", query):
                out += results
        return out

    def preflight(self) -> None:
        config.massive_config()

    def extract(self, raw: list[dict]) -> list[dict]:
        return [{"date": r["date"], **{f: r.get(f) for f in FIELDS}} for r in raw if r.get("date")]
