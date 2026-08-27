"""Massive.com grouped daily bars.

Endpoint:
    GET {base_url}/v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted=true&apiKey=...

Returns one aggregate bar per ticker for the given trading date. Use this as the
template for other Massive endpoints (tickers, trades, quotes) — one source each.

Run:
    uv run sourcing-py run massive_grouped_daily --date 2024-08-15
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sourcing_py.common import config, http, ratelimit
from sourcing_py.common.source import Source

# Provider key for the shared cross-process rate limiter. All Massive sources share it.
_PROVIDER = "massive"

# Upstream single-letter field -> our snake_case field.
_FIELD_MAP = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "transactions",
    "t": "window_start_ms",
}


class MassiveGroupedDailySource(Source):
    name = "massive_grouped_daily"
    weekdays_only = True  # US stock market is closed weekends — don't waste calls

    def fetch(self, params: dict[str, Any]) -> dict:
        date = params.get("date")
        if not date:
            raise ValueError("massive_grouped_daily requires --date YYYY-MM-DD")

        cfg = config.massive_config()
        url = f"{cfg['base_url']}/v2/aggs/grouped/locale/us/market/stocks/{date}"
        self._date = date  # stash for extract()

        # Space requests to stay under the provider's per-minute cap (free tier).
        rpm = cfg["rate_limit_per_min"]
        ratelimit.throttle(_PROVIDER, 60.0 / rpm if rpm else 0.0)

        with http.get_client() as client:
            return http.get_json(
                client,
                url,
                params={"adjusted": "true", "apiKey": cfg["api_key"]},
            )

    def partition_date(self, params: dict[str, Any]) -> date | None:
        d = params.get("date")
        return date.fromisoformat(d) if d else None

    def preflight(self) -> None:
        config.massive_config()  # raises ConfigError if the key is missing

    def cadence_hint(self) -> str | None:
        rpm = config.massive_config()["rate_limit_per_min"]
        if not rpm:
            return None
        return (
            f"pacing Massive at {rpm} req/min (~{60 / rpm:.0f}s apart), "
            "shared across all Massive programs via state/rate/massive.lock"
        )

    def extract(self, raw: dict) -> list[dict]:
        results = raw.get("results") or []
        records = []
        for r in results:
            record: dict[str, Any] = {"ticker": r["T"], "date": self._date}
            for upstream, field in _FIELD_MAP.items():
                record[field] = r.get(upstream)
            records.append(record)
        return records
