"""Stock splits reference data from Massive (`/v3/reference/splits`) into `stock_splits`.

EOD bars are fetched with `adjusted=true`, which only applies splits known at fetch time.
A split that executes after a bar was downloaded leaves that bar on the old share basis;
derive-features turns this table into `daily_bars.split_factor` to back-adjust those bars.

Run:
    uv run sourcing-py etrade fetch-splits [--fromdate 2024-08-16]
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, date, datetime

import httpx
import pyarrow as pa

from ..common import config, http, ratelimit
from ..utils import trading_calendar as tc
from . import db

_PROVIDER = "massive"
_PAGE_LIMIT = 1000


def _pages(client: httpx.Client, cfg: dict, fromdate: str):
    """Yield each page's results, following `next_url` (which omits the apiKey)."""
    rpm = cfg["rate_limit_per_min"]
    url: httpx.URL | None = httpx.URL(
        f"{cfg['base_url']}/v3/reference/splits",
        params={"execution_date.gte": fromdate, "limit": _PAGE_LIMIT, "apiKey": cfg["api_key"]},
    )
    while url is not None:
        ratelimit.throttle(_PROVIDER, 60.0 / rpm if rpm else 0.0)
        page = http.get_json(client, url)
        yield page.get("results") or []
        nxt = page.get("next_url")
        # passing params= would replace next_url's cursor query, so merge the key in instead
        url = httpx.URL(nxt).copy_merge_params({"apiKey": cfg["api_key"]}) if nxt else None


def _normalize(results: list[dict], fetched_at: datetime) -> list[dict]:
    rows = []
    for r in results:
        sf, st = r.get("split_from"), r.get("split_to")
        if not (r.get("id") and r.get("ticker") and r.get("execution_date") and sf and st):
            continue
        rows.append({
            "id": r["id"],
            "ticker": r["ticker"],
            "execution_date": date.fromisoformat(r["execution_date"]),
            "split_from": float(sf),
            "split_to": float(st),
            "fetched_at": fetched_at,
        })
    return rows


def fetch_splits(
    *, fromdate: str | None = None, db_path=None, client: httpx.Client | None = None
) -> dict:
    """Refresh stock_splits for execution_date >= fromdate (default: the earliest bar).

    The window is replaced wholesale so splits that upstream later cancels disappear.
    """
    cfg = config.massive_config()
    fetched_at = datetime.now(UTC)
    with db.connect(db_path) as con:
        if fromdate is None:
            first_bar = con.execute("SELECT min(date) FROM daily_bars").fetchone()[0]
            fromdate = (first_bar or tc.ANCHOR).isoformat()

        rows: list[dict] = []
        pages = 0
        with nullcontext(client) if client is not None else http.get_client() as c:
            for results in _pages(c, cfg, fromdate):
                rows.extend(_normalize(results, fetched_at))
                pages += 1
        rows = list({r["id"]: r for r in rows}.values())

        cols = ["id", "ticker", "execution_date", "split_from", "split_to", "fetched_at"]
        tbl = pa.table({c: [r[c] for r in rows] for c in cols}) if rows else None
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM stock_splits WHERE execution_date >= ?", [fromdate])
            if tbl is not None:
                con.register("_splits", tbl)
                con.execute(
                    f"INSERT INTO stock_splits ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _splits "
                    "ON CONFLICT (id) DO UPDATE SET ticker = excluded.ticker, "
                    "execution_date = excluded.execution_date, split_from = excluded.split_from, "
                    "split_to = excluded.split_to, fetched_at = excluded.fetched_at"
                )
                con.unregister("_splits")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        matched, total = con.execute(
            "SELECT count(DISTINCT s.ticker) FILTER (WHERE y.symbol IS NOT NULL), count(*) "
            "FROM stock_splits s LEFT JOIN symbols y ON y.symbol = s.ticker"
        ).fetchone()

    return {
        "fromdate": fromdate,
        "pages": pages,
        "splits_fetched": len(rows),
        "splits_total": total,
        "tickers_in_universe": matched,
    }
