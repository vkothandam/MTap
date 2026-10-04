"""Run the news aggregator for stocks with market cap above a threshold.

Market cap is derived from the E*TRADE fundamentals store (read-only):
    market_cap_usd = QTCO (latest total common shares outstanding, in millions) * 1e6 * latest close
so the >$Nmm filter is `QTCO_value * close > N` (N in millions of dollars). Only real
operating stocks are targeted (`symbols.fundamentals_status = 'ok'`); ETFs/funds/indexes
returned no statement data (status 'none' or NULL) and are excluded.

Queries Google News RSS per symbol (`q=<SYM>+stock`), the two-level aggregator deep-fetching
each article. Runs in resumable chunks so progress persists across a long run.

    uv run python scripts/run_news_by_mktcap.py --limit 100          # bounded test: top 100
    uv run python scripts/run_news_by_mktcap.py                      # full run (all qualifying)
    uv run python scripts/run_news_by_mktcap.py --min-mktcap 500 --chunk 50
"""

from __future__ import annotations

import argparse
import os

import duckdb

from sourcing_py.common import config
from sourcing_py.news import aggregator

_GOOGLE_FEED = {
    "name": "google",
    "url_template": "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
    "tag": "symbol",
    "fetch_article": True,
}

_MKTCAP_SQL = """
WITH shares AS (
    SELECT symbol, value AS qtco,
           row_number() OVER (PARTITION BY symbol ORDER BY fiscal_end DESC) rn
    FROM fundamentals WHERE line_item = 'QTCO' AND value IS NOT NULL
),
px AS (
    SELECT symbol, close,
           row_number() OVER (PARTITION BY symbol ORDER BY date DESC) rn
    FROM daily_bars WHERE close IS NOT NULL
)
SELECT s.symbol, s.qtco * p.close AS mktcap_mm
FROM shares s
JOIN px p USING (symbol)
JOIN symbols y USING (symbol)
WHERE s.rn = 1 AND p.rn = 1
  AND y.fundamentals_status = 'ok'          -- real operating stock, not an ETF/fund/index
  AND s.qtco * p.close > ?                   -- market cap (in $mm) above threshold
ORDER BY mktcap_mm DESC
"""


def select_symbols(min_mktcap_mm: float, limit: int | None) -> list[str]:
    etrade_db = config.etrade_config()["db_path"]
    con = duckdb.connect(str(etrade_db), read_only=True)
    try:
        sql = _MKTCAP_SQL + ("\nLIMIT ?" if limit else "")
        params = [min_mktcap_mm, limit] if limit else [min_mktcap_mm]
        return [r[0] for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-mktcap", type=float, default=500.0, help="threshold in $ millions (default 500)")
    ap.add_argument("--limit", type=int, default=None, help="cap to the top N by market cap (test runs)")
    ap.add_argument("--chunk", type=int, default=50, help="symbols per resumable chunk (default 50)")
    ap.add_argument("--days", type=int, default=None, help="lookback window; default = config lookback_days")
    ap.add_argument("--per-domain-rps", type=float, default=None,
                    help="max request starts per second PER DOMAIN (default = config, 10). "
                         "Fetches fan out across many publisher hosts and run concurrently.")
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="global cap on in-flight requests (default = config, 20)")
    args = ap.parse_args()

    # news_config() reads these env vars at call time (once per chunk), so setting them here
    # tunes the whole run without touching settings.toml.
    if args.per_domain_rps is not None:
        os.environ["NEWS_PER_DOMAIN_RPS"] = str(args.per_domain_rps)
    if args.max_concurrency is not None:
        os.environ["NEWS_MAX_CONCURRENCY"] = str(args.max_concurrency)

    symbols = select_symbols(args.min_mktcap, args.limit)
    cfg = config.news_config()
    print(
        f"selected {len(symbols)} stock(s) with market cap > ${args.min_mktcap:,.0f}mm "
        f"(Google News feed, {cfg['per_domain_rps']:g}/sec per domain, "
        f"{cfg['max_concurrency']} concurrent)",
        flush=True,
    )
    if not symbols:
        return 0

    totals = {"new_articles": 0, "skipped_seen": 0, "failed_feeds": 0}
    n_chunks = (len(symbols) + args.chunk - 1) // args.chunk
    for i in range(0, len(symbols), args.chunk):
        chunk = symbols[i:i + args.chunk]
        idx = i // args.chunk + 1
        summary = aggregator.aggregate(symbols=chunk, days=args.days, resume=True, feeds=[_GOOGLE_FEED])
        for k in totals:
            totals[k] += summary[k]
        print(
            f"  chunk {idx}/{n_chunks} ({chunk[0]}..{chunk[-1]}): "
            f"+{summary['new_articles']} new, {summary['skipped_seen']} seen, "
            f"{summary['failed_feeds']} feed fail; running total {summary['total_articles']:,} articles",
            flush=True,
        )

    print(
        f"done — {totals['new_articles']:,} new article(s), {totals['skipped_seen']:,} already seen, "
        f"{totals['failed_feeds']} feed fetch failure(s) across {len(symbols)} stock(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
