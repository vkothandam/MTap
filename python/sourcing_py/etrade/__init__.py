"""E*TRADE fundamentals sourcing (DuckDB-backed).

Two phases:
  1. build-symbols — union unique tickers from recent daily_summary files into a
     `symbols` master table (no auth needed).
  2. fetch — resolve each symbol's XID and pull balance sheet / income statement /
     SEC filings from the wsod REST API into normalized long/tidy tables. Requires
     the browser session written by the Node E*TRADE service (state/etrade/session.json).

See README.md in this package for the operational guide.
"""
