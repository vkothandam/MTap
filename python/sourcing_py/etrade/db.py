"""DuckDB store for E*TRADE fundamentals.

The DB is the concise, normalized system-of-record; Parquet for ML (TFT) is a later
export step. Schema:

  symbols      — one row per ticker (Phase 1 seeds it; Phase 2 fills xid/company/etc.;
                 industry mapping fills industry_code).
  daily_bars   — Phase 1a: one EOD OHLCV bar per (symbol, date), loaded from the
                 daily_summary jsonl files. The price time series TFT trains on.
                 Phase 1c (features.py) adds derived columns: day_idx, 10-K/10-Q filing
                 flags, results_window classification, and forward VWAP aggregates.
  fundamentals — long/tidy: one row per (statement, period, fiscal_end, line_item);
                 dedupes the repeated column labels that verbose JSON would carry.
  sec_filings  — one row per filing (10-K / 10-Q).
  trading_calendar — date dimension: one row per NYSE (XNYS) session with a global,
                     contiguous day_idx (1 = first session >= 2020-01-01). The source of
                     truth for daily_bars.day_idx; built by etrade/features.py.
  sectors         — TRBC economic-sector dimension (11 rows); populated from MBin's
                    trading.db (see industry.py). Names stored once, not per symbol.
  industries      — TRBC industry dimension (59 rows), each FK to a sector.
  symbol_industry — join table (symbol, industry_code): a symbol may belong to MANY
                    industries (and thus many sectors, via industries.sector_code).

All DDL is idempotent (CREATE TABLE IF NOT EXISTS), so opening an existing DB is safe.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from ..common import config

_DDL = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol              TEXT PRIMARY KEY,
    xid                 BIGINT,        -- resolved in Phase 2 via symbol-lookup
    issue_type          TEXT,          -- Phase 2
    exchange            TEXT,          -- Phase 2
    company_name        TEXT,          -- Phase 2
    auto_invest         BOOLEAN,       -- Phase 2
    has_fund_commentary BOOLEAN,       -- Phase 2
    first_seen          DATE,          -- earliest daily_summary date the ticker appears
    last_seen           DATE,          -- latest daily_summary date
    days_seen           INTEGER,       -- number of daily_summary days it appears in
    fundamentals_status TEXT,          -- Phase 2 outcome: 'ok' (fetched) | 'none' (no data:
                                       --   ETF/fund/invalid symbol). NULL = not yet attempted.
    fundamentals_checked_at TIMESTAMP, -- when Phase 2 last attempted this symbol
    xid_resolved_at     TIMESTAMP,
    updated_at          TIMESTAMP
    -- NB: sector/industry live in symbol_industry (a symbol can have several), not here.
);

CREATE TABLE IF NOT EXISTS sectors (
    sector_code TEXT PRIMARY KEY,      -- TRBC economic-sector code, e.g. '53'
    sector_name TEXT,                  -- e.g. 'Consumer Cyclicals'
    updated_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS industries (
    industry_code TEXT PRIMARY KEY,    -- TRBC industry code, e.g. '532040'
    industry_name TEXT,                -- e.g. 'Household Goods'
    sector_code   TEXT,               -- FK -> sectors.sector_code
    source_url    TEXT,               -- provenance carried from MBin's Industries table
    updated_at    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS symbol_industry (
    symbol        TEXT,               -- FK -> symbols.symbol
    industry_code TEXT,               -- FK -> industries.industry_code
    mapped_at     TIMESTAMP,          -- when this membership was last (re)established
    PRIMARY KEY (symbol, industry_code)  -- many industries per symbol allowed
);

CREATE TABLE IF NOT EXISTS daily_bars (
    symbol          TEXT,
    date            DATE,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    volume          DOUBLE,     -- upstream volumes can be fractional
    vwap            DOUBLE,
    transactions    BIGINT,
    window_start_ms BIGINT,
    source          TEXT,       -- upstream `_source`
    ingested_at     TIMESTAMP,  -- upstream `_ingested_at` (when the jsonl was produced)
    run_id          TEXT,       -- upstream `_run_id`
    loaded_at       TIMESTAMP,  -- when Phase 1a loaded this row into DuckDB
    -- Phase 1c derived features (features.py; rewritten in full on every derive-features run)
    day_idx              BIGINT,    -- global trading-day index (-> trading_calendar.date)
    is_10k               SMALLINT,  -- 1 on the (snapped) trading day of a 10-K filing, else 0
    is_10q               SMALLINT,  -- 1 on the (snapped) trading day of a 10-Q filing, else 0
    results_window       TEXT,      -- earnings-window class relative to nearest filing (t)
    vwap_nx_1d           DOUBLE,    -- vol-weighted vwap over the next 1 trading day
    vwap_nx_10d          DOUBLE,    -- vol-weighted vwap over the next 10 trading days
    vwap_end_week        DOUBLE,    -- current day .. last session of current ISO week (incl.)
    vwap_nx_qtr          DOUBLE,    -- current day .. next 10-K/10-Q filing day (incl.)
    vwap_pct_prev_day    DOUBLE,    -- vwap % change vs previous trading day (fraction; NULL on 1st bar)
    features_computed_at TIMESTAMP, -- when derive-features last wrote this row
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS trading_calendar (
    date     DATE PRIMARY KEY,   -- an XNYS (NYSE) session date
    day_idx  BIGINT,             -- global consecutive index, 1 = first session >= 2020-01-01
    calendar TEXT                -- calendar name the sessions came from, e.g. 'XNYS'
);

CREATE TABLE IF NOT EXISTS fundamentals (
    xid                 BIGINT,
    symbol              TEXT,
    statement           TEXT,      -- 'balance_sheet' | 'income_statement'
    period_type         TEXT,      -- 'q' | 'a'
    fiscal_end          DATE,      -- reporting period end date
    line_item           TEXT,      -- stable upstream field code (e.g. 'ATCA', 'RTLR')
    value               DOUBLE,    -- as reported; scale per thousand_multiplier
    raw_label           TEXT,      -- human label (e.g. 'Total current assets')
    period_length       INTEGER,   -- income statement only: periods covered (e.g. 3)
    period_length_units TEXT,      -- e.g. 'M' (months); NULL for balance sheet
    thousand_multiplier BOOLEAN,   -- upstream scaling flag that applies to `value`
    fetched_at          TIMESTAMP,
    PRIMARY KEY (xid, statement, period_type, fiscal_end, line_item)
);

CREATE TABLE IF NOT EXISTS sec_filings (
    xid          BIGINT,
    symbol       TEXT,
    form_type    TEXT,     -- '10-K' | '10-Q'
    date_filed   DATE,
    html_doc_key TEXT,     -- opaque document key; embeds the SEC accession number
    fetched_at   TIMESTAMP,
    PRIMARY KEY (xid, form_type, html_doc_key)
);
"""

# Additive column migrations for DBs created before a column existed. DuckDB's
# ADD COLUMN IF NOT EXISTS makes these idempotent and cheap on an already-migrated DB.
_MIGRATIONS = (
    "ALTER TABLE symbols ADD COLUMN IF NOT EXISTS fundamentals_status TEXT",
    "ALTER TABLE symbols ADD COLUMN IF NOT EXISTS fundamentals_checked_at TIMESTAMP",
    # Industry membership moved off `symbols` (1:1 FK) to the symbol_industry join table
    # (many-to-many). Drop the interim per-symbol columns if an older DB still has them.
    "ALTER TABLE symbols DROP COLUMN IF EXISTS industry_code",
    "ALTER TABLE symbols DROP COLUMN IF EXISTS industry_mapped_at",
    # Phase 1c derived feature columns on daily_bars (populated by features.py).
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS day_idx BIGINT",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS is_10k SMALLINT",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS is_10q SMALLINT",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS results_window TEXT",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS vwap_nx_1d DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS vwap_nx_10d DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS vwap_end_week DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS vwap_nx_qtr DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS vwap_pct_prev_day DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS features_computed_at TIMESTAMP",
)


def db_path() -> Path:
    return config.etrade_config()["db_path"]


@contextmanager
def connect(path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open (creating parent dirs and tables) the fundamentals DuckDB. Auto-closes."""
    p = Path(path) if path is not None else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p))
    try:
        con.execute(_DDL)
        for stmt in _MIGRATIONS:
            con.execute(stmt)
        yield con
    finally:
        con.close()
