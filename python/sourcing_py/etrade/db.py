"""DuckDB store for E*TRADE fundamentals.

The DB is the concise, normalized system-of-record; Parquet for ML (TFT) is a later
export step. Schema:

  symbols      — one row per ticker (Phase 1 seeds it; Phase 2 fills xid/company/etc.)
  daily_bars   — Phase 1a: one EOD OHLCV bar per (symbol, date), loaded from the
                 daily_summary jsonl files. The price time series TFT trains on.
  fundamentals — long/tidy: one row per (statement, period, fiscal_end, line_item);
                 dedupes the repeated column labels that verbose JSON would carry.
  sec_filings  — one row per filing (10-K / 10-Q).

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
    PRIMARY KEY (symbol, date)
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
