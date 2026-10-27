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
  sec_filings  — one row per filing (10-K / 10-Q), from E*TRADE.
  sec_edgar_tickers / sec_edgar_filings — SEC EDGAR ticker -> CIK map and each company's
                 8-K / 10-K / 10-Q / 20-F / 40-F filings with acceptance times (edgar.py).
  results_announcements — derived: one row per financial-results release and the session
                 that reacts to it (features.py).
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
    -- Pre-market / after-hours from 1-min bars (extended_hours files; eod.py second pass).
    -- `volume` already covers the whole day: it exceeds pre + regular + after-hours by the
    -- auction prints that are in no minute bar, so these never replace it.
    pre_market_price     DOUBLE,    -- first pre-market trade (04:00-09:29 ET bar open)
    pre_market_last      DOUBLE,    -- last pre-market trade (close of the last bar before 09:30)
    pre_market_volume    DOUBLE,    -- sum of 04:00-09:29 minute bars
    regular_volume       DOUBLE,    -- sum of 09:30-close minute bars (close 16:00, 13:00 half days)
    after_hours_price    DOUBLE,    -- last after-hours trade (close minute .. 19:59 ET)
    after_hours_volume   DOUBLE,    -- sum of close-minute .. 19:59 bars (the 16:00 minute counts here)
    ext_fetched_at       TIMESTAMP, -- when the extended-hours bars were fetched (naive UTC)
    -- VWAPs from the same minute bars (massive_ext, 2026-10); NULL on rows fetched before.
    -- Raw as fetched: prices x ext_split_factor and volumes / ext_split_factor to adjust.
    regular_vwap         DOUBLE,    -- sum(bar vw*v)/sum(v) over 09:30-close bars (NOT the daily vwap: that has auction prints)
    after_hours_vwap     DOUBLE,
    pre_market_vwap_0800 DOUBLE, pre_market_tpvwap_0800 DOUBLE, pre_market_volume_0800 DOUBLE,
    pre_market_vwap_0900 DOUBLE, pre_market_tpvwap_0900 DOUBLE, pre_market_volume_0900 DOUBLE,
    pre_market_vwap_0915 DOUBLE, pre_market_tpvwap_0915 DOUBLE, pre_market_volume_0915 DOUBLE,
    pre_market_vwap_0925 DOUBLE, pre_market_tpvwap_0925 DOUBLE, pre_market_volume_0925 DOUBLE,
    pre_market_vwap_0930 DOUBLE, pre_market_tpvwap_0930 DOUBLE, pre_market_volume_0930 DOUBLE,
    -- Phase 1c derived features (features.py; rewritten in full on every derive-features run)
    day_idx              BIGINT,    -- global trading-day index (-> trading_calendar.date)
    is_10k               SMALLINT,  -- 1 on the (snapped) trading day of a 10-K filing, else 0
    is_10q               SMALLINT,  -- 1 on the (snapped) trading day of a 10-Q filing, else 0
    results_window       TEXT,      -- earnings window vs the results releases (results_announcements)
    days_since_results   INTEGER,   -- sessions since the latest release by this day (0 = release session)
    days_to_results      INTEGER,   -- sessions to the next scheduled release; the estimate if not in the data yet
    days_to_results_is_est SMALLINT, -- 1 when days_to_results is the estimate
    days_to_results_est  INTEGER,   -- sessions to the expected next release (year-ago + 364d; < 0 overdue)
    results_window_pit   TEXT,      -- results_window from days_since_results + days_to_results_est
    vwap_nx_1d           DOUBLE,    -- vol-weighted vwap over the next 1 trading day
    vwap_nx_10d          DOUBLE,    -- vol-weighted vwap over the next 10 trading days
    vwap_end_week        DOUBLE,    -- current day .. last session of current ISO week (incl.)
    vwap_nx_qtr          DOUBLE,    -- current day .. next 10-K/10-Q filing day (incl.)
    vwap_pct_prev_day    DOUBLE,    -- vwap % change vs previous trading day (fraction; NULL on 1st bar)
    split_factor         DOUBLE,    -- price multiplier for splits not yet applied upstream (1.0 = none);
                                    --   adjusted price = raw * factor, adjusted volume = raw / factor
    ext_split_factor     DOUBLE,    -- split_factor for the extended-hours columns (keyed on ext_fetched_at)
    pre_market_pct_prev_close DOUBLE, -- pre_market_last vs the previous bar's close (fraction, adjusted)
    after_hours_pct_close     DOUBLE, -- after_hours_price vs the same day's close (fraction, adjusted)
    features_computed_at TIMESTAMP, -- when derive-features last wrote this row
    PRIMARY KEY (symbol, date)
);

-- Derived by derive-features: cap-weighted close-to-close return of the market
-- (sector_code '00') and of each TRBC sector, one row per (date, sector).
CREATE TABLE IF NOT EXISTS market_daily (
    date        DATE,
    sector_code TEXT,
    cap_ret     DOUBLE,
    n           INTEGER,
    PRIMARY KEY (date, sector_code)
);

-- Derived by derive-features: macro inputs per session date (yields as of the previous
-- session, CPI from the first session of month M+2).
CREATE TABLE IF NOT EXISTS macro_daily (
    date              DATE PRIMARY KEY,
    ust_3m_chg_bp     DOUBLE,
    ust_2y_chg_bp     DOUBLE,
    ust_10y_chg_bp    DOUBLE,
    ust_10y_2y_slope  DOUBLE,
    ust_10y_3m_slope  DOUBLE,
    ust_10y_z252      DOUBLE,
    cpi_yoy           DOUBLE,
    core_cpi_yoy      DOUBLE,
    cpi_yoy_chg       DOUBLE,
    core_cpi_mom      DOUBLE
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

CREATE TABLE IF NOT EXISTS stock_splits (
    id             TEXT PRIMARY KEY, -- upstream Massive split id
    ticker         TEXT,             -- = daily_bars.symbol
    execution_date DATE,             -- first session that trades on the post-split basis
    split_from     DOUBLE,           -- shares before (e.g. 1 for a 10-for-1, 10 for a 1-for-10)
    split_to       DOUBLE,           -- shares after
    fetched_at     TIMESTAMP
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

CREATE TABLE IF NOT EXISTS sec_edgar_tickers (
    symbol     TEXT PRIMARY KEY,  -- = symbols.symbol
    cik        BIGINT,            -- SEC Central Index Key (tickers can share one: GOOG/GOOGL)
    sec_ticker TEXT,              -- as SEC writes it, e.g. 'BRK-B' for BRK.B
    title      TEXT,              -- SEC company name
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sec_edgar_filings (
    cik         BIGINT,
    accession   TEXT,       -- e.g. '0000320193-26-000071'
    form        TEXT,       -- '8-K' | '10-K' | '10-Q' | '20-F' | '40-F' (+ '/A' amendments)
    filing_date DATE,       -- EDGAR filing date
    report_date DATE,       -- 8-K: event date; 10-K/10-Q: period end
    accepted_at TIMESTAMP,  -- acceptance time (New York) from the filing's header; 8-K results releases only
    items       TEXT,       -- 8-K item codes, e.g. '2.02,9.01' (2.02 = results release)
    primary_doc TEXT,       -- main document file name in the filing
    fetched_at  TIMESTAMP,
    PRIMARY KEY (cik, accession)
);

CREATE TABLE IF NOT EXISTS results_announcements (
    symbol      TEXT,
    day_idx     BIGINT,     -- the session that first reacts to the release (t = 0)
    date        DATE,       -- that session's date
    source      TEXT,       -- '8-K 2.02' | '10-K' | '10-Q' (EDGAR) | 'etrade 10-K' | 'etrade 10-Q'
    scheduled   BOOLEAN,    -- the quarter's results release; false = an extra 2.02 (e.g. preliminary)
    accepted_at TIMESTAMP,  -- EDGAR acceptance time (New York); NULL for E*TRADE dates
    timing      TEXT,       -- 'before_open' | 'market_hours' | 'after_close'; NULL if unknown
    accession   TEXT,
    PRIMARY KEY (symbol, day_idx)
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
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS split_factor DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS results_window_pit TEXT",
    # Earnings windows moved from 10-K/10-Q filing dates to the results releases.
    "ALTER TABLE daily_bars DROP COLUMN IF EXISTS days_since_filing",
    "ALTER TABLE daily_bars DROP COLUMN IF EXISTS days_to_filing_est",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS days_since_results INTEGER",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS days_to_results INTEGER",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS days_to_results_is_est SMALLINT",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS days_to_results_est INTEGER",
    # Pre-market / after-hours (eod.py loads them; features.py derives the split factor and pcts).
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS pre_market_price DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS pre_market_last DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS pre_market_volume DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS regular_volume DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS after_hours_price DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS after_hours_volume DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS ext_fetched_at TIMESTAMP",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS ext_split_factor DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS pre_market_pct_prev_close DOUBLE",
    "ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS after_hours_pct_close DOUBLE",
    # Extended-hours VWAPs and cut-off sums (massive_ext 2026-10), then the per-bar features
    # derived from them (features.py).
    *(f"ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS {c} DOUBLE" for c in (
        "regular_vwap", "after_hours_vwap",
        *(f"pre_market_{k}_{c}" for c in ("0800", "0900", "0915", "0925", "0930")
          for k in ("vwap", "tpvwap", "volume")))),
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
