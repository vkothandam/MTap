"""DuckDB store for the news processing layer.

Separate DB from the E*TRADE fundamentals store (default state/news/news.duckdb) so the
two subsystems stay independent. The canonical symbol universe (and company names, needed
for verified symbol attribution) still lives in the E*TRADE `symbols` table — we read it
READ-ONLY via `load_symbol_universe()`; this store never owns symbols.

Schema (all DDL idempotent — CREATE TABLE IF NOT EXISTS):

  news_articles   — one row per unique article; the row's presence == "already processed",
                    so re-runs over the overlapping multi-month RSS windows skip it cheaply.
  article_symbols — verified (article, symbol) attributions (many-to-many). Only matches
                    corroborated in the article text are stored; noisy candidates dropped.
  article_sentiment — per-article FinBERT output (pos/neg/neu probs + aggregate signed score).
  symbol_sentiment_daily — (deferred / on-demand) per-symbol, per-date sentiment rollup.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from ..common import config
from ..common.errors import ConfigError

_DDL = """
CREATE TABLE IF NOT EXISTS news_articles (
    article_id     TEXT PRIMARY KEY,  -- stable hash of the feed entry's guid/link
    feed           TEXT,              -- feed that surfaced it, e.g. 'yahoo' | 'google'
    feed_symbol    TEXT,              -- the symbol whose feed this came from (tentative)
    site           TEXT,              -- publisher domain of the (resolved) url
    url            TEXT,              -- article url (resolved past redirects when deep-fetched)
    title          TEXT,
    summary        TEXT,              -- RSS-provided blurb
    body_text      TEXT,              -- extracted main article text (NULL if not fetched/failed)
    published_at   TIMESTAMP,         -- from the feed entry
    lang           TEXT,
    extract_status TEXT,              -- 'ok' | 'failed' | 'skipped' (no deep fetch needed)
    fetched_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS article_symbols (
    article_id   TEXT,               -- FK -> news_articles.article_id
    symbol       TEXT,               -- FK -> (E*TRADE) symbols.symbol; verified matches only
    match_method TEXT,               -- 'ticker+name' | 'name' | 'cashtag' (evidence, strongest)
    matched_at   TIMESTAMP,
    PRIMARY KEY (article_id, symbol)  -- one article can affect many symbols
);

CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id TEXT PRIMARY KEY,     -- FinBERT scores the article text, not per-symbol
    label      TEXT,                 -- 'positive' | 'negative' | 'neutral' (argmax)
    score_pos  DOUBLE,               -- softmax probabilities (sum ~ 1)
    score_neg  DOUBLE,
    score_neu  DOUBLE,
    score_agg  DOUBLE,               -- aggregate signed score = score_pos - score_neg, in [-1,1]
    model      TEXT,                 -- e.g. 'ProsusAI/finbert'
    scored_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS symbol_sentiment_daily (
    symbol         TEXT,             -- deferred / on-demand rollup (segment 3)
    date           DATE,             -- published_at::date
    n_articles     INTEGER,
    mean_score_agg DOUBLE,
    pos_count      INTEGER,
    neg_count      INTEGER,
    neu_count      INTEGER,
    updated_at     TIMESTAMP,
    PRIMARY KEY (symbol, date)
);
"""

# Additive migrations for DBs created before a column existed (idempotent). None yet.
_MIGRATIONS: tuple[str, ...] = ()


def db_path() -> Path:
    return config.news_config()["db_path"]


@contextmanager
def connect(path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open (creating parent dirs + tables) the news DuckDB. Auto-closes."""
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


def load_symbol_universe(etrade_db_path: Path | None = None, *, required: bool = True) -> list[dict]:
    """Read (symbol, company_name) from the E*TRADE `symbols` table, READ-ONLY.

    This is the canonical universe for symbol attribution. We open the fundamentals DB in
    read-only mode so a running E*TRADE job is never blocked. Returns a list of
    {"symbol", "company_name"} dicts (company_name may be None for Phase-1-only rows).
    """
    p = Path(etrade_db_path) if etrade_db_path is not None else config.etrade_config()["db_path"]
    if not p.exists():
        if required:
            raise ConfigError(
                f"E*TRADE symbols DB not found at {p}. Run `sourcing-py etrade build-symbols` "
                "first (it seeds the symbol universe the news layer attributes articles to), "
                "or set ETRADE_DB_PATH."
            )
        return []
    con = duckdb.connect(str(p), read_only=True)
    try:
        rows = con.execute("SELECT symbol, company_name FROM symbols ORDER BY symbol").fetchall()
    finally:
        con.close()
    return [{"symbol": r[0], "company_name": r[1]} for r in rows]


def bulk_upsert(con, table: str, columns: list[str], conflict: str, update: str,
                rows: list[dict]) -> None:
    """Set-based upsert of `rows` into `table` via a registered Arrow batch (mirrors
    etrade.fundamentals._bulk_upsert). Rows must be pre-deduped on the conflict key —
    a set-based ON CONFLICT errors if the same PK appears twice in one INSERT."""
    if not rows:
        return
    import pyarrow as pa

    tbl = pa.table({c: [r.get(c) for r in rows] for c in columns})
    con.register("_news_upsert_batch", tbl)
    try:
        cols = ", ".join(columns)
        con.execute(
            f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _news_upsert_batch "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {update}"
        )
    finally:
        con.unregister("_news_upsert_batch")
