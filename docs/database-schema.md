# Database schema

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with: uv run python scripts/gen_schema_docs.py
     Source of truth: each store's `_DDL` in the module noted per section. -->

MTap keeps two independent [DuckDB](https://duckdb.org) stores. Both are created and migrated idempotently on open (`CREATE TABLE IF NOT EXISTS` + additive `ALTER`), so the diagrams below reflect a freshly-opened DB. Foreign keys are logical only (DuckDB stores them as comments); the edges are drawn from the relationships in the generator.

## E*TRADE fundamentals store

The concise, normalized system-of-record for the E*TRADE collector. `symbols` is the canonical ticker universe; the price series, fundamentals, filings, and TRBC sector/industry dimensions hang off it.

- **Default path:** `state/etrade/fundamentals.duckdb`
- **DDL source:** [`python/sourcing_py/etrade/db.py`](../python/sourcing_py/etrade/db.py)

```mermaid
erDiagram
    daily_bars {
        VARCHAR symbol PK
        DATE date PK
        DOUBLE open
        DOUBLE high
        DOUBLE low
        DOUBLE close
        DOUBLE volume
        DOUBLE vwap
        BIGINT transactions
        BIGINT window_start_ms
        VARCHAR source
        TIMESTAMP ingested_at
        VARCHAR run_id
        TIMESTAMP loaded_at
        BIGINT day_idx
        SMALLINT is_10k
        SMALLINT is_10q
        VARCHAR results_window
        DOUBLE vwap_nx_1d
        DOUBLE vwap_nx_10d
        DOUBLE vwap_end_week
        DOUBLE vwap_nx_qtr
        DOUBLE vwap_pct_prev_day
        TIMESTAMP features_computed_at
    }
    fundamentals {
        BIGINT xid PK
        VARCHAR symbol
        VARCHAR statement PK
        VARCHAR period_type PK
        DATE fiscal_end PK
        VARCHAR line_item PK
        DOUBLE value
        VARCHAR raw_label
        INTEGER period_length
        VARCHAR period_length_units
        BOOLEAN thousand_multiplier
        TIMESTAMP fetched_at
    }
    industries {
        VARCHAR industry_code PK
        VARCHAR industry_name
        VARCHAR sector_code
        VARCHAR source_url
        TIMESTAMP updated_at
    }
    sec_filings {
        BIGINT xid PK
        VARCHAR symbol
        VARCHAR form_type PK
        DATE date_filed
        VARCHAR html_doc_key PK
        TIMESTAMP fetched_at
    }
    sectors {
        VARCHAR sector_code PK
        VARCHAR sector_name
        TIMESTAMP updated_at
    }
    symbol_industry {
        VARCHAR symbol PK
        VARCHAR industry_code PK
        TIMESTAMP mapped_at
    }
    symbols {
        VARCHAR symbol PK
        BIGINT xid
        VARCHAR issue_type
        VARCHAR exchange
        VARCHAR company_name
        BOOLEAN auto_invest
        BOOLEAN has_fund_commentary
        DATE first_seen
        DATE last_seen
        INTEGER days_seen
        VARCHAR fundamentals_status
        TIMESTAMP fundamentals_checked_at
        TIMESTAMP xid_resolved_at
        TIMESTAMP updated_at
    }
    trading_calendar {
        DATE date PK
        BIGINT day_idx
        VARCHAR calendar
    }
    sectors ||--o{ industries : "sector_code"
    symbols ||--o{ symbol_industry : "symbol"
    industries ||--o{ symbol_industry : "industry_code"
    symbols ||--o{ daily_bars : "symbol"
    trading_calendar ||--o{ daily_bars : "date -> day_idx"
    symbols ||--o{ fundamentals : "xid / symbol"
    symbols ||--o{ sec_filings : "xid / symbol"
```

## News processing store

Independent store for the news layer. It never owns symbols: `article_symbols.symbol` and `symbol_sentiment_daily.symbol` reference the E*TRADE `symbols` table, read READ-ONLY via `load_symbol_universe()` (see the cross-store note below).

- **Default path:** `state/news/news.duckdb`
- **DDL source:** [`python/sourcing_py/news/db.py`](../python/sourcing_py/news/db.py)

```mermaid
erDiagram
    article_sentiment {
        VARCHAR article_id PK
        VARCHAR label
        DOUBLE score_pos
        DOUBLE score_neg
        DOUBLE score_neu
        DOUBLE score_agg
        VARCHAR model
        TIMESTAMP scored_at
    }
    article_symbols {
        VARCHAR article_id PK
        VARCHAR symbol PK
        VARCHAR match_method
        TIMESTAMP matched_at
    }
    news_articles {
        VARCHAR article_id PK
        VARCHAR feed
        VARCHAR feed_symbol
        VARCHAR site
        VARCHAR url
        VARCHAR title
        VARCHAR summary
        VARCHAR body_text
        TIMESTAMP published_at
        VARCHAR lang
        VARCHAR extract_status
        TIMESTAMP fetched_at
    }
    symbol_sentiment_daily {
        VARCHAR symbol PK
        DATE date PK
        INTEGER n_articles
        DOUBLE mean_score_agg
        INTEGER pos_count
        INTEGER neg_count
        INTEGER neu_count
        TIMESTAMP updated_at
    }
    news_articles ||--o{ article_symbols : "article_id"
    news_articles ||--|| article_sentiment : "article_id"
```

## Cross-store reference

The news store's `article_symbols.symbol` and `symbol_sentiment_daily.symbol` columns point at `symbols.symbol` in the **E*TRADE fundamentals store**. There is no database foreign key across the two files; the news layer opens the E*TRADE DB read-only to load the symbol universe (`news/db.py::load_symbol_universe`).
