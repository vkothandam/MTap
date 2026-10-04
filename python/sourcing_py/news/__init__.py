"""News processing layer — three logical segments over a normalized DuckDB store.

  1. aggregator — per-symbol RSS across configurable feeds, two-level (parse the feed,
     then deep-fetch + extract the article body for feeds that need it), history-aware
     (already-seen articles are skipped).  -> news_articles
  2. sentiment  — local FinBERT scores each article (positive/negative/neutral + an
     aggregate signed score) and attributes it to affected symbols with a VERIFIED text
     match (ticker corroborated by company name), discarding noisy hits.
                  -> article_sentiment + article_symbols
  3. daily      — (deferred / on-demand) roll per-article sentiment up to one score per
     symbol per date.  -> symbol_sentiment_daily

Mirrors the etrade/ subsystem: own `news` CLI verb, a DuckDB store, and reuse of the
shared common/ utilities (http, ratelimit, failures, config, errors).
"""
