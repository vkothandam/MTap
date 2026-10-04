# News processing (Python, DuckDB)

A per-symbol **news sentiment** feature layer for the TFT model, in three logical segments
over a normalized local **DuckDB** store (`state/news/news.duckdb`, separate from the
E*TRADE fundamentals store):

1. **aggregator** — pull per-symbol news from configurable RSS feeds (Yahoo Finance +
   Google News to start), two-level and history-aware. → `news_articles`
2. **sentiment** — score each article locally with **FinBERT** (`ProsusAI/finbert`) and
   attribute it to affected symbols with a *verified* text match. → `article_sentiment` +
   `article_symbols`
3. **daily** *(deferred / on-demand)* — roll per-article sentiment up to one score per
   symbol per date. → `symbol_sentiment_daily`

Like the `etrade/` collector this is a **symbol-keyed, DB-backed** subsystem with its own
`news` CLI verb, reusing the shared `common/` utilities: `http` (retry/backoff),
`ratelimit` (cross-process pacing, lock `state/rate/news.lock`), `failures` (per-symbol
feed errors), `config`, and `errors`. The canonical **symbol universe** (and company names,
needed for verified attribution) is read **READ-ONLY** from the E*TRADE `symbols` table —
this store never owns symbols.

## Configure

`[news]` in `shared/config/settings.toml` (see `settings.example.toml`), all env-overridable:

| Key | Env | Default |
|---|---|---|
| `db_path` | `NEWS_DB_PATH` | `./state/news/news.duckdb` |
| `requests_per_sec` | `NEWS_REQUESTS_PER_SEC` | `2` |
| `finbert_model` | `NEWS_FINBERT_MODEL` | `ProsusAI/finbert` |
| `lookback_days` | `NEWS_LOOKBACK_DAYS` | `30` |
| `request_timeout` | — | `30` |

Feeds are an array-of-tables — each `[[news.feeds]]` has `name`, a `url_template` with a
`{symbol}` placeholder, `tag` (`"symbol"` marks the feed as tagged with the queried symbol,
which becomes a *candidate* for verification), and `fetch_article` (bool) marking feeds that
need the deep body fetch. Defaults ship Yahoo + Google, both `fetch_article = true`.

## Segment 1 — aggregate

```bash
cd python
uv run sourcing-py news aggregate --symbols AAPL,MSFT      # specific symbols
uv run sourcing-py news aggregate --all --days 30           # every symbol, 30-day lookback
uv run sourcing-py news aggregate --all --no-resume         # force re-fetch of known articles
```

Two passes per `(symbol × feed)`: **(A)** GET the RSS and parse it with `feedparser`;
**(B)** for feeds flagged `fetch_article`, GET the entry link (following redirects — Google
News links redirect to the publisher) and extract the main body text with `trafilatura`.
Feeds not so flagged store the RSS content directly (`extract_status='skipped'`).

**History:** each article is keyed by a stable `article_id` (hash of feed + guid/link);
if its row already exists we skip it entirely (no re-fetch, no re-parse), which is what
makes the overlapping multi-month windows Yahoo/Google return cheap to re-poll. `--no-resume`
forces a full re-fetch/re-extract. Per-symbol feed failures are logged to
`state/news/failures.jsonl`.

## Segment 2 — analyze (FinBERT + verified attribution)

```bash
uv run sourcing-py news analyze            # score articles lacking a sentiment row
uv run sourcing-py news analyze --no-resume  # re-score everything
```

**Sentiment:** FinBERT scores `title + '. ' + (body_text or summary)` → softmax
`score_pos/neg/neu`, argmax `label`, and an aggregate signed `score_agg = score_pos −
score_neg` in `[-1, 1]`. The model + tokenizer load lazily as a process-wide singleton;
the **first run downloads** `ProsusAI/finbert` into the HuggingFace cache (one-time
network). `transformers` + `torch` are core deps — missing → a `ConfigError` with a hint.
`--resume` scores only articles with no `article_sentiment` row.

**Verified attribution** (`matcher.py`): feed tagging is noisy — a Google search for ticker
`MANE` returns a mix of *MANE* and *Bagmane*. So the feed tag is a **candidate**, not truth;
every candidate must be corroborated in the article text, combining two signals:

- **ticker** — a `$MANE` cashtag, or a standalone `MANE` token matched on **word boundaries**
  (so *Bagmane* can't yield `MANE`), required to be an uppercase run and not a common word.
- **company name** — the normalized company name (Inc/Corp/Ltd/… stripped) appears as a
  whole phrase.

Decision (favouring precision): `name` present → accept; `ticker`+`name` → accept
(strongest, `ticker+name`); explicit `$TICKER` cashtag → accept; a **bare ticker token with
no name/cashtag support → rejected** (this is the *Bagmane* noise case). Only accepted
`(article, symbol)` rows land in `article_symbols`; `match_method` records the evidence.

## Segment 3 — build-daily (deferred / on-demand)

```bash
uv run sourcing-py news build-daily                          # all dates
uv run sourcing-py news build-daily --fromdate 2026-07-01 --todate 2026-08-01
```

One set-based SQL rollup joining `article_symbols → article_sentiment → news_articles`,
grouped by `(symbol, published_at::date)` into `symbol_sentiment_daily` (`n_articles`,
`mean_score_agg`, pos/neg/neu counts). Idempotent (upsert on `(symbol, date)`). Not part of
the routine pipeline yet — run it when the daily feature is needed for TFT.

## Convenience

```bash
uv run sourcing-py news run --all --days 30    # aggregate -> analyze in one go
```

## Storage schema

```
news_articles(article_id PK, feed, feed_symbol, site, url, title, summary, body_text,
              published_at, lang, extract_status['ok'|'failed'|'skipped'], fetched_at)
              -- one row per unique article; presence == "already processed" (skip on re-run)

article_symbols(article_id, symbol, match_method['ticker+name'|'name'|'cashtag'], matched_at,
                PRIMARY KEY (article_id, symbol))  -- verified matches only; many-to-many

article_sentiment(article_id PK, label, score_pos, score_neg, score_neu, score_agg,
                  model, scored_at)  -- FinBERT, per-article text

symbol_sentiment_daily(symbol, date, n_articles, mean_score_agg,
                       pos_count, neg_count, neu_count, updated_at,
                       PRIMARY KEY (symbol, date))  -- segment 3, on-demand
```
