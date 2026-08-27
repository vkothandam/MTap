# E*TRADE fundamentals (Python, DuckDB)

Builds a normalized, local **DuckDB** store for ML (TFT): the master symbol table
(Phase 1), the end-of-day **price bars** from the `daily_summary` files (Phase 1a), and
per-stock **fundamentals** from E*TRADE's wsod REST API — balance sheet, income statement,
and SEC filings (Phase 2). The DB is the concise system-of-record (no repeated verbose
JSON); Parquet export for TFT is a later step.

Unlike the file-emitting `sources/` (jsonl/parquet per date), this is a **symbol-keyed,
DB-backed** collector, so it lives in its own package with its own CLI verb rather than
the `Source` file-writer contract. It still reuses the shared `common/` utilities:
`http` (retry/backoff), `ratelimit` (cross-process pacing), `failures` (resume), and
`config` (settings/env).

## Headless E*TRADE login — required for Phase 2

Phase 2 calls the authenticated wsod API and **requires the browser session captured by
the Node E*TRADE service**: `state/etrade/session.json`. This Python client reads that
file and attaches the credentials to every request — it does **not** log in itself.

> **Auth scheme (from live sampling, 2026-08).** The wsod API authenticates with a
> **bearer token** (`authorization: Bearer <token>`) plus browser `origin`/`referer`
> headers — not the cookie/`stk1`/`stk2` scheme originally assumed. `client._bearer_token()`
> reads the token from the session (checks `accessToken`/`access_token`/`bearer`/`token`
> or an `authorization` header) and still passes through any `cookies`/`stk1`/`stk2` the
> session carries. **Confirm which key the puppeteer login writes the token under and
> tighten this once it's merged.**

> **Status.** As of this writing the Node puppeteer login is **scaffolded but not yet
> merged** (`node/package.json` has no `puppeteer`; `serve etrade` is unregistered; the
> `etrade/` service dir is docs-only). Until it lands and has written `session.json`,
> `sourcing-py etrade fetch` fails fast with a clear message. **Phase 1 (build-symbols)
> needs no login and runs today.** See
> [../../../node/src/services/etrade/README.md](../../../node/src/services/etrade/README.md)
> for the login/OTP flow that produces the session.

Once the login is available: run `serve etrade` (Node), complete the OTP, confirm
`state/etrade/session.json` exists, then run Phase 2.

## Configure

`[etrade]` in `shared/config/settings.toml` (see `settings.example.toml`), all env-overridable:

| Key | Env | Default |
|---|---|---|
| `base_url` | `ETRADE_BASE_URL` | `https://etrade.api.wsod.com/etrade-api/1.0` |
| `stocks_per_sec` | `ETRADE_STOCKS_PER_SEC` | `2` |
| `db_path` | `ETRADE_DB_PATH` | `./state/etrade/fundamentals.duckdb` |
| `session_path` | `SESSION_PATH` | `./state/etrade/session.json` |

## Phase 1 — master symbols table (no login)

Union unique `ticker`s from a recent window of `out/daily_summary/daily_summary-<date>.jsonl`
into the DuckDB `symbols` table (idempotent upsert; tracks first_seen/last_seen/days_seen):

```bash
cd python
uv run sourcing-py etrade build-symbols --days 90
# or an explicit window:
uv run sourcing-py etrade build-symbols --fromdate 2026-04-08 --todate 2026-08-14
```

## Phase 1a — EOD price bars (no login)

Load the actual end-of-day OHLCV bars the `daily_summary` files hold (Phase 1 only took
distinct tickers) into the `daily_bars` table — the price time series Phase 2 fundamentals
augment for TFT. Uses DuckDB's native `read_json_auto` (bulk, not row-by-row); upserts on
`(symbol, date)`, so re-loading a corrected day refreshes it and re-runs are idempotent.
Afterwards the `symbols` master's first_seen/last_seen/days_seen are refreshed from the full
`daily_bars` history.

```bash
cd python
uv run sourcing-py etrade ingest-eod                       # all files (default)
uv run sourcing-py etrade ingest-eod --days 250            # last 250 trading days
uv run sourcing-py etrade ingest-eod --fromdate 2024-08-16 --todate 2026-08-14
```

Reference run: 500 files / ~2 years → **5.7M bars, 15,886 symbols in ~37 s**, DB ≈ 313 MB.
Files that fail to parse are reported (and the run exits non-zero) rather than aborting.

## Phase 2 — fundamentals (needs session.json)

Resolve each symbol's XID (`symbol-lookup`) and fetch balance sheet + income statement
(period `q` and `a`) and SEC filings (`10-K`, `10-Q`) into the normalized tables.

```bash
uv run sourcing-py etrade fetch --symbols GOOGL,AAPL   # specific symbols
uv run sourcing-py etrade fetch --all --resume          # every symbol; skip completed
uv run sourcing-py etrade fetch --retry-failed          # rerun only logged failures
```

- **Throttle:** paced to ~`stocks_per_sec` stocks/second (each stock ≈ 7 calls) via the
  cross-process lock `state/rate/etrade.lock`; `http.get_json` retries transient 429/5xx.
- **Failures & resume:** per-symbol failures are logged to `state/etrade/failures.jsonl`;
  `--resume` skips symbols already stored, `--retry-failed` reruns just the logged ones and
  drains the log as they succeed.

> The `_normalize_*` helpers in `fundamentals.py` were **finalized against real sampled
> responses** (GOOG/GOOGL, 2026-08); trimmed captures live in `tests/fixtures/etrade/` and
> back the normalization tests. Statement line items are keyed by the upstream **stable
> code** (`ATCA`, `RTLR`, …) with the human label kept as `raw_label`; period ends are
> aligned positionally from `summaryData` since deeper tree entries omit their own `date`.
> The full end-to-end scrape still awaits the login (only the shape is confirmed).

## Storage schema

```
symbols(symbol PK, xid, issue_type, exchange, company_name, auto_invest,
        has_fund_commentary, first_seen, last_seen, days_seen, xid_resolved_at, updated_at)

daily_bars(symbol, date, open, high, low, close, volume, vwap, transactions,
           window_start_ms, source, ingested_at, run_id, loaded_at,
           PRIMARY KEY (symbol, date))  -- Phase 1a: one EOD OHLCV bar per symbol/day

fundamentals(xid, symbol, statement['balance_sheet'|'income_statement'],
             period_type['q'|'a'], fiscal_end, line_item, value, raw_label,
             period_length, period_length_units, thousand_multiplier, fetched_at)
             -- long/tidy: one row per (line-item code, period); line_item is the stable
             -- upstream code, raw_label the human name. period_length[_units] are
             -- income-statement only (e.g. 3 'M'); thousand_multiplier scales value.

sec_filings(xid, symbol, form_type['10-K'|'10-Q'], date_filed, html_doc_key, fetched_at)
             -- one row per filing; html_doc_key is the opaque doc id (embeds the accession).
             -- Filings attach to the primary share class (e.g. GOOGL, not GOOG).
```

The long/tidy `fundamentals` layout keeps the store compact and makes selective
pivot-to-wide for TFT straightforward (filter by `line_item`/`period_type`, then export
Parquet via `common/writer.py` or DuckDB `COPY ... TO`).
