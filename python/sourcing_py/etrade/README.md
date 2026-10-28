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

## Authentication — required for Phase 2

Phase 2 calls the authenticated wsod API and needs `state/etrade/session.json`. This
Python client is a **pure consumer** of that file — it never logs in itself. There are
two ways to produce it:

**1. Bearer token — primary / verified.** Live sampling (2026-08) showed the wsod API
authenticates with a **bearer token** (`authorization: Bearer <token>`) plus browser
`origin`/`referer` headers. Copy a short-lived token from a logged-in browser session and
seed it:

```jsonc
// state/etrade/session.json
{ "accessToken": "<paste the Bearer token here>" }
```

This is the path the full 15,886-symbol run used end to end. Tokens are short-lived; when
one expires the fetch aborts cleanly on 401/403 — re-seed a fresh token and re-run with
`--resume` to continue where it stopped.

**2. Puppeteer login — fallback / automation.** The Node E*TRADE service logs in (handling
the SMS/OTP challenge) and writes the session for you:

```bash
cd python
uv run sourcing-py etrade login           # runs `npm run serve -- etrade`, waits for session.json
uv run sourcing-py etrade login --once     # stop the service as soon as the session lands
```

`login` shells out to the Node service, streams its output (so you can complete the OTP),
and waits until `session.json` is written before reporting readiness; it then leaves the
service running to keep the session alive (Ctrl-C to stop, or use `--once`). It needs the
Node prerequisites — `.env` with `ET_USERNAME`/`ET_PASSWORD` + SMS gateway + OTP host, and
a reachable Redis; see
[../../../node/src/services/etrade/README.md](../../../node/src/services/etrade/README.md).

> **Caveat.** The puppeteer service persists **cookies + `stk1`/`stk2`** headers (plus any
> bearer token it captured from the page's own XHRs). Whether the cookie/`stk` session
> *alone* authenticates the fundamentals REST API is **unverified** — only the bearer
> token is proven. If `fetch` then returns 401/403, fall back to option 1 with a fresh
> token. `client._bearer_token()` reads a token from `accessToken`/`access_token`/`bearer`/
> `token` or an `authorization` header, and `_headers()` always forwards any
> `cookies`/`stk1`/`stk2` the session carries.

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

## Phase 1b — sector / industry classification (no login)

Populate the `sectors` and `industries` dimensions and set each symbol's `industry_code`
from the sibling **MBin** repo's SQLite trading DB (`../MBin/data/db/trading.db`), which
already holds the TRBC classification (11 sectors, 59 industries) in its `Industries` +
`Symbol_Industry` tables.

```bash
cd python
uv run sourcing-py etrade map-industries                       # default MBin path
uv run sourcing-py etrade map-industries --trading-db /path/to/trading.db
```

Read-only against MBin; idempotent (upserts the dims, then rebuilds `symbol_industry` for
our symbols). Memberships live in the **`symbol_industry` join table**, so a symbol may
belong to **many** industries (and thus many sectors) — MBin is 1:1 today (TSLA →
industry `531010` *Automobiles & Auto Parts* only, which nests under sector `53` *Consumer
Cyclicals* — one industry, its sector derived via `industries.sector_code`, **not** two
memberships), but the schema needs no change if that grows. Symbols
absent from MBin (many ETFs/warrants) simply get no membership rows. Configurable via
`[etrade].trading_db_path` / env `MBIN_TRADING_DB`. Reference run: **11 sectors, 59
industries; 5,808 of 15,886 symbols classified** (2,499 MBin symbols are outside our
universe). Join to read a symbol's labels (one row per membership):

```sql
SELECT si.symbol, i.industry_name, sec.sector_name
FROM symbol_industry si
JOIN industries i   ON si.industry_code = i.industry_code
JOIN sectors    sec ON i.sector_code    = sec.sector_code;
```

## Phase 2 — fundamentals (needs session.json)

Resolve each symbol's XID (`symbol-lookup`) and fetch balance sheet + income statement
(period `q` and `a`) and SEC filings (`10-K`, `10-Q`) into the normalized tables.

```bash
uv run sourcing-py etrade fetch --symbols GOOGL,AAPL   # specific symbols
uv run sourcing-py etrade fetch --all --resume          # every symbol; skip completed
uv run sourcing-py etrade fetch --retry-failed          # rerun only logged failures
```

- **Concurrency & throttle:** symbols are fetched concurrently (`workers`, default 10,
  network-only in worker threads) while every DuckDB write stays on the main thread in one
  atomic per-symbol transaction. The cross-process lock `state/rate/etrade.lock` caps the
  aggregate request rate at ~`stocks_per_sec × 7` calls/s, so `ETRADE_STOCKS_PER_SEC=4`
  gives ~4 full stocks/second; `http.get_json` retries transient 429/5xx. Writes use a
  set-based Arrow bulk-upsert (`_bulk_upsert`) — ~1k rows/symbol land in one INSERT.
- **No-data is not failure:** ETFs/funds/SPACs return HTTP **556** (no statement data) and
  delisted/warrant/unit tickers return HTTP **400** (symbol not found). Both are recorded as
  `symbols.fundamentals_status='none'` (committed) — **not** logged as failures — so
  `--resume` skips them instead of re-hitting the API each run. Successful symbols get
  `fundamentals_status='ok'`.
- **Failures & resume:** genuine/transient errors are logged to `state/etrade/failures.jsonl`;
  `--resume` skips any symbol with a committed outcome (status set, or — for pre-column rows —
  any stored fundamentals), `--retry-failed` reruns just the logged ones and drains the log
  as they succeed. History accumulates: each run UPSERTs the ~5 periods a call returns onto
  the `fundamentals` PK, so older periods are kept, not overwritten.

Reference full run (2026-08, 15,886 symbols, single bearer token, `ETRADE_STOCKS_PER_SEC=4`):
**6,801 ok / 9,014 no-data, ~5.48M fundamentals rows + ~48k filings**, ~8 symbols/second.

> The `_normalize_*` helpers in `fundamentals.py` were **finalized against real sampled
> responses** (GOOG/GOOGL, 2026-08); trimmed captures live in `tests/fixtures/etrade/` and
> back the normalization tests. Statement line items are keyed by the upstream **stable
> code** (`ATCA`, `RTLR`, …) with the human label kept as `raw_label`; period ends are
> aligned positionally from `summaryData` since deeper tree entries omit their own `date`.

## Phase 1c — derived features (no login)

Compute the calendar-aligned, point-in-time feature columns the downstream TFT panel needs
and write them back onto `daily_bars`. Depends on Phase 1a bars (all four VWAPs, `day_idx`)
and Phase 2 filings (the 10-K/10-Q flags, `vwap_nx_qtr`), plus the SEC EDGAR results releases
(the earnings windows).

```bash
cd python
uv run sourcing-py etrade fetch-splits                       # Massive splits -> stock_splits (~40s free tier)
uv run sourcing-py etrade fetch-edgar                        # SEC filings + release times (see below)
uv run sourcing-py etrade derive-features                    # all symbols
uv run sourcing-py etrade derive-features --symbols GOOGL,AAPL   # scope to a subset
```

**Split adjustment.** EOD bars are downloaded with `adjusted=true`, which only applies the
splits known at download time. Bars stored earlier are never re-downloaded, so a later split
would leave them on the old share basis. `fetch-splits` refreshes `stock_splits` from Massive
`/v3/reference/splits`, by default for every execution date from the first bar onward.

`derive-features` then sets `daily_bars.split_factor` on each bar. The factor is the product
of `split_from / split_to` over splits that meet all three conditions:

- executed after the bar's date
- executed after the bar's `ingested_at` date (so splits Massive already applied are skipped)
- executed on or before today (announced future splits are ignored)

Raw OHLCV stays untouched. Every derived VWAP column, and the exporter, use
`price × split_factor` and `volume ÷ split_factor`.

Known gap: in a few cases Massive's own adjustment missed a split. The download-time rule
can't detect these. As of 2026-10-04 there were 12, all but 2 of them delisted or renamed
tickers. In the CS universe the two are BVC (2026-01-05) and CLBK (2026-07-21).

**Results releases (SEC EDGAR).** A company's earnings press release is furnished on an
8-K with item 2.02 ("Results of Operations and Financial Condition"), usually days or weeks
before the 10-K/10-Q. `fetch-edgar` maps every CS/DR symbol to its SEC CIK
(`company_tickers.json`; SEC writes BRK-B for our BRK.B) and stores each company's 8-K,
10-K, 10-Q, 20-F and 40-F filings (with amendments) from 400 days before the first bar
(`data.sec.gov/submissions`, incl. the older history files). For every 8-K with item 2.02 it
then fetches the acceptance time from the filing's EDGAR header (`<ACCEPTANCE-DATETIME>`, New
York time). That is one small request per release, ~1 hour for the full universe the first
time; later runs only fetch the new ones. The submissions JSON's own `acceptanceDateTime` is
not used: it is labelled UTC but for some filings is Eastern time plus twice the offset.

- SEC requires a User-Agent with a contact email and at most 10 requests/second:
  `[edgar] user_agent / requests_per_sec / workers` in `settings.toml` (or
  `EDGAR_USER_AGENT`, `EDGAR_REQUESTS_PER_SEC`, `EDGAR_WORKERS`). Defaults: 8 req/s, 4 workers.
- `--symbols A,B` refreshes a subset; `--since YYYY-MM-DD` changes the history start.
- Exits 1 if any company or release time failed to fetch; re-running retries just those.
- ADRs that file 6-K/20-F (no 2.02) and preferreds have no releases; see the fallbacks below.

`derive-features` turns them into **`results_announcements`**, one row per (symbol, results
session):

- **t = 0** is the session that first trades on the news: the acceptance day's session if
  accepted before the 16:00 close, otherwise the next session (a Friday-evening release reacts
  on Monday). `timing` is `before_open` (< 09:30), `market_hours` or `after_close`. Without a
  fetched time the filing date's session is used.
- **Scheduled vs unscheduled.** A release belongs to the first 10-K/10-Q filed from one
  session before it to 50 sessions after it. The last release of each group is the quarter's
  scheduled results; earlier ones (preliminary results, guidance updates under 2.02) are
  `scheduled = false`. Only scheduled releases are anticipated by the exact pre windows
  (`days_to_results`). Whether a release was the scheduled one is only known from later
  filings, so the point-in-time estimate doesn't use it.
- **Fallbacks:** a 10-K/10-Q that no release belongs to is its own announcement (source
  `10-K`/`10-Q`); a symbol with no EDGAR filings at all uses E*TRADE's `sec_filings` dates
  (source `etrade 10-K`/`etrade 10-Q`).

The trading-day grid comes from the generic `sourcing_py/utils/trading_calendar.py` helper
(wraps `exchange_calendars`' authoritative **XNYS** calendar), materialized into the
`trading_calendar` dimension table. Columns written on `daily_bars`:

- **`day_idx`** — global contiguous NYSE trading-day index (`1` = first session on/after
  2020-01-01); shared across symbols, the source of truth for cross-symbol calendar alignment.
- **`is_10k` / `is_10q`** — `1` on the trading day a 10-K/10-Q was filed (t=0); a filing that
  lands on a weekend/holiday snaps forward to the next session.
- **Earnings windows**, in sessions relative to a results announcement (t = 0):
  `Pre_Earnings_Runup` (t−10..t−2), `Earnings_Eve` (t−1), `Earnings_Day` (t=0),
  `Post_Earnings_Reaction` (t+1..t+3), `Post_Earnings_Drift` (t+4..t+15), else
  `Normal_Trading`. The post windows (from the latest announcement, any kind) are checked
  first, so a recent release wins over an upcoming one.
  - **`days_since_results`** — sessions since the latest announcement (0 on its day; NULL
    before the first).
  - **`days_to_results`** / **`results_window`** — sessions to the next **scheduled**
    announcement's actual date. Companies announce these dates weeks ahead, so this is known
    on the day in practice. Where the next one isn't in the data yet (the latest quarter), the
    estimate below is used and **`days_to_results_is_est`** = 1.
  - **`days_to_results_est`** / **`results_window_pit`** — strictly point in time: the
    expected next announcement is a year-ago one + 364 days (same weekday),
    skipping anniversaries within 20 sessions after the latest announcement (that quarter's,
    already out); negative once overdue, and up to 10 sessions overdue reads as
    `Earnings_Eve`.
- **Forward VWAPs** (volume-weighted `SUM(vwap*volume)/SUM(volume)`): `vwap_nx_1d` (next
  trading day), `vwap_nx_10d` (next 10), `vwap_end_week` (through the last session of the
  current ISO week), `vwap_nx_qtr` (through the next 10-K/10-Q filing day). NULL where the
  forward window is empty. `vwap_nx_qtr` still runs to the next E*TRADE 10-K/10-Q filing.
- **`vwap_pct_prev_day`** — backward-looking vwap % change vs the previous trading day
  (`(vwap − prev_vwap)/prev_vwap`, a fraction: 0.05 = +5%). NULL on a symbol's first bar.
- **`split_factor`** — price multiplier for splits not yet applied upstream (1.0 = none).
  See *Split adjustment* above.

> **Full, idempotent recompute.** Every derived column is reset and rewritten on each run
> (one transaction, set-based SQL). Because the forward-looking columns for existing rows
> change as new bars land, **re-run `fetch-splits` + `derive-features` after each `ingest-eod`/`fetch`.**

## Phase 2a — TFT panel export (no login)

Writes the training panel consumed by the downstream ML code: `out/tft/panel.parquet` plus a
`panel.meta.json` sidecar (schema version, row/symbol counts, day_idx/date range, filters,
git SHA). The Parquet file and its sidecar are the **only** contract with the ML side.

```bash
cd python
uv run sourcing-py etrade fetch-splits && uv run sourcing-py etrade derive-features  # first
uv run sourcing-py etrade export-tft --include-inference
uv run sourcing-py etrade export-tft --out /tmp/panel.parquet --min-bars 250 --issue-types CS
```

- One row per `(symbol, day_idx)` with **split-adjusted, otherwise raw values**. Per-symbol scaling happens at
  training time (`GroupNormalizer`).
- **Universe** (flags shown with their defaults): `--issue-types CS` (common stocks only since 2026-10-27; `CS,DR` adds ADRs), `--min-bars 120`,
  `--min-median-vwap 1`, `--min-median-volume 50000`.
- **Target:** `target_tomorrow_vwap` (`vwap_nx_1d`) and `target_return`
  (`vwap_nx_1d / vwap - 1`). Rows with a NULL target are dropped. `--include-inference`
  keeps each symbol's bar on the latest session, with a NULL target.
- **Statics:** `symbol`, `industry_code`, `exchange`, `issue_type`. Missing values become
  `'unknown'`.
- **Known calendar features:** `day_of_week`, `month`, `is_month_end`, `is_quarter_end`,
  `sessions_gap_next` (calendar days to the next session).
- **Observed:**
  - prices and ratios: `today_vwap`, `vwap_pct_prev_day`, `intraday_spread_pct`,
    `close_position_pct`, `open_position_pct`
  - volume: `volume_velocity` (volume ÷ trailing 20-session mean, today included),
    `transactions_per_volume`
  - news: `daily_sentiment`, `sentiment_volume`, `has_news`. Joined from the news DB
    (read-only ATTACH) and zero-filled.
  - filings: `is_10k`, `is_10q`
  - earnings calendar (schema 3): `results_window`, `days_since_results`, `days_to_results`,
    `days_to_results_is_est` (exact scheduled dates, estimate past the data), and
    `results_window_pit`, `days_to_results_est` (strictly point in time)
- **Not exported, because they leak the future:** `vwap_nx_10d`, `vwap_end_week`,
  `vwap_nx_qtr`.

## Storage schema

```
symbols(symbol PK, xid, issue_type, exchange, company_name, auto_invest,
        has_fund_commentary, first_seen, last_seen, days_seen,
        fundamentals_status['ok'|'none'|NULL], fundamentals_checked_at,
        xid_resolved_at, updated_at)  -- fundamentals_status drives Phase 2 --resume

daily_bars(symbol, date, open, high, low, close, volume, vwap, transactions,
           window_start_ms, source, ingested_at, run_id, loaded_at,
           -- Phase 1c derived features (features.py; rewritten in full each derive-features run):
           day_idx, is_10k, is_10q, results_window, days_since_results,
           days_to_results, days_to_results_is_est, days_to_results_est,
           results_window_pit, vwap_nx_1d, vwap_nx_10d,
           vwap_end_week, vwap_nx_qtr, vwap_pct_prev_day, features_computed_at,
           PRIMARY KEY (symbol, date))  -- Phase 1a: one EOD OHLCV bar per symbol/day

trading_calendar(date PK, day_idx, calendar)  -- Phase 1c: NYSE (XNYS) session dimension;
             -- day_idx is the global contiguous trading-day index (1 = first session
             -- >= 2020-01-01) and the source of truth for daily_bars.day_idx.

fundamentals(xid, symbol, statement['balance_sheet'|'income_statement'],
             period_type['q'|'a'], fiscal_end, line_item, value, raw_label,
             period_length, period_length_units, thousand_multiplier, fetched_at)
             -- long/tidy: one row per (line-item code, period); line_item is the stable
             -- upstream code, raw_label the human name. period_length[_units] are
             -- income-statement only (e.g. 3 'M'); thousand_multiplier scales value.

sec_filings(xid, symbol, form_type['10-K'|'10-Q'], date_filed, html_doc_key, fetched_at)
             -- one row per filing; html_doc_key is the opaque doc id (embeds the accession).
             -- Filings attach to the primary share class (e.g. GOOGL, not GOOG).

sec_edgar_tickers(symbol PK, cik, sec_ticker, title, fetched_at)
             -- fetch-edgar: our symbol -> SEC CIK (several symbols can share a CIK)
sec_edgar_filings(cik, accession, form, filing_date, report_date, accepted_at, items,
                  primary_doc, fetched_at, PRIMARY KEY (cik, accession))
             -- 8-K/10-K/10-Q/20-F/40-F (+ /A); items = 8-K item codes ('2.02,9.01');
             -- accepted_at (New York time) only for 8-Ks with item 2.02
results_announcements(symbol, day_idx, date, source, scheduled, accepted_at, timing,
                      accession, PRIMARY KEY (symbol, day_idx))
             -- derive-features: one row per results session (t = 0); source '8-K 2.02',
             -- '10-K'/'10-Q' (no release found) or 'etrade 10-K'/'etrade 10-Q' (no EDGAR)
```

The long/tidy `fundamentals` layout keeps the store compact and makes selective
pivot-to-wide for TFT straightforward (filter by `line_item`/`period_type`, then export
Parquet via `common/writer.py` or DuckDB `COPY ... TO`).
