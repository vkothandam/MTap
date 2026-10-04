# Next session: Parquet export + TFT training

Handoff notes for picking up the work. Feature spec: [Parquet.txt](Parquet.txt). Design
background: [stock-prediction-plan.md](stock-prediction-plan.md) §3–§8. Schema:
[database-schema.md](database-schema.md).

## Where things stand (as of 2026-10-04)

**Done and populated in the real DB** (`state/etrade/fundamentals.duckdb`, ~667 MB):

- `derive-features` (`python/sourcing_py/etrade/features.py`) has been run on the real DB:
  **5,706,444 bars, 15,886 symbols, 1,694 calendar sessions, 47,952 filings mapped.**
- Derived columns on `daily_bars`: `day_idx`, `is_10k`, `is_10q`, `results_window`,
  `vwap_nx_1d`, `vwap_nx_10d`, `vwap_end_week`, `vwap_nx_qtr`, `vwap_pct_prev_day`,
  `features_computed_at`.
- `trading_calendar(date, day_idx, calendar)` dimension table, built by the reusable
  `python/sourcing_py/utils/trading_calendar.py` (XNYS via `exchange_calendars`;
  `day_idx = 1` on 2020-01-02).
- Checked on AAPL: `day_idx` is contiguous, `is_10q=1` on 2026-07-31 (`Earnings_Day`) with the
  correct buckets around it, `vwap_pct_prev_day = -0.0776` that day, and `vwap_nx_1d` is NULL
  on the latest bar.
- 27 tests in `tests/test_etrade.py` pass. `gen_schema_docs.py --check` is clean.
- Backup: `state/etrade/fundamentals.snapshot.duckdb` (~320 MB, from before derive-features).

**Nothing is committed.** Everything since the initial commit is still in the working tree,
including the untracked `features.py`, `utils/`, `industry.py`, `login.py`, `news/` and
`docs/`. You may want to commit before starting the exporter.

**Re-run rule:** run `sourcing-py etrade derive-features` after every `ingest-eod` / `fetch`.
The forward columns, including the target `vwap_nx_1d`, change as new bars land.

## Next step 1: the Parquet exporter

Proposed command: `sourcing-py etrade export-tft`, output `panel.parquet`. The work is
DuckDB SQL followed by `COPY ... TO 'panel.parquet' (FORMAT parquet)`, with one row per
`(symbol, day_idx)`.

Field mapping from [Parquet.txt](Parquet.txt) to its source:

| Panel column | TFT role | Source / formula | Status |
|---|---|---|---|
| `target_tomorrow_vwap` | target | `daily_bars.vwap_nx_1d` | ready |
| `symbol` | static cat | `daily_bars.symbol` | ready |
| `industry_code` | static cat | `symbol_industry` (many-to-many) | **needs collapse**, see decision 2 |
| `exchange` | static cat | `symbols.exchange` | ready (NULL for non-fetched symbols) |
| `issue_type` | static cat | `symbols.issue_type` | ready (NULL for non-fetched symbols) |
| `day_idx` | time index | `daily_bars.day_idx` | ready |
| `calendar` | known cat | see decision 3 | **unclear** |
| `today_vwap` | observed | `daily_bars.vwap` | ready |
| `vwap_pct_prev_day` | observed | `daily_bars.vwap_pct_prev_day` | ready |
| `intraday_spread_pct` | observed | `(high - low) / vwap` | compute at export |
| `close_position_pct` | observed | `(close - vwap) / vwap` | compute at export |
| `open_position_pct` | observed | `(open - vwap) / vwap` | compute at export |
| `volume_velocity` | observed | `volume / AVG(volume) OVER 20 trailing sessions` | compute at export |
| `transactions_per_volume` | observed | `transactions / volume` | compute at export |
| `daily_sentiment` | observed | `news.duckdb` `symbol_sentiment_daily.mean_score_agg` | cross-store join |
| `sentiment_volume` | observed | `symbol_sentiment_daily.n_articles` | cross-store join |
| `is_10k` / `is_10q` | observed | `daily_bars.is_10k` / `is_10q` | ready |

Implementation notes:

- **Cross-store join:** `ATTACH 'state/news/news.duckdb' AS news (READ_ONLY)` and LEFT JOIN on
  `(symbol, date)`. Fill missing days with `n_articles = 0` and `mean_score_agg = 0` (neutral).
  Most symbol-days have no news.
- Guard every ratio with `NULLIF(denominator, 0)`.
- **`volume_velocity`:** use `RANGE BETWEEN 19 PRECEDING AND CURRENT ROW` on `day_idx`,
  matching the RANGE-on-day_idx approach in `features.py`. Decide whether the 20-day mean
  includes the current day. The first 19 bars per symbol have a partial window.
- **Drop rows where the target is NULL** (each symbol's last bar), or keep them only as
  inference rows.
- Emit **raw values, not normalized ones.** TFT's `GroupNormalizer` scales per symbol
  (plan §6). The "Normalized" label in Parquet.txt means TFT normalizes them at training time.
- **Leakage:** every observed feature must be known at close of day t. All the fields listed
  above meet this. The forward VWAP columns other than the target (`vwap_nx_10d`,
  `vwap_end_week`, `vwap_nx_qtr`) are future values and **must not** be used as inputs.
- **Universe filter (plan §7):** consider `days_seen >= N`, and possibly
  `fundamentals_status='ok'`.
- Add tests in `tests/test_etrade.py` using the existing `_bar` / `_seed_ramp` /
  `_insert_filing` helpers. The exporter could live in a new `etrade/export.py`.

## Next step 2: training

- **Dependency:** `pytorch-forecasting` and its `lightning` are not installed yet. Check that
  they work with the `torch>=2.2` pin in `python/pyproject.toml` before running `uv add`.
- `TimeSeriesDataSet` roles:
  - `group_ids=["symbol"]` and `time_idx="day_idx"` with `allow_missing_timesteps=True`.
    `day_idx` is global, so a symbol's missing bars show up as gaps.
  - Static categoricals: `industry_code`, `exchange`, `issue_type`.
  - Known: `day_idx`, calendar features.
  - Unknown: the observed reals listed above.
  - `target="target_tomorrow_vwap"` with
    `GroupNormalizer(groups=["symbol"], transformation="log")`.
- Embedding sizes from Parquet.txt: `symbol` 16-D, `industry_code` 4-D, the rest small. These
  are set through `embedding_sizes` on the TFT model.
- **Split:** walk-forward / expanding window by `day_idx`, never a random split. Metrics:
  quantile loss plus directional accuracy.
- Encoder/decoder lengths are still TBD. A starting point: about 60 sessions of history and a
  1-day horizon, since the target is next-day VWAP.

## Open decisions to settle first

1. **Target form.** Raw `vwap_nx_1d` (the current spec) or a return
   (`vwap_nx_1d / vwap - 1`). Returns are more stationary across symbols.
2. **`industry_code` as multi-hot.** Parquet.txt says "Mapped via Multi-Hot", but a TFT static
   categorical takes one value per series. In practice MBin is 1:1 today, so one
   `industry_code` per symbol works. 10,078 symbols have no industry, so they need an
   `"unknown"` level. Real multi-hot would need static real columns, one per industry.
3. **`calendar` column.** The spec says it "marks operational vs weekend/holiday", but
   `daily_bars` only contains trading sessions, so every row would get the same value.
   `trading_calendar.calendar` is just `'XNYS'`. Options: drop it; replace it with day-of-week,
   month and quarter-end known features; or add a "next day is a holiday / long weekend" flag.
4. **ETF / no-fundamentals symbols** (9,014 of them). Keep them using price and sentiment only,
   or exclude them.

## Useful commands

```bash
cd python
uv run pytest tests/test_etrade.py -q
uv run sourcing-py etrade derive-features          # after any ingest-eod / fetch
uv run python scripts/gen_schema_docs.py           # after any DDL change (CI runs --check)
# quick look at the real DB (use the absolute path, read-only):
uv run python -c "import duckdb; c=duckdb.connect('/Users/vkothandaraman/development/pprojects/MTap/state/etrade/fundamentals.duckdb', read_only=True); print(c.execute('SELECT count(*) FROM daily_bars').fetchone())"
```

The DB path resolves from the repo root (`ETRADE_DB_PATH`). A cwd-relative path from
`python/` will not find it.
