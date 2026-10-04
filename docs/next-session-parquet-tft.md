# Parquet export + TFT training: plan

Feature spec: [Parquet.txt](Parquet.txt). Design background: [stock-prediction-plan.md](stock-prediction-plan.md)
§3–§8. Schema: [database-schema.md](database-schema.md).

Branch `feat/derive-features-news` (commit `9942f17`) holds all work through derive-features.

## Data facts (real DB, 2026-10-04)

These differ from earlier assumptions, so check them before tuning anything:

| Fact | Consequence |
|---|---|
| Bars span **2024-08-16 → 2026-08-14**, `day_idx` 1164–1663 (~500 sessions) | Only ~4 walk-forward folds with a 60-session encoder |
| 15,886 symbols; 8,909 have ≥450 bars, 1,711 have <60 | Universe filter required |
| `vwap_nx_1d` NULL on 122,584 rows (67,754 intra-symbol gaps + last bars); 39,188 bars have NULL vwap/volume/transactions | Target is NULL whenever the next session's bar is missing → drop those rows |
| `vwap_nx_1d` is an **absolute price** (AAPL 2026-08-13: 305.75), not a % change | Cross-symbol scale handled by `GroupNormalizer(log)` per symbol; evaluate in % terms |
| `symbol_industry` is exactly 1:1 (5,808 symbols, 59 codes) | Single `industry_code` categorical + `"unknown"`; no multi-hot |
| `symbol_sentiment_daily` is **empty**; `news_articles` covers only 2026-07-26 → 08-25 (741 symbol links) | Sentiment ≈ all-zero in training but non-zero at inference → exclude from v1 features |
| `pytorch-forecasting 1.8.0` + `lightning 2.6.6` resolve against installed torch 2.13; MPS available, 64 GB RAM | Local training is feasible; no pin changes |

## Decisions (settled 2026-10-04)

1. **Target:** raw next-day VWAP (`target_tomorrow_vwap`) with
   `GroupNormalizer(groups=["symbol"], transformation="log")`. The panel also carries
   `target_return = vwap_nx_1d / vwap - 1` so switching is a training-config flag, not a re-export.
2. **Universe:** liquid common stock + ADRs: `issue_type IN ('CS','DR')`, ≥120 bars,
   median VWAP ≥ $1, median volume ≥ 50k. Thresholds are exporter flags.
3. **`industry_code`:** one value per symbol, `"unknown"` when unmapped.
4. **`calendar`:** replaced by known-future features `day_of_week`, `month`, `is_month_end`,
   `is_quarter_end`, `sessions_gap_next` (calendar days to the next session; >1 marks
   weekends/holidays). All come from `trading_calendar`.
5. **Sentiment:** exported (zero-filled + `has_news` flag) but off by default in training until
   news history accumulates.
6. **Code location:** ML lives in a self-contained top-level `ml/` project, to be extracted into
   its own repo later. **`panel.parquet` + `panel.meta.json` is the only contract.** `ml/` never
   imports `sourcing_py`, never reads the DuckDB stores or `settings.toml`, and its deps never
   enter `python/pyproject.toml`.

## Step 1: exporter (`sourcing-py etrade export-tft`) — DONE 2026-10-04

The first real run produced 1,773,489 rows across 3,730 symbols (3,685 of them inference
rows). It covers day_idx 1164–1663, the file is 109 MB, and the export takes about 1.4 s.
There are no duplicate keys. The only NULLs are the target on inference rows and
`vwap_pct_prev_day` on each symbol's first bar, so the ML side should fill that with 0.

**Open data issue: corporate actions.** 993 rows across 469 symbols have a next-day move
above 50%, up to 11.8×. For example, WOLF on 2025-09-26 went from $1.33 to $17.08. Massive's
`adjusted=true` only adjusts as of each day's fetch, so splits that happen later are never
applied to earlier daily files. The stored history is therefore unadjusted across reverse
splits and restructurings. Fix options:

- (a) Source a splits table (Polygon-style `/v3/reference/splits`) and back-adjust in
  derive-features.
- (b) Exclude or mask rows near jumps where `|return| > X`, as an exporter flag or in the
  ML loss.
- (c) Re-fetch the history with fresh `adjusted=true`.

(a) is the correct fix. (b) is a stopgap.


New module `python/sourcing_py/etrade/export.py`. It runs DuckDB SQL, then
`COPY ... TO '<out_root>/tft/panel.parquet' (FORMAT parquet)`, with one row per
`(symbol, day_idx)` and **raw, un-normalized values**.

| Panel column | TFT role | Source / formula |
|---|---|---|
| `target_tomorrow_vwap` | target | `daily_bars.vwap_nx_1d` |
| `target_return` | alt target | `vwap_nx_1d / NULLIF(vwap,0) - 1` |
| `symbol` | group id / static cat | `daily_bars.symbol` |
| `industry_code` | static cat | `symbol_industry`, else `'unknown'` |
| `exchange`, `issue_type` | static cat | `symbols`, else `'unknown'` |
| `day_idx` | time index | `daily_bars.day_idx` |
| `date` | metadata | `daily_bars.date` |
| `day_of_week`, `month`, `is_month_end`, `is_quarter_end`, `sessions_gap_next` | known | `trading_calendar` |
| `today_vwap` | observed | `vwap` |
| `vwap_pct_prev_day` | observed | `daily_bars.vwap_pct_prev_day` |
| `intraday_spread_pct` | observed | `(high - low) / NULLIF(vwap,0)` |
| `close_position_pct` | observed | `(close - vwap) / NULLIF(vwap,0)` |
| `open_position_pct` | observed | `(open - vwap) / NULLIF(vwap,0)` |
| `volume_velocity` | observed | `volume / NULLIF(AVG(volume) OVER (PARTITION BY symbol ORDER BY day_idx RANGE BETWEEN 19 PRECEDING AND CURRENT ROW), 0)`; the trailing window includes today |
| `transactions_per_volume` | observed | `transactions / NULLIF(volume,0)` |
| `daily_sentiment`, `sentiment_volume`, `has_news` | observed | `ATTACH news.duckdb (READ_ONLY)`; LEFT JOIN `(symbol,date)`, zero-filled |
| `is_10k`, `is_10q` | observed | `daily_bars` |

Rules:

- Compute the window features over **all** bars first, then filter rows. Otherwise dropping a
  row would distort its neighbours' 20-day means.
- Drop rows where the target or `vwap` is NULL. `--include-inference` keeps each symbol's
  latest bar, with a NULL target, for prediction.
- **No leakage:** never export `vwap_nx_10d`, `vwap_end_week` or `vwap_nx_qtr`. Never export
  `results_window` either, because `Pre_Earnings_Runup` and `Earnings_Eve` are assigned from
  an upcoming filing date that isn't known at close of day t.
- Write a `panel.meta.json` sidecar with `schema_version`, row/symbol counts, the `day_idx`
  range, the filter thresholds, the export timestamp and the git SHA.
- CLI summary: rows, symbols, day_idx range, and the share of rows with news.
- Tests in `tests/test_etrade.py` using `_bar` / `_seed_ramp` / `_insert_filing`. Cover the
  ratio formulas, the 20-day window over a gap, NULL-target dropping, the universe filter,
  the `'unknown'` fills, and the absence of leakage columns.
- Run `gen_schema_docs.py` only if DDL changes (it shouldn't).

## Step 2: training program (`ml/`)

```
ml/
  pyproject.toml        # pytorch-forecasting>=1.8, lightning>=2.6, torch, pandas, pyarrow
  tft_vwap/
    config.py           # dataclass: panel path, target, enc/dec length, feature toggles, folds
    dataset.py          # panel.parquet -> TimeSeriesDataSet (checks meta schema_version)
    train.py            # CLI: walk-forward folds, checkpoints + metrics per fold
    evaluate.py         # metrics + naive baseline
  tests/                # synthetic in-memory panel; no dependency on MTap stores
```

- **TimeSeriesDataSet:**
  - `group_ids=["symbol"]`, `time_idx="day_idx"`, `allow_missing_timesteps=True`.
  - Static categoricals: `symbol` (16-D), `industry_code` (4-D), `exchange`, `issue_type`
    (low-dim, set via `embedding_sizes`).
  - Known reals: `day_idx` plus the calendar features.
  - Unknown reals: the observed list, plus the target itself as an encoder input.
  - Missing reals are filled with 0 and flagged.
  - `add_relative_time_idx=True`, `add_target_scales=True`.
- **Model:** `TemporalFusionTransformer`, `QuantileLoss`, encoder 60, decoder 1. Start small
  (`hidden_size` 32, attention heads 2) and train on MPS.
- **Walk-forward folds** (expanding window, by `day_idx`): the first validation block starts
  at about 1414, so the first fold trains on ~250 sessions. Validation blocks are 60
  sessions: 1414–1473, 1474–1533, 1534–1593, 1594–1663. The encoder context may reach
  back into the training region, but no validation target is ever trained on.
- **Metrics per fold:**
  - quantile loss
  - MAPE on the P50 forecast
  - directional accuracy: the sign of `pred/today_vwap - 1` vs `target_return`
  - P10–P90 interval coverage
  - all reported against the naive baseline "tomorrow VWAP = today VWAP"
- **Throughput guard:** ~4–5k symbols × ~400 rows gives about 2M samples. Use
  `limit_train_batches` plus early stopping. A `--symbols N` flag samples a subset for fast
  iteration.

## Suggested build order

1. `export.py`, its CLI wiring and tests, then run it on the real DB and sanity-check the panel.
2. Scaffold `ml/` and write a `dataset.py` test on a synthetic panel.
3. `train.py` and `evaluate.py`: a single-fold smoke run on `--symbols 200`, then the full
   walk-forward.

## Useful commands

```bash
cd python
uv run pytest tests/test_etrade.py -q
uv run sourcing-py etrade derive-features          # after any ingest-eod / fetch
uv run python scripts/gen_schema_docs.py           # after any DDL change (CI runs --check)
uv run python -c "import duckdb; c=duckdb.connect('/Users/vkothandaraman/development/pprojects/MTap/state/etrade/fundamentals.duckdb', read_only=True); print(c.execute('SELECT count(*) FROM daily_bars').fetchone())"
```

The DB path resolves from the repo root (`ETRADE_DB_PATH`). A cwd-relative path from
`python/` will not find it. Re-run `derive-features` after every `ingest-eod` / `fetch`
before exporting.
