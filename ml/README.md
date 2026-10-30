# tft-vwap: next-day VWAP with a Temporal Fusion Transformer (and intraday, below)

Self-contained training project for the MTap TFT panel. Its **only input** is
`panel.parquet` + `panel.meta.json`, written by `sourcing-py etrade export-tft`. Nothing here
imports `sourcing_py` or reads the DuckDB stores, so the directory can move to its own repo
unchanged.

```bash
cd ml
uv sync --extra dev
uv run pytest -q                                                    # synthetic panel, ~5 s
uv run tft-vwap --symbols 200 --folds 3 --max-epochs 3              # quick real-data smoke run
uv run tft-vwap                                                     # all symbols, all 4 folds
uv run tft-vwap --target vwap                                       # forecast the raw price instead
uv run tft-vwap-report runs/<run_id> --symbol AAPL                  # per-stock calibration report
uv run tft-vwap-predict runs/<run_id> --side up --move 0.01 --confidence 0.7   # tomorrow's screener
uv run tft-vwap-predict runs/<run_id> --symbol AAPL                 # tomorrow's spread for one stock
uv run tft-vwap-predict runs/<run_id> --asof 2026-10-02             # forecast a past session (10-05)
uv run tft-vwap-grade runs/<run_id>                                 # grade forecasts that have happened
```

`--panel` defaults to `../out/tft/panel.parquet`. Export with `--include-inference` so the
last session can serve as a forecast target.

## What is modelled

- **Series:** `vwap_pct_prev_day` (default, `--target return`) or `today_vwap`
  (`--target vwap`), forecast **one session ahead**: the distribution of next-day VWAP's move
  vs today's VWAP.
  The panel's `target_tomorrow_vwap` is not used as the series. It sits on day t's row, so the
  encoder (and `add_target_scales`) would see tomorrow's price. Instead it is the label when
  scoring.
- **Normaliser:** `--normalizer encoder` (default) scales each sample by its own encoder window.
  `group` uses one per-symbol scale over the whole training span. Price targets use a log
  transform.
- **Loss:** QuantileLoss at P5, P10, …, P95 (19 quantiles = the forecast distribution).
  pytorch-forecasting scores the loss in target units, so price-target samples are weighted by 1 / the symbol's median price. That
  makes the loss roughly a % loss instead of being dominated by $300 stocks.
- **Inputs** (every field, by category: [FEATURES.md](FEATURES.md)):
  - statics: `symbol` (16-D embedding), `industry_code` (4-D), `exchange`, `issue_type`; a static
    with one value in a fold's training rows is left out (`--no-drop-constant-statics` keeps it),
    so on the CS-only panel `issue_type` is dropped
  - known: the calendar features
  - observed: price/volume ratios and 10-K/10-Q flags, with `volume_velocity` log1p'd
  - earnings calendar, dated from the 8-K results releases: sessions since the last release
    and to the next one, plus the Pre_Earnings_Runup / Earnings_Eve / Earnings_Day /
    Post_Earnings_Reaction / Post_Earnings_Drift flags. `--earnings exact` (default) counts to
    the next release's actual (pre-announced) date, `pit` to the year-ago estimate, `off`
    drops them.
  - sentiment: off unless `--use-sentiment` (the news history is still near-empty)

## Walk-forward evaluation

`n_folds` test blocks of `fold_len` sessions (default 4 × 60) tile the end of the panel, or
end at `--test-end <day_idx>`. To compare a new run with an old one on a longer panel, pass
the old run's last `test_end` (in its `config.json` folds). Then both are trained and
tested on the same sessions. For each fold:

| Segment | Targets | Use |
|---|---|---|
| train | `< es_start` | fit |
| early-stop | `es_start .. test_start-1` (20 sessions) | early stopping and checkpoint selection |
| test | `test_start .. test_end` | reported metrics, never seen during training or selection |

Encoders may look back across segment boundaries, but targets never do. Each evaluated
forecast is the single longest-encoder sample for that (symbol, day). It is kept only when the
panel has both that day's row and the previous session's row: the previous row supplies the
naive baseline (`today_vwap`) and the label (`target_tomorrow_vwap`).

## Metrics (per fold, `metrics.json` and the printed table)

All metrics are scale-free and compared with the naive forecast "tomorrow's VWAP = today's":

| Metric | Meaning |
|---|---|
| `mape` / `naive_mape`, `mape_skill` | P50 error; skill = 1 − mape/naive (> 0 beats naive) |
| `median_ape` / `naive_median_ape` | robust to the heavy news-day tails |
| `directional_accuracy`, `up_rate` | sign of the P50 move vs the actual move; `up_rate` is the always-up hit rate |
| `interval_coverage` | share of actuals inside the outermost quantiles, P5–P95 (`interval_nominal` 0.9) |
| `quantile_loss` / `naive_quantile_loss` | pinball loss on returns; naive treats today's price as every quantile |

## Distribution calibration (the probability calls)

The 19 quantiles give each forecast a CDF (linear between quantiles, clamped to [5%, 95%]
outside them). From it, every test row gets P(move ≥ +1/2/3%) and P(move ≤ −1/2/3%). These
are scored on the **pooled** test folds:

- **Probability calls** (`calibration_signals.csv`): for each event and call level (60/70/80/90%),
  take the rows where the forecast probability is at least that level. Report how many there were
  (`signals`), the mean forecast probability, how often the event happened (`hit_rate`) and how often
  it happens anyway (`base_rate`). A trustworthy 70% call has `hit_rate` ≈ `mean_prob`. It is
  useful when that is well above `base_rate`.
- **Quantile calibration** (`calibration_quantiles.csv`): share of actuals at or below each P_tau.
  This should equal tau. Above tau means the forecast is too low there; below means too high.
- **Grid** (`calibration_grid.csv`): the all-stock version of the per-stock grid below.

Events, call levels, probability buckets and quantiles are `TrainConfig.moves` (±0.5% … ±3%),
`.confidences`, `.prob_buckets` and `.quantiles`. The first three only affect scoring, so
the report and predict commands use the current defaults even on older runs.

## Per-stock calibration report (`tft-vwap-report`)

This command re-scores a finished run's saved `fold_*/test_predictions.parquet`. Nothing is
retrained. `train` runs it automatically at the end. `--moves` and `--prob-buckets` change the
grid. To explore the same numbers interactively (any move, band, stock, industry or day range,
and run-vs-run comparison), use `../dashboard` (`uv run forecast-dash --runs ../ml/runs`). It
reads only the run files; see the forecast-run contract in `dashboard/README.md`.
`test_predictions.parquet` carries `date` and `industry_code`, so a run can be graded without
the panel. Older runs get them joined from the panel in `config.json`.

- **Grid** (`grid_{overall,by_symbol,by_industry,by_day}.csv`): one row per event (move ≥ +m or
  ≤ −m) and forecast-probability bucket (5–10%, 10–20%, …, 90–95%). Each row has:
  - `n`
  - `mean_prob`: what the model said
  - `hit_rate`: what happened
  - `deviation = hit_rate − mean_prob`. Below 0 means overconfident; above 0 means too cautious.
  - `z = deviation / binomial standard error`

  Read a row as "AAPL: when the model said 70–80% for ≥ +1%, it came true 55% (n=31)". At
  ~60 test days per stock per fold most per-stock cells are thin, so always check `n` and `z`.
  `|z| < 2` is within noise. `grid_by_day` shows regime days, when every stock misses together.
- **Ladder** (`ladder_{overall,by_symbol}.csv`): the move the model gives at each confidence c.
  Upside is q(1−c), so the 70% upside move is P30. Downside is q(c). `hit_rate` is how often the
  actual move cleared it (ideal: c).
- **Bias** (`bias_{by_symbol,by_day}.csv`): built from the PIT, u = forecast CDF at the actual
  move.
  - `bias = 0.5 − mean(u)`. Above 0 is **optimistic** (positive bias: actuals landed below the
    forecast). Below 0 is **pessimistic**.
  - `bias_z` tests it against a calibrated uniform PIT. The label is `calibrated` when
    |bias_z| < 2.
  - `up_deviation` / `down_deviation` are the n-weighted grid deviations per side.
  - With thousands of stocks, about 5% get labelled by chance alone. Compare the label counts
    with that before reading anything into single names.

The printout shows:
- the all-stock grid as a pivot (cell = came true / model said (n))
- the ladder
- the most optimistic and most pessimistic stocks with n ≥ `--min-n`
- with `--symbol`, that stock's grid, ladder and bias

## Tomorrow's forecast, frozen model (`tft-vwap-predict`)

This loads `fold_<k>/best.ckpt`; the default is the latest fold, whose training data is the
newest. The checkpoint is only read: there is no training and no weight update. The
checkpoint's fitted encoders and normalisers are applied to each symbol's last
`encoder_length` panel sessions. The decoder step is the session after the panel's last day;
its calendar comes from `panel.meta.json["next_session"]`, so export with
`--include-inference`.

- `forecast_<date>.{parquet,csv}`: one row per symbol trading on the last session. Columns:
  - today's VWAP
  - `q05..q95` as fractional moves vs today's VWAP
  - the ladder `up_60..up_90` / `down_60..down_90`
- `forecast_<date>_moves.csv`: one row per symbol and move. Columns:
  - the model probability and its bucket
  - `hist_*`: that stock's test `n` / `hit_rate` / `deviation` in the same bucket
  - `all_*`: the same for all stocks

  Example: model 72% for ≥ +1% → AAPL's 70–80% calls for +1% came true 55% (n=31); all
  stocks' came true 73%. If `report/` is missing it is built first.
- Modes:
  - no flags: count of stocks per event and call level
  - `--symbol X`: X's ladder and full probability spread vs history
  - `--side up|down --move m --confidence c`: screener, every stock with P(event) ≥ c
- `--asof D` forecasts the panel session after D from the panel cut at D. The decoder's
  calendar comes from that real next row, so the forecast is what the model would have said on
  D's evening. Every forecast file has an `asof` column with its reference session.
- After writing, predict runs `tft-vwap-grade`. For every `forecast_<date>.parquet` whose date
  is now in the panel, it writes `actuals.parquet`: `date, symbol, actual_move` (vwap(date) /
  vwap(asof) − 1, both from the same export) and `actual_vwap` (the forecast's `today_vwap` ×
  (1 + move), so a split in between doesn't distort the grade). The dashboard's Forecast vs
  actual tab reads it. To grade yesterday's forecast, refresh and re-export the panel, then run
  `tft-vwap-grade runs/<run_id>`.

## 5-year series: input groups, pre-market states, baselines, screener (2026-10-27)

The design and reasons: [MODEL-STRATEGY.md](MODEL-STRATEGY.md) Q11–Q16; the inputs:
[FEATURES.md](FEATURES.md) sections 11 and 16; results: [docs/LEARNINGS.md](../docs/LEARNINGS.md)
`tag-ml-daily`. Train with `PYTHONWARNINGS="ignore::UserWarning:sklearn.utils.validation"`
(FEATURES.md section 12).

```bash
P=../out/tft/exp5y-r1/panel.parquet       # schema 4, export-tft --include-inference, frozen
C="--panel $P --earnings exact --n-folds 15 --min-train-sessions 252 --fold-len 63 --early-stop-len 21 --test-end 1713"  # 15 folds
G="--session-shape --after-hours"         # the base's groups (R1c); R1 was --market --macro --after-hours --session-shape
ODD="--folds 1,3,5,7,9,11,13"
B=runs/20261028-233305Z-return-4438sym    # the base, R1c (it beat R0; MODEL-STRATEGY Q21)

uv run tft-vwap $C                                    # R0  baseline
uv run tft-vwap $C $G                                 # R1c the base (R1: all four groups)
uv run tft-vwap $C $G $ODD --premarket optional       # R4  A: 25% night, 25% prices only, 50% everything
uv run tft-vwap $C $G $ODD --premarket on --pm-groups px        # R4p B: prices always on
uv run tft-vwap $C $G $ODD --premarket on --pm-groups px,full   # R4f B: everything on
uv run tft-vwap-overlay $B --panel $P --inputs px   $ODD   # R6p C -> $B-overlay-px
uv run tft-vwap-overlay $B --panel $P --inputs full $ODD   # R6f C -> $B-overlay-full
uv run tft-vwap-gbm $C $G                             # R7  plain HGB per quantile
uv run tft-vwap-gbm $C $G --premarket on --pm-groups px,full $ODD   # R7f HGB + full pre-market
uv run tft-vwap-gbm $C $G --premarket optional --pm-groups px,full --open --vol --tail-iter 600 \
    --label R5   # R5: one HGB for 09:30 (runs/<id>~px) and 09:45 (runs/<id>); runs/queue-r5.sh
uv run tft-vwap-recal runs/<id> --by pm,width --window 4 --label R5-cal   # -> runs/<id>-cal
uv run tft-vwap-recal runs/<id> --by pm,width --sessions 20 --label R5-cal20s   # rolling -> runs/<id>-cal20s
uv run tft-vwap-event-overlay runs/<R0> --panel $P    # R6c band scale on event days -> runs/<R0>-event
```

- **Pause and resume.** `touch runs/<id>/STOP` stops at the next epoch end (or `--max-hours H`);
  `uv run tft-vwap --resume runs/<id>` continues. A resume refuses if the panel changed.
- **Time left.** A `[progress]` line per epoch, `runs/<id>/progress.json` every 30 s, and
  `uv run tft-vwap-status [runs/<id> | runs]`: fold and run time left as earliest–latest
  (estimate) and an ETA. Earliest assumes early stopping as soon as possible; latest, every
  fold to `max_epochs`; the estimate uses the folds already finished.
- **States of a pre-market run.** An `optional` run writes `runs/<id>/` (full: the data's own
  flags), `runs/<id>~px/` (prices on, full off) and `runs/<id>~night/` (both off). Each has
  `config.json` (`view_of`, `state`) and `fold_k/test_predictions.parquet` + `metrics.json`, so the
  dashboard and compare treat it as a run. A px-only run's main dir is the px state.
- **Live forecasts per state:** `uv run tft-vwap-predict runs/<id> --state night|px|full` writes
  `forecast_<date>.*` into that state's dir (no name clashes; `tft-vwap-grade` grades each dir).
  `--symbols-file F` forecasts only those symbols into `forecast_<date>_subset.*`, not graded.
- **Compare:** `uv run tft-vwap-compare runs/A runs/B [--state night|px|full] [--state-b …]
  [--only-pm px|full]`. Rows match on (symbol, date) when both runs have dates, else (symbol,
  day_idx); the report says which. Per-fold table, earnings split, bootstrap CI. The bake-off:
  - night: `runs/R4 runs/R1 --state night --only-pm full`
  - 09:30: `runs/R4 runs/R4p --state px --only-pm px`, and R4 vs `runs/R1-overlay-px`
  - 09:45: `runs/R4 runs/R4f --state full --only-pm full`, and R4 vs `runs/R1-overlay-full`
- **Model cards:** every training command writes `runs/<id>/MODEL.md` (`model_card/`): front
  matter `label`, `forecast` (night, 09:30, 09:45, 09:30/09:45, intraday), `summary`, `kind`,
  `status`; then purpose, inputs, folds, result and the command. It is written when the run starts
  (status running), on a budget pause (paused) and at the end (done, with the pooled test result).
  `tft-vwap`, `tft-vwap-gbm`, `tft-vwap-overlay`, `tft-vwap-event-overlay`, `tft-intraday` and
  `screener-train` take `--label R7f --about "one line" --purpose "the question"`; without
  `--label` the label is `<kind> <MMDD-HHMM>`. A rewrite keeps the label, summary and purpose
  already in the card, so `--resume` keeps them (the card flags may also be given on resume).
  A `--premarket optional` run also writes cards for its `~night` and `~px` siblings. The dashboard
  shows the label and the card. `uv run python scripts_model_cards.py` replaces the cards of the
  known research runs with hand-written verdicts; re-run it when a verdict changes. `runs/` is
  gitignored, so the script is the record and the NAS backup holds the files.
- **Overlays** train for test fold k on the base run's out-of-fold forecasts of earlier folds
  only; folds with fewer than 2 earlier folds are marked `thin` (metrics.json, summary.json).
- **GBM (R7):** one sample per (symbol, day), last-session inputs plus `vwap_pct_prev_day` lags
  0, 1, 2, 5, 20; one `HistGradientBoostingRegressor(loss="quantile")` per quantile, sorted; fixed
  `--max-iter` (no early stop). `--premarket on` (groups as features) or `optional` (pre-market
  training rows also seen with the full group blank; scored as 09:45 in the run dir and as 09:30 in
  `<id>~px`). `--open` (the open for every stock), `--vol` (past volatility) and `--tail-iter N`
  (N iterations for P15 and below and P85 and above): FEATURES.md section 18.
- **Recalibration:** `uv run tft-vwap-recal runs/<id> [--window 4] [--by pm,width] [--folds …]
  [--label …]` writes `runs/<id>-cal`, the same run with each quantile shifted, for test fold k,
  by the tau-quantile of (actual − P_tau) over the `window` test folds before k (never k itself),
  per group: `pm` = with or without full pre-market, `width` = P5–P95 width tercile (cut on the
  earlier folds). Folds with no earlier fold are skipped. Live, the shifts come from the
  latest test blocks. It fixes the steady pessimism (R7f mean |q err| 0.78 → 0.10 pts) at no
  pinball cost; MODEL-STRATEGY Q35. **`--sessions N`** (rolling, the chosen mode) calibrates each
  day on the run's forecasts of the N sessions before it, from any fold (days with fewer than
  2,000 calibration rows are left out), and writes `runs/<id>-cal<N>s`; live, the shifts come from
  the last N graded days. On R5 with N = 20: 0.79 → 0.18 pts, where the 4-fold window gives 0.57,
  for 0.25% pinball (Q36).
- **Handoff to intraday:** `uv run tft-vwap-handoff runs/<id> [--live] [--state night|px|full]`
  writes `runs/<id>/handoff.parquet` + `handoff.meta.json`: the out-of-fold daily quantiles (one
  row per symbol and forecast date) and their summaries, past-only record and day context.
  Columns and rules: [docs/intraday-model.md](../docs/intraday-model.md) §1.
- **Variable importance:** each fold writes `fold_k/importance.json` (variable-selection weights
  over ≤ 20 test batches); `summary.json` has `importance_by_group` (base, earnings, market,
  macro, session_shape, after_hours, premarket_px, premarket_full).

### Screener (`ml/screener/`, R8)

Ranks the liquid stocks by expected range from the open, `max(high − open, open − low)/open`
(MODEL-STRATEGY.md Q15). Separate package; reads the panel and run files only.

```bash
uv run screener-train --panel $P --base-run runs/<R5> [--folds …] [--settings settings.toml]
uv run screener-score --date D --model runs/screener/<id> --prices prices-D.jsonl \
    [--pm premarket-D.jsonl] --forecast runs/<R5>~px/forecast_D.parquet --state px|full \
    [--symbols-file shortlist-D.csv]          # -> out/screener/screener-D.parquet + -top20.csv
uv run screener-outcomes --panel $P --days 60  # -> out/screener/outcomes.parquet (dashboard)
```

- `screener-train` fits a px and a full model and writes the R8 evaluation to `summary.json`:
  top-20 realised range vs the universe median, a random 20, naive ATR, TFT P5–P95 width alone
  and pre-market volume alone, with paired bootstrap CIs. The model for fold k is fit only on
  days before fold k (fold 0 is not evaluated).
- `screener-D.parquet`: `symbol, date, state, score, pred_range, rule_pass, rank, lean, open,
  last, prev_close, prev_vwap, dollar_volume_20d, split_day` + features. `lean` (TFT median
  up/down) is not used for ranking. Rule filters come from `[screener]` in settings.
- Live scoring assumes no split today; `exclude_split_day` bites only in training/evaluation.

## Outputs

```
runs/<UTC stamp>-<target>-<N>sym/
  config.json       # TrainConfig, panel meta, fold boundaries
  summary.json      # per-fold results, test table, pooled metrics + calibration
  calibration_{signals,quantiles,grid}.csv   # pooled over the test folds
  report/           # tft-vwap-report: grid_*, ladder_*, bias_* CSVs
  forecast_<date>.{parquet,csv}, forecast_<date>_moves.csv   # tft-vwap-predict
  actuals.parquet   # tft-vwap-grade: date, symbol, actual_move, actual_vwap
  fold_<k>/
    best.ckpt                 # best early-stop checkpoint
    metrics.json              # early-stop + test metrics, test calibration, epochs, timings
    test_predictions.parquet  # symbol, day_idx, q05..q95 (prices), prev_vwap, actual
    logs/metrics.csv          # train/val loss curves
```

## Intraday: the forward VWAP from any second (`tft-intraday`)

A second package, `tft_intraday`, in the same project and under the same boundary rules. Its
only input is the intraday panel `out/tft_intraday/panel/` (`date=<D>/part.parquet` +
`panel.meta.json`), written by `sourcing-py intraday export-tft`. Standing at a decision second
t, one model forecasts six targets as 19 quantiles: the VWAP of the next 1 / 5 / 15 / 60
minutes, of the rest of the day and of the whole day, each as a log move from the last price.
Every field, the targets, folds, baselines and grading: [FEATURES-intraday.md](FEATURES-intraday.md).

```bash
uv run tft-intraday --symbols 50 --max-epochs 1 --limit-train-batches 20   # smoke run, ~5 min
uv run tft-intraday                                  # the default folds, all symbols, ~1 h on MPS
uv run tft-intraday --daily-folds                    # one fold per test date
uv run tft-intraday --baselines-only                 # baselines only, no training
uv run tft-intraday --daily-run none --no-group      # ablations: no daily forecast / no peers
uv run tft-intraday-report runs/intraday/<run_id>    # re-score a run
```

The morning daily forecast is joined from `--daily-run` (default
`runs/20261011-173109Z-return-3785sym`), fold `--daily-fold` 1 only: its test window holds
every backfill day out of sample.

```
runs/intraday/<UTC stamp>-intraday-<N>sym[-baselines]/
  config.json       # model, targets, horizon_s, step_s, eval_stride_s, daily_run, folds, config
  summary.json      # per-fold results and the by_target table
  report/           # by_target, by_day, by_tod parquet (tft, climatology, momentum, daily)
  fold_<k>/
    best.ckpt, metrics.json, logs/metrics.csv
    baselines.json            # climatology and momentum quantiles per 15-minute slot
    intraday_<target>.parquet # symbol, date, t_s, mask, actual, q05..q95, clim_q*, mom_q*, ref_trail,
                              # ref_session, y, last, prev_vwap (+ has_daily, daily_q* for r_close)
```

The prediction files are the dashboard's intraday contract
(`dashboard/forecast_dash/intraday_contract.py`); the newest run by name is the one shown.
