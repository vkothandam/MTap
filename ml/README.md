# tft-vwap: next-day VWAP with a Temporal Fusion Transformer

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
- **Inputs:**
  - statics: `symbol` (16-D embedding), `industry_code` (4-D), `exchange`, `issue_type`
  - known: the calendar features
  - observed: price/volume ratios and 10-K/10-Q flags, with `volume_velocity` log1p'd
  - sentiment: off unless `--use-sentiment` (the news history is still near-empty)

## Walk-forward evaluation

`n_folds` test blocks of `fold_len` sessions (default 4 × 60) tile the end of the panel. For
each fold:

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
grid.

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

## Outputs

```
runs/<UTC stamp>-<target>-<N>sym/
  config.json       # TrainConfig, panel meta, fold boundaries
  summary.json      # per-fold results, test table, pooled metrics + calibration
  calibration_{signals,quantiles,grid}.csv   # pooled over the test folds
  report/           # tft-vwap-report: grid_*, ladder_*, bias_* CSVs
  forecast_<date>.{parquet,csv}, forecast_<date>_moves.csv   # tft-vwap-predict
  fold_<k>/
    best.ckpt                 # best early-stop checkpoint
    metrics.json              # early-stop + test metrics, test calibration, epochs, timings
    test_predictions.parquet  # symbol, day_idx, q05..q95 (prices), prev_vwap, actual
    logs/metrics.csv          # train/val loss curves
```
