# forecast-dash: interactive calibration dashboard for quantile forecasts

This Streamlit + Plotly app grades next-day move forecasts. For each forecast it asks: when the
model said P(move ≥ +1%) = 70%, how often did it happen? You can slice that by stock,
industry, day range, move size and probability band, and compare runs side by side.

The project is self-contained. It imports nothing from `ml/` or `sourcing_py`; a test enforces
that. Its only input is a run directory on disk (the **forecast-run contract** below), so it can
move to its own repo or into MBin unchanged. Any model that writes these files is graded the
same way, so the modelling can change and every run still gets the same evaluation.

## Quick start

```bash
cd dashboard
uv sync --extra dev                             # first time only
uv run forecast-dash --runs ../ml/runs          # start: opens http://localhost:8501
```

To stop it, press Ctrl+C in the terminal where it runs. If it was started in the background or
that terminal is gone: `pkill -f forecast-dash`.

To use a custom port, different run, or skip telemetry:
```bash
uv run forecast-dash --runs ../ml/runs/<run_id> --port 8502
FORECAST_DASH_RUNS=../ml/runs uv run forecast-dash --port 8503
```

To run tests:
```bash
uv run pytest -q && uv run ruff check .
```

Notes:
- `--runs` can also come from `FORECAST_DASH_RUNS` env var. It may point at a single run or at a
  directory of runs; with a directory of runs, the newest is shown first.

## Forecast-run contract

```
<run>/fold_<k>/test_predictions.parquet   one row per (symbol, forecast day); one file per test fold
    required: symbol, day_idx, prev_vwap, actual, q05..q95   (quantiles and actual in price units)
    optional: date, industry_code
<run>/config.json                         optional; {"config": {"panel": "<path>", ...}}
<run>/forecast_<date>.parquet             optional; tomorrow: symbol, q05..q95 as fractional
                                          moves (0.012 = +1.2%), optional industry_code, today_vwap,
                                          asof (the session the forecast was made from)
<run>/actuals.parquet                     optional; date, symbol, actual_move, actual_vwap for
                                          forecast days that have happened (tft-vwap-grade)
```

- Prices in the Forecast tab are `today_vwap × (1 + q)`. If a forecast file has no `today_vwap`,
  the $ columns are empty and only the % views apply.
- Forecast vs actual joins each `forecast_<date>.parquet` to `actuals.parquet` on (date, symbol).
  `actual_move` is the realised move vs the forecast's `asof` session; `actual_vwap` is on the
  forecast's own price basis, so a split in between doesn't distort the grade.

- The quantile levels are read from the column names (`qNN` → NN/100). At least 3 columns are
  needed, and every fold must use the same set.
- Everything is converted to fractional moves vs `prev_vwap`. Rows with no `actual`, or with
  `prev_vwap` ≤ 0, are dropped.
- If `date` or `industry_code` is missing, the app joins it from the panel recorded in
  `config.json`. If that is not possible, `day_idx` stands in for the date and the industry is
  `unknown`.

## Maths (`forecast_dash/calib.py`)

- **Forecast CDF F(x).** Linear between the quantiles, clamped to [P5, P95] outside them.
- **Event probabilities.**
  - Upside event "move ≥ +m": P = 1 − F(m).
  - Downside event "move ≤ −m": P = F(−m).
- **Grid cell** (event × probability bucket). Each cell has:
  - `n`
  - `mean_prob` ("said")
  - `hit_rate` ("came true")
  - `deviation` = came true − said. Below 0 means overconfident.
  - `z`, the binomial z-score
  - a Wilson 95% CI
- **Ladder at confidence c.** The upside move is q(1 − c) and the downside move is q(c). If the
  forecast is calibrated, the actual clears it c of the time.
- **PIT.** u = F(actual) and bias = 0.5 − mean(u).
  - Positive bias is optimistic: actuals land below the forecast.
  - Negative bias is pessimistic.
  - The noise s.e. is √(1/12/n), and |z| ≥ 2 is flagged.
- **Run summary.**
  - Pinball (quantile) loss vs a naive forecast of a 0% move at every level.
  - P5–P95 coverage vs its nominal 90%.
  - Mean PIT.
  - Median P5–P95 width.
  - ECE, the n-weighted mean |deviation| over the grid.

## Sidebar (applies to every tab)

- run and folds
- side: up, down or both
- move sizes: presets plus any custom %
- probability bucket width: 5% or 10%
- industries and stocks (empty = all)
- test-day range
- min n per grid point, which hides thin cells

## Tabs

Every tab opens with an **In short** box: plain-English findings computed from the current
filters (`forecast_dash/insights.py`). ✅ fine · ⚠️ watch · ❌ problem · ℹ️ context. The
"ℹ️ About this page" popover next to each title explains the page. Every chart, table and
sub-panel has its own ⓘ (what it shows, how to read it, and for tables what each column means),
every table column header has a hover tooltip (e.g. `ci_lo` / `ci_hi`, `lift`, `share`), and
the metrics and controls have `?` tooltips. All of this text is in `forecast_dash/help_text.py`
(`HELP`, `PANELS`, `COLUMNS`).

| Tab | What it shows | How to read it |
| --- | --- | --- |
| Summary | The executive summary. Headline numbers (forecasts, quantile skill vs naive, P5–P95 coverage, bias, ECE). A call-reliability heatmap (event × call level, came true / said). Calls by side vs the base rate. Quantile calibration. A per-fold stability table. The full calls table. Call levels and the headline level are selectable | Start here. The In short box says whether the distribution beats naive, whether the bands are the right width, which way the model leans, how each side's calls at the headline level held up, the most and least trustworthy call, and whether the folds (time periods) agree. If they don't, pooled numbers mix regimes. |
| Calibration | Reliability curves (one per event), deviation heatmap, ladder bars | Points on the diagonal are calibrated. Below it, or red cells, mean the model said more than happened (overconfident). Blue cells mean it was too cautious. Bubble size grows with n, and the error bars are 95% CIs. |
| Threshold explorer | Pick an event and a call level c. Headline figures for calls with P ≥ c, a hit-rate-vs-c curve, a hit-rate-vs-move curve, and a per-stock table with min-calls and min-accuracy filters | Look for the call level where the hit rate is high enough and there are still enough calls (grey bars, log axis). Compare it with the base rate: a call is only useful if it beats how often the event happens anyway. |
| Across stocks | Funnel plot of per-stock bias vs n with ±2σ noise bounds, bias by industry, a sortable table | Stocks outside the funnel have real bias. Expect about 4.5% outside from noise alone, and the In short box shows that expected count. Box- or lasso-selecting a point opens it in Stock detail. |
| By day | Cross-sectional PIT bias per day with the noise band, regime days marked, and an optional event's came-true − said per day | Regime days are days when the whole market moved against the forecast together. Calibration errors grouped on a few days point to a market-level factor, not a per-stock fault. |
| Stock detail | One stock: fan chart (P5–P95 and P25–P75 bands, P50, actual dots; red dots fall outside the outer band), PIT histogram, ladder, that stock's grid | A flat PIT histogram means calibrated. Mass on the left means actuals fell below the forecast (optimistic). A U shape means the bands are too narrow. |
| Forecast → Price ranges | Pick a forecast day (newest first; *open* = not graded yet, *graded* = actuals in). Screener on that `forecast_<date>.parquet`. Sliders: percentile band (default P10–P90), one-sided confidence (floor/cap $), max band width %, min price, and side / move / P ≥. The table gives each stock's VWAP range in $ (low, median, high, floor, cap, width), P(move) and its historical came-true rate in the same probability bucket. One stock: price-range chart, every percentile in $ and %, the $ ladder and the probability-per-move curve against history. On a graded day the table adds `actual_vwap` and `inside`, and the chart draws the actual. CSV download | The low–high range is where the model expects tomorrow's VWAP with the chosen central probability. Read "model says 72%" next to "this stock came true 58% (n=31) in the 70–80% bucket". That historical rate is the answer to "how far do I trust this number". |
| Forecast → Forecast vs actual | Defaults to the day picked in Price ranges, if graded. Once a forecast day has happened and `tft-vwap-grade` has run: band coverage vs nominal, misses above/below, direction hit rate, bias, quantile skill vs naive, event calls said vs came true. Median-vs-actual scatter, PIT histogram, coverage and bias by day (with more than one graded day), a graded table with `outside_by`, and one stock's band and actual in $ for each graded day | Coverage near nominal with misses split evenly means the bands are honest. More misses above than below (negative bias) means the model was too pessimistic that day. One day is one market regime, so judge it over several days. |
| Compare runs | Reliability overlay for one event across runs, a summary table, and calls by run and side at each call level | In short names the winner per metric (or says there is no clear winner), which run's calls held best at the chosen call level, and warns when the rows differ or are too few. By default only (stock, day) rows present in every run are used, so the comparison is fair. Lower pinball and ECE are better. Coverage should be near nominal and bias near 0. |

## Layout

```
forecast_dash/
  contract.py   discover runs, load + validate, convert to fractional moves, panel fallback join
  calib.py      pure numpy/pandas maths (above)
  charts.py     pure functions: tables -> plotly Figures
  insights.py   pure functions: tables -> ranked plain-English findings (the In short boxes)
  help_text.py  the ⓘ text: per page, per chart / table / sub-panel, per column
  app.py        Streamlit UI: sidebar, tabs, caching (st.cache_resource, per run and per move)
  cli.py        forecast-dash entry point -> streamlit run app.py
tests/          maths, contract, charts, insights, AppTest smoke over every tab, import boundary
```
