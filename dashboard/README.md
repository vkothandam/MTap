# forecast-dash: interactive calibration dashboard for quantile forecasts

This Streamlit + Plotly app grades next-day move forecasts. For each forecast it asks: when the
model said P(move ≥ +1%) = 70%, how often did it happen? You can slice that by stock,
industry, day range, move size and probability band, and compare runs side by side.

The project is self-contained. It imports nothing from `ml/` or `sourcing_py`; a test enforces
that. Its only input is a run directory on disk (the **forecast-run contract** below), so it can
move to its own repo or into MBin unchanged. Any model that writes these files is graded the
same way, so the modelling can change and every run still gets the same evaluation.

```bash
cd dashboard
uv sync --extra dev
uv run forecast-dash --runs ../ml/runs          # opens http://localhost:8501
uv run forecast-dash --runs ../ml/runs/<run_id> --port 8502
uv run pytest -q && uv run ruff check .
```

`--runs` can also come from `FORECAST_DASH_RUNS`. It may point at a single run or at a
directory of runs; with a directory of runs, the newest is shown first.

## Forecast-run contract

```
<run>/fold_<k>/test_predictions.parquet   one row per (symbol, forecast day); one file per test fold
    required: symbol, day_idx, prev_vwap, actual, q05..q95   (quantiles and actual in price units)
    optional: date, industry_code
<run>/config.json                         optional; {"config": {"panel": "<path>", ...}}
<run>/forecast_<date>.parquet             optional; tomorrow: symbol, q05..q95 as fractional
                                          moves (0.012 = +1.2%), optional industry_code, today_vwap
```

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

| Tab | What it shows | How to read it |
| --- | --- | --- |
| Calibration | Reliability curves (one per event), deviation heatmap, ladder bars | Points on the diagonal are calibrated. Below it, or red cells, mean the model said more than happened (overconfident). Blue cells mean it was too cautious. Bubble size grows with n, and the error bars are 95% CIs. |
| Threshold explorer | Pick an event and a call level c. Headline figures for calls with P ≥ c, a hit-rate-vs-c curve, a hit-rate-vs-move curve, and a per-stock table with min-calls and min-accuracy filters | Look for the call level where the hit rate is high enough and there are still enough calls (grey bars, log axis). Compare it with the base rate: a call is only useful if it beats how often the event happens anyway. |
| Across stocks | Funnel plot of per-stock bias vs n with ±2σ noise bounds, bias by industry, a sortable table | Stocks outside the funnel have real bias. Expect about 4.5% outside from noise alone, and the header shows that expected count. Box- or lasso-selecting a point opens it in Stock detail. |
| By day | Cross-sectional PIT bias per day with the noise band, regime days marked, and an optional event's came-true − said per day | Regime days are days when the whole market moved against the forecast together. Calibration errors grouped on a few days point to a market-level factor, not a per-stock fault. |
| Stock detail | One stock: fan chart (P5–P95 and P25–P75 bands, P50, actual dots; red dots fall outside the outer band), PIT histogram, ladder, that stock's grid | A flat PIT histogram means calibrated. Mass on the left means actuals fell below the forecast (optimistic). A U shape means the bands are too narrow. |
| Tomorrow | Screener on `forecast_<date>.parquet` by side, move and P ≥. It shows each stock's historical came-true rate in the same probability bucket (that stock and all stocks), plus one stock's probability-per-move curve against history and its ladder | Read "model says 72%" next to "this stock came true 58% (n=31) in the 70–80% bucket". That historical rate is the answer to "how far do I trust this number". |
| Compare runs | Reliability overlay for one event across runs, and a summary table | By default only (stock, day) rows present in every run are used, so the comparison is fair. Lower pinball and ECE are better. Coverage should be near nominal and bias near 0. |

## Layout

```
forecast_dash/
  contract.py   discover runs, load + validate, convert to fractional moves, panel fallback join
  calib.py      pure numpy/pandas maths (above)
  charts.py     pure functions: tables -> plotly Figures
  app.py        Streamlit UI: sidebar, tabs, caching (st.cache_resource, per run and per move)
  cli.py        forecast-dash entry point -> streamlit run app.py
tests/          maths, contract, charts, AppTest smoke over every tab, import boundary
```
