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
uv run forecast-dash --runs ../ml/runs --intraday-runs ../ml/runs/intraday   # the default
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

## Sections

The sidebar **Section** radio switches between three views (it follows `?section=home`,
`cross-day` or `intraday`, so a link opens one):

- **Home**: one card per section with the newest run's headline numbers and an *Open ...*
  button. Cross-day shows P10-P90 coverage, pinball skill, direction hit, the latest forecast day
  and the live-grade line. Intraday shows, for the 1 / 5 / 15 / 60 minute horizons, coverage,
  skill vs climatology and vs momentum, and direction hit vs the last price (from
  `report/by_target.parquet` if the run has it, else computed). A card says so when there is no run (`tft-intraday` has not
  been run).
- **Cross-day**: the next-day forecast tabs below, with their own sidebar filters.
- **Intraday**: six tabs for the intraday model, described under *Intraday* below.

## Sidebar (applies to every Cross-day tab)

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
| Compare runs | Reliability overlay for one event across runs, a summary table, and calls by run and side at each call level | In short names the winner per metric (or says there is no clear winner), which run's calls held best at the chosen call level, and warns when the rows differ or are too few. By default only (stock, day) rows present in every run are used, so the comparison is fair. Lower pinball and ECE are better. Coverage should be near nominal and bias near 0. The table also has `vs_first` (pinball vs the first chosen run), direction (all moves and \|move\| > 2%), rank IC, median error, P10–P90 coverage, the quantile calibration error and its grade (≤ 0.5 pts excellent, ≤ 1 good, ≤ 2 fair, else poor). A models table and an ⓘ per run show each run's `MODEL.md` card. |

## Intraday

Intraday runs are found under `<runs>/intraday` by default; override with
`--intraday-runs DIR` or `FORECAST_DASH_INTRADAY_RUNS`. The sidebar picks the run, the **target**
and the **reference**.

- Targets: *Next 1 / 5 / 15 / 60 min* (VWAP of the next minutes, cut at the close), *Rest of day*
  (VWAP from t to the close) and *Whole-day VWAP* (what the daily model forecasts).
- Reference: *Trailing VWAP* (default), *Last price* or *Session VWAP*. Quantiles and outcomes are
  shifted onto it, so skill vs the random walk ("the VWAP ahead equals the reference") and the
  direction call change; coverage and pinball loss do not.

### Intraday run contract (`forecast_dash/intraday_contract.py`)

```
<run>/config.json                          model == "tft_intraday", targets, horizon_s, folds
                                           (test_dates per fold), daily_run, step_s, ...
<run>/fold_<k>/intraday_<target>.parquet   one per target; a row per decision point
                                           (symbol, date, t_s) on the fold's test dates
<run>/report/by_target.parquet             optional, precomputed per-target scores
```

Columns of `intraday_<target>.parquet`: `symbol`, `date` (YYYY-MM-DD), `t_s` (seconds since the
09:30 ET open), `tod_bucket` (15-minute slot 0..25), `sector_code` (may be null), `rvol`, `mask`
(1 = valid, other rows are ignored), `actual`, `q05..q95` (19 quantiles), `clim_q05..clim_q95`
(climatology baseline; older runs lack it and fall back to momentum), `mom_q05..mom_q95`
(momentum baseline), `ref_trail`, `ref_session`, `y`, `last` ($ at t), `prev_vwap` ($). For
`r_close` also `daily_q05..daily_q95` and `has_daily`. **`actual`, `q*`, `clim_q*`, `mom_q*` and `ref_*` are
log moves from the last price at t** (the last price is 0). `ref_trail` is the trailing VWAP of
the horizon's length (for `r_rod`: the last 60 minutes; for `r_close`: the session VWAP so far), `ref_session` the session
VWAP so far.

`report/by_target.parquet` columns: `target`, `model` (`tft` | `climatology` | `momentum` |
`daily`), `n`, `coverage80`, `pinball`, `skill_clim`, `skill_rw`, `skill_mom`, `skill_daily`, `dir_last`,
`dir_trail`, `dir_session`, `pit_bias`, `width80_med` (+ `_lo` / `_hi` bootstrap bounds). Home
reads the `tft` rows; nothing else needs the report.

### Intraday tabs

Each opens with an **In short** box, and every chart, table and column has its ⓘ / tooltip, as in
Cross-day.

| Tab | What it shows | How to read it |
| --- | --- | --- |
| Summary | One row per target (decision points, P10-P90 coverage, pinball, skill vs climatology, random walk and momentum, direction hit vs last / trailing / session VWAP, PIT bias, median width), a skill-by-horizon chart, and the chosen target's skill by test day | Coverage near 80% means the bands are right. Skill above 0 beats the baseline; climatology is the one that matters, momentum and the random walk are easy bars. Direction 50% is a coin flip; the one vs the trailing VWAP flatters any forecast near the last price. A bad day usually is a market-wide move. |
| Calibration | Share of outcomes below each quantile, and P(above reference) said vs came true | Points on the diagonal are calibrated. Below the diagonal in the probability chart means overconfident. |
| Time of day | Per 15-minute slot: coverage, band width, skill and direction hit | The open and close differ from midday. A slot with coverage far from 80% or negative skill is where the model needs work. Thin slots are noisy. |
| Session replay | Pick a test day and stock: last price, session VWAP so far, the VWAP that followed, and the predicted P10-P90 band and median at each decision time, in $. For *Whole-day VWAP* also the morning daily band | The red line (what followed) should stay inside the blue band about 80% of the time. See how the band narrows or shifts as the day reveals prices. |
| Across stocks | Per stock and per sector: coverage and skill, a coverage-vs-skill chart, and a *min decision points* filter | Left of the 80% line the stock's bands are too narrow; below 0 the model loses to climatology. |
| Intraday vs daily | Always the whole-day VWAP: pinball loss of the intraday forecast and of the morning daily forecast by time of day, and the intraday skill | The intraday model should win more as the day goes on. Only decision points with a morning forecast (`has_daily`) count. |

## Layout

```
forecast_dash/
  contract.py   discover runs, load + validate, convert to fractional moves, panel fallback join
  calib.py      pure numpy/pandas maths (above)
  charts.py     pure functions: tables -> plotly Figures
  insights.py   pure functions: tables -> ranked plain-English findings (the In short boxes)
  help_text.py  the ⓘ text: per page, per chart / table / sub-panel, per column
  app.py        Streamlit UI: Section switch, Cross-day sidebar and tabs, caching
  ui.py         shared Streamlit helpers (_chart, _table, _in_short, ...), run caches, sections
  home.py       Home page: one card per section
  intraday_page.py     the Intraday section (six tabs)
  intraday_contract.py discover / load intraday runs
  intraday_calib.py    intraday scores, built on calib.py (pure)
  cli.py        forecast-dash entry point -> streamlit run app.py
tests/          maths, contract, charts, insights, AppTest smoke over every tab, import boundary
```

## Model cards (`MODEL.md`)

A run directory may hold a `MODEL.md`. Its front matter gives `label` (R0, R1c, …), `forecast`
(night, 09:30, 09:45, 09:30/09:45 or intraday) and a one-line `summary`; the markdown below it
holds the purpose, inputs, folds, result and command. The dashboard then shows the run as
"R1c · <directory>" in the pickers (the daily run picker and the Intraday run picker), names it by
its label in Compare runs, lists the chosen models in a table, and opens the card from an ⓘ. A
run without a card is shown by its directory name. Every trainer (`tft-vwap`, `tft-vwap-gbm`,
the overlays, `tft-intraday`, `screener-train`) writes the card itself (`--label`, `--about`,
`--purpose`); `ml/scripts_model_cards.py` adds hand-written verdicts to the research runs.

```markdown
---
label: R1c
forecast: night
summary: BASE: R0 + session shape + after-hours (no market, no macro)
---

**Purpose.** ...
```
