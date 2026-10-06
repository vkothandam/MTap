"""The "About this page" text for each page: key -> (title, markdown)."""

TERMS = """
**Words used on every page**
- **forecast**: for one stock and one test day, the model's 19 quantiles (P5 … P95) of the
  next day's VWAP move.
- **event**: a move of at least +m (`>= +1.0%`, upside) or at most −m (`<= -1.0%`, downside).
  The forecast gives every event a probability. It is linear between the quantiles and capped to
  5–95%.
- **said**: the probability the model gave. **came true**: how often the event happened.
- **call at P ≥ c**: every forecast that gave the event at least probability c.
- **base rate**: how often the event happened across all forecasts. A call is only useful if it
  comes true more often than this.
- **deviation** = came true − said. Below 0 means **overconfident**, above 0 means **too
  cautious**.
- **PIT** u: where the actual move fell in the forecast distribution (0 = below P5, 1 = above
  P95). It should be uniform.
  **bias** = 0.5 − mean(u). Above 0 is **optimistic**: actual moves landed below the forecast.
  Below 0 is **pessimistic**. **z** is the bias in noise units; |z| ≥ 2 is flagged.
- **fold**: one block of test days. The model never trained on these days.
"""

HELP: dict[str, tuple[str, str]] = {
    "overview": ("Forecast calibration dashboard", f"""
This dashboard grades the forecasts of one or more training runs. For each (stock, test day) the
model forecast a distribution of the next day's move. The question is always the same: *when the
model said 70%, did it happen about 70% of the time?*

**Sidebar (applies to every page)**
- **Run**: a training run directory (`fold_*/test_predictions.parquet`).
- **Folds**: which test blocks to include. The folds are consecutive time periods, so picking
  one fold looks at one market period.
- **Side / Move sizes / Extra move size**: the events to score. Each size is used as +m and/or
  −m, depending on Side.
- **Probability bucket width**: how finely forecasts are grouped by what the model said (5% or
  10%).
- **Industries / Stocks / Test days**: narrow the forecasts. Empty means all.
- **Min n per grid point**: hide grid cells with fewer forecasts than this.

**Pages**: *Summary* gives the verdict on the run. *Calibration* breaks it down by event and
probability. *Threshold explorer* asks at which call level an event becomes reliable. *Across
stocks* and *By day* find where the misses come from. *Stock detail* covers one stock.
*Tomorrow* screens the latest forecast. *Compare runs* puts runs side by side.

Each page starts with an **In short** box: findings computed from the current filters.
✅ fine · ⚠️ watch · ❌ problem · ℹ️ context.
{TERMS}"""),

    "summary": ("Summary", """
The executive summary of the selected run and filters.

**Controls**
- **Call levels**: the confidence levels to grade, e.g. 60/70/80/90%.
- **Headline call level**: the level the *In short* box talks about.

**Top numbers**
- **forecasts / stocks**: rows after the sidebar filters.
- **quantile skill**: how much lower the quantile (pinball) loss is than a naive "no move"
  forecast. It grades the whole distribution. Above 0 beats naive.
- **P5–P95 coverage**: share of actual moves inside the outer band. Target 90%. Below the target
  means the bands are too narrow.
- **bias**: 0.5 − mean PIT over all forecasts. + optimistic, − pessimistic. Within ±0.01 is
  small.
- **probability error (ECE)**: the average |came true − said| over the calibration grid of the
  selected moves, weighted by n.

**Call reliability (event × call level)**: each cell is the calls with P(event) ≥ the call
level. The text is *came true / said*. The colour is the deviation: red = overconfident (came
true less often than said), blue = too cautious, white = about right. Hover for the call count.

**Calls by side**: the same, averaged over the selected moves of each side and weighted by
calls. Solid line = came true, dotted = said, dashed = base rate. A solid line above the dashed
one means the calls carry information. Solid close to dotted means the number can be taken at
face value.

**Quantile calibration**: for each forecast quantile Pτ, the share of actual moves that landed
below it. Ideal is the diagonal (P30 → 30%). Below the diagonal at the low quantiles and above it
at the high ones means the bands are too narrow.

**Fold stability**: the same scores per test block, with each side's calls at the headline
level. Pooled numbers can hide a model that is fine in one period and off in another. If the
folds disagree, read the pooled tables with care.

**Calls table**: every (event, call level) row.
- `n`: calls; `share`: their share of all forecasts
- `mean_prob`: said; `hit_rate`: came true; `ci_lo` / `ci_hi`: its 95% interval
- `base_rate`; `deviation`
- `lift` = came true / base rate. Above 1 means the call beats picking at random.
"""),

    "calibration": ("Calibration", """
Does each probability mean what it says? This page groups the forecasts by event and by
**probability bucket** (what the model said: 5–10%, 10–20%, …, 90–95%).

**Said vs came true (reliability diagram)**
- One line per event. Each point is a bucket.
- x = mean probability the model said; y = how often it came true.
- The point size grows with the number of forecasts, and the error bars are 95% intervals.
- On the dashed diagonal = calibrated. **Below** it = overconfident. Above it = too cautious.

**Deviation heatmap**: rows = events, columns = buckets. The cell text is *came true / said*.
The colour is came true − said: **red = overconfident**, **blue = too cautious**, white = about
right. The darker the colour, the bigger the gap. Cells with fewer than *Min n* forecasts are
hidden.

**Ladder**: for a confidence c, the model's own c-move is P(1−c) on the upside and P(c) on the
downside. "70% sure of at least +x%" means x is P30. The bars show how often the actual move
cleared that move, per side. Ideal = the dashed line (c). Bars below it mean the model's
confident moves are too far out.

**Grid table**
- `move`, `side`, `event`, `bucket`, `n`
- `mean_prob` (said), `hit_rate` (came true), `deviation`
- `z` = deviation in binomial noise units. |z| < 2 is within noise.
- `ci_lo` / `ci_hi`: 95% Wilson interval on the hit rate
"""),

    "threshold": ("Threshold explorer", """
At what call level does an event become reliable, and how many calls are left at that level?

**Controls**: the event, and the call level c (P(event) ≥ c, from 5% to 95%).

**Top numbers**
- **calls**: forecasts at or above the level, and their share of all forecasts.
- **came true** with its 95% interval.
- **model said (mean)**: the mean probability of those calls.
- **base rate**: how often the event happened over all forecasts.

**Hit rate as the call level rises**: the x-axis sweeps c from 5% to 95%.
- blue line = came true, with a shaded 95% band
- orange dotted = model said
- grey dashed = base rate
- grey bars (right axis, log scale) = number of calls

A good model has blue tracking orange and rising well above the base rate. If blue flattens or
falls at the high levels, the model's most confident calls are not more accurate.

**Hit rate by move size**: at the chosen call level, the came-true rate for every selected move,
one line per side. The dashed line marks the call level.

**Stocks table**: the calls at this level, grouped by stock.
- `calls`, `came_true`, `said`, `deviation = came_true − said`
- **Min calls per stock** and **Min came-true rate** filter the table.
- With few calls per stock, a high came-true rate can be luck. Check `calls`.
"""),

    "stocks": ("Across stocks", """
Is any stock (or industry) consistently forecast too high or too low, beyond what chance would
produce?

**Per-stock bias vs sample size (funnel)**
- One dot per stock: x = its test forecasts, y = bias (0.5 − mean PIT).
- The dashed grey lines are ±2σ = ±2·√(1/(12n)), the range a calibrated stock stays inside 95%
  of the time.
- Dots outside are flagged: red = optimistic (actual moves landed below the forecast), blue =
  pessimistic. Grey dots are within noise.
- With thousands of stocks about 4.5% land outside by chance. Compare the flagged count with the
  "expected from noise" count above the chart.
- Box- or lasso-select dots to open the first one in *Stock detail*.

**Bias by industry**: the same bias per industry. The error bars are ±2σ. A bar whose error bar
does not reach 0 is a real lean. Red = optimistic, blue = pessimistic.

**Table**
- `n`, `mean_pit` (0.5 = centred), `bias`
- `bias_z`: bias in noise units; |z| ≥ 2 is flagged
- `label`: calibrated / optimistic / pessimistic
- **Min forecasts per stock** drops thin stocks.
"""),

    "days": ("By day", """
Do the misses come in market-wide waves? Each day's forecasts across all stocks are pooled.

**Top panel**
- Dark line: that day's PIT bias (+ optimistic, − pessimistic).
- Grey shaded band: ±2σ for that day's number of forecasts.
- Red / blue dots: **regime days** (|z| ≥ 2), when the whole cross-section missed the same way.
  Red = optimistic, the market fell short of the forecasts. Blue = pessimistic, the market beat
  them.
- **Overlay event** (optional): a green line with that event's came true − said per day.

**Bottom panel**: forecasts per day.

A per-stock model cannot foresee market-wide shocks, so regime days are expected. Clusters of
same-colour days point to a regime the model has not adapted to.

**Regime days table**: `day`, `n`, `mean_pit`, `bias`, `bias_z` (with the overlay, also
`mean_prob`, `hit_rate`, `deviation`).
"""),

    "stock": ("Stock detail", """
One stock's test forecasts. Pick it here, or select it in *Across stocks*.

**Header**: industry, number of test forecasts, bias and z, and the label.

**Forecast range vs actual (fan chart)**: the x-axis is test days and the y-axis is the next-day
move.
- light band = P5–P95; darker band = P25–P75; blue line = P50
- dots = actual moves: black inside P5–P95, **red outside**
- About 10% red dots is expected. Many more means the bands are too narrow for this stock.

**PIT histogram**: where each actual move fell in its forecast, in 10 bins. Flat at 10% (dashed
line) is calibrated.
- Heavy on the left: the actual was below the forecast, so the model is optimistic.
- Heavy on the right: pessimistic.
- U-shaped: the bands are too narrow. A hump in the middle: the bands are too wide.

**Ladder, heatmap, said vs came true**: the same charts as *Calibration*, for this stock only.
Per stock most cells are thin, so check `n` in the hover.
"""),

    "tomorrow": ("Tomorrow", """
The latest frozen-model forecast (`forecast_<date>.parquet`) next to how similar calls fared in
the test folds.

**Controls**: side, move size and call level. The list keeps stocks with P(event) ≥ the level.

**Screener table**
- `prob`: tomorrow's probability of the event; `bucket`: its probability bucket
- `hist_n` / `hist_hit_rate` / `hist_deviation`: **this stock's** test forecasts in the same
  bucket for the same event, how often they came true, and came true − said
- `all_n` / `all_hit_rate` / `all_deviation`: the same for **all stocks**

Read a row as: "the model says 72%. This stock's 70–80% calls came true 55% (n=31); all stocks'
came true 66%." Trust the number more when `hist_hit_rate` and `all_hit_rate` are close to
`prob`, and the `n` values are not tiny.

**Min historical n / Min historical came-true**: keep only stocks whose own history backs the
call.

**One stock: tomorrow's spread vs its history**
- lines = the model's probability per move size (solid upside, dotted downside)
- diamonds = this stock's past came-true rate in the same bucket
- crosses = all stocks' rate

Below that are the stock's ladder (the move at 60–90% confidence) and the per-move table.
"""),

    "compare": ("Compare runs", """
Which training run's forecasts are better? Use this after changing the model or the training.

**Only rows present in every chosen run**: keep the comparison fair by scoring each run on the
same (stock, day) rows. Untick it when the runs tested different folds (they share no rows).

**In short**
- the winner on each metric, and which run wins most often
- whose calls at the chosen call level held up best

**Said vs came true by run**: the reliability curve for one event, one line per run. The line
closer to the diagonal is better calibrated.

**Calls by run**: for each run and side, the calls at each level: came true (solid) vs said
(dotted).

**Metrics table** (one column per run)
- `rows`
- `pinball` / `naive_pinball`: quantile loss of the model and of "no move". Lower is better.
- `pinball_skill` = 1 − pinball / naive. Higher is better.
- `coverage` vs `coverage_nominal`: share inside P5–P95 vs the 90% target
- `mean_pit` / `bias`: overall lean; 0 is best
- `median_width`: typical P5–P95 width. Narrower is only better at equal coverage.
- `ece`: average |came true − said| over the grid. Lower is better.
"""),
}
