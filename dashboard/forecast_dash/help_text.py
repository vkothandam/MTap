"""The ⓘ help text: each page (HELP), each chart / table / sub-panel (PANELS) and each table
column (COLUMNS, column_help)."""

import re

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
*Forecast* shows a day's VWAP ranges and grades the days that have happened. *Compare runs* puts runs side by side.

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

    "tomorrow": ("Forecast: price ranges", """
A frozen-model forecast (`forecast_<date>.parquet`) as a **VWAP price range** per stock, next
to how similar calls fared in the test folds.

**Forecast day**: every forecast in the run, newest first. *open* = the day hasn't been graded
yet (normally the next session); *graded* = its actuals are in, so the table adds
`actual_vwap` and `inside` (actual within the band) and the one-stock chart draws the actual.

**Price range controls**
- **Percentile band**: the low and high percentile of the range. P10–P90 = an 80% central
  range: the model gives 10% to VWAP below `low` and 10% above `high`.
- **One-sided confidence**: `floor` = the VWAP the model is that confident it stays **above**,
  `cap` = the VWAP it is that confident it stays **below**.
- **Max band width / Min price**: drop wide-range or low-priced stocks.

Prices are `ref_vwap × (1 + move)`. `ref_vwap` is the VWAP of the session the forecast was made
from (split-adjusted), and the move quantiles are what the model forecasts.

**% deviation controls**: side, move size and P(event) ≥. The table keeps stocks with
P(event) ≥ the level (set it to 0 to see every stock).
- `prob`: the probability of the event; `bucket`: its probability bucket
- `hist_n` / `hist_hit_rate` / `hist_deviation`: **this stock's** test forecasts in the same
  bucket for the same event, how often they came true, and came true − said
- `all_n` / `all_hit_rate` / `all_deviation`: the same for **all stocks**

Read a row as: "the model says 72%. This stock's 70–80% calls came true 55% (n=31); all stocks'
came true 66%." Trust the number more when `hist_hit_rate` and `all_hit_rate` are close to
`prob`, and the `n` values are not tiny.

**One stock**
- the forecast VWAP at every percentile, with the chosen band shaded and the reference VWAP
- a table of every percentile in $ and %, and the one-sided ladder in $
- the probability-per-move curve against its history (diamonds = this stock, crosses = all)
"""),

    "tomorrow_actual": ("Forecast vs actual", """
Each live forecast once its day has closed: what the model said for the day against what
happened. Actuals come from `actuals.parquet`, written by the model's grade step (for ml/:
`uv run tft-vwap-grade <run>`, also run by every `tft-vwap-predict`) after the panel is
refreshed with that day's bars.

**Controls**: the graded day, the percentile band to grade, and an event (side, move, P ≥) to
check said vs came true.

**Metrics**
- **band held**: share of stocks whose actual VWAP landed inside the band. Compare with the
  band's nominal width (P10–P90 = 80%).
- **direction right**: the forecast median had the same sign as the actual move
- **bias**: 0.5 − mean PIT. + optimistic (actuals below the forecast), − pessimistic
- **quantile skill**: pinball loss vs a naive "no move" forecast; > 0 beats it
- **event calls**: of the stocks with P(event) ≥ the level, the mean probability said and the
  share where it came true

**Charts**: PIT histogram (flat = calibrated); forecast median vs actual move per stock (red =
outside the band); and, with more than one graded day, band coverage and bias by day.

**Table**: per stock the forecast `low`/`median`/`high` VWAP, the actual VWAP and move,
`inside`, `pit` (where the actual fell in the forecast distribution) and `outside_by` (how far
past the band edge, as a move).

**One stock**: its band and actual VWAP over every graded day, in $.

A single day is one market regime: stocks move together, so coverage on one day swings with
the market. Judge the model on several graded days.
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

**Metrics table** (one row per run)
- `rows`
- `pinball` / `naive_pinball`: quantile loss of the model and of "no move". Lower is better.
- `pinball_skill` = 1 − pinball / naive. Higher is better.
- `coverage` vs `coverage_nominal`: share inside P5–P95 vs the 90% target
- `mean_pit` / `bias`: overall lean; 0 is best
- `median_width`: typical P5–P95 width. Narrower is only better at equal coverage.
- `ece`: average |came true − said| over the grid. Lower is better.
"""),
}

# ---- per panel: the ⓘ next to every chart, table and sub-panel ------------------------------

PANELS: dict[str, tuple[str, str]] = {
    # Summary
    "sum_heatmap": ("Calls: came true / said", """
One cell per **event** (row) and **call level** (column). A call at P ≥ c is every forecast
that gave the event at least probability c.

- Cell text is `came true / said`: how often those calls happened vs the average probability the
  model gave them.
- Colour is the deviation (came true − said). **Red** means the model said more than happened
  (overconfident). **Blue** means it was too cautious. White is about right.
- Cells with fewer calls than *Min n per grid point* (sidebar) are hidden.
"""),
    "sum_sides": ("Calls by side", """
The calls table rolled up per **side** (upside events vs downside events), averaged over the
selected moves and weighted by the number of calls.

- **Solid**: came true. **Dotted**: what the model said. **Dashed**: the base rate, how often
  the event happens anyway.
- Good calls have the solid line on the dotted one (honest) and well above the dashed one
  (informative).
- One forecast can count under several moves, so `n` counts (call, move) pairs.
"""),
    "sum_quantiles": ("Quantile calibration", """
For every forecast quantile Pk: the share of actual moves that landed **below** it.

- Calibrated forecasts sit on the diagonal: 10% of actuals below P10, 50% below P50, and so on.
- A line **above** the diagonal means the actuals land low, so the forecast is optimistic.
  **Below** it means they land high (pessimistic).
- An S shape (above on the left, below on the right) means the bands are too narrow.
"""),
    "sum_folds": ("Fold stability", """
The same headline numbers, one row per **fold**. A fold is one block of test days the model never
trained on, so the rows are separate time periods.

If the folds agree, the pooled numbers can be trusted. If they don't, the run mixes market regimes
and the pooled figures average them away. Each side's calls use the **Headline call level**.
"""),
    "sum_calls": ("Calls table", """
The numbers behind the heatmap and the side chart: one row per (event, call level). Hover a
column header for its meaning.

How to read a row: *of the `n` forecasts that gave this event at least `confidence`, the model
said `mean_prob` on average and `hit_rate` came true (95% CI `ci_lo`–`ci_hi`). The event
happens `base_rate` of the time anyway, so the call is `lift`× more likely than chance.*
"""),
    # Calibration
    "cal_reliability": ("Said vs came true", """
The reliability curve. Forecasts are grouped by the probability the model gave (the bucket), one
line per event.

- **x**: the average probability said in the bucket. **y**: how often the event came true.
- Points on the **diagonal** are calibrated. **Below** it means overconfident, **above** it means
  too cautious.
- Bigger markers mean more forecasts. Error bars are the 95% CI, so a point whose bar crosses the
  diagonal is within noise.
"""),
    "cal_heatmap": ("Deviation heatmap", """
The same grid as the reliability curve: one cell per (event, probability bucket).

- Cell text is `came true / said`. Colour is came true − said: **red** = overconfident,
  **blue** = too cautious.
- Look for patterns: a red band at high buckets means the model is too sure when it is most
  sure.
"""),
    "cal_ladder": ("Ladder", """
At each confidence c, every forecast gives its own **upside ladder move** (VWAP ≥ the
(1 − c) quantile) and **downside ladder move** (VWAP ≤ the c quantile). If the forecast is
calibrated, the actual clears each one c of the time.

- **Bars**: the share of forecasts whose actual cleared its own ladder move, with 95% CI.
- **Dashed line**: the ideal (= c).
- Bars below the line mean the ranges are too narrow at that confidence.
"""),
    "cal_grid": ("Grid table", """
One row per (event, probability bucket): the data behind the reliability curve and the
heatmap. Hover a column header for its meaning.

`z` says whether a deviation is real: |z| ≥ 2 is unlikely to be noise at this `n`.
"""),
    # Threshold explorer
    "thr_curve": ("Hit rate as the call level rises", """
Sweep the call level c from 5% to 95%. For the calls with P(event) ≥ c:

- **blue**: how often they came true, with its 95% CI band
- **orange dotted**: what the model said on average
- **grey dashed**: the base rate, how often the event happens anyway
- **grey bars** (right axis, log scale): how many calls are left

Pick the level where the hit rate is high enough and the bars still show enough calls.
"""),
    "thr_move": ("Hit rate by move size", """
At the chosen call level: for each move size, how often the calls came true, per side. The
dashed line is the call level.

Points below the line mean that, for that move size, the calls came true less often than the
level promised. Bigger moves are rarer and usually have fewer calls, so their error bars are
wider.
"""),
    "thr_stocks": ("Stocks", """
The calls at the chosen event and call level, split by stock. Use the sliders to keep stocks with
enough calls and a high enough hit rate.

A high `came_true` on few `calls` can be luck. Look for stocks with both.
"""),
    # Across stocks
    "st_funnel": ("Funnel: per-stock bias vs sample size", """
One dot per stock: its bias (0.5 − mean PIT) against how many test forecasts it has.

- The grey **funnel** is ±2σ of pure noise. It narrows as n grows, because big samples should be
  close to 0.
- Dots **outside** the funnel have a real lean: red = optimistic (actuals below the forecast),
  blue = pessimistic.
- About 4.5% of stocks fall outside from noise alone.
- Box- or lasso-select dots to list them. The first one opens in Stock detail.
"""),
    "st_industry": ("Bias by industry", """
The PIT bias pooled per industry. Bars are ±2σ noise: a bar whose whisker doesn't cross 0 is a
real lean for that industry. **Red** = optimistic, **blue** = pessimistic.
"""),
    "st_table": ("Bias table", """
One row per stock with at least *Min forecasts per stock*, sorted by `bias_z` (most pessimistic
first). Hover a column header for its meaning.
"""),
    # By day
    "day_chart": ("Bias per day", """
Top panel: for each test day, the PIT bias across all selected stocks.

- **grey band**: ±2σ noise for that day's sample size
- **dots**: regime days (|z| ≥ 2), when the whole cross-section missed in the same direction.
  Red = the market fell short of the forecast, blue = it beat it.
- **green line** (with an overlay event): that day's came true − said for the event

Bottom panel: forecasts per day.
"""),
    "day_regimes": ("Regime days", """
The days where |bias_z| ≥ 2: the whole cross-section missed together. That points to a
market-wide move the model couldn't see, not a per-stock fault. Hover a column header for its
meaning.
"""),
    # Stock detail
    "sd_fan": ("Fan chart", """
This stock's forecast over every test day, as a next-day move.

- **light band**: the outer quantiles (P5–P95). **dark band**: P25–P75. **line**: the median
  (P50).
- **dots**: what actually happened. Red dots fell outside the outer band.

About 10% of dots should be red. Many more means the bands are too narrow for this stock.
"""),
    "sd_pit": ("PIT histogram", """
Where each actual fell in its forecast distribution (u = 0: below everything, 1: above).

- **Flat** at the dashed line means calibrated.
- Mass on the **left**: actuals fell below the forecast (optimistic).
- Mass on the **right**: pessimistic.
- A **U** shape means the bands are too narrow. A hump means they are too wide.
"""),
    "sd_ladder": ("Ladder for this stock", """
The Calibration ladder for this stock only. At each confidence c, the bars show how often the
actual cleared its own upside move (VWAP ≥ the (1 − c) quantile) and its downside move (VWAP ≤
the c quantile). The ideal is the dashed line (= c).
"""),
    "sd_heatmap": ("Deviation for this stock", """
The Calibration heatmap for this stock only: came true / said per (event, probability bucket).
**Red** = overconfident, **blue** = too cautious. One stock has few forecasts per cell, so expect
more noise.
"""),
    "sd_reliability": ("Said vs came true for this stock", """
The reliability curve for this stock only. Points on the diagonal are calibrated. The error bars
are wide because one stock has few forecasts per bucket.
"""),
    # Forecast: price ranges
    "tm_table": ("Price-range screener", """
One row per stock in the chosen forecast file that passes the filters, sorted by `prob`.

- **Price columns**: the forecast VWAP range for the forecast day, in $. Every price is
  `ref_vwap × (1 + forecast move)`.
- **`prob`**: the model's probability of the chosen event.
- **`hist_*` / `all_*`**: the track record behind that probability. Over the test period, take
  this stock's forecasts (hist) or all stocks' forecasts (all) whose probability fell in the same
  bucket. How often did the event come true? A big gap between `prob` and `hist_hit_rate` means
  the number deserves less trust for this stock.

Hover a column header for its meaning.
"""),
    "tm_range": ("Forecast VWAP by percentile", """
The picked stock's forecast VWAP at every percentile Pk. The model gives k% chance that the
VWAP ends at or below that price.

- **green band**: the chosen percentile band (e.g. P10–P90 = 80% central range)
- **dashed line**: the reference VWAP the forecast was made from
- **black line** (graded day): the actual VWAP

A steep curve means a wide, uncertain forecast. A flat one means a tight forecast.
"""),
    "tm_pct": ("Every percentile", """
The picked stock's whole forecast distribution: each percentile as a move and as a VWAP price.
Hover a column header for its meaning.
"""),
    "tm_ladder": ("One-sided levels in $", """
For each confidence c:

- **floor**: the VWAP should stay at or above this price with probability c (the (1 − c)
  quantile)
- **cap**: the VWAP should stay at or below this price with probability c (the c quantile)

Each is shown in $ and as a move from the reference VWAP. Higher confidence means a lower floor
and a higher cap.
"""),
    "tm_spread": ("Probability per move vs history", """
For each move size (x), per side:

- **line**: the probability the model gives this stock for the event (up: ≥ +m, down: ≤ −m)
- **diamonds**: how often the event came true for this stock's past test forecasts in the same
  probability bucket
- **crosses**: the same for all stocks

Diamonds near the line mean this stock's probabilities have been honest. Hover a diamond for its
n: a few forecasts prove little.
"""),
    "tm_spread_table": ("Probability per move: numbers", """
The numbers behind the chart: one row per event for the picked stock. Hover a column header for
its meaning.
"""),
    # Forecast vs actual
    "lv_scatter": ("Forecast median vs actual move", """
One dot per stock on the graded day.

- **x**: the forecast median move (P50). **y**: what actually happened.
- The **diagonal** is a perfect median.
- **Red** dots fell outside the chosen band.

A cloud tilted along the diagonal means the median had direction skill. A cloud shifted above or
below means the whole market moved against the forecast.
"""),
    "lv_pit": ("PIT histogram for the day", """
Where each stock's actual fell in its forecast distribution on this day.

- **Flat** means calibrated.
- Mass on the **right** means actuals beat the forecast (pessimistic).
- Mass on the **left** means optimistic.

One day is one market regime, so a lopsided histogram here is normal. Look at it over several
days.
"""),
    "lv_days": ("Graded days", """
Only shown with more than one graded day.

- **Top**: the share of stocks inside the chosen band each day vs its target (the band's width,
  e.g. 80%).
- **Bottom**: that day's PIT bias. Red = optimistic, green = pessimistic.

Days that bounce around the target are fine. A persistent gap is a problem.
"""),
    "lv_table": ("Graded table", """
One row per stock on the graded day, sorted by the size of the miss (`outside_by`). Hover a column
header for its meaning.
"""),
    "lv_fan": ("This stock over every graded day", """
The picked stock's forecast bands and actual VWAP for each graded day, in $.

- **light band**: the outer quantiles. **dark band**: P25–P75. **line**: the median.
- **dots**: the actual VWAP. Red dots fell outside the outer band.
"""),
    "lv_range": ("Forecast VWAP by percentile vs actual", """
The picked stock on the chosen graded day.

- **curve**: the VWAP at each percentile
- **green band**: the chosen band
- **dashed line**: the reference VWAP
- **black line**: the actual VWAP

The actual should land inside the band about as often as its width says.
"""),
    # Compare runs
    "cmp_reliability": ("Said vs came true by run", """
The reliability curve for one event, one line per run. The line closer to the diagonal is better
calibrated. Bigger markers mean more forecasts, and the error bars are 95% CIs.
"""),
    "cmp_table": ("Metrics by run", """
One row per run, scored on the same rows when *Only rows present in every chosen run* is
ticked. Rows match on (stock, date) when every chosen run has real dates, otherwise on
(stock, day_idx); the checkbox help says which.

- Lower `pinball` and `ece` are better. Higher `pinball_skill` is better.
- `coverage` should be near `coverage_nominal`, and `bias` near 0.
- Narrower `median_width` is only better if coverage holds.

Hover a column header for its meaning.
"""),
    "cmp_calls_chart": ("Calls by run and side", """
For each run and side, the calls at each call level: came true (**solid**), what the model said
(**dotted**) and the base rate (**dashed**). A better run has solid close to dotted and well
above dashed.
"""),
    "cmp_calls": ("Calls by run", """
The full calls table for every run: one row per (run, event, call level). Hover a column header
for its meaning.
"""),
}

# ---- per column: header tooltips (and the "Columns" list in each ⓘ) -------------------------

COLUMNS: dict[str, str] = {
    "symbol": "stock ticker",
    "industry_code": "the stock's industry",
    "run": "training run (its directory name)",
    "move": "the event's move size as a fraction: +0.01 = VWAP at least 1% above the reference "
            "(upside), −0.01 = at least 1% below (downside)",
    "side": "up: the event is a rise of at least +m. down: a fall of at least −m",
    "event": "the event in words: `>= +1.0%` = next-day VWAP at least 1% above the reference",
    "confidence": "call level c: a call is every forecast that gave the event at least this "
                  "probability",
    "n": "how many forecasts this row is computed from. Small n means noisy rates and wide CIs",
    "mean_prob": "said: the average probability the model gave the event over these forecasts",
    "hit_rate": "came true: the share of these forecasts where the event actually happened",
    "ci_lo": "low end of the 95% confidence interval (Wilson) for hit_rate. With only n "
             "forecasts the true came-true rate is uncertain; it is very likely at least this. "
             "If mean_prob is below ci_lo, the model was too cautious beyond noise",
    "ci_hi": "high end of the 95% confidence interval (Wilson) for hit_rate: the true "
             "came-true rate is very likely at most this. If mean_prob is above ci_hi, the "
             "model was overconfident beyond noise",
    "base_rate": "how often the event happened across all forecasts, called or not. A call is "
                 "only useful if hit_rate beats this",
    "share": "the share of all forecasts that were calls at this level (n / all rows): how "
             "often the model makes this call",
    "deviation": "came true − said. Below 0: overconfident (said more than happened). Above 0: "
                 "too cautious. 0 is ideal",
    "lift": "hit_rate / base_rate: how many times more often the event happens when the model "
            "calls it. 1 = no better than chance; higher is better",
    "bucket": "the probability band the forecast's P(event) fell in (e.g. 70%-80%)",
    "bucket_i": "the bucket's number, 0 = lowest (for sorting)",
    "z": "deviation in noise units: (hit_rate − mean_prob) / its binomial standard error. "
         "|z| ≥ 2 is unlikely to be noise",
    "fold": "test block: a stretch of days the model never trained on",
    "first_day": "first test day in the fold",
    "last_day": "last test day in the fold",
    "rows": "forecasts (stock × day)",
    "pinball": "quantile (pinball) loss of the forecast, on the move. Lower is better",
    "naive_pinball": "the same loss for a naive forecast of a 0% move at every quantile",
    "pinball_skill": "1 − pinball / naive_pinball: how much lower the quantile loss is than a "
                     "naive 'no move' forecast. Above 0 beats naive; higher is better",
    "coverage": "the share of actual moves inside the outer band (P5–P95). Should be near "
                "the band's width (90%)",
    "coverage_nominal": "the target coverage: P95 − P5 = 90%",
    "median_width": "the typical (median) P5–P95 band width, as a move. Narrower is only "
                    "better if coverage holds",
    "ece": "probability error: the n-weighted average |came true − said| over the grid of the "
           "selected moves. Lower is better",
    "mean_pit": "the average PIT u = forecast CDF at the actual: where the actual fell in the "
                "forecast (0 = below all of it, 1 = above). 0.5 is ideal",
    "bias": "0.5 − mean PIT. Positive: optimistic (actuals landed below the forecast). "
            "Negative: pessimistic. 0 is ideal",
    "bias_z": "bias in noise units: bias / √(1/12/n). |z| ≥ 2 is a real lean, not noise",
    "label": "calibrated (|bias_z| < 2), optimistic (bias_z ≥ 2) or pessimistic (bias_z ≤ −2)",
    "day": "the forecast day",
    "calls": "this stock's forecasts with P(event) at or above the call level",
    "said": "the average probability the model gave on those calls",
    "came_true": "the share of those calls where the event happened",
    "ref_vwap": "reference VWAP: the VWAP of the session the forecast was made from. Every "
                "price = ref_vwap × (1 + forecast move)",
    "width": "the band's width as a move: (high − low) / ref_vwap. 4% = the range spans 4% "
             "of the price",
    "actual_vwap": "the VWAP that printed on the forecast day, on the forecast's price basis "
                   "(a split in between doesn't distort it)",
    "actual_move": "the realised move: actual VWAP vs the reference VWAP",
    "inside": "whether the actual VWAP fell inside the low–high band",
    "pit": "where the actual fell in the forecast: 0.10 = at the forecast's P10, 0.5 = at the "
           "median. 0 or 1 = outside every quantile",
    "outside_by": "how far the actual landed beyond the band, as a move: + above high, − "
                  "below low, 0 inside",
    "prob": "the model's probability of the chosen event for this stock",
    "hist_n": "this stock's past test forecasts whose probability fell in the same bucket",
    "hist_hit_rate": "how often the event came true for those forecasts: this stock's track "
                     "record at a probability this size",
    "hist_deviation": "hist_hit_rate − the average probability said on them. Below 0: the model "
                      "has been overconfident for this stock in this bucket",
    "all_n": "all stocks' past test forecasts whose probability fell in the same bucket",
    "all_hit_rate": "how often the event came true for those forecasts (all stocks)",
    "all_deviation": "all_hit_rate − the average probability said on them (all stocks)",
    "percentile": "forecast percentile Pk: the model gives k% chance that the VWAP ends at or "
                  "below this level",
    "VWAP": "the VWAP price at that percentile = ref_vwap × (1 + move)",
    "VWAP ≥ (floor)": "the VWAP should stay at or above this price with this confidence (the "
                      "(1 − c) percentile)",
    "floor move": "the floor as a move from the reference VWAP",
    "VWAP ≤ (cap)": "the VWAP should stay at or below this price with this confidence (the c "
                    "percentile)",
    "cap move": "the cap as a move from the reference VWAP",
}

# Columns that mean something else in one table: "<panel>:<column>".
COLUMNS.update({
    "tm_pct:move": "the forecast move at that percentile, vs the reference VWAP",
    "tm_ladder:confidence": "one-sided confidence c",
    "lv_table:came_true": "whether the chosen event happened for this stock (1) or not (0)",
    "day_regimes:n": "stocks forecast that day",
    "day_regimes:mean_prob": "the overlay event: average probability said that day",
    "day_regimes:hit_rate": "the overlay event: share of stocks where it happened that day",
    "st_table:n": "test forecasts for the stock",
})

_PATTERNS: list[tuple[str, str]] = [
    (r"low \(P(\d+)\)",
     "the band's low end in $: the model gives {0}% chance the VWAP ends at or below it"),
    (r"high \(P(\d+)\)",
     "the band's high end in $: the model gives {0}% chance the VWAP ends at or below it"),
    (r"median \(P50\)", "the forecast median VWAP in $: an even chance of ending above or below"),
    (r"floor \((\d+%)\)", "the VWAP should stay at or above this price with {0} confidence"),
    (r"cap \((\d+%)\)", "the VWAP should stay at or below this price with {0} confidence"),
    (r"P\((.+)\)", "the model's probability of the event `{0}` for this stock"),
    (r"(up|down)_calls",
     "{0}side calls at the headline call level, counted per (forecast, move) pair"),
    (r"(up|down)_said", "the average probability the model gave those {0}side calls"),
    (r"(up|down)_came_true", "how often those {0}side calls happened"),
    (r"(up|down)_deviation", "{0}side came true − said. Below 0: overconfident"),
]


def column_help(name: str, panel: str | None = None) -> str | None:
    """Plain-English meaning of a table column, or None. `panel` picks a table-specific one."""
    name = str(name)
    if panel and COLUMNS.get(f"{panel}:{name}"):
        return COLUMNS[f"{panel}:{name}"]
    if name in COLUMNS:
        return COLUMNS[name]
    for pat, text in _PATTERNS:
        if m := re.fullmatch(pat, name):
            return text.format(*m.groups())
    return None


# ---- Intraday section ------------------------------------------------------------------------

ID_TERMS = """
**Words used on every intraday page**
- **decision point**: one (stock, day, second t). At t the model forecasts a VWAP that is still
  ahead. Points where no valid answer exists (e.g. too close to the close) are left out.
- **target**: what is forecast. *Next 1 / 5 / 15 / 60 min* = the VWAP of the next minutes (cut at
  the close). *Rest of day* = the VWAP from t to the close. *Whole-day VWAP* = the same number
  the daily model forecasts.
- **forecast**: 19 quantiles (P5 … P95) of that VWAP, as a move from the last price at t.
- **reference**: what the move is measured against. *Trailing VWAP* = the VWAP of the same
  length just before t (rest of day: the last 60 minutes; whole day: the session VWAP so far). *Last price* and
  *Session VWAP* are the other choices. It changes direction and skill-vs-random-walk, not the
  band coverage.
- **random walk**: the naive forecast that the VWAP ahead equals the reference.
- **climatology**: the baseline that matters. The training days' spread of each move from the
  last price, per 15-minute slot, the same for every stock and second. It knows nothing about
  the moment, so skill vs climatology is the model's real edge.
- **momentum**: "the VWAP ahead equals the trailing VWAP", with the training days' spread around
  it. An easy bar: the VWAP ahead stays near the last price while the trailing VWAP lags behind
  it, so anything centred near the last price beats momentum. For the same reason the direction
  call vs the trailing VWAP looks good for any forecast; read the one vs the last price.
- **skill** = 1 − pinball loss / baseline's pinball loss. Above 0 beats the baseline.
- **P10–P90 coverage**: the share of outcomes inside the P10–P90 band. It should be 80%.
"""

HELP.update({
    "id_summary": ("Intraday summary", f"""
How the intraday model did on every target, on the days it never trained on.

**Controls (sidebar)**: the run, the target the other pages follow, and the reference.

The table has one row per target: how many decision points, P10–P90 coverage, pinball loss,
skill vs climatology, the random walk and momentum, how often the median called the direction right
against each reference, PIT bias and the typical band width.
{ID_TERMS}"""),
    "id_calibration": ("Intraday calibration", """
Is the model honest about its own uncertainty, for the chosen target and reference? Two views:
the share of outcomes below each quantile, and the probability the VWAP ahead is above the
reference as said vs what happened.
"""),
    "id_tod": ("Time of day", """
The same scores per 15-minute slot of the session. The open and the close behave differently
from midday, so a pooled number can hide a slot where the bands are too narrow or the model has
no edge.
"""),
    "id_replay": ("Session replay", """
One stock on one day. At each decision time the chart shows the price, the session VWAP so far,
the VWAP that actually followed, and the band the model predicted at that moment. Use it to see
how the forecasts looked at the time, not just their averages.
"""),
    "id_stocks": ("Across stocks", """
The scores per stock and per sector. Look for stocks where the bands are systematically too
narrow, or where the model has no edge over climatology.
"""),
    "id_vs_daily": ("Intraday vs daily", """
The whole-day VWAP is what the daily model forecasts each morning. This page compares the
intraday model's forecast of the same number, made at each time of day, with that morning
forecast on the same outcomes. The intraday model should win more as the day reveals prices.
Only decision points that have a morning daily forecast count.
"""),
})

PANELS.update({
    "id_sum_chart": ("Skill by horizon", """
Per target, the pinball skill vs each baseline: climatology, momentum and the random walk. A bar
above 0 beats that baseline; below 0 is worse than it. Climatology is the one that matters;
momentum and the random walk are easy to beat.
"""),
    "id_sum_table": ("Scores per target", """
One row per target, on the chosen reference. Coverage should be near 80%. Skill above 0 beats
the baseline. The direction columns are how often the median was on the right side of that
reference (50% = coin flip). PIT bias above 0 means outcomes landed below the forecast.
"""),
    "id_sum_days": ("Scores per test day", """
The chosen target's scores per test day. A day with much worse skill or coverage is usually a
market-wide move, not a stock-level fault.
"""),
    "id_sum_days_chart": ("Skill by test day", """
Skill vs each baseline on each test day for the chosen target. Dips are days the model
struggled.
"""),
    "id_cal_quantiles": ("Quantile calibration", """
For each forecast quantile, the share of outcomes that landed below it. Points on the diagonal
are calibrated. Above it: outcomes fell below the quantile more often than promised, so the
forecast is too high (optimistic). Below it: too low.
"""),
    "id_cal_grid": ("P(above reference): said vs came true", """
The model's probability that the VWAP ahead ends above the reference, grouped in probability
buckets, against how often it did. Bubbles on the diagonal are calibrated; bubble size grows with
n and the error bars are 95% intervals. Below the diagonal: overconfident.
"""),
    "id_tod_chart": ("Scores by time of day", """
Four panels over the session. *Coverage*: should sit on the 80% line. *Width*: how wide the
P10–P90 band is. *Skill*: above 0 beats the baseline. *Direction*: above 50% the median called
the side right.
"""),
    "id_tod_table": ("Time-of-day table", """
The numbers behind the chart, one row per 15-minute slot. Slots with few decision points are
noisy.
"""),
    "id_replay": ("Session replay", """
In dollars, over the session. **Grey**: last price. **Green**: session VWAP so far. **Red**: the
VWAP that followed each decision time (what the model tried to forecast). **Blue band**: the
predicted P10–P90 range at that time, dotted line its median. For *Whole-day VWAP* a purple
band shows the morning daily forecast. The red line should stay inside the blue band about 80%
of the time.
"""),
    "id_stocks_chart": ("Coverage vs skill per stock", """
One bubble per stock. Left of the vertical line the bands are too narrow for that stock; right
of it too wide. Below the horizontal line the model loses to climatology (momentum for runs
without it). Bubble size grows with n.
"""),
    "id_stocks_table": ("Scores per stock", """
One row per stock, best skill vs climatology first. Stocks with few decision points are noisy.
"""),
    "id_stocks_sectors": ("Scores per sector", """
The same scores grouped by sector. A sector with poor coverage points to a shared cause.
"""),
    "id_vs_chart": ("Intraday vs daily pinball", """
Whole-day VWAP pinball loss by time of day for the intraday model and for the morning daily
forecast, on the same outcomes. Lower is better. The bars show the intraday skill: positive
means the intraday forecast is better.
"""),
    "id_vs_table": ("Intraday vs daily table", """
The numbers behind the chart, one row per 15-minute slot.
"""),
})

COLUMNS.update({
    "target": "what is forecast: the VWAP of the next 1 / 5 / 15 / 60 minutes, the rest of the "
              "day, or the whole day",
    "time": "clock time (ET) at the start of the 15-minute slot",
    "tod_bucket": "15-minute slot of the session, 0 = 09:30",
    "sector_code": "the stock's sector",
    "date": "the day",
    "coverage80": "share of outcomes inside the P10–P90 band. It should be 80%: below means "
                  "the bands are too narrow, above too wide",
    "skill_rw": "1 − pinball / the random walk's pinball (VWAP ahead = reference). Above 0 "
                "beats it",
    "skill_clim": "1 − pinball / the climatology baseline's pinball (the usual spread of the move "
                  "from the last price at that time of day). Above 0 is real skill",
    "skill_mom": "1 − pinball / the momentum baseline's pinball. Above 0 beats it, but it is an "
                 "easy bar",
    "dir_last": "how often the median was on the right side of the last price",
    "dir_trail": "how often the median was on the right side of the trailing VWAP. Flattering: "
                 "the trailing VWAP lags the price, so a forecast near the last price gets it",
    "dir_session": "how often the median was on the right side of the session VWAP so far",
    "pit_bias": "0.5 − mean PIT: above 0 optimistic (outcomes landed below the forecast), "
                "below 0 pessimistic. 0 is ideal",
    "width80_med": "typical (median) P10–P90 band width as a log move: 0.5% ≈ a band 0.5% wide",
    "intraday_pinball": "the intraday model's pinball loss on the whole-day VWAP. Lower is better",
    "daily_pinball": "the morning daily forecast's pinball loss on the same outcomes. Lower is "
                     "better",
    "skill": "1 − intraday / daily pinball. Above 0: the intraday forecast is better",
    "id_stocks_sectors:n": "decision points in the sector",
})

PANELS["home_intraday"] = ("Intraday scores by horizon", """
The newest intraday run on its test days, for the four short horizons (the VWAP of the next
1 / 5 / 15 / 60 minutes). *Coverage* should be near 80%. *Skill vs climatology* above 0 is real
skill: it beats the usual spread of the move from the last price at that time of day. *Skill vs
momentum* is an easier bar. *Direction* is how often the median called which side of the last
price the VWAP ahead landed on (50% = coin flip).
""")
COLUMNS["horizon"] = "how far ahead the forecast VWAP runs: the next 1, 5, 15 or 60 minutes"

HELP.update({
    "wl_today": ("Today's list", """
The stocks most likely to travel far from today's open, ranked by the screener's score (the
80th-percentile range). The list is made at 09:30 (pre-market price known) or 09:45 (pre-market
volume known too). **Sidebar sliders** re-apply the rule filters and re-rank; they start at the
`[screener]` settings. `lean` is the daily model's median direction. It is shown, not used to
rank.
"""),
    "wl_outcome": ("Screener: picks vs outcome", """
Past lists graded against what happened. For each day the top picks' realised range from the open
(max of high - open and open - low, over the open) is compared with the median of every stock
in that list. **Hit rate** is the share of picks above that median; 50% is chance. Needs
`outcomes.parquet` (`screener-outcomes`).
"""),
})
PANELS.update({
    "wl_table": ("Top picks", """
One row per passing stock, best score first. `score` and `pred_range` are fractions of the open
price: how far the stock may travel from the open on the day. Read it as a shortlist to look at,
not a direction call.
"""),
    "wl_chart": ("Score of the top picks", """
The screener score of each pick, best first. A steep drop means only the first few are clearly
ahead of the rest.
"""),
    "wl_outcome_chart": ("Picks vs universe", """
Bars: the mean realised range from the open of that day's top picks. Line: the median over every
stock in the same list. Bars above the line on most days mean the screener finds the movers.
"""),
    "wl_outcome_table": ("Graded lists", """
One row per graded list: how many picks and stocks, the picks' mean realised range, the universe
median, the ratio between them, the hit rate and the mean predicted range.
"""),
})
COLUMNS.update({
    "state": "px = list made at 09:30 (pre-market price known), full = at 09:45 (pre-market volume too)",
    "rank": "position among the stocks that pass the rules (1 = highest score)",
    "score": "the screener's 80th-percentile range from the open, as a fraction of the open",
    "pred_range": "the screener's typical (mean-model) range from the open, as a fraction",
    "lean": "up or down: the daily model's median move direction. Not used for ranking",
    "open": "today's opening price",
    "gap_prev_close": "the open vs the previous close, as a fraction",
    "tft_width90": "the daily model's P5-P95 width vs the previous VWAP: how unsure it is",
    "prev_range": "yesterday's (high - low) / VWAP",
    "atr14_pct": "14-day average true range as a fraction of the close",
    "dollar_volume_20d": "mean of volume x VWAP over 20 sessions, in $",
    "pm_vol_pct_10d": "pre-market volume ratio: log1p(100 x pre-market volume / 10-day mean)",
    "earn_near": "1 if a results release is within a session of today",
    "picks": "stocks in the top list that day",
    "universe": "stocks in that day's list that have an outcome",
    "top_range": "mean realised range from the open of the picks",
    "universe_median": "median realised range from the open of every stock in the list",
    "ratio": "top_range divided by universe_median; above 1 = the picks moved more",
    "hit_rate": "share of picks whose range beat the universe median",
    "predicted": "mean predicted range of the picks",
})
