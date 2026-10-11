# Model fields

Every panel column and what the TFT does with it. The panel (`panel.parquet`, schema 3, written
by `sourcing-py etrade export-tft`) is the model's only input: one row per stock and trading
session. The groups are the tuples in [tft_vwap/config.py](tft_vwap/config.py), and `prepare()`
in [tft_vwap/dataset.py](tft_vwap/dataset.py) applies the transforms listed here.

**Leakage rule:** every input on day t's row is known at the close of day t. The model forecasts
day t+1, so the only inputs it has for t+1 itself are the calendar ones (known in advance).

Unless stated otherwise, prices and volumes are split-adjusted and missing reals become 0.

## At a glance: every column by its TFT role

The TFT sorts its fields into the roles below. Defaults are shown (`--target return`,
`--earnings exact`).

| TFT role | columns | how the model uses them | section |
| --- | --- | --- | --- |
| **Target, the predicted series** | `vwap_pct_prev_day` | Its past values (up to day t) feed the encoder. The model forecasts its value on day t+1: tomorrow's VWAP vs today's, the % move. | 1 |
| Static categoricals | `symbol`, `industry_code`, `exchange`, `issue_type` | One embedding per stock | 2 |
| Known-future reals | `day_of_week`, `month`, `is_month_end`, `is_quarter_end`, `sessions_gap_next` | Read for the encoder days and for day t+1 | 3 |
| Observed (past-only) reals | `intraday_spread_pct`, `close_position_pct`, `open_position_pct`, `volume_velocity`, `transactions_per_volume`, `is_10k`, `is_10q`, the earnings inputs, the optional sentiment inputs | Read for the encoder days only | 4–7 |
| Added by the model | `relative_time_idx`, `encoder_length`, the target's centre and scale | Made by pytorch-forecasting | 8 |
| **Model output** (not in the panel) | `q05` … `q95`, `up_<c>`/`down_<c>`, `prob` | The forecast: 19 percentiles of tomorrow's move, and the confidence moves and event probabilities read off them | 1 |
| Answer key, never an input | `target_tomorrow_vwap`, `target_return` | Day t+1's VWAP, stored on day t's row. Used only to score forecasts. | 1 |
| Index and display | `day_idx`, `date` | Time index | 9 |
| Text the flags come from | `results_window`, `results_window_pit` | Turned into the 0/1 window flags | 6 |
| Not an input with the default target | `today_vwap` | Converts forecast moves into prices. It is the target with `--target vwap`. | 1 |

## 1. What the model predicts

**The model forecasts tomorrow's VWAP as a % move from today's VWAP.** The predicted column is
`vwap_pct_prev_day`, a session's VWAP / the previous session's VWAP − 1. Its value on day t+1
is

    vwap_pct_prev_day(t+1) = VWAP(t+1) / VWAP(t) − 1   (tomorrow's % deviation from today's VWAP)

The model forecasts this one number for each stock as 19 quantiles, P5, P10 … P95. Each sample
is an encoder of up to 60 sessions of history (at least 20) followed by one decoder step,
tomorrow.

**Tomorrow's VWAP is in the training data.** In training, the answer for a sample whose history
ends on day t is the target column on day t+1's row: tomorrow's move, as a percentage. The
panel also stores the same answer on day t's own row, but those two columns are never inputs.
On day t they are the future, so feeding them in would hand the model the answer:
- `target_tomorrow_vwap`: tomorrow's VWAP as a price.
- `target_return`: the same % move, equal to the next row's `vwap_pct_prev_day`.

AAPL, the last rows of the panel:

| date | `today_vwap` | `vwap_pct_prev_day` (target) | `target_tomorrow_vwap` | `target_return` |
| --- | --- | --- | --- | --- |
| 2026-10-01 | 329.6008 | −1.59% | 333.1653 | +1.08% |
| 2026-10-02 | 333.1653 | **+1.08%** ← 10-01's `target_return` | 333.3621 | +0.06% |
| 2026-10-05 | 333.3621 | **+0.06%** | 333.1968 | −0.05% |
| 2026-10-06 | 333.1968 | **−0.05%** | NULL | NULL |

A sample whose history ends on 10-01 is trained to predict 10-02's +1.08%. The last row
(10-06) has no tomorrow yet. It is the live forecast row: `tft-vwap-predict` adds a 10-07
decoder row that holds only the calendar inputs (from the meta's `next_session`), and the model
forecasts 10-07's move.

**What comes out.** The quantiles are the same forecast in both files, saved two ways:

| file | one row per | columns |
| --- | --- | --- |
| `fold_<k>/test_predictions.parquet` (backtest) | stock and forecast day | `symbol`, `day_idx` and `date` (the forecast day t+1), `prev_vwap` (VWAP on t), `actual` (VWAP on t+1), `q05` … `q95` as **prices**, i.e. `prev_vwap` × (1 + move), `industry_code` |
| `forecast_<date>.parquet` (live) | stock | `date` (the forecast session), `asof` (the last session used), `fold`, `symbol`, `industry_code`, `today_vwap`, `q05` … `q95` as **fractional moves** from `today_vwap`, and `up_<c>`/`down_<c>`: the move reached with confidence c (P(move ≥ `up_70`) = 70%, P(move ≤ `down_70`) = 70%) |

For example, AAPL's forecast for 2026-10-07: `today_vwap` 333.20, `q05` −2.47%, `q50` −0.14%,
`q95` +2.55%. That is a 90% range for tomorrow's VWAP of 324.96 … 341.71, with a median of
332.72.

### The quantiles: what they mean, how they are made, what is derived from them

**What one quantile means.** `qNN` is the move that tomorrow's VWAP falls at or below with
probability NN%:

    P( move(t+1) ≤ q_τ ) = τ,   τ = 0.05, 0.10, … 0.95   (TrainConfig.quantiles)

- `q50` is the median forecast: an even chance tomorrow is above or below it.
- `q05` … `q95` is the 90% range; `q10` … `q90` 80%, and `q25` … `q75` 50%.
- The spread between them is the model's uncertainty. A wide range means an uncertain day, e.g.
  around earnings.

AAPL for 2026-10-07 (moves from `today_vwap` 333.20):

| | `q05` | `q10` | `q25` | `q50` | `q75` | `q90` | `q95` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| move | −2.47% | −2.06% | −0.95% | −0.14% | +0.81% | +1.76% | +2.55% |
| VWAP | 324.96 | 326.34 | 330.02 | 332.72 | 335.89 | 339.07 | 341.71 |

So the model gives an even chance that tomorrow's VWAP is within 330.02 … 335.89, and a 90%
chance it is within 324.96 … 341.71.

**How the model learns them.** The TFT has one output per quantile level: 19 numbers per stock
per forecast day, all predicted together. Training minimises the **quantile (pinball) loss**,
averaged over the 19 levels and all samples. For a level τ and the error e = actual − q_τ:

    loss_τ = τ · e         if e ≥ 0  (the actual came in above the quantile)
    loss_τ = (τ − 1) · e   if e < 0  (the actual came in below it)

For `q90`, an actual above it costs 0.9 per unit of error and one below it costs 0.1. The loss
is smallest when 90% of actuals fall below `q90`, so each output learns its own percentile.
pytorch-forecasting scales each sample by its encoder window's centre and scale
(`EncoderNormalizer`), then scales the outputs back, so the loss is scored in the target's own
units, the fractional move.

**After the model.** The 19 outputs aren't forced to be in order, so each row is sorted
(`score_frame`, `forecast`) to make `q05` ≤ `q10` ≤ … ≤ `q95`. The backtest file then turns
them into prices, `prev_vwap` × (1 + q); the live file keeps them as moves.

**What is derived from them** (live forecast files and the calibration report):

| field | how it is calculated | AAPL, 2026-10-07 |
| --- | --- | --- |
| `prob` for a move m (`forecast_<date>_moves.csv`, moves ±0.5% … ±3%) | The forecast's distribution, read by straight-line interpolation between neighbouring quantiles (`prob_below`). Down moves: P(move ≤ m). Up moves: P(move ≥ m) = 1 − P(move ≤ m). Beyond `q05`/`q95` the model says nothing more, so probabilities are clamped to 5% … 95%. | P(≥ +1%) = 20.0%: +1% sits at about `q80`. P(≤ −2%) = 10.5% |
| `bucket`, `hist_hit_rate`, `all_hit_rate` (same file) | The `prob` band (5–10%, 10–20% … 90–95%), and how often calls in that band came true in the backtest, for this stock and for all stocks | `≤ −2%` falls in the 10–20% band. AAPL's past calls in that band came true 16% of the time (n = 25), and all stocks' 16.1% |
| `up_<c>` (c = 60, 70, 80, 90) | The quantile at 1 − c (`quantile_at`, interpolated): a c chance that tomorrow's move is **at least** this. It is a floor, and can be negative. | `up_70` = `q30` = −0.84%: 70% chance tomorrow's VWAP is ≥ 330.39 |
| `down_<c>` | The quantile at c: a c chance that the move is **at most** this. It is a ceiling. | `down_70` = `q70` = +0.55%: 70% chance it is ≤ 335.02 |

**How the quantiles are graded** (`tft-vwap` summary, `tft-vwap-report`, `tft-vwap-grade`, the
dashboard). Every grade compares the backtest forecasts with what actually happened:

| measure | calculation | a good forecast |
| --- | --- | --- |
| Pinball loss, and skill | The training loss above, on the test days. The naive forecast puts every quantile at a 0 move (tomorrow = today). Skill = 1 − model loss / naive loss. | Skill > 0, higher is better |
| Range coverage | The share of actuals inside `q05` … `q95` (also `q10` … `q90`, `q25` … `q75`) | 90% (80%, 50%) |
| Per-level calibration (`calibration_quantiles.csv`) | For each τ, the share of actuals ≤ `q_τ` | τ itself |
| Probability grid (`grid_*.csv`) | For each move and `prob` band: how often the move happened (hit rate) vs the average `prob`. Deviation = hit rate − `prob`, with z against the binomial noise | Deviation ≈ 0. Negative means overconfident. |
| Ladder hits | How often the actual cleared `up_<c>` / `down_<c>` | c |
| PIT and bias | PIT is where the actual landed in the forecast distribution, `prob_below(actual)`. Bias = 0.5 − mean PIT. | PIT evenly spread, mean 0.5. Bias > 0: forecasts too high (optimistic); < 0: too low (pessimistic). |
| Direction | Whether the sign of `q50` matched the sign of the actual move | > 50% |

**With `--target vwap`** the series is `today_vwap`, a price normalised on a log scale. The
model then forecasts tomorrow's VWAP price directly, and the forecast files convert it to the
same moves. `return` is the default; the price target is weighted by 1 / the stock's median
price (`loss_weight`) so expensive stocks don't dominate the loss.

**Series breaks** (`series_breaks`, on by default): `prepare()` splits a symbol into separate
series wherever a different security takes over the ticker:
- more than 20 sessions without a bar: a reused ticker or a relisting (BNY was a closed-end fund
  before BNY Mellon took the ticker)
- a one-session VWAP move of 10x either way: a missed reverse split, or a bankruptcy share
  exchange

The latest part keeps the ticker and earlier parts become `SYMBOL~1`, `~2` …, so no encoder
spans a break. The move across the break counts as unknown (0), like a series' first bar. On the
panel to 2026-10-06 this splits 22 symbols (4,630 rows renamed, none dropped), and the largest
move left is +529%. Runs saved before the setting existed load with it off.

## 2. Static categoricals (one value per stock)

| column | embedding | source |
| --- | --- | --- |
| `symbol` | 16 | ticker |
| `industry_code` | 4 | lowest code in `symbol_industry` (E*TRADE/MBin), `unknown` if none |
| `exchange` | 2 | `symbols.exchange` |
| `issue_type` | 1 | `CS` common stock, `DR` depositary receipt (ADR) |

## 3. Known-future reals (calendar, also on the decoder step t+1)

| column | meaning |
| --- | --- |
| `day_of_week` | 1 = Monday … 5 = Friday |
| `month` | 1 … 12 |
| `is_month_end` | 1 if the next session is in a new month |
| `is_quarter_end` | 1 if the next session is in a new quarter |
| `sessions_gap_next` | calendar days to the next session (3 over a weekend, more over holidays) |

## 4. Observed reals: price and volume (encoder only)

| column | formula (day t) |
| --- | --- |
| `vwap_pct_prev_day` | VWAP / previous session's VWAP − 1. This is the target (section 1), so its past values are already in the encoder. It is an extra input only with `--target vwap` |
| `intraday_spread_pct` | (high − low) / VWAP |
| `close_position_pct` | (close − VWAP) / VWAP |
| `open_position_pct` | (open − VWAP) / VWAP |
| `volume_velocity` | volume / mean volume of the last 20 sessions incl. t; **log1p** in `prepare()` (news days run 100x+) |
| `transactions_per_volume` | trades / volume (small = block trades) |

## 5. Observed reals: filing days

| column | meaning |
| --- | --- |
| `is_10k` | 1 on the session a 10-K was filed (a weekend/holiday filing moves to the next session) |
| `is_10q` | the same for a 10-Q |

These are E*TRADE's filing dates, often days after the results release (section 6).

## 6. Observed reals: financial-results calendar (`--earnings exact|pit|off`, default `exact`)

The windows are counted in sessions from a results announcement, t = 0:

| window | sessions from the announcement |
| --- | --- |
| Pre_Earnings_Runup | −10 … −2 |
| Earnings_Eve | −1 |
| Earnings_Day | 0 |
| Post_Earnings_Reaction | +1 … +3 |
| Post_Earnings_Drift | +4 … +15 |
| Normal_Trading | otherwise |

**Announcement dates** (`results_announcements` in the DuckDB; `sourcing-py etrade
fetch-edgar`, then `derive-features`):
- The results press release is an SEC 8-K with item 2.02. t = 0 is the first session to trade
  on it, from EDGAR's acceptance time: before the 16:00 close → that day's session, after it →
  the next session.
- Several 2.02s before one 10-K/10-Q (preliminary results, then the full release): the last is
  the quarter's **scheduled** release; earlier ones are unscheduled, so they get post windows
  but are never anticipated by pre windows.
- No 2.02 (some smaller filers): the 10-K/10-Q filing date. No EDGAR filings at all: E*TRADE's
  10-K/10-Q dates.

**Post windows** come from the latest announcement on or before day t, of any kind. They are
checked first, so a recent release wins over an upcoming one.

**Pre windows** need the next scheduled date. There are two versions:
- `exact` (the default): the actual date of the next scheduled release. Companies announce
  their results dates weeks ahead, so on day t the date is normally public, though not in our
  data. Where the next release isn't in the panel yet (the latest quarter, and the forecast
  row), the estimate below is used and `days_to_results_is_est` = 1. Train and forecast rows
  therefore differ: training rows almost always have the exact date, the live forecast the
  estimate. The estimate's accuracy is below. For the same reason a `tft-vwap-predict
  --asof` backfill with an `exact` model sees dates that came out after the as-of day.
- `pit`: strictly point in time. The expected date is a year-ago announcement + 364 days
  (52 weeks, same weekday), moved to a session. It ignores `scheduled`, which later filings
  decide. Anniversaries less than 20 sessions after the latest announcement are that
  quarter's and are skipped. Up to 10 sessions overdue reads as Earnings_Eve.

How good the estimate is (panel to 2026-10-06; days 10 sessions before a scheduled release):
on the exact session for 38%, within ±1 for 53%, within ±5 for 84%, no estimate for 0.8%. So
`pit`'s pre windows (Runup, Eve) are on the right sessions about half the time. Post windows
and Earnings_Day never use the estimate, in either mode.

| column | mode | meaning | `prepare()` |
| --- | --- | --- | --- |
| `days_since_results` | both | sessions since the latest announcement (0 on its day) | NULL (none yet) → 63; clip 0 … 63 |
| `days_to_results` | exact | sessions to the next scheduled announcement; the estimate where it isn't known | NULL → 63; clip −10 … 63 |
| `days_to_results_is_est` | exact | 1 where `days_to_results` is the estimate | |
| `days_to_results_est` | pit | sessions to the expected next announcement; negative once overdue | NULL → 63; clip −10 … 63 |
| `is_pre_earnings_runup`, `is_earnings_eve`, `is_earnings_day`, `is_post_earnings_reaction`, `is_post_earnings_drift` | exact | one-hot of the panel's `results_window` | |
| the same with a `_pit` suffix | pit | one-hot of the panel's `results_window_pit` | |

Caveats:
- Coverage of the 3,763 panel symbols: 2,982 (79%) have 8-K 2.02 releases, 107 only 10-K/10-Q
  dates, and 674 (18%) none. Those always read as Normal_Trading, 63 sessions from a release:
  - 252 of the 265 ADRs and 380 common stocks are foreign private issuers (20-F/40-F). They
    put out results on 6-K, which has no item codes.
  - 35 tickers aren't in SEC's ticker list (renamed or delisted, e.g. BK is now BNY).
  - 7 others, mostly banks that file with their bank regulator, not the SEC (OZK, PFBC).

Runs trained before these inputs existed load with `earnings = "off"`, so a frozen model
keeps reading exactly its original inputs.

## 7. Observed reals: news sentiment (off unless `--use-sentiment`)

| column | meaning |
| --- | --- |
| `daily_sentiment` | mean article score for the stock that day (0 if no news) |
| `sentiment_volume` | number of articles |
| `has_news` | 1 if any article |

Off by default because the news history is still near-empty.

## 8. Added by the model (pytorch-forecasting)

- `relative_time_idx`: position within the encoder window.
- `encoder_length`: how much history this sample has.
- The target's centre and scale (`add_target_scales`), from the encoder normaliser by default
  or a per-symbol one (`--normalizer group`).

## 9. In the panel but not inputs

| column | why |
| --- | --- |
| `day_idx`, `date` | time index (`day_idx` = NYSE session number) and display |
| `target_tomorrow_vwap` | tomorrow's VWAP on today's row: the answer key for scoring (`actual` in `test_predictions.parquet`). It would leak as an input. |
| `target_return` | `target_tomorrow_vwap` / `today_vwap` − 1, tomorrow's % move on today's row, for analysis. The model learns the same number from the next row's `vwap_pct_prev_day`. |
| `today_vwap` | today's VWAP price: converts the forecast moves into prices. It is the target only with `--target vwap`. |
| `results_window`, `results_window_pit` | the section-6 windows as text; the model gets their one-hot flags |
| `loss_weight` (made in `prepare()`) | 1 / the symbol's median price, the loss weight for `--target vwap` |

## 10. Kept out of the panel on purpose (they would leak)

| column (`daily_bars`) | why |
| --- | --- |
| `vwap_nx_10d`, `vwap_end_week`, `vwap_nx_qtr` | forward VWAPs over the next 10 sessions / rest of the week / up to the next filing |
| `vwap_nx_1d` | not dropped: it is exported as `target_tomorrow_vwap`, the answer key on day t's row (section 1) |
