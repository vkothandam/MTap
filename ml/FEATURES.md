# Daily model (tft_vwap): fields, training and testing

Every panel column and what the TFT does with it. The panel (`panel.parquet`, schema 4 since
2026-10-27, written by `sourcing-py etrade export-tft`; schema-3 panels and runs still load) is the model's only input: one row per stock and trading
session. The groups are the tuples in [tft_vwap/config.py](tft_vwap/config.py), and `prepare()`
in [tft_vwap/dataset.py](tft_vwap/dataset.py) applies the transforms listed here.

- **Fields** (sections 1–11): what the model predicts, every column by its role, and the columns
  not used yet.
- **Training and testing** (sections 12–16): how runs are trained and graded, the results so
  far, the parameters you can change, candidate inputs, and how to add one and test it.
- **Current state** (section 17): the production run, the daily routine, the runs to keep, known
  data issues and standing decisions.
- **Schema-4 input groups** (section 18): market/sector, macro, session shape, after-hours and
  the two pre-market groups, with their flags and the night / 09:30 / 09:45 forecast states.
- **Screener** (section 19): the 09:30 watch list's target and features (`ml/screener`).

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
| Observed reals, schema 4 (each group behind a flag) | market/sector (13), macro (10), session shape (4), after-hours (4) | Read for the encoder days only; off unless `--market`, `--macro`, `--session-shape`, `--after-hours` | 18 |
| Known-future reals, schema 4 (`--premarket`) | prices: `pm_last_pct_prev_vwap`, `open_pct_prev_vwap`, `pm_px_known`; full: `pm_vwap_pct_prev_vwap`, `pm_vol_pct_10d`, `pm_range_pct`, `pm_full_known` | Day d's own pre-market on the decoder step; masked in training so one model gives the night, 09:30 and 09:45 forecasts | 18 |
| Carried for live and the screener, not TFT inputs | `vol_10d_mean`, `volume`, `export_id` | `vol_10d_mean` makes the live `pm_vol_pct_10d`; `volume` feeds the screener's ATR and dollar volume | 18, 19 |

### Inputs by run

Columns the model reads, by group (`--target return`):

| group | `--earnings exact` (default) | `--earnings pit` (the 10-11 retrain) | production (`20261005-025909Z-return-3712sym`) |
| --- | --- | --- | --- |
| static categoricals | 4 | 4 | 4 |
| known-future reals | 5 | 5 | 5 |
| target history (`vwap_pct_prev_day`) | 1 | 1 | 1 |
| price/volume and filing reals | 7 | 7 | 7 |
| earnings reals | 8 | 7 | none: trained before they existed |
| sentiment reals | 3 with `--use-sentiment`, else none | the same | none |
| series breaks | on | on | off |

No full-panel run has used `exact` yet. The 10-11 retrain is `20261011-173109Z-return-3785sym`
(section 13).

**The 5-year series** (planned 2026-10-27, section 18 and [README.md](README.md)): R0 has the
columns above (`exact`, series breaks on). R1 adds market/sector 13, macro 10, session shape 4
and after-hours 4. R4 adds the 7 pre-market columns (`--premarket optional`), R4p only the 3
price ones always on, R4f all 7 always on. R6p/R6f (overlays) and R7 (HGB) read R1's inputs plus
the pre-market ones as stated in section 18.

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
| `issue_type` | 1 | `CS` common stock, `DR` depositary receipt (ADR). The 5-year panel is `CS` only, so the input is constant there |

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

**Stale since 2026-08-24.** E*TRADE's filings were last fetched then (no token since), so both
flags are 0 on every session after it, including the live forecast rows. The test blocks end
2026-08-14 and are unaffected. EDGAR (`sec_edgar_filings`) has the same forms up to 2026-10-09
(section 15 C).

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

## 11. Extended-hours totals in `daily_bars` (added 2026-10-15)

From Massive 1-minute bars (`sourcing-py massive-ext`), loaded by `ingest-eod`. Since schema 4
(2026-10-27) the panel uses minute-bar VWAPs and volumes built from the same bars, as the
after-hours and pre-market groups (section 18); the totals below stay in `daily_bars` and are
not exported themselves. As of 2026-10-27 this Mac's DuckDB doesn't have them yet: the files are
on the NAS.

| column (`daily_bars`) | meaning |
| --- | --- |
| `pre_market_price`, `pre_market_last` | first and last pre-market trade (04:00–09:29 ET) |
| `pre_market_volume`, `regular_volume`, `after_hours_volume` | minute-bar volume before the open, 09:30 to the close, and from the close minute to 19:59 (the 16:00 minute counts as after-hours). `volume` stays the whole-day figure: it is about 19% above their sum, the auction prints that are in no minute bar. |
| `after_hours_price` | last after-hours trade |
| `pre_market_pct_prev_close` | `pre_market_last` vs the previous session's close, split-adjusted (fraction) |
| `after_hours_pct_close` | `after_hours_price` vs the same day's close, split-adjusted (fraction) |

Timing: day t's after-hours is known by 20:00 ET on day t, so it is a fair observed input for
forecasting day t+1. Day t+1's pre-market is known only on the morning of t+1, so it is usable
only if the forecast is made after it, before the open.

## 12. How training and testing work

`uv run tft-vwap` ([train.py](tft_vwap/train.py)) trains one model per walk-forward fold. Each
model is tested on sessions it never saw.

- **Data:** the panel covers 2024-08-16 … 2026-10-06: 536 sessions, 3,763 stocks, 1.91M rows.
  `export-tft` keeps common stocks and ADRs (the 2-year panel; the 5-year series from 2026-10-27
  keeps common stocks only, `--issue-types CS`) that have:
  - at least 120 bars
  - a median VWAP of at least $1
  - a median volume of at least 50,000
- **Samples:** one per stock and day with at least 20 sessions of history before it. The encoder
  reads up to 60 sessions. The decoder is the next session.

**Folds.** Four test blocks of 60 sessions. Each has a 20-session early-stop block before it and
trains on everything earlier. These are the folds of production and of the 10-11 retrain, pinned
by `--test-end 1663`:

| fold | trains on targets up to | early-stop block | test block | training samples |
| --- | --- | --- | --- | --- |
| 0 | 2025-08-01 | 2025-08-04 … 08-29 | 2025-09-02 … 11-24 | 0.75M |
| 1 | 2025-10-27 | 2025-10-28 … 11-24 | 2025-11-25 … 2026-02-23 | 0.97M |
| 2 | 2026-01-23 | 2026-01-26 … 02-23 | 2026-02-24 … 05-19 | 1.18M |
| 3 | 2026-04-21 | 2026-04-22 … 05-19 | 2026-05-20 … 08-14 | 1.40M |

Encoders may read back across these boundaries; targets never do. Without `--test-end` the folds
end at the panel's last session.

**5-year folds (from 2026-10-27).** `--n-folds 0 --min-train-sessions 252 --fold-len 63
--early-stop-len 21` makes as many 63-session test blocks as fit after a year of training: 16 on
the 5-year panel, about 2022-11 → 2026-10 (the 2022 bear market, the 2023 rally, the April 2025
sell-off). Each has a 21-session early-stop block before it. Fold 0 trains on about 1 year, fold
15 on about 4.6. Training samples are one per (stock, day) (`dedupe_train`), so a stock's last
training day no longer appears in ~41 samples. Runs can pause and resume (`touch runs/<id>/STOP`,
`--resume runs/<id>`, `tft-vwap-status`). The pre-market bake-off uses the 8 odd folds
(`--folds 1,3,...,15`) on rows where the pre-market data exists.

**Training.**
- The loss is the quantile loss over the 19 levels (section 1), with Adam at learning rate 1e-3
  and batches of 512.
- An epoch is 500 random batches: 256k samples, a fifth to a third of a fold's samples.
- A fold trains for up to 15 epochs. It stops after 3 epochs without a better early-stop loss and
  keeps the best checkpoint, `fold_<k>/best.ckpt`.
- The 10-11 run stopped after 7, 8, 5 and 8 epochs.
- All four folds take about 1.5 h on this Mac.

**Outputs.**
- The printed summary and `summary.json`: loss against the naive forecast, direction and
  coverage, per fold and pooled.
- `fold_<k>/test_predictions.parquet` and `calibration_*.csv`.
- `tft-vwap-report` adds `report/`, and the dashboard reads all of it.
- The grades are the table in section 1.

```
cd ml
export PYTHONWARNINGS="ignore::UserWarning:sklearn.utils.validation"  # else 13 h of warning spam
uv run tft-vwap --symbols 200 --max-epochs 2           # smoke test, minutes
uv run tft-vwap --earnings pit --test-end 1663         # full run on production's folds, ~1.5 h
uv run tft-vwap-report runs/<run_id>                   # grids, ladders, bias tables
cd ../dashboard && uv run forecast-dash --runs ../ml/runs   # Compare runs tab
```

**Pitfalls** found so far:
- `sourcing-py etrade export-tft --help` has no help: it runs the export, without the inference
  rows. Always pass `--include-inference`.
- Without the `PYTHONWARNINGS` setting above, a full run takes 13 h, not 1.5 h. The warning
  ("X does not have valid feature names") is harmless; it comes from pytorch-forecasting's
  target-scale transform.
- pytorch-forecasting scores the loss in target units, so a price target (`--target vwap`) needs
  1/price loss weights, or dear stocks dominate.
- A per-stock GroupNormalizer on the raw price loses badly to the naive forecast. The
  EncoderNormalizer (the default) fixes it.
- `model.predict()` changes the `trainer_kwargs` dict it is given. Pass a fresh one each call.
- Evaluation sets need `longest_encoder()` (dataset.py), else each stock-day is scored once per
  encoder length from 20 up.
- "`VIRTUAL_ENV=.venv-test` does not match" from uv is harmless.

## 13. Results so far, and what they point to

- **Production** is `20261005-025909Z-return-3712sym`: no earnings inputs, no series breaks.
- **The 10-11 retrain** `20261011-173109Z-return-3785sym` added `--earnings pit` and series
  breaks, on the same folds.

The comparison below uses the same 863k forecasts for both, leaving out the 22 split symbols:

| measure | production | 10-11 retrain | aim |
| --- | --- | --- | --- |
| pinball loss | reference | −2.7% (lower on 62% of test days) | lower |
| skill vs the naive forecast | 0.239 | 0.260 | higher |
| P5–P95 coverage | 86.7% | 88.1% | 90% |
| P25–P75 coverage | 43.7% | 46.9% | 50% |
| direction of `q50` | 57.8% | 58.8% | above 50% |
| Earnings_Day pinball loss | reference | −3.0% | lower |
| Earnings_Day P5–P95 coverage | 42% | 49% | 90% |

- By fold the pinball change is −0.5%, −0.0%, −8.3% and −1.1%. Production's fold 2 was weak.
- Including the 22 split symbols the gain is −4.0%, much of it from AKTS.
- Details are in [docs/LEARNINGS.md](../docs/LEARNINGS.md) (`tag-ml-daily`).

**What the numbers point to**, largest gap first:
1. **Earnings days.** The 90% band holds on only 49% of results days. The model doesn't see:
   - whether a release has already come out after the close
   - how the stock traded after hours
   - the surprise

   And with `pit`, the Eve and Runup flags fall on the right session only about half the time.
   Candidates are in section 15.
2. **The bands are a little too narrow everywhere:** 88% coverage against 90%, and 47% against
   50%. Fixes are better inputs, more training, or a calibration layer.
3. **The skill score flatters.** The naive forecast has zero width, so beating it is easy. A
   real baseline is needed before small gains can be judged.
4. **Fold 0 trains on one year.** Five years of data would give every fold more history and
   more market regimes.
5. **The retrain changed two things at once,** so the earnings and series-break shares of the
   gain are unknown. Change one thing per run.

### Improving the testing

| improvement | why | status |
| --- | --- | --- |
| A baseline: each stock's trailing 60-session quantiles of the move | the honest yardstick (MODEL-STRATEGY.md Q1, Q2) | not built |
| Two seeds per configuration | The day-by-day comparison only measures noise across days. Re-training with another seed shows how much a gain varies from training luck alone. | not done |
| One change per run (ablation) | to know what each input is worth | practice |
| Test the latest sessions | `--test-end 1663` leaves 2026-08-17 … 10-06 untested. Move it once production's folds aren't needed. | |
| Grades by segment: earnings days, ADRs, low-priced and thin stocks, volatile days | an input can help one segment and hurt another | partly in the dashboard (by industry, by day) |
| A calibration layer | show raw and calibrated grades side by side | allowed since 2026-10-14, not built |
| A paired run comparison with a per-day confidence interval | The 10-11 numbers came from a scratch script. The dashboard's Compare runs tab scores runs on the same rows but has no interval. | not built |

## 14. Training parameters you can change

All are `TrainConfig` fields in [config.py](tft_vwap/config.py), and each run saves them in
`config.json`. Those with a flag can be set on the `tft-vwap` command line. For the rest, edit
the default in config.py for now.

| parameter | flag | default | what it does | ideas |
| --- | --- | --- | --- | --- |
| `target` | `--target` | `return` | the predicted series (section 1) | |
| `normalizer` | `--normalizer` | `encoder` | scale each sample by its window, or one scale per stock (`group`) | `group` once there are 5 years |
| `earnings` | `--earnings` | `exact` | the results-calendar inputs (section 6) | a full `exact` run against the `pit` run |
| `series_breaks` | | on | section 1 | keep |
| `use_sentiment` | `--use-sentiment` | off | section 7 | only once news is backfilled |
| `symbols` | `--symbols` | all | a random subset, for quick trials | 200 for smoke tests |
| `n_folds`, `fold_len`, `early_stop_len` | `--n-folds` | 4, 60, 20 | the walk-forward layout | more folds with 5 years |
| `test_end` | `--test-end` | the panel's last session | pins the folds | 1663 to compare with production |
| `only_folds` | `--folds` | all | run some folds only | `--folds 3` for a quicker check |
| `encoder_length`, `min_encoder_length` | | 60, 20 | sessions of history per sample | 120, once there are 5 years |
| `hidden_size` | `--hidden-size` | 32 | network width | 16 or 64 |
| `attention_head_size`, `hidden_continuous_size` | | 2, 16 | attention heads; width of each real input's encoding | 4 heads |
| `dropout` | | 0.1 | randomly switches off parts of the network while training, so it can't lean on any one path | 0.2 if test scores fall behind early-stop scores |
| `learning_rate` | `--learning-rate` | 1e-3 | step size | 3e-4 with more epochs |
| `gradient_clip_val` | | 0.1 | caps each update | |
| `embedding_sizes` | | symbol 16, industry 4, exchange 2, issue type 1 | the learned numbers per category (MODEL-STRATEGY.md glossary) | symbol 8: less room to memorise one stock's past jumps (MODEL-STRATEGY.md Q6) |
| `batch_size` | `--batch-size` | 512 | samples per step | |
| `max_epochs`, `limit_train_batches`, `patience` | `--max-epochs`, `--limit-train-batches` | 15, 500, 3 | training length | `--limit-train-batches 1000`: today an epoch sees 18–34% of the samples, and runs stop after 5–8 epochs |
| `quantiles` | | P5 … P95 | the output levels | keep: grading and the dashboard read them |
| `moves`, `confidences`, `prob_buckets` | | | scoring only, never the model | |
| `seed` | `--seed` | 42 | the subset choice and the training randomness | a second seed per comparison |

## 15. Candidate inputs to add or change

**The timing rule.** Every candidate must be known before the forecast is made. Today the
forecast runs after midnight ET from the bars up to the previous close. Moving it to after the
close or to before the open is open decision 1 in MODEL-STRATEGY.md. A value known only later,
such as day t+1's pre-market, fits only with a later cutoff. Training must then use the same
clock time.

**Data to load first.**
- This Mac's DuckDB `daily_bars` covers 2024-08-16 … 2026-10-06 and has no pre-market or
  after-hours columns. The 5-year bars (first file 2021-10-21) and the extended-hours files are
  on the NAS ([docs/STATUS.md](../docs/STATUS.md)).
- Load them with `daily-sourcing.sh` and `MTAP_DAILY_INGEST_ONLY=1` (see
  [docs/TODO.md](../docs/TODO.md)).
- Then run, in order:
  - `fetch-edgar`. The stored 8-Ks start 2023-07-13. By default it reaches back to 400 days
    before the first bar. Without it, the 2021–2023 rows have no results dates and read as
    Normal_Trading.
  - `derive-features`
  - `export-tft`

  The panel grows about 2.5×.
- **Survivorship.** Of the 2,923 tickers whose bars stopped before 2026-09, 2,557 have no issue
  type in `symbols`.
  - Most are likely delisted or renamed tickers that E*TRADE no longer returns. Some may never
    have been common stocks.
  - `export-tft` keeps only `CS`/`DR`, so it drops them, and the panel over-represents stocks
    that survived.
  - They need an issue type before a 5-year retrain, for example from Massive's ticker
    reference.

`Qn` below refers to the Q&A log in [MODEL-STRATEGY.md](MODEL-STRATEGY.md).

**A. From data already collected**

| candidate | from | known | role | why it may help |
| --- | --- | --- | --- | --- |
| A results release already out: an 8-K 2.02 accepted after day t's close and before the cutoff | `results_announcements` (acceptance time) | at the cutoff | known-future 0/1 for t+1 | tomorrow is certainly a results day. With the after-midnight cutoff this catches after-close releases; a pre-open cutoff also catches most pre-market ones. |
| After-hours move `after_hours_pct_close`, and after-hours volume / `volume` | `daily_bars` (section 11) | 20:00 ET on t | observed real | the first reaction to after-close results and news |
| Pre-market move of t+1, `pre_market_pct_prev_close` | `daily_bars` (section 11) | the morning of t+1 | known-future real | the strongest single clue to t+1's VWAP, but only for a pre-open forecast |
| Market and sector moves: SPY, QQQ, IWM, the sector ETFs (XLK …), VXX | `daily_bars` (all present for the panel's 536 sessions) | close of t | observed reals, the same for every stock that day (the sector one by industry) | tells calm days from market-wide sell-offs |
| Overnight gap: open(t) / close(t−1) − 1 | `daily_bars` | open of t | observed real | |
| Trailing volatility: the spread of `vwap_pct_prev_day` and the mean `intraday_spread_pct` over 5 and 20 sessions | the panel | close of t | observed reals | Band width is the most predictable part (Q2). The TFT can work it out from the encoder, but explicit columns usually help, and LightGBM needs them. |
| Dollar volume: median VWAP × volume over 20 sessions | the panel | close of t | observed real | small, thin stocks move more |
| Other 8-K items (agreements, leadership changes, other events …) | `sec_edgar_filings`: 148,765 8-Ks since 2023-07 with their item codes, 45,443 of them 2.02 | the filing date; acceptance times are fetched for 2.02 only | observed 0/1 per item group | the event type matters more than sentiment (Q3) |
| Calendar events: monthly options expiry (third Friday), quarterly index rebalances, the Russell reconstitution (June), half-day sessions | the exchange calendar | ahead | known-future reals | volume and closing-auction spikes move VWAP |

**B. Needs new collection**

| candidate | source | note |
| --- | --- | --- |
| News per stock and day: count, sentiment, and a coverage flag | Massive `/v2/reference/news` backfill (paid plan, since 2016), then the scrapers | "No news collected" must not look like "no news" (Q6). `--use-sentiment` exists, but the panel has 0 news rows. |
| Earnings surprise, guidance, analyst rating changes | Benzinga through Massive, if the plan includes partner data | open decision 3 |
| Short interest, options implied volatility | no source yet | implied volatility is the market's own forecast of the band width |

**C. Changes to current inputs**

| input | change | why |
| --- | --- | --- |
| earnings mode | a full `exact` run, compared with the 10-11 `pit` run | `exact` is the default but has never been trained on the full panel. Its live row uses the estimate (section 6), so compare the test grades with live grading. |
| `is_10k`, `is_10q` | take the filing dates from EDGAR (`sec_edgar_filings`, current) instead of E*TRADE (stale since 2026-08-24, section 5) | the flags are 0 on every live row today |
| `industry_code` | the sector level, or a better choice than the lowest code | the lowest code of a many-to-many map is arbitrary for multi-industry stocks |
| missing values become 0 | a 0/1 "known" flag wherever 0 is also a real value | e.g. `transactions_per_volume` when the trade count is missing; the news columns |
| `volume_velocity` | add a 5-session version | catches volume spikes sooner |
| history window | 2 years → 5 | more results days per stock (about 8 → 20) and several market regimes. Fix survivorship first (above). |

## 16. How to add or change an input, and test it

1. **Decide when it is known**, relative to the forecast cutoff (section 15). Day t's row may hold
   only what is known before the forecast for t+1. A value describing t+1 itself goes in the
   known-future group.
2. **Get it into DuckDB.** Raw bars come in through `ingest-eod`. Derived columns are computed
   in `derive-features` ([features.py](../python/sourcing_py/etrade/features.py)).
3. **Export it.** Add it to the panel query in
   [export.py](../python/sourcing_py/etrade/export.py). A known-future column also goes into the
   calendar and into `next_session` in `panel.meta.json`. When the model starts requiring the
   column, bump the panel's schema version and `SUPPORTED_SCHEMA_VERSIONS` in config.py.
4. **Give it a role** in [config.py](tft_vwap/config.py): `KNOWN_REALS`, an observed group, or a
   new optional group.
   - An optional group gets a `TrainConfig` switch and a flag in train.py.
   - Put the switch in `ADDED_INPUTS` with its "off" value, so runs saved before it still load
     with their original inputs. `earnings`, `series_breaks`, `market`, `macro`,
     `session_shape`, `after_hours` and `premarket` work this way.
5. **Clean it** in `prepare()` ([dataset.py](tft_vwap/dataset.py)): fill missing values, clip,
   and take the log of heavy tails. A new categorical also needs an `embedding_sizes` entry.
6. **Check the forecast path.** `tft-vwap-predict` builds the decoder row in `future_rows()`.
   Calendar columns are one row copied to every stock; per-stock known-future columns (the
   pre-market group) come from `panel.next.parquet`, which `export-tft` writes and
   `export-next` fills on the morning. A new per-stock known-future input goes there too.
7. **Document it** in this file: the at-a-glance table and its section.
8. **Test it** in [tests/test_features.py](tests/test_features.py). Check its values, and that
   day t's value uses nothing from after the cutoff.
9. **Smoke run:** `uv run tft-vwap --symbols 200 --max-epochs 2`. This takes minutes and catches
   errors, not gains.
10. **Full run on production's folds:** `uv run tft-vwap --earnings pit --test-end 1663` plus the
    new flag, with `PYTHONWARNINGS` set as in section 12. It takes about 1.5 h. Change one
    thing per run.
11. **Compare** it with the run it should beat, in the dashboard's Compare runs tab on the same
    rows. Check:
    - the pinball loss
    - the coverage of each band
    - the earnings-day numbers
    - each fold: a real gain holds in most folds, not just one
12. **Record it** in [docs/LEARNINGS.md](../docs/LEARNINGS.md) under `tag-ml-daily`. Update
    [MODEL-STRATEGY.md](MODEL-STRATEGY.md)'s Current position if the result changes the plan.

## 17. Current state (2026-10-25; 5-year retrain added 2026-10-27)

**5-year retrain (built 2026-10-27, not run).** Schema 4, the new groups, the masked pre-market
states, the overlays (R6p/R6f), the HGB baseline (R7), the screener and the morning/night jobs are
on branch `feat/daily-5y-inputs`. Before the runs: the NAS deploy, S1 (one live morning), S7
(inference time on the NAS), then the 5-year load (`ingest-eod`, `derive-features`, `export-tft
--include-inference --out out/tft/exp5y/panel.parquet`) and R0–R8 (commands in
[README.md](README.md), the decisions in [MODEL-STRATEGY.md](MODEL-STRATEGY.md) Q11–Q16, results
to [docs/LEARNINGS.md](../docs/LEARNINGS.md)). Production stays as below until a run beats it.

**Production** is `runs/20261005-025909Z-return-3712sym --fold 3`. Switching to the 10-11 retrain
(`runs/20261011-173109Z-return-3785sym --fold 3`) is the user's open decision. The latest live
forecast is for 2026-10-07. This Mac's panel ends 2026-10-06, so nothing after that has been
forecast or graded.

**Daily routine.**
- The NAS fetches the day's Massive bars at 06:40 ([docs/STATUS.md](../docs/STATUS.md)).
- This Mac then loads them by hand, since nothing schedules it yet (docs/TODO.md):

```
scripts/schedule/daily-sourcing.sh --ingest-only   # ingest-eod, fetch-splits, fetch-edgar,
                                                   # derive-features, export-tft --include-inference
cd ml && uv run tft-vwap-predict runs/20261005-025909Z-return-3712sym --fold 3
```

`tft-vwap-predict` writes `forecast_<date>.*` and then grades every earlier forecast whose day is
now in the panel, into the run's `actuals.parquet`. The dashboard's Forecast page reads both. The
live grades so far are in [docs/LEARNINGS.md](../docs/LEARNINGS.md) (`tag-ml-daily`).

**Runs in `ml/runs`:**

| run | size | keep? |
| --- | --- | --- |
| `20261005-025909Z-return-3712sym` | 230M | yes: production, live forecasts and `actuals.parquet` |
| `20261011-173109Z-return-3785sym` | 208M | yes: the candidate, and the intraday model reads its fold 1 (`daily_run` in [tft_intraday/config.py](tft_intraday/config.py)) |
| `20261004-231546Z-return-3712sym` | 117M | the first full run (one `forecast_2026-08-17`); can go |
| `20261004-*-200sym`, `20261005-025856Z-return-10sym` | 2–4M each | smoke tests; can go |
| `pit-test1663.log` | 3M | the 10-11 run's log |
| `intraday/` | 1.7G | the intraday model ([FEATURES-intraday.md](FEATURES-intraday.md)) |

Re-check the intraday model whenever the daily model it reads changes (MODEL-STRATEGY.md Q5).
Backups from before the EDGAR change: `state/etrade/fundamentals.pre-edgar.duckdb` and
`out/tft/panel.schema2.*`.

**Known data issues:**
- PSKY's 2026-10-05 bar is about twice its history (a 2-for-1 split Massive applied on 10-05):
  a fake +107% move. Flagged, not patched.
- XRPN's VWAP goes 20.6 → 39.5 → 19.6 with no split: a Massive glitch.
- About 12 splits Massive never adjusted (BVC, CLBK …). **Standing decision: ignore them.** A
  "raw jump = split ratio" detector was tried and reverted; don't re-add it. Series breaks
  (section 1) are a different rule: a 10x move or a 20-session gap.
- `is_10k` / `is_10q` stale since 2026-08-24 (section 5).
- 18% of the panel stocks have no results dates (section 6).
- Survivorship: delisted tickers without an issue type are dropped (section 15).

**Standing decisions:**
- Live forecasts are for the next session only. No wider `--asof` backfill.
- The goal is the percentile distribution of the move, graded per stock, to show each stock's
  bias (for example "AAPL's 70% call for +2% comes true 55% of the time").
- Retraining and recalibration are allowed (since 2026-10-14). The aim is 5 years of bars plus
  news, and the model choice (TFT, LightGBM, Chronos-2) is open, compared through the run-file
  contract ([dashboard/README.md](../dashboard/README.md)).
- `ml/` never imports `sourcing_py` or reads DuckDB: `panel.parquet` and its meta are the only
  input. Every dashboard chart and table needs its ⓘ help text (a test checks).

## 18. Schema-4 input groups (added 2026-10-27)

Built by `derive-features` ([features.py](../python/sourcing_py/etrade/features.py):
`market_daily`, `macro_daily`, `pm_px_expr`, `pm_full_expr`) and exported by `export-tft`. Each
group has its own `TrainConfig` switch, is in `ADDED_INPUTS` (runs saved before it load with it
off), and is NULL-filled with 0 in `prepare()` after its known flag is set. Day-t lags check
`day_idx`, so a gap in a stock's bars never pairs it with an older session.

**Market / sector** (`--market`, observed, day t):

| column | meaning |
| --- | --- |
| `mkt_cap_ret` | cap-weighted close-to-close return of day t over common stocks with bars on both days; each return clipped to ±50%. Cap = the quarter-end share snapshot × the split-consistent close, carried forward by price |
| `sec_50_cap_ret` … `sec_63_cap_ret` | the same within each of the 11 TRBC sectors (50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 63) |
| `own_sector_cap_ret` | the stock's own sector's return (the first two digits of `industry_code`), else the market's |

Delisted tickers with no cap fall back to shares × close, then to equal weight with a warning.

**Macro** (`--macro`, observed; one value per day for every stock):

| column | meaning |
| --- | --- |
| `ust_3m_chg_bp`, `ust_2y_chg_bp`, `ust_10y_chg_bp` | 1-session change of the 3-month, 2-year, 10-year Treasury yield, in bp |
| `ust_10y_2y_slope`, `ust_10y_3m_slope` | curve slopes, percentage points |
| `ust_10y_z252` | the 10-year yield vs its past 252 sessions (z-score, past only) |
| `cpi_yoy`, `core_cpi_yoy` | CPI and core CPI year on year, computed from the index levels |
| `cpi_yoy_chg` | change in `cpi_yoy` from the month before |
| `core_cpi_mom` | core CPI month on month |

Timing: yields lag 1 session (`[macro] yield_lag_sessions = 1`: day t's yields are not served by
the next afternoon); CPI for month M is used from the first session of M+2. No raw levels.
Macro is the group most able to memorise dates; check its importance and folds (README.md).

**Session shape** (`--session-shape`, observed, day t):

| column | formula |
| --- | --- |
| `high_position_pct`, `low_position_pct` | (high − VWAP) / VWAP, (low − VWAP) / VWAP |
| `gap_open_pct` | open / previous close − 1 |
| `gap_open_vs_prev_vwap_pct` | open / previous VWAP − 1 |

**After-hours** (`--after-hours`, observed, day t; known by 20:00 ET, so the night forecast
from 20:20 ET has it):

| column | formula |
| --- | --- |
| `ah_vwap_pct_rth_vwap` | after-hours VWAP (minute-bar `vw`, 16:00–19:59) / regular-session VWAP (09:30–15:59) − 1 |
| `ah_vol_pct_10d` | log(1 + 100 × after-hours volume / mean daily volume of the 10 sessions before t) |
| `close_pct_rth_vwap` | close / regular-session VWAP − 1 |
| `ah_known` | 1 when both VWAPs exist (not scaled) |

Rows before the minute-bar history starts (the first week of the 5 years, S5) have `ah_known` 0.
Existing `daily_bars` rows get these only after a `massive-ext` re-run, `ingest-eod
--extended-only` and `derive-features`.

**Pre-market** (`--premarket off|on|optional`, `--pm-groups px|px,full`, **known-future**: row d
holds day d's own pre-market up to the panel meta's `premarket_cutoff`, 09:30 ET; bars count by
their start time). All from Massive minute bars in training; never ChartIQ or the watchlist.

| group | column | formula |
| --- | --- | --- |
| prices (09:30) | `pm_last_pct_prev_vwap` | last pre-market trade before 09:30 / previous session's VWAP − 1, clipped ±30% |
| | `open_pct_prev_vwap` | day d's open / previous VWAP − 1, clipped ±30% |
| | `pm_px_known` | 1 when the last price, the open and the previous VWAP all exist |
| full (09:45) | `pm_vwap_pct_prev_vwap` | pre-market VWAP (minute `vw`) / previous VWAP − 1, clipped ±30% |
| | `pm_vol_pct_10d` | log(1 + 100 × pre-market volume / `vol_10d_mean`), clipped [0, 6] |
| | `pm_range_pct` | (pre-market high − low) / previous VWAP, clipped [0, 1] |
| | `pm_full_known` | 1 when all of these exist |

- Both flags are 0, and the values 0, when the data is missing, the previous session is not
  exactly one session back, or a split executes on d. The pre-market columns and flags are not
  scaled.
- **Masks (`optional`).** In training `pm_full_known` is off for 50% of samples
  (`--pm-mask-full`) and `pm_px_known` for 25% (`--pm-mask-px`), from one shared draw, so full
  is never on while prices are off. A masked group's values are set to 0 with its flag. The
  early-stop mask is fixed, so epochs compare.
- **States.** One `optional` run scores three ways: `night` (both off), `px` (prices on), `full`
  (the data's own flags). The run dir is the full state; `<run>~px` and `<run>~night` sit beside
  it, each with `config.json` {`view_of`, `state`}. `tft-vwap-predict --state` writes into the
  matching dir; `--symbols-file` (the 09:30 shortlist) writes `forecast_<date>_subset.*`, which
  is not graded.
- **Live.** `panel.next.parquet` has the next session's rows with both flags 0 and
  `vol_10d_mean`. On the morning, `export-next --prices prices-D.jsonl` sets the prices group
  (watchlist mid or ChartIQ, S1 checks the gap to Massive) and `--premarket premarket-D.jsonl`
  the full group (Massive, the same definitions as training). An incomplete record leaves its
  group at 0; a cutoff or date mismatch is an error.
- `vol_10d_mean`: mean Massive daily volume of the 10 sessions before d, per stock
  (`derive-features`). Not a TFT input.
- `volume`: day t's split-adjusted daily volume, for the screener (section 19). Not a TFT input.

**Overlays and the baseline.** `tft-vwap-overlay --inputs px|full` (R6p/R6f) fits one
gradient-boosted adjustment per quantile on R1's out-of-fold test quantiles plus the pre-market
group, for test fold k from folds before k only. `tft-vwap-gbm` (R7) is a plain HGB per quantile
on the last-day inputs plus lags 1, 2, 5 and 20 of `vwap_pct_prev_day`, sorted so quantiles never
cross; fixed `--max-iter`, no early stop.

## 19. Screener features (`ml/screener`, added 2026-10-27)

Ranks the liquid universe at 09:30 (state `px`) and 09:45 (state `full`) by expected range from
the open. Reads the panel and run files only.

- **Target:** `range_from_open` = max(high − open, open − low) / open on day d; HGB on its log,
  plus a 0.8-quantile HGB for a "big range" score. `hl_range_open` = (high − low) / open is
  stored for the dashboard. Day d's high and low appear only in the target.
- **Features** (all known by 09:31; [features.py](screener/features.py)):

| feature | meaning |
| --- | --- |
| `gap_prev_close`, `gap_prev_vwap` | d's open vs the previous close, the previous VWAP |
| `tft_width90`, `tft_width50`, `tft_median` | the TFT forecast for d: P5–P95 and P25–P75 width, and the median move, vs the previous VWAP. Out-of-fold test forecasts only in training |
| `prev_range`, `prev_range_open` | t's (high − low) / VWAP, and t's range from its open |
| `atr14_pct` | 14-session mean true range / close |
| `log_dollar_volume` | log1p of the 20-session mean volume × VWAP |
| `mkt_ret`, `sector_ret` | t's `mkt_cap_ret`, `own_sector_cap_ret` |
| `days_to_results`, `days_since_results`, `earn_near` | moved forward to d; `earn_near` = a release within a session |
| `day_of_week` | 1 … 5 |
| `pm_last_pct_prev_vwap` | both states |
| `pm_vwap_pct_prev_vwap`, `pm_vol_pct_10d`, `pm_range_pct` | `full` state only |

- Pre-market values with their flag 0 become NaN (HGB handles missing).
- **Rules** (`[screener]` in settings, the Watchlist sliders): `min_price`, `min_dollar_volume`,
  `min_pm_vol_pct`, `min_abs_gap_pct`, `exclude_split_day`, `earnings = any|only|exclude`,
  `top_n`. The TFT median is shown as a lean (up/down), not ranked on. Live scoring assumes no
  split on d.
- **Output:** `out/screener/screener-D.parquet` (the universe, score, `rule_pass`, rank),
  `-top20.csv`; `screener-outcomes` grades past lists into `outcomes.parquet`.
- **Kept only if** (R8) its top 20 beats the naive top 20 by ATR with a CI clear of 0; also
  compared with a random 20, the TFT width alone and the pre-market volume ratio alone.
