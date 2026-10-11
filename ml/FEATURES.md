# Model fields

Every column the TFT reads, by category. The panel (`panel.parquet`, schema 3, written by
`sourcing-py etrade export-tft`) is the only input. The groups are the tuples in
[tft_vwap/config.py](tft_vwap/config.py); `prepare()` in [tft_vwap/dataset.py](tft_vwap/dataset.py)
applies the transforms listed here.

**Leakage rule:** every input on day t's row is known at the close of day t. The model forecasts
day t+1, so the only inputs it has for t+1 itself are the calendar ones (known in advance).

Unless stated otherwise, prices and volumes are split-adjusted and missing reals become 0.

## 1. Series and label

| column | role |
| --- | --- |
| `vwap_pct_prev_day` | the series with `--target return` (default): today's VWAP vs the previous session's, as a fraction |
| `today_vwap` | the series with `--target vwap` (price, log-normalised) |
| `target_tomorrow_vwap` | the scoring label only, never an input: it sits on day t's row but is day t+1's VWAP |

Each sample is an encoder of up to 60 sessions (at least 20), and the decoder is the one next
session. The model forecasts 19 quantiles, P5 … P95, of the series at t+1.

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
| `vwap_pct_prev_day` | VWAP / previous session's VWAP − 1. Not an extra input when it is the series |
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

These are E*TRADE's filing dates, often days after the results release of section 6.

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
| `target_return` | target_tomorrow_vwap / today_vwap − 1, for analysis |
| `results_window`, `results_window_pit` | the section-6 windows as text; the model gets their one-hot flags |
| `loss_weight` (made in `prepare()`) | 1 / the symbol's median price, the loss weight for `--target vwap` |

## 10. Kept out of the panel on purpose (they would leak)

| column (`daily_bars`) | why |
| --- | --- |
| `vwap_nx_10d`, `vwap_end_week`, `vwap_nx_qtr` | forward VWAPs over the next 10 sessions / rest of the week / up to the next filing |
| `vwap_nx_1d` | exported only as the label `target_tomorrow_vwap` |
