# Model fields

Every column the TFT reads, by category. The panel (`panel.parquet`, schema 2, written by
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

Together these are the Earnings_Day of section 6.

## 6. Observed reals: financial-results calendar (on by default, `--no-earnings` to drop)

The windows are relative to the 10-K/10-Q filing, counted in sessions:

| window | sessions from the filing |
| --- | --- |
| Pre_Earnings_Runup | −10 … −2 |
| Earnings_Eve | −1 |
| Earnings_Day | 0 |
| Post_Earnings_Reaction | +1 … +3 |
| Post_Earnings_Drift | +4 … +15 |
| Normal_Trading | otherwise |

The pre windows need the next filing date, which is not known on day t. So they come from an
**expected** filing date: the stock's 10-K/10-Q of a year earlier plus 364 days (52 weeks, same
weekday), moved to a session. Anniversaries less than 20 sessions after the latest filing are
that quarter's and are skipped.

On every filing between 2024-08 and 2026-08 that has a year-ago filing, the expected date was:
- exact for 31%
- within ±1 session for 53%
- within ±5 for 86%
- within ±10 for 91%

The post windows use the actual filing, which by then is in the past.

| column | meaning | `prepare()` |
| --- | --- | --- |
| `days_since_filing` | sessions since the latest 10-K/10-Q filed by day t (0 on the filing day) | NULL (no filing yet) → 63; clip 0 … 63 |
| `days_to_filing_est` | sessions from t to the expected next filing; negative once it is overdue and nothing is filed yet | NULL → 63; clip −10 … 63 |
| `is_pre_earnings_runup` | expected filing 2 … 10 sessions ahead | one-hot of the panel's `results_window_pit` |
| `is_earnings_eve` | expected filing next session, or due and not yet filed (up to 10 sessions overdue) | 〃 |
| `is_post_earnings_reaction` | 1 … 3 sessions after a filing | 〃 |
| `is_post_earnings_drift` | 4 … 15 sessions after a filing | 〃 |

A recent actual filing wins over an expected one: the post windows are checked first.

Caveats:
- A 10-K/10-Q filing is often days after the earnings press release (an 8-K). These windows
  are relative to the filing, not the release.
- ADRs (which file 20-F) and stocks with no 10-K/10-Q in `sec_filings` always read as
  Normal_Trading, at 63 from a filing. That is about 21% of panel rows.

Runs trained before these inputs existed load with `use_earnings = False`, so a frozen model
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
| `results_window_pit` | the section-6 window as text; the model gets its one-hot flags |
| `loss_weight` (made in `prepare()`) | 1 / the symbol's median price, the loss weight for `--target vwap` |

## 10. Kept out of the panel on purpose (they would leak)

| column (`daily_bars`) | why |
| --- | --- |
| `results_window` | the hindsight window: its pre-earnings buckets use the actual upcoming filing date. Section 6 is its point-in-time version |
| `vwap_nx_10d`, `vwap_end_week`, `vwap_nx_qtr` | forward VWAPs over the next 10 sessions / rest of the week / up to the next filing |
| `vwap_nx_1d` | exported only as the label `target_tomorrow_vwap` |
