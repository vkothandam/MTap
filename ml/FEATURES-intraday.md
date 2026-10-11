# Intraday model fields

> **Status: design.** This is the field contract for the intraday model. The collector that
> feeds it is built (`sourcing-py intraday store`); the bars, the panel export and
> `tft_intraday` come next and need a few weeks of collected sessions first. Until then,
> the files and modules named here do not exist yet.

Every column the intraday TFT (`tft_intraday`) sees, and what it predicts.
- **From sourcing**, the inputs are the panel (`out/tft_intraday/panel/date=<d>/`, written by `sourcing-py intraday export-tft`) and the live snapshot (`out/intraday_live/snapshot-<date>.parquet`). The snapshot has the same feature columns and no targets.
- **On the ML side**, the daily model's forecast for the day is joined from its own run files (§5, §12).
- The groups are the tuples in `tft_intraday/config.py` (to be written).

**One sample** = one stock, standing at a decision second t of session D. t can be any second; the training panel has one every 15 s.
- Its encoder is the 1-minute buckets ending exactly at t, back to the open: bucket j covers [t − 60(j+1) s, t − 60j s).
- Its one decoder step is t itself.

**Leakage rule:** every input is known at second t.
- Static inputs are known before the open of D. They come from D−1's close, the daily forecast made after D−1, the calendar, and results releases accepted before 09:30.
- Observed inputs for a bucket use only feed events before that bucket ends.
- Anything after t appears only as a target.

**Units:**
- Price-like fields are logs vs `prev_vwap`, the VWAP of D−1: `ln(x / prev_vwap)`. ×100 is roughly %.
- Volume-like fields are divided by `avg_vol_10d`, the mean daily volume of D−10 … D−1, split-adjusted.
- Missing reals become 0 unless a `has_*` flag says otherwise.

## At a glance: every column by its TFT role

| TFT role | columns | how the model uses them | section |
| --- | --- | --- | --- |
| **Targets, the predicted series** (6) | `L_1m`, `L_5m`, `L_15m`, `L_60m`, `L_rod`, `L_close_vwap` | The VWAP of the next 1 / 5 / 15 / 60 minutes, of the rest of the day, and of the whole day, each vs `prev_vwap`. **Never inputs**, not even their past values. | 1 |
| Static categoricals (7) | `sector_code`, `industry_code`, `marketcap_category`, `exchange`, `issue_type`, `day_of_week`, `is_half_day` | One embedding each. There is no symbol embedding. | 2 |
| Static reals (47) | prior session (8), results calendar (7), **today's daily forecast** (27), the daily forecast across the sector (5) | Read once per sample: what was known before the open | 3–5 |
| Known reals (4 + 1 categorical) | `t_s`, `session_frac`, `minutes_to_close`, `expected_vol_remaining`, `tod_bucket` | Read for the encoder buckets and the decision step | 6 |
| Observed reals (60, +2 with quotes) | price vs VWAP, volume, sub-minute, returns, day so far, results released today, other symbols, the daily forecast playing out, staleness | Read for the encoder buckets only | 4, 7–13 |
| Added by the model | `relative_time_idx`, `encoder_length`, the target scales | Made by pytorch-forecasting | 14 |
| **Model output** (not in the panel) | per target: `q05` … `q95`; per reference: `p_up`, `up_<c>`/`down_<c>`, `prob` | The forecast | 1 |
| Answer key, never an input | the six targets, `window_min_actual_k`, `mask_k` | Computed from the feed after t | 1 |
| Keys, anchors, provenance | `symbol`, `date`, `phase_s`, `step`, `prev_vwap`, `anchor_price`, raw reference VWAPs, `source`, `daily_run`, … | Not inputs | 15 |

## 1. What the model predicts

**The model forecasts the volume-weighted average price of windows that start at t.** For a horizon k:

    fvwap_k(t) = Σ price · volume / Σ volume    over the feed events in (t, t+k]
    L_k(t)     = ln( fvwap_k(t) / prev_vwap )

| target | window | what it shows |
| --- | --- | --- |
| `L_1m`, `L_5m`, `L_15m`, `L_60m` | (t, t+k], cut at 16:00 (13:00 on half days) | the volume-weighted direction over the next interval |
| `L_rod` | (t, close] | the rest of the day |
| `L_close_vwap` | the whole session: (VWAP so far · volume so far + `fvwap_rod` · rest-of-day volume) / total volume | ln(VWAP(D) / VWAP(D−1)): **the daily model's target**, in log units, updated through the day |

The model forecasts each target as 19 quantiles, P5 … P95. It is one TFT with six targets (a `MultiLoss` of six 19-quantile losses) and `prediction_length` = 1. Each target already summarises its whole window, so the horizons are separate targets, not decoder steps.

**Why a forward VWAP, not the price at t+k:** the last price bounces between the bid and the ask. The window's VWAP is where volume actually traded, i.e. the volume-weighted momentum direction.

**The answers come from the feed at full resolution**: the L0 events, not the 1-minute bars. They are stored on the decision row, the same way `target_return` sits on day t's row in the daily panel.

Unlike the daily model, the targets' past values are **never** inputs. The 60-minute answer of a row from 10 minutes ago already contains the next 50 minutes. Their known-at-t counterparts, the trailing VWAPs `L_trail_k` (§7), are the inputs instead.

**Masks:**
- `window_min_actual_k` is the window's real length after it is cut at the close.
- `mask_k` = 0 when:
  - the window has less volume than `[intraday].min_window_vol`, or
  - for `L_rod` and `L_close_vwap`, the session VWAP is unreliable (a feed gap over 5 minutes).
- A masked target gets zero loss weight.

AAPL on 2026-10-07 at 10:41:37 (*illustrative intraday values; the daily numbers are real*). `prev_vwap` = 333.20, which is the daily forecast's `today_vwap`:

| field | price | log vs `prev_vwap` |
| --- | --- | --- |
| last price | 334.10 | `y` = +0.270% |
| trailing 5-minute VWAP, 10:36:37 … 10:41:37 | 333.95 | `L_trail_5` = +0.225% (an input) |
| VWAP of 10:41:37 … 10:46:37 | 334.25 | `L_5m` = +0.315% (the answer) |

**What comes out.** The same quantiles are saved two ways:

| file | one row per | columns |
| --- | --- | --- |
| `fold_<k>/test_predictions.parquet` (backtest) | stock, decision second, target | `symbol`, `date`, `t_s`, `phase_s`, `target` (`fvwap_1m` … `fvwap_rod`, `close_vwap`), `window_min_actual`, `prev_vwap`, `anchor_price`, `actual`, and `q05` … `q95` as **log levels** vs `prev_vwap` |
| `intraday_<date>.parquet` (live) | stock, decision second, target, reference | `symbol`, `ts`, `target`, `reference` (`trail_vwap`, `last`, `session_vwap`), `ref_value`, `q05` … `q95` as **fractional moves from the reference**, `p_up`, `up_<c>`/`down_<c>`, plus context columns `session_vwap`, `vwap_dev`, `vol_frac_10d`, `rvol`, `L_trail_5` |

### The quantiles: what they mean, how they are made, what is derived from them

**What one quantile means**, the same as in the daily model:

    P( L_k ≤ q_τ ) = τ,   τ = 0.05, 0.10, … 0.95

**Reading the direction off a reference.** The quantiles are levels vs `prev_vwap`. Every reference r is known at t, so the move from r is an exact shift:

    move_τ = exp( q_τ − ln(r / prev_vwap) ) − 1

There are three references:
- `trail_vwap`, the default, is the "momentum" view. It is the trailing k-minute VWAP for horizon k, and the session VWAP so far for `L_rod`. "Will the next 5 minutes trade above the last 5?"
- `last` is the last price.
- `session_vwap` is the day's VWAP so far.

`L_close_vwap` is also reported against `prev_vwap`, so it reads exactly like the daily forecast.

AAPL, `L_5m` (illustrative):

| | `q10` | `q25` | `q50` | `q75` | `q90` |
| --- | --- | --- | --- | --- | --- |
| level vs `prev_vwap` | +0.08% | +0.17% | +0.25% | +0.33% | +0.42% |
| move vs `trail_vwap` (+0.225%) | −0.15% | −0.06% | +0.03% | +0.11% | +0.20% |

`L_trail_5` = +0.225% sits at about the 42nd percentile, so `p_up` vs the trailing VWAP is 58%. Against the last price (+0.270%, the 56th percentile) it is 44%. So the next 5 minutes will more likely trade above the last 5 minutes, but a little below the current tick.

**How the model learns them.**
- The loss is the pinball loss of `ml/FEATURES.md` §1, averaged over 19 levels × 6 targets, with masked windows at weight 0.
- The targets are normalised per symbol (`GroupNormalizer`), fitted on the training fold only.
- `EncoderNormalizer` is **not** used: it would fit on the encoder's target values, and those overlap the future.
- The loss is scored in the targets' own unit, the log level.

**After the model.**
- Each row's 19 outputs are sorted, so `q05` ≤ … ≤ `q95`.
- The backtest file keeps log levels.
- The live file converts them into moves from each reference.

**What is derived from them:**

| field | how it is calculated |
| --- | --- |
| `p_up` | 1 − `prob_below(ref)`. Straight-line interpolation between neighbouring quantiles, clamped to 5% … 95%, as in the daily `prob_below` |
| `prob` for a move m (`[intraday].moves`, default ±0.1%, ±0.25%, ±0.5%, ±1%) | P(move ≥ m) for up moves, P(move ≤ m) for down moves, same interpolation and clamp |
| `up_<c>` / `down_<c>` (c = 60, 70, 80, 90) | the move at quantile 1 − c / c: a c chance the window's VWAP is at least / at most this far from the reference |

**How the quantiles are graded** (`tft-intraday-report`). The measures are the same as for the daily model (`ml/FEATURES.md` §1), per target:

| measure | calculation | a good forecast |
| --- | --- | --- |
| Pinball loss and skill | Skill = 1 − model / baseline, against two baselines. **Momentum continues**: every quantile at `L_trail_k`. **Morning's daily forecast**: for `L_close_vwap` only, the daily quantiles `dq05` … `dq95`. | Skill > 0. Beating the second means the day so far improves on the morning forecast. |
| Range coverage, per-level calibration, probability grid, ladder hits, PIT and bias | as in the daily model | as in the daily model |
| Direction | the sign of the `q50` move vs each reference; also a volume-weighted version | > 50% |

Every measure is also split by `tod_bucket`, `rvol` tercile, `source` (feed) and `daily_fold`.

## 2. Static categoricals

| column | values | source |
| --- | --- | --- |
| `sector_code` | the sector of `industry_code` | `symbol_industry → industries.sector_code` |
| `industry_code` | as in the daily model | lowest code in `symbol_industry` |
| `marketcap_category` | mega / large / mid / small / micro, as of D−1 | shares × prev close, or MBin `Market_Cap` (see Open items) |
| `exchange` | as in the daily model | `symbols.exchange` |
| `issue_type` | `CS`, `DR` | as in the daily model |
| `day_of_week` | 0 … 4 | calendar |
| `is_half_day` | 0/1 | calendar (210-minute session) |

No `symbol` embedding: with ~1,000 stocks and a few dozen sessions it would memorise. It is tried as an ablation after 40+ sessions.

## 3. Static reals: the prior session (`prior_day`)

These come from the daily panel's row for **D−1**. D's own row is never read: its `vwap_pct_prev_day` is VWAP(D)/VWAP(D−1) − 1, which is the answer to `L_close_vwap` (§16).

| column | formula |
| --- | --- |
| `log_dollar_vol_10d` | ln(`avg_vol_10d` × prev close): liquidity |
| `vwap_pct_prev_day` | VWAP(D−1) / VWAP(D−2) − 1 |
| `intraday_spread_pct`, `close_position_pct` | D−1's, as in `ml/FEATURES.md` §4 |
| `volume_velocity` | D−1's, log1p |
| `is_10k`, `is_10q` | a filing on D−1 |
| `days_since_prev_session` | calendar days from D−1 to D: D−1's `sessions_gap_next` |

## 4. The results calendar (`earnings`)

These use the same EDGAR 8-K item 2.02 release dates as the daily model (`ml/FEATURES.md` §6). Sourcing builds them from `results_announcements` acceptance times, **as of 09:30 on D**, and point in time only.

| column | role | meaning |
| --- | --- | --- |
| `days_since_results` | static | sessions since the latest release accepted before 09:30 on D. 0 if it was accepted after D−1's close. NULL → 63, clip 0 … 63 |
| `days_to_results_est` | static | the `pit` estimate (a year-ago release + 364 days); negative once overdue. NULL → 63, clip −10 … 63 |
| `is_pre_earnings_runup_pit`, `is_earnings_eve_pit`, `is_earnings_day_pit`, `is_post_earnings_reaction_pit`, `is_post_earnings_drift_pit` | static | the window D is in, from the two values above |
| `released_today` | observed | 1 from the second a 2.02 release is accepted, if that is after D−1's 16:00 |
| `results_age_min` | observed | minutes since that acceptance, clipped 0 … 390; 0 when there is none |

**Why the daily panel's `results_window` for D is not used:** its Earnings_Day counts a release accepted during D's session, which is not known at 09:30. Here, a release accepted during the session shows up through `released_today` at the second it is accepted.

**Why only `pit`:** `exact` uses dates published after D−1, so training rows and live rows would differ.

**Coverage** is the same as in the daily model: 79% of stocks. Foreign filers always read as no release.

## 5. Static reals: today's daily forecast (`daily_fc`, `group_fc`)

**Every intraday sample of D gets the daily model's next-day forecast for D.** That forecast was made after D−1's close, for VWAP(D) / VWAP(D−1) − 1. So it is a prior on `L_close_vwap`, and on where the price should spend the day. ML joins it (`tft_intraday/daily_join.py`) from the daily model's run files, never from sourcing.

**Where each day's forecast comes from:**

| intraday days | daily file | matched on | conversion |
| --- | --- | --- | --- |
| MBin backfill (2025-12-16 … 2026-01-12) | `<daily run>/fold_1/test_predictions.parquet`. Its test window is 2025-11-25 … 2026-02-23, so these days are out of sample. | `date` = D (the forecast day) | prices → `q / prev_vwap − 1` |
| collected days (from A onward) | `<daily run>/forecast_<D>.parquet`, written each evening by `tft-vwap-predict --fold 3`: `date` = D, `asof` = D−1 | `date` = D | already moves from `today_vwap` = VWAP(D−1) = `prev_vwap` |
| a collected day with no forecast file | `tft-vwap-predict <run> --fold 3 --asof <D−1>` writes the missing file: frozen model, panel cut at D−1 | same | same |

**Rules for the join:**
- **Never use a fold that trained on D.** Fold 3 is used only for D ≥ 2026-05-20, its test start. Fold 1 is used only inside its test window. A test asserts that the backfill uses fold 1 and never fold 3.
- **One daily run is pinned** in `ml/config/paths.toml` `[daily].run`, and it is recorded in the intraday run's config.
  - Production `20261005-025909Z-return-3712sym` and the 2026-10-11 `pit` retrain `20261011-173109Z-return-3785sym` have identical folds, so either works.
  - With an `exact`-earnings run, an `--asof` backfill sees results dates that came out after D−1. The `pit` run is therefore preferred.
  - Switching the daily run means rebuilding the join and training a new intraday run. It is not a recalibration.
- **Log units:** `dq_τ = ln(1 + q_τ)`, on the same scale as `y` and the targets.
- **Series breaks:** a ticker reused later is `SYMBOL~1` in the older folds, e.g. in December 2025's rows. The join strips the `~n` suffix and matches on ticker and `date`. On any one date only one part exists.
- **Price check:** the daily `prev_vwap` / `today_vwap` must match the intraday `prev_vwap` within 1 bp. A mismatch means a split was adjusted differently. That row then gets `has_daily_forecast` = 0.

**The fields.** AAPL's forecast for 2026-10-07 has `today_vwap` 333.20 and fold 3. Its moves are `q05` −2.47%, `q50` −0.14%, `q95` +2.55%.

| column | how it is calculated | AAPL, 2026-10-07 |
| --- | --- | --- |
| `dq05` … `dq95` (19) | ln(1 + q_τ) | −2.50%, …, −0.14%, …, +2.52% |
| `d_width` | `dq90` − `dq10` | 3.83% |
| `d_skew` | (`dq90` − `dq50`) − (`dq50` − `dq10`) | −0.05% |
| `d_p_up` | 1 − `prob_below(0)` | 44% (0 sits at about `q56`) |
| `hist_n` | number of this stock's daily forecasts graded before D | |
| `hist_pit_bias` | 0.5 − their mean PIT: > 0 means too optimistic | |
| `hist_cov80` | the share of actuals inside `q10` … `q90` | |
| `hist_dir_hit` | the share where the sign of `q50` was right | |
| `has_daily_forecast` | 1 if matched. When 0 (outside the daily universe, or a failed price check), the fields above are 0. | |

The `hist_*` stats use only out-of-sample daily forecasts **dated before D**: all folds' `test_predictions`, plus the graded live forecasts in `actuals.parquet`. They never use the run's pooled `calibration_grid.csv` or report, because those include days after D.

**Not taken, because they add nothing:**
- `up_<c>` / `down_<c>` are the quantiles themselves: `up_70` = `q30` and `down_70` = `q70`.
- `prob` and the `_moves.csv` hit rates are functions of the quantiles and of the pooled grid.

**Across the sector (`group_fc`, static).** These are taken over every stock the daily model forecast for D, excluding the stock itself, with `sector_code` read from the daily panel:

| column | meaning |
| --- | --- |
| `sec_d_median` | the sector's mean `dq50` |
| `sec_d_p_up` | the sector's mean `d_p_up` |
| `d_rel` | `dq50` − `sec_d_median`: this stock's call relative to its sector |
| `sec_d_disp` | the standard deviation of `dq50` within the sector |
| `sec_fc_bias_20d` | the sector's mean (0.5 − PIT) over its last 20 graded days before D |

## 6. Known reals (time and the volume clock)

| column | meaning |
| --- | --- |
| `t_s` | seconds since 09:30 ET (the linear clock) |
| `session_frac` | `t_s` / session length: 0 → 1 at the close, so half days line up |
| `minutes_to_close` | (session end − t) / 60 |
| `tod_bucket` (categorical) | the 15-minute slot, 0 … 25; its own embedding gives each part of the day its own variance |
| `expected_vol_remaining` | 1 − `typical_frac(t)`: the share of a normal day's volume still to come |

## 7. Observed reals: price vs VWAP (`price_vwap`)

| column | formula (bucket ending at t) |
| --- | --- |
| `y` | ln(last price / `prev_vwap`) |
| `L_trail_1`, `_5`, `_15`, `_60` | ln(trailing k-minute VWAP / `prev_vwap`): the known-at-t counterparts of the targets |
| `slope_1_5`, `slope_5_15` | `L_trail_1` − `L_trail_5`, `L_trail_5` − `L_trail_15`: is the volume-weighted level rising or falling? |
| `L_session` | ln(session VWAP so far / `prev_vwap`) |
| `vwap_dev` | price / session VWAP − 1 |

## 8. Observed reals: volume and the volume clock (`volume`, `volume_clock`)

| column | formula |
| --- | --- |
| `vol_frac_10d` | cumulative volume / `avg_vol_10d`. It rises all day: about 1.0 at the close on a normal day. |
| `rvol` | `vol_frac_10d` / `typical_frac(t)`: 2.0 means twice the usual volume by this time |
| `vol_bucket` | this bucket's volume / `avg_vol_10d` |
| `sv_1`, `sv_5`, `sv_15` | signed volume (tick rule) over 1 / 5 / 15 minutes / `avg_vol_10d`: buying vs selling pressure |
| `L_vtrail_1pct`, `L_vtrail_5pct` | ln(VWAP of the last 1% / 5% of `avg_vol_10d` traded / `prev_vwap`): momentum measured in volume, not minutes |

## 9. Observed reals: sub-minute, returns and volatility (`sub_minute`, `returns_vol`)

| column | formula |
| --- | --- |
| `vslope_10s`, `vslope_30s` | VWAP slope over the last 10 s / 30 s, from L0 |
| `burst_10s` | the largest 10-second volume in the bucket / the bucket's volume |
| `ret_1m`, `ret_5`, `ret_15`, `ret_60` | log returns of the last price |
| `rv_5`, `rv_15`, `rv_60` | realised volatility over 5 / 15 / 60 minutes |

## 10. Observed reals: the day so far (`day_sofar`)

These carry the whole session into every bucket, even when the encoder is cut short.

| column | formula |
| --- | --- |
| `y_open` | ln(day open / `prev_vwap`): the gap |
| `y_high_sofar`, `y_low_sofar`, `range_pos` | the day's high and low vs `prev_vwap`; (price − low) / (high − low) |
| `or15_hi`, `or15_lo`, `or30_vwap`, `or_break` | opening-range levels vs `prev_vwap`; a breakout as −1 / 0 / +1 |
| `cum_sv` | signed volume since the open / `avg_vol_10d` |
| `rv_sofar`, `max_dd_sofar`, `max_ru_sofar` | volatility, the largest drawdown and the largest run-up since the open |
| `frac_above_vwap` | the share of the session so far spent above the session VWAP |
| `z_trail_5` | `L_trail_5` / `rv_sofar`: the move in units of the stock's own volatility |

## 11. Observed reals: other symbols, indirectly (`group_ctx`)

These are aggregates over all symbols at the same second t, **excluding the stock itself**. No named peer or index series are used.
- A sector with fewer than 5 members falls back to the universe.
- A symbol staler than `[intraday].peer_max_staleness_s` is left out.

| column | meaning |
| --- | --- |
| `mkt_trail_5`, `mkt_trail_15` | the universe's volume-weighted mean `L_trail_5` / `L_trail_15` |
| `mkt_breadth` | the share of the universe above its own session VWAP |
| `sec_trail_5`, `sec_rvol`, `sec_breadth` | the same for the sector (`sec_rvol` is the median) |
| `excess_5` | `L_trail_5` − `sec_trail_5`: leading or lagging the sector |
| `rank_trail_5`, `rank_rvol` | percentile ranks within the universe at t |

## 12. Observed reals: how the daily forecast is playing out (`daily_fc`, `group_fc`)

ML computes these from §5's quantiles and the bucket's prices. They use the same `prob_below`, clamped to 5% … 95%.

| column | meaning | AAPL at 10:41:37 (illustrative) |
| --- | --- | --- |
| `d_pit_now` | `prob_below(y)` on `dq05` … `dq95`: where the price sits in today's forecast | `y` +0.270% is between `dq55` and `dq60`, so 0.59 |
| `d_pit_vwap` | `prob_below(L_session)`: the session VWAP so far vs the forecast of the day's VWAP. This compares like with like. | |
| `sec_pit_now`, `mkt_pit_now` | the median `d_pit_now` of the sector / the universe (excluding self) | |
| `pit_excess` | `d_pit_now` − `sec_pit_now` | |
| `sec_frac_above_dq70`, `sec_frac_below_dq30` | the share of the sector already above its own `dq70` / below its `dq30` | |

If the whole sector is running at its daily P75, the morning's forecasts are running hot for the group. That says something about this stock before its own price shows it.

## 13. Observed reals: quality and quotes (`quality`, `quotes`)

| column | meaning |
| --- | --- |
| `staleness_s` | t − the stock's last feed event. E*TRADE MIDCAP updates every 20 s; Massive every 1 s. |
| `spread_bps`, `imbalance` | bid/ask spread and size imbalance. **Off by default**: only E*TRADE MAIN has quotes, so they are not swap-stable. |

## 14. Added by the model (pytorch-forecasting)

- `relative_time_idx`: the position within the encoder.
- `encoder_length`: how many buckets since the open. It ranges from 1 at 09:31 to 390.
- The targets' centre and scale, from the per-symbol `GroupNormalizer`.

## 15. In the panel but not inputs

| column | why |
| --- | --- |
| `symbol`, `date`, `phase_s`, `step` | keys. A series is (`symbol`, `date`, `phase_s`), and `t_s` = 60 · `step` + `phase_s`. |
| `prev_vwap`, `anchor_price` | turn log levels back into prices and % moves |
| `trail_vwap_1/5/15/60`, `session_vwap` (prices) | the references for reading off the direction |
| `window_min_actual_k`, `mask_k` | target masks (§1) |
| `source`, `vwap_src`, `vwap_reliable`, `n_events` | feed provenance; the report splits by `source` |
| `daily_run`, `daily_fold`, `daily_asof` (ML join) | which daily forecast was used; the report splits by `daily_fold` |

## 16. Kept out on purpose

| what | why |
| --- | --- |
| the targets' past values | the forward windows of earlier rows overlap the future |
| the daily panel's bar columns on **D's** row (`vwap_pct_prev_day`, `target_*`, spreads) | D's own VWAP: the answer to `L_close_vwap` |
| the daily panel's `results_window` for D | it counts a release accepted during D's session (§4) |
| the daily run's pooled `calibration_grid.csv` / `_moves.csv` hit rates | they include graded days after D |
| `phase_s` | predictions must not depend on where t falls within the minute |
| a symbol embedding | it memorises with this few sessions (§2) |
| named peer or index series | other symbols enter only as aggregates (§11) |
| MBin's VWAP column | it resets when the scraper restarts |
| E*TRADE's "10 Day Volume" | only MAIN has it. `avg_vol_10d` from `daily_bars` is used; E*TRADE's figure is a cross-check. |
| raw prices and volumes | not comparable across stocks; only the normalised forms are inputs |

**Size:**
- 7 static categoricals.
- 47 static reals: 8 `prior_day`, 7 `earnings`, 27 `daily_fc`, 5 `group_fc`.
- 5 known: 4 reals and `tod_bucket`.
- 60 observed reals, 62 with `quotes`.
- Without the daily forecast (`daily_fc`, `group_fc` off): 15 static reals and 53 observed.
- Each group can be switched off on its own, which is how the report's lift tests work.
