# Intraday model fields

Every column the intraday TFT (`tft_intraday`) sees, what it predicts, and how it is graded.
- **From sourcing**, the input is the panel `out/tft_intraday/panel/date=<D>/part.parquet` plus `panel.meta.json`, written by `sourcing-py intraday export-tft` (`python/sourcing_py/intraday/export.py`). Its history so far is MBin's E*TRADE watchlist CSVs (2025-12-15 … 2026-01-12, 12 sessions), brought in by `sourcing-py intraday ingest-mbin`; sessions from the iMac collector are added the same way.
- **On the ML side**, the daily model's morning forecast for D is joined from a finished `tft_vwap` run (§5).
- The column lists are in `panel.meta.json` (`features`); ML reads them from there. The groups that can be switched off are in `tft_intraday/config.py`.

**One sample** = one stock, standing at a decision second t of session D.
- t steps every 15 s from the open, with one random phase per date shared by all symbols (`phase_s`, so the cross-symbol aggregates line up). Live, t can be any second: every window is anchored at t, not at a clock minute.
- Its encoder is the 15 s steps ending exactly at t: up to 80 of them (20 minutes). The day-so-far fields (§9) carry the rest of the session.
- Its one decoder step is t itself.

**No bars.** Every window is computed at the feed's own resolution (≈2 s MAIN, ≈21 s MidCap) from cumulative sums over the feed's polls:

    PV(x), V(x)       = Σ price · Δvolume, Σ Δvolume over the polls up to time x
    VWAP over (a, b]  = (PV(b) − PV(a)) / (V(b) − V(a))

- Each volume change is priced at the midpoint of the previous and current last price.
- The first volume change of the session is priced at `day_open` (the opening cross), not at the premarket print.
- The session runs from the last premarket poll to the first poll at or after the close + 30 s, so the closing cross is in it. Half days close at 13:00 (exchange calendar).

**Leakage rule:** every input is known at second t.
- Static inputs are known before the open of D: D−1's daily bar, the calendar, results releases accepted before 09:30, and the daily forecast made after D−1's close.
- Observed inputs at a step use only polls at or before that step.
- Anything after t appears only as a target.

**Units:**
- Price levels are logs vs D−1's VWAP, `prev_vwap`, without mixing bases: `L(x) = ln(x / prev_close_feed) + ln(close(D−1) / vwap(D−1))`. The feed's previous close and `daily_bars` close(D−1) must agree within 1%, else the symbol is dropped for the day (`basis_mismatch` in the meta).
- Volumes are divided by `avg_vol_10d`, the mean `daily_bars` volume of D−10 … D−1. It is null (and the symbol dropped for the day) when a split falls inside that window.
- Targets and the references (§1) are log moves from the last price at t.
- Missing reals become 0.

## At a glance: every column by its TFT role

| TFT role | columns | how the model uses them | section |
| --- | --- | --- | --- |
| **Targets** (6) | `r_1m`, `r_5m`, `r_15m`, `r_60m`, `r_rod`, `r_close` | The VWAP of the next 1 / 5 / 15 / 60 minutes, of the rest of the day, and of the whole day, each as a log move from the last price at t. **Never inputs**, not even their past values. | 1 |
| Static categoricals (4) | `sector_code`, `watchlist`, `day_of_week`, `is_half_day` | One embedding each. There is no symbol embedding. | 2 |
| Static reals (6 + 8) | the prior session and the results calendar (6); the morning daily forecast (8) | Read once per sample | 3, 5 |
| Known (4 reals + 1 categorical) | `t_s`, `session_frac`, `minutes_to_close`, `expected_vol_remaining`, `tod_bucket` | Read for the encoder steps and the decision step | 4 |
| Observed reals (45) | price vs VWAP, volume, sub-minute, returns and volatility, day so far, results released today, other symbols, staleness | Read for the encoder steps only | 6–10 |
| Added by the model | `relative_time_idx`, `encoder_length`, the target scales | Made by pytorch-forecasting | 11 |
| **Model output** | per target: `q05` … `q95` | The forecast | 1 |
| Answer key, never an input | the six targets, `tmask` | Computed from the feed after t | 1 |
| Keys and references | `symbol`, `date`, `step`, `phase_s`, `last`, `prev_vwap`, `prev_close`, `avg_vol_10d`, `ref_trail_*`, `ref_session` | Not inputs | 12 |

## 1. What the model predicts

**The model forecasts the volume-weighted average price of windows that start at t,** as a move from the last price `last_t`:

    r_k(t) = ln( VWAP over (t, t+k] / last_t )

| target | window | valid when (`tmask` bit) |
| --- | --- | --- |
| `r_1m`, `r_5m`, `r_15m`, `r_60m` | (t, t+k], cut at the close | the window ends inside the feed's coverage, has volume, and has no feed gap over max(30 s, 2.5 × the symbol's poll interval); 10 minutes for `r_60m`. `r_1m` also needs ≥ 2 polls inside, which drops most MidCap rows. Bits 0–3. |
| `r_rod` | (t, close] | the day is covered to the close, no gap over 10 minutes. Bit 4. |
| `r_close` | the whole session | always, from `daily_bars`: `ln(VWAP(D) / VWAP(D−1)) − y_t`. **The daily model's target**, re-based to the last price, so it is valid on partial days too. Bit 5. |

`tmask` holds the six validity bits; a masked target gets zero loss weight and is not graded. Partial days (12-15, 12-26, 12-29, 01-02, 01-05, 01-12) have no valid `r_rod` and a part of `r_60m`.

**Why a forward VWAP, not the price at t+k:** the last price bounces between the bid and the ask. The window's VWAP is where volume actually traded.

**Why moves from the last price, not levels vs `prev_vwap`:** the move is what a decision at t acts on, and its scale is the same at 09:31 and 15:59. Levels are one shift away: `level = r + y_t`.

**Row alignment.** The panel stores each decision's targets on the decision row, the way `target_return` sits on day t's row in the daily panel. A TFT sample reads its target from the row after its encoder, so `tft_intraday/dataset.py` moves the targets and `tmask` one row down and adds one padding row per (symbol, date) series. The sample whose encoder ends at step s then predicts the targets of decision s, and its encoder never contains them.

**One model, six targets.** One TFT with `prediction_length` = 1 and a `MaskedMultiLoss` of six 19-quantile pinball losses (P5 … P95). The validity bitmask is the dataset weight; the loss decodes it per target and takes the mean over the valid rows of the batch (an all-masked batch scores 0). The targets are normalised per symbol (`GroupNormalizer(groups=["symbol"])`), never per (symbol, date), which would scale by the day's own answers.

**What comes out** (`fold_<k>/intraday_<target>.parquet`, the dashboard contract): one row per test decision on the 60 s stride, every value a log move from the last price at t.

| column | meaning |
| --- | --- |
| `symbol`, `date`, `t_s`, `tod_bucket`, `sector_code`, `rvol` | the decision and its context |
| `mask`, `actual` | the target's validity bit and the answer (null when masked) |
| `q05` … `q95` | the model's quantiles, sorted |
| `clim_q05` … `clim_q95`, `mom_q05` … `mom_q95` | the climatology and momentum baselines (§13) |
| `ref_trail`, `ref_session` | the trailing VWAP for this horizon (`MOMENTUM_REF`) and the session VWAP so far, as moves from `last`: the references the direction is read against |
| `y`, `last`, `prev_vwap` | to turn moves back into prices and levels |
| `has_daily`, `daily_q05` … `daily_q95` | `r_close` only: the morning daily forecast as a move from the last price, `dq − y_t` |

The files are deliberately not named `test_predictions`, so the daily loaders never read them.

**Reading the direction.** Every reference r is known at t, so the move vs r is an exact shift, `q_τ − ref`. There are three:
- `ref_trail` (default) is the "momentum" view: will the next 5 minutes trade above the last 5?
- the last price (0);
- `ref_session`, the day's VWAP so far.

## 2. Static categoricals

| field | how it is calculated |
| --- | --- |
| `sector_code` | from `symbol_industry` → `industries` (the smallest code when a symbol has several); `unknown` when missing |
| `watchlist` | the E*TRADE watchlist the symbol came from (Main, MidCap, …) |
| `day_of_week`, `is_half_day` | the calendar |

Group ids (`symbol`, `date`) are encoded with an `add_nan` label encoder, so every test date (always unseen) and new symbols encode. They are not embedded.

## 3. Static reals: the prior session and the results calendar

| field | how it is calculated |
| --- | --- |
| `log_dollar_vol_10d` | ln(`avg_vol_10d` × close(D−1)) |
| `vwap_pct_prev_day` | VWAP(D−1) / VWAP(D−2) − 1, on one split basis |
| `intraday_spread_pct` | (high − low) / VWAP, of D−1 |
| `close_position_pct` | (close − VWAP) / VWAP, of D−1 |
| `days_since_prev_session` | calendar days from D−1 to D |
| `days_since_results` | sessions since the last results release accepted before the open (`results_announcements`, SEC 8-K 2.02); 0 for one after D−1's close; 63 when none, clipped 0 … 63 |

## 4. Known (time and the volume clock)

| field | how it is calculated |
| --- | --- |
| `t_s` | seconds since the open |
| `session_frac` | `t_s` / session length |
| `minutes_to_close` | (session length − `t_s`) / 60 |
| `expected_vol_remaining` | 1 − the typical cumulative volume share at `session_frac` |
| `tod_bucket` (categorical) | `t_s` // 900: 15-minute slots 0 … 25, 26 for the closing seconds |

The typical curve is the median symbol's cumulative volume share by session fraction, averaged over **full sessions before D** (`out/tft_intraday/profiles/<D>.json`); a default U-shape until one exists.

## 5. Static reals: the morning daily forecast (`daily_join.py`)

The daily model forecasts VWAP(D) from data up to D−1, so its forecast for D is known before the open. It comes from one fold of a pinned `tft_vwap` run (`IntradayConfig.daily_run`, `daily_fold`): **fold 1 of `20261011-173109Z-return-3785sym`, whose test window 2025-11-25 … 2026-02-23 holds every backfill day out of sample.** The later folds trained on December, so they are never used for it. Series-break suffixes (`A~1`) are stripped; the latest part wins.

| field | how it is calculated |
| --- | --- |
| `has_daily` | 1 when the run has a forecast for (symbol, D), else 0 and the rest are 0 |
| `d_med` | the forecast median as a log move from VWAP(D−1): `ln(q50 / prev_vwap)` |
| `d_width` | P90 − P10 of the same |
| `d_skew` | (P90 + P10 − 2·P50) / width |
| `d_p_up` | share of the 19 quantiles above VWAP(D−1) |
| `hist_mae`, `hist_cov80` | the daily model's absolute median error and P10–P90 coverage on this symbol over the 20 sessions **before D** (rows of D are shifted out) |
| `sec_d_med` | the mean `d_med` of the other symbols in the sector that day |

`--daily-run none` drops them (14 → 6 static reals).

## 6. Observed reals: price vs VWAP

| field | how it is calculated |
| --- | --- |
| `y` | `L(last_t)` |
| `L_trail_1`, `_5`, `_15`, `_60` | `L` of the VWAP over (t − k, t]; the last price when the window had no volume |
| `L_session` | `L` of the session VWAP so far |
| `slope_1_5`, `slope_5_15` | `L_trail_1 − L_trail_5`, `L_trail_5 − L_trail_15` |
| `vwap_dev` | last / session VWAP − 1 |

## 7. Observed reals: volume

| field | how it is calculated |
| --- | --- |
| `vol_frac_10d` | session volume so far / `avg_vol_10d` |
| `rvol` | `vol_frac_10d` / the typical share by now (floored at 0.02): > 1 is a busier day than usual |
| `vol_step` | volume in the last 15 s / `avg_vol_10d` |
| `sv_1`, `sv_5`, `sv_15` | signed volume (tick rule: a volume change takes the sign of the last price move, an unchanged price keeps the previous sign) over the last k minutes / `avg_vol_10d` |

## 8. Observed reals: sub-minute, returns and volatility

| field | how it is calculated |
| --- | --- |
| `vslope_10s`, `vslope_30s` | ln(VWAP of (t − w, t] / VWAP of (t − 2w, t − w]); 0 without volume |
| `ret_1`, `_5`, `_15`, `_60` | ln(last_t / last price k minutes before) |
| `rv_5`, `rv_15`, `rv_60` | √Σ of squared poll-to-poll log returns over the last k minutes |

## 9. Observed reals: the day so far

| field | how it is calculated |
| --- | --- |
| `y_open` | `L(day_open)` |
| `y_high_sofar`, `y_low_sofar`, `range_pos` | `L` of the session high and low so far (with the open), and where the last price sits between them (0.5 when flat) |
| `cum_sv` | signed volume since the open / `avg_vol_10d` |
| `rv_sofar` | realised volatility since the open |
| `frac_above_vwap` | share of the day's decision steps with the last price above the session VWAP |
| `z_trail_5` | (`L_trail_5` − `y`) / `rv_sofar`, clipped ±10 |
| `released_today`, `results_age_min` | 1 from the second a release accepted after D−1's close is public (`accepted_at` ≤ t), and minutes since, clipped 0 … 390 |

## 10. Observed reals: other symbols, and quality

Other symbols enter only as aggregates at the same decision second, **each excluding the symbol itself**. A sector with fewer than 5 other members falls back to the universe. `--no-group` drops this group.

| field | how it is calculated |
| --- | --- |
| `mkt_trail_5`, `mkt_trail_15` | mean of the others' `L_trail_k − y` (trailing VWAP vs last price: basis-free) |
| `mkt_breadth` | share of the others above their session VWAP |
| `sec_trail_5`, `sec_rvol`, `sec_breadth` | the same within the sector (`sec_rvol` is the mean log `rvol`) |
| `excess_5` | own `L_trail_5 − y` minus `sec_trail_5` |
| `rank_trail_5`, `rank_rvol` | the symbol's rank among all symbols at that second, 0 … 1 |
| `staleness_s` | t − the symbol's last poll, clipped 0 … 600 |
| `since_trade_s` | t − the last poll with a volume change, clipped 0 … 600 |

## 11. Added by the model (pytorch-forecasting)

- `relative_time_idx`: the position within the encoder.
- `encoder_length`: 4 … 80 steps.
- The targets' per-symbol centre and scale (`add_target_scales`).

## 12. In the panel but not inputs

| column | why |
| --- | --- |
| `symbol`, `date`, `step` | keys: a series is (`symbol`, `date`), `t_s = phase_s + 15 · step` |
| `phase_s` | provenance; predictions must not depend on where t falls |
| `last`, `prev_vwap`, `prev_close`, `avg_vol_10d` | turn moves into prices and volumes |
| `ref_trail_1m` … `ref_trail_60m`, `ref_session` | the trailing and session VWAPs as moves from `last`: the momentum baseline and the dashboard's references |
| `tmask` | target validity (§1) |

`panel.meta.json` records, per date: the phase, coverage, rows dropped and why (`no_premarket`, `no_daily_bars`, `basis_mismatch`, `no_avg_vol`), the share of valid rows per target, and on full days `vwap_check`: the feed's session VWAP vs `daily_bars` VWAP (median 0.5–3 bp on the December full days).

## 13. Training, baselines and grading

**Walk-forward by date** (`make_folds`). Each test block's early-stop dates are the ones just before it, enough to reach 600k rows; training gets every date before those.

| fold | train | early stop | test |
| --- | --- | --- | --- |
| 0 | 12-15 … 12-24 (5 sessions) | 12-26 | 12-29, 12-30 |
| 1 | 12-15 … 12-31 (9 sessions) | 01-02, 01-05 | 01-12 |

Early stopping is scored on one decision per symbol every 5 minutes, the test on one every minute, each at the longest encoder. `--daily-folds` tests each date on its own.

**Baselines** (`baselines.py`, fitted on the training dates only):
- **Climatology, the one that matters**: the training days' quantiles of each target per `tod_bucket` (a slot with < 200 rows uses the whole day), the same for every stock and second. It knows nothing about the moment, so skill vs climatology is the model's real edge.
- **Momentum**: "the VWAP ahead equals the trailing VWAP" — `ref_trail` plus the training residuals' quantiles per `tod_bucket`. `r_rod` uses the 60-minute trailing VWAP, `r_close` the session VWAP. **An easy bar**: the VWAP ahead stays near the last price while the trailing VWAP lags behind it, so anything centred near the last price beats momentum by 25–30% at 1–60 minutes. For the same reason `dir_trail` is ≈ 70% for any such forecast; `dir_last` is the direction call to read.
- **Random walk**: the point forecast "no move from `ref_trail`".
- **Morning daily forecast**: for `r_close` only, `daily_q*`.

**Grading** (`tft-intraday-report`, `report/{by_target, by_day, by_tod}.parquet`), per target and model, on the valid rows:

| measure | calculation | a good forecast |
| --- | --- | --- |
| `pinball` | mean over the 19 levels | low |
| `skill_clim`, `skill_rw`, `skill_mom`, `skill_daily` | 1 − pinball / the baseline's, on the same rows | > 0; `skill_clim` is the real edge |
| `coverage80` | share of actuals inside P10–P90 | ≈ 0.80 |
| `dir_last`, `dir_trail`, `dir_session` | the median called the side of that reference right | > 0.5 |
| `pit_bias` | 0.5 − mean PIT; > 0: actuals land low in the band | ≈ 0 |
| `width80_med` | median P10–P90 width | narrow, at the right coverage |

Forecasts made every minute overlap (60 of them share a 60-minute window), so `by_target` gives 90% intervals for `skill_clim`, `skill_rw` and `skill_mom` from a bootstrap over blocks of one session hour per day, not over rows.

## 14. Not built yet, or kept out

| what | why |
| --- | --- |
| live forecasts (`tft-intraday-predict --live` on the collector's stream) | next round; the panel's columns are what it will read |
| the targets' past values | the forward windows of earlier rows overlap the future |
| D's own daily bar (VWAP, spreads, volume) | the answer to `r_close` |
| bid/ask spread and imbalance | only E*TRADE MAIN has quotes, so they are not swap-stable |
| `marketcap_category`, `exchange`, `issue_type` | no source in `symbols` yet |
| the daily forecast playing out (`y − d_med` etc.) and bursts | derivable by the model from `y` and `d_med`; to add if the report shows a gap |
| a symbol embedding | it memorises with this few sessions |
| named peer or index series | other symbols enter only as aggregates (§10) |
| MBin's VWAP column | it resets when the scraper restarts |
| E*TRADE's "10 Day Volume" | only MAIN has it; `avg_vol_10d` comes from `daily_bars` |
| raw prices and volumes | not comparable across stocks |
