# Forecasting design brief: daily, intraday and the momentum goal

**Last updated: 2026-10-30.** A starting point for a session that evaluates alternative designs. It
gives the overall picture and enough detail to begin. The full specs are linked at the end; where
this page and a spec disagree, the spec wins.

## 1. The final goal

**From 09:30 on, rank the stocks by where they will move next. Keep two lists:**
- likely to move **up** (positive territory);
- likely to move **down** (negative territory).

Each pick carries a size and a calibrated probability.

- **Who uses it.** One trader, one list, refreshed through the day. Night forecasts are not needed
  (decided 2026-10-29, [MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) Q35).
- **What "good" means, in order:**
  1. **Calibrated quantiles.** P_tau has tau of the actuals below it, as the dashboard shows it
     (mean/max |q err|, grade, ECE, coverage, bias).
  2. **Pinball loss**, as the tie-breaker.
  3. For a ranking: **rank IC** and the realised move of the top N against simple baselines.
- **Preferences:** one model per job, no ensembles, and no source-specific workarounds in live
  inputs.

**Where it stands.** The pieces exist, but there is no signed momentum ranking yet.

| piece | answers | state |
| --- | --- | --- |
| Daily model | how far, and with what uncertainty, today's VWAP lands from yesterday's VWAP | chosen (R5-cal20s), not yet live |
| Screener | which stocks will move the most from the open (either direction) | kept (R8b), not yet live |
| Intraday model | where the next 1–60 min VWAP lands from the last price | one small prototype, no edge on direction |
| **Signed momentum ranking** (up list / down list) | which stocks will keep going, and which way | **open: the subject of new designs** |

## 2. Daily forecast

**Intention.** Before and at the open, give every stock a 19-quantile band (P5 … P95) for today's
whole-day VWAP. It sets the day's expectation: the median move, the width, the lean and the
tails. The intraday model and the screener both start from it.

**Target.** `vwap_pct_prev_day(t+1)` = VWAP(t+1) / VWAP(t) − 1, as 19 quantiles. Prices are
`prev_vwap × (1 + q)`.

**Forecast states.** One model is scored two ways:

| state | when | what it knows of today |
| --- | --- | --- |
| 09:30 (`<run>~px`) | at the open | the open price and the last pre-market trade |
| 09:45 (run dir) | after Massive's minute bars reach 09:30 | also pre-market VWAP, volume and range |

**Chosen model.** **R5-cal20s** (09:45) and **R5~px-cal20s** (09:30). There are two parts:
- **R5** is one HistGradientBoosting regressor per quantile, sorted so the quantiles never cross
  (`tft-vwap-gbm --premarket optional --open --vol --tail-iter 600`).
- **Rolling recalibration** (`tft-vwap-recal --sessions 20 --by pm,width`) then shifts each day's
  quantiles by the residual quantiles of the 20 sessions before it, per pre-market × width group.

Results (6 odd folds, 1.27M rows):
- pinball 0.00669;
- mean / max |q err| 0.18 / 0.31 pts, grade excellent;
- P5–P95 coverage 89.5%.

The TFTs (R0–R4) lost to it.

**Input columns** (panel `out/tft/exp5y-r1/panel.parquet`, schema 4; full definitions in
[ml/FEATURES.md](../ml/FEATURES.md)):

| group | columns | known when |
| --- | --- | --- |
| target history | `vwap_pct_prev_day`; R5 also uses lags 1, 2, 5, 20 | day t close |
| stock identity | `symbol` (TFT embedding only), `industry_code`, `exchange`, `issue_type` | static |
| calendar | `day_of_week`, `month`, `is_month_end`, `is_quarter_end`, `sessions_gap_next` | known ahead |
| day-t price and volume | `intraday_spread_pct`, `close_position_pct`, `open_position_pct`, `volume_velocity`, `transactions_per_volume` | day t close |
| filings and results | `is_10k`, `is_10q`; results windows from SEC 8-K 2.02 (`Pre_Earnings_Runup` −10..−2, `Earnings_Eve`, `Earnings_Day`, `Post_Earnings_Reaction` +1..+3, `Post_Earnings_Drift` +4..+15), days to and since results | day t (`--earnings exact\|pit`) |
| market / sector | `mkt_cap_ret`, `own_sector_cap_ret`, `sec_50…63_cap_ret` (11 sectors) | day t close |
| macro | Treasury yield 1-day changes, slopes, 10y z-score; CPI and core CPI YoY / MoM | lagged (yields 1 session, CPI M+2) |
| session shape | `high_position_pct`, `low_position_pct`, `gap_open_pct`, `gap_open_vs_prev_vwap_pct` | day t close |
| after-hours | `ah_vwap_pct_rth_vwap`, `ah_vol_pct_10d`, `close_pct_rth_vwap`, `ah_known` | 20:00 ET |
| **pre-market, 09:30** | `pm_last_pct_prev_vwap`, `open_pct_prev_vwap`, `pm_px_known`; R5's `f_open_all` (the open for every stock) | 09:30 |
| **pre-market, 09:45** | `pm_vwap_pct_prev_vwap`, `pm_vol_pct_10d`, `pm_range_pct`, `pm_full_known` | 09:45 (bars to 09:30) |
| GBM volatility | `f_vol5/20/60` (std of the daily move), `f_absmove20`, `f_spread20` | day t close |

Not used, and why:
- News sentiment: optional and off.
- E*TRADE fundamentals: about 5 quarters with no filing dates, so they leak.
- ChartIQ and watchlist volumes: they don't match the training source.
- Raw price levels: stocks are 1000× apart in price; ratios are used instead.

## 3. Screener (09:30 / 09:45 watch list)

**Intention.** Pick the 20 liquid stocks (≥ $5M dollar volume, ≥ $3) with the biggest expected
move from the open, in either direction.

- **Target:** `range_from_open` = max(high − open, open − low) / open. The model is an HGB on its
  log, plus a 0.8-quantile HGB for a "big range" score.
- **Features:**
  - the gap vs the previous close and the previous VWAP;
  - the daily forecast's P5–P95 and P25–P75 width and median (out of fold);
  - yesterday's range;
  - ATR(14);
  - dollar volume;
  - the market and sector moves;
  - results proximity;
  - day of week;
  - the pre-market price, plus VWAP, volume and range at 09:45.
- **Result (R8b, 882 days):**
  - the top 20 move 12.46% from the open, vs 10.90% for the ATR top 20 (+1.56 pts, CI clear);
  - that is above the universe median on 99% of days.
- **Gap to the goal.** It ranks **size**, not **sign**. The daily median is shown as an up/down
  lean but is not ranked on.

## 4. Intraday forecast

**Intention.** After the open, update the view second by second: where the price is heading over
the next minutes and the rest of the day. This is what drives an up/down momentum list that
changes during the day.

**Today's prototype** (`ml/tft_intraday`, branch `intraday-model`, 12 sessions of the E*TRADE
feed). It is one TFT with six targets. Each target is the forward VWAP as a log move from the last
price at decision second t:
- `r_1m`, `r_5m`, `r_15m`, `r_60m`;
- `r_rod` (the rest of the day);
- `r_close` (the whole day).

| horizon | skill vs climatology |
| --- | --- |
| 1 min | +2.0% |
| 5 min | +2.9% |
| 15 min | +2.6% |
| 60 min | ~0 |
| rest of day | −17% |

**Direction is about 50%: no direction edge.**

**Input columns** (spec: [ml/FEATURES-intraday.md](../ml/FEATURES-intraday.md)). Prices are logs
vs the previous VWAP; volumes are ÷ the 10-day average volume.

| group | columns |
| --- | --- |
| static | `sector_code`, `watchlist`, `day_of_week`, `is_half_day`, `log_dollar_vol_10d`, the prior day's move / spread / close position, `days_since_results` |
| daily forecast for D | `has_daily`, `d_med`, `d_width`, `d_skew`, `d_p_up`, `hist_mae`, `hist_cov80`, `sec_d_med` |
| time and volume clock | `t_s`, `session_frac`, `minutes_to_close`, `expected_vol_remaining`, `tod_bucket` |
| price vs VWAP | `y`, `L_trail_1/5/15/60`, `L_session`, `slope_1_5`, `slope_5_15`, `vwap_dev` |
| volume | `vol_frac_10d`, `rvol`, `vol_step`, signed volume `sv_1/5/15` (tick rule) |
| sub-minute, returns, volatility | `vslope_10s/30s`, `ret_1/5/15/60`, `rv_5/15/60` |
| day so far | `y_open`, high / low so far, `range_pos`, `cum_sv`, `rv_sofar`, `frac_above_vwap`, `z_trail_5`, `released_today`, `results_age_min` |
| other stocks (self excluded) | `mkt_trail_5/15`, `mkt_breadth`, `sec_trail_5`, `sec_rvol`, `sec_breadth`, `excess_5`, `rank_trail_5`, `rank_rvol` |
| feed quality | `staleness_s`, `since_trade_s` |

**The daily handoff.** `ml/runs/<daily run>/handoff.parquet` is the only thing intraday should read
from the daily model ([intraday-model.md](intraday-model.md) §1). It has one row per (symbol, D),
out of fold:
- `dq05…dq95`;
- `d_med`, `d_width`, `d_width50`, `d_width90`;
- `d_skew`, `d_tail`, `d_p_up`, `d_width_rel`;
- `hist_mae`, `hist_cov80`, `hist_bias`, `hist_n`;
- `sec_d_med`, `mkt_d_med`, `d_width_rank`;
- `d_state`, `fold`, `source`.

It is built from R0 today and should be rebuilt from R5-cal20s.

**Data available for intraday:**

| source | resolution | history |
| --- | --- | --- |
| Massive minute bars (`bars_1m`, NAS) | 1 min, with `vw` | 2021-10-29 → (5 years) |
| Massive REST 1-second bars | 1 s | 2024-10-21 → 2026-10-16 (2 years) |
| Massive WebSocket 1-second | 1 s, live | 2026-10-14 → |
| E*TRADE watchlist feed | ~2 s / ~21 s | 12 sessions + 2026-10-12 → |

**Proposed but not built** ([MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) Q31–Q34):
- a 1-minute grid from `bars_1m` over 5 years, with a 60-minute encoder and a decision every
  5 min;
- volume relative to the time of day, moves in units of the stock's volatility, effort vs result,
  and regime summaries;
- 1-second summaries as an ablation;
- a volatility-scaled climatology baseline;
- feature groups screened on HGB before any TFT.

The draft [intraday-volume-pressure-plan.md](intraday-volume-pressure-plan.md) covers a volume
above/below price "pressure mass" and a horizon-free target: which side is hit first, and how fast.

## 5. Constraints any design must respect

- **No leakage.** Every input is known at its decision time:
  - for the daily model: day t close, or 09:30 / 09:45 for pre-market;
  - for intraday: second t.
  Day d's high, low and VWAP appear only in targets. Live data never enters the history files.
- **Out of fold only for stacked models.** A model that reads another model's forecast trains on
  that model's out-of-fold test forecasts, never its training-day outputs. The handoff and the
  recal guarantee this.
- **Walk-forward folds.**
  - 15 test blocks of 63 sessions (2023-01-20 → 2026-10-26).
  - Each block has a 21-session early-stop block in front of it.
  - Each fold trains on everything before its early-stop block (at least 252 sessions).
  - Intraday should test on the same blocks ([intraday-model.md](intraday-model.md) §3).
- **The same source in training and live.** Training uses Massive only. Live pre-market prices
  come from the watchlist mid or ChartIQ (prices only); full pre-market comes from Massive at
  09:45.
- **Live timeline (ET).**
  - 09:29:30: a Massive snapshot gives the shortlist.
  - 09:30: the 09:30 forecast and screener on the shortlist.
  - 09:45: the full forecast for all ~4.4k stocks.
  - After that: intraday updates.
  - Scoring must take seconds.
- **Size.** About 4.4k stocks × 1,255 sessions daily. Intraday at a 15 s step over 5 years does
  not fit in memory, so it needs streaming or sampling.

## 6. How to evaluate a new design

- **Run contract.** Write `ml/runs/<id>/fold_k/test_predictions.parquet`, with columns:
  - symbol, day_idx, date, prev_vwap, actual, `q05…q95` as prices, industry_code.

  Add `config.json`, `metrics.json`, `summary.json` and a `MODEL.md` card. The dashboard, the
  `tft-vwap-compare`, `tft-vwap-recal` and `tft-vwap-handoff` commands, and the screener then work
  unchanged.
- **Tools:**
  - `tft-vwap-compare A B` gives a paired pinball difference with a bootstrap CI, per fold and for
    earnings days.
  - The dashboard's Compare runs tab gives calibration on common rows.
  - `tft-vwap-recal --sessions 20` checks whether calibration alone closes the gap.
- **Baselines to beat:**

  | job | baseline | score |
  | --- | --- | --- |
  | daily quantiles | R5-cal20s | pinball 0.00669, 0.18 pts \|q err\| |
  | direction at 09:30 | "today moves the way the open moved vs the previous VWAP" | 70.7% hit rate, rank IC 0.53; no model beats it |
  | direction at night | "the way yesterday closed vs its VWAP" | 61.3% |
  | biggest movers | ATR top 20 | 10.90% from the open; R8b 12.46% |
  | intraday | climatology | volatility-scaled for the revision |

- **For a signed momentum ranking**, report:
  - the hit rate and rank IC of the up and down lists separately;
  - the realised signed move of the top N against the open rule's top N;
  - calibration of P(up).

## 7. What we have learned so far

- **The models win on size and calibration, not direction.** The daily models' gain over R0
  (skill 27% → 40%) is band width and calibration. Direction stays at the one-input rule's level.
- **Error drifts by regime.** In any single month every model is off by about 2 pts (market-wide
  mood). Fold-window recal lags this; a rolling 20-session recal follows it. A design that adapts
  to the recent market state is worth testing.
- **HGB beat the TFT** on the same inputs, and it trains in minutes. The open price for every
  stock (`f_open_all`) and the volatility features helped most, especially for stocks with no
  pre-market.
- **The intraday prototype** has a small edge in band quality at 1–15 min and none on direction.
  It saw only 12 sessions.

## 8. Alternatives worth evaluating

1. **A signed momentum target.** For example:
   - P(VWAP of the rest of the day > the price at t);
   - or the open-to-close move in volatility units;
   - ranked separately for the up and down lists.
   Compare it with simply ranking by the daily `d_p_up` / median after 09:30.
2. **Screener with sign.** Split `range_from_open` into its up and down parts (high − open,
   open − low), each with its own model, giving two lists.
3. **Intraday on 5 years of minute bars**, the Q31–Q34 design. Screen features on HGB first; the
   TFT only if HGB plateaus.
4. **Horizon-free targets**: first-passage, i.e. which of ±k·σ is hit first and when. See the
   volume-pressure plan.
5. **Regime and state inputs**: market breadth, VIX-like dispersion, the rolling recal shifts
   themselves as features, cross-sectional ranks.
6. **Other model families** (phase 2): LightGBM/XGBoost, AutoGluon, PatchTST, Chronos. Each must
   run fold by fold on the same run contract.
7. **Ranking losses** (LambdaRank, pairwise) instead of quantile regression, when the output is a
   list.

## 9. Where the details are

| topic | doc |
| --- | --- |
| every daily column and its formula | [ml/FEATURES.md](../ml/FEATURES.md) (§18 schema 4, §19 screener) |
| every intraday column | [ml/FEATURES-intraday.md](../ml/FEATURES-intraday.md) |
| handoff, intraday data, fold alignment | [intraday-model.md](intraday-model.md) |
| current model choices and results | [ml/MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) **Scoreboard**; Q35 (goal), Q36 (rolling recal), Q31–Q34 (intraday) |
| terms, metrics, how to read reports | [ml/GUIDE.md](../ml/GUIDE.md) |
| experiment results with examples | [LEARNINGS.md](LEARNINGS.md) `tag-ml-daily` ("Direction and move accuracy") |
| commands | [ml/README.md](../ml/README.md) |
| hosts, schedule, data locations | [STATUS.md](STATUS.md) |
