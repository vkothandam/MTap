# Intraday model: design

**v2 (2026-11-06):** the next model is a sequence model (TFT) on pocket targets over durations, with the GBM
as Phase 2. Its columns, targets and tests: [intraday-features.md](intraday-features.md); decisions: MODEL-STRATEGY Q40.
This file stays as the record of the Stage A/B design and the Step 0 probe.

Status: design agreed 2026-11-01; Step 0 probe implemented (`ml/scripts_intraday_probe.py`), full build
not started. Supersedes the predictor part of [intraday-redesign.md](intraday-redesign.md) (the analog
engine stays as the Space page's "similar moments" view). The old prototype `tft_intraday` is not the
base. Terms are defined in LEARNINGS `tag-ml-intraday` ("Terms used in the intraday results") and the
MODEL-STRATEGY glossary.

**Goal.** From 09:30 on, every liquid stock gets a calibrated view of how far it can go up and down, how
soon, and how long that view stays valid. A fast sort gives the top 20 up / top 20 down; a 40–100
shortlist gets detailed readings and price / time / volume triggers. Options must be possible later.

**Evidence that shapes it.** The stock's own price path alone had no edge (analog experiment, and HGB
on the same 10 inputs, [analog-pockets.md](analog-pockets.md)). New information has to come from the
daily forecast, market / sector, volume at price, order flow and events; the design tests each of
those in order. GBM beat the TFT on daily (R5); rolling recalibration made the daily quantiles
excellent (R5-cal20s).

## 1. Terms

| term | meaning | example |
| --- | --- | --- |
| **t** | the decision minute (09:31 … 15:59), every minute | 10:42 |
| **p_t** | the last trade price before t; every distance is measured from it | $50.00 |
| **σ** | the stock's typical daily move: std of the VWAP-to-VWAP log change over the prior 20 sessions. Distances are in σ so stocks compare | σ = 2% → 0.5σ = 1% = $0.50 |
| **normal volume** | the prior-20-session mean daily volume × the market's intraday volume curve (probe); per-stock minute medians in the full build | |
| **rvol** | actual volume ÷ normal volume for the same minutes | 2.0 = twice the usual |
| **d (delay)** | how old the newest data is: 15 min for Stage A (Massive only), 0 for Stage B (Massive to t − 15 + E*TRADE polls) | |
| **excess** | a value minus its climatology (the typical value for the same time of day × liquidity × σ group) | 0 = nothing unusual |
| **point-in-time (PIT)** | an input at t uses only data known at t − d; enforced by tests | |
| **nats** | log-score unit: +x nats = the model put e^x times the base rate's probability on what happened | +0.011 ≈ 1.1% more; +0.69 = twice |
| **market cap** | as of the previous close: shares outstanding (Massive ticker overview, latest quarter-end snapshot) × the previous close | AAPL 14.59B shares × $250 = $3.65T → mega |
| **cap band** | mega ≥ $200B, large $10–200B, mid $2–10B, small $250M–2B, micro < $250M (user, 2026-11-01) | |

The "±0.6σ typical range" holds the middle ~45% of outcomes if moves were normal (±1σ ~68%); real
intraday moves have fatter tails, so the measured share is reported, not assumed.

## 2. Horizons: windows from now, not a moment

Every output describes the whole window from now to t + h ("at any time within the next h minutes"),
not the price at exactly t + h.
- Windows are nested and all start now: (t, t+1], (t, t+2], (t, t+4] … (t, close]. Window outputs only
  grow with h, and the model is constrained to respect that (monotone in h).
- The list 1, 2, 4 … 128 min, close is only where the curve is sampled for training; h is an input to
  one model, so the user sees three numbers per direction (pressure, horizon score, reliability
  half-life, §5), not a table per horizon.
- The race and the hazard have no horizon: they end when a level is touched.
- Inside windows too: the window VWAP and the "when" outputs (minutes until the high). The price at
  exactly t + h is stored for options pricing only.
- Pockets use disjoint time bins ([0,1), [1,2), [2,4) … [128, close)) because they say when the volume
  trades.

## 3. Inputs

All in σ, rvol or shares. Window w = the last w minutes before t − d, w ∈ {5, 15, 60, since the open}.

**A. Fixed for the day:** `gap_prev_close`, `gap_prev_vwap`; pre-market `pm_last`, `pm_vwap`, `pm_vol`,
`pm_range` (to 09:15 when d = 15 and t ≤ 09:45, else to 09:30); `prev_clv` (yesterday's close in its
range); daily forecast (R5-cal20s) `d_med`, `d_iqr`, `d_w90`, `d_pup`; results window and days to /
since results (EDGAR 8-K); `sigma`, `dollar_vol`, `log_mcap`; news count later.

**B. Today so far:** `px_prev`, `px_open`, `px_prevclose`; `vwap_sofar`, `px_vwap`; `band_px`,
`band_vwap` (where p_t and today's VWAP so far sit in today's daily forecast band, as a percentile);
`from_high`, `from_low`, `day_range`, `min_since_high`, `min_since_low`; `rvol_sofar`; `minute`,
`to_close`; the last 3 daily moves.

**C. Recent path, per window w:** `ret_w`; `rvol_w`, `rvol_accel`; `range_w`, `clv_w` (0 = at the window
low, 1 = at its high); `vwapdev_w`; `imb_w` signed flow = (buy − sell) ÷ volume. Minute tier: buy share
= Φ(Δp ÷ σ_1m) per bar (bulk classification); 1 s tier: tick rule per second.

**D. Volume at price (VAP):** today's volume before t − d in 0.25σ bins around p_t: `poc`, `va_lo`,
`va_hi` (15%–85% of volume), `node_up`, `node_dn` (nearest bin ≥ 2× the median bin), `thin_up`,
`thin_dn` (share of the next 1σ with < 20% of the POC volume: air pockets), `above_share`, and with a
30-min decay `above_dec`, `bulge30`, `sag30`. Yesterday's VAP and τ = 5 / 120 in the full build.

**E. Market and sector:** `mkt_w`, `sec_w` = the prior-dollar-volume-weighted mean 1-min return of the
eligible universe (bars_1m has no SPY) and of the stock's sector (industry code's first 2 digits), in
their own σ; `resid_w` = the stock's move minus the market's; `breadth` (share up since the open);
`rank_open`, `rank_rvol`, `rank_imb` (rank across stocks this minute).

**F. Tiers and sources:**
- Minute tier (Massive minute bars, 5 years): A–E.
- 1 s tier (Massive 1-second bars, 2024-10-21 →; 5-year backfill on the NAS): `imb_w` by tick rule and
  poll-grid versions of C for the last 15 min; blank before, with `tier_1s_known`.
- Live Stage B: Massive up to t − 15, E*TRADE polls (~2 s) for the last 15 min; in training the same
  features come from 1 s bars on a 2 s grid with a 1.25 s lag, so the seam is invisible. Measured:
  price matches 90–95% exactly, volume within 0.002–0.006%, VWAP 0.2–0.4 bp on 2 s polls.
- Dropped: zigzag legs and price-impact λ (VAP chosen instead).

## 4. Targets

Per window (t, t+h], in σ from p_t, regular session, bars from minute t on only.
| name | definition |
| --- | --- |
| `up_h`, `dn_h` | highest / lowest price within the window; wick (bar highs/lows) and supported (beyond which only 5% of the window's volume traded) |
| `t_up_h`, `vol_up_h` | minutes until the high; volume until the high |
| touch | `tu_<level>_<h>` / `td_…`: the window's high ≥ p_t·e^L / low ≤ p_t·e^−L, levels 0.5σ, 1σ, 0.5%, 1%; h = 15, 60 min, close |
| race | at ±k·σ (k = 0.25, 0.5, 1): up first / down first / neither by the close |
| `bulge_h`, `sag_h` | future pockets: Σ volume share × distance above / below p_t |
| `occ_up_h(y)`, `occ_dn_h(y)` | minutes spent above p_t + yσ / below p_t − yσ |
| volume balance | share of the next 60 min's volume traded above p_t (> 50% = the event) |
| `U`, `Dn`, `lean` | Stage A's target: U = Σ_h excess `up_h`, equal weight per grid point; Dn likewise; lean = U − Dn |
| `px_at_h` | the price at exactly t + h, for options only |

## 5. Outputs

| # | reading | definition | stage |
| --- | --- | --- | --- |
| 1 | lean score, rank | predicted lean; top 20 up / 20 down; ranks 21–50 watch list (join ≤ 20, leave > 35) | A |
| 2 | P(up first) | +0.5σ touched before −0.5σ | A |
| 3 | pressure A_up / A_dn | extra up / down potential over typical, in σ and % | B |
| 4 | horizon score T½ | minutes by which half of that extra potential has played out | B |
| 5 | reliability half-life R½ | minutes after which the skill over typical halves (measured out of sample per group) | B |
| 6 | ranges | P5 … P95 of `up_h`, `dn_h` for any h; "chance of reaching +x% within h" curves | B |
| 7 | percentile at a level | where a chosen price sits in that distribution | B |
| 8 | race | P(up first / down first / neither) at 0.25, 0.5, 1σ, with median minutes to hit | B |
| 9 | volatility | predicted range `up_h` + `dn_h` | B |
| 10 | momentum hazard + expiry | chance the run continues; price / time / volume triggers past which it's usually too late | B |
| 11 | pockets | future volume mass above / below, how high, how soon, how long; 2-D map with similar past moments | B |
| 12 | options store | quantile curves, touch probabilities, `px_at_h` per stock and minute | B |

Every reading is shown both as a percentile and as a probability ("+2% is the P70 of the 60-min high"
= 30% chance to touch it); both come from one monotone curve. Three different "percentiles" are named
apart in the UI: **outcome percentile** (a level inside this stock's forecast), **edge** (this forecast
vs the typical one for its group), **rank** (this stock among all stocks now). Up and down are
symmetric and both can be likely at once (choppy stocks), so direction comes from the race and lean.
Average height = the area under the "chance of reaching at least +x" curve; reported with the share
that comes from the extrapolated tail. Pressure when price probabilities are even: volume balance,
pocket mass, time balance and the race (reference price p_t). A daily volume-pressure target: later.

## 6. Model

```
 bars_1m (5y) ─┐                  ┌─ Stage A  ranker, all ≥$20M stocks, every minute, d = 15
 agg_1s (5y) ──┼─ feature builder ┤     → top 20 up / 20 down + watch list → fast E*TRADE list
 daily R5 ─────┤  (one code path) │
 market/sector ┘                  └─ Stage B  distribution heads, shortlist, d = 0 (stitched)
                                        → readings 3–12 (+ similar past moments from tree leaves)
                                  ── rolling 20-session recalibration per group ──
```
- LightGBM (monotone in h, leaf output for similar moments, LambdaRank); HGB as the cross-check and in
  the probe.
- Stage A: regression on lean (rank-gaussed per minute) and a classifier on the ±0.5σ race; challenger
  LambdaRank with each minute as a query; trained at d = 15 and d = 0.
- Stage B (after Stage A passes): rows stacked over h (log-uniform sample), h as a feature; quantile
  heads (7, sorted) for `up_h`, `dn_h`, `vwap_h`, pockets; multiclass race; binary hazard. A and T½
  fitted to the predicted curve: excess(h) = A·(1 − e^{−h·ln2/T½}).
- Calibration: quantiles shifted by the prior 20 sessions' residuals per group (R5-cal20s pattern);
  probabilities by rolling isotonic per group. Nightly recal, monthly retrain.
- Data: universe ≥ $20M prior 20-session mean dollar volume and σ > 0.3%, PIT; every minute scored.
  15 walk-forward blocks; the last 3 months sealed for one final verdict; CIs by bootstrapping days.
- A new standalone package in `ml/` (reads NAS parquet read-only; no imports of `tft_intraday`,
  `tft_vwap` or `sourcing_py`; the daily forecast enters as a parquet file).

## 7. First build and the go / no-go

1. Universe and row counts (measured: ~5,550 stock-days a day, ~1,475 with ≥ $20M dollar volume,
   2024-03).
2. Feature builder with no-look-ahead tests.
3. Targets and per-group climatology.
4. Stage A at d = 15 and d = 0.
5. The ladder, one input group at a time: price path (expect no edge) → + daily forecast → +
   market/sector → + VAP → + signed flow → + 1 s tier → + events.
6. **Go** only if rank IC (per minute, averaged per day) and the top-20 realised move beat every
   baseline with a CI clear of 0: climatology; daily-only; the open rule (rank by move since the
   open); step 1. Otherwise stop and write it up.
7. Docs: LEARNINGS, MODEL-STRATEGY Scoreboard + Q, STATUS Changes.

Open: the top-20 realised-move margin that counts as useful (the screener beat ATR by +1.56 pts).

## 8. Step 0 probe (implemented): `ml/scripts_intraday_probe.py`

Run: `uv run python scripts_intraday_probe.py [--parts features,ladder,scan,recs,retime] [--quick]`;
status in `<repo>/temp/intraday_probe/status.txt` (was `/tmp/mtap/…` before 2026-11-14), results in `report.txt` and `results.json` there.
Tests: `ml/tests/test_intraday_probe.py` (features and model inputs never move when bars from t − d on
change; targets match a brute-force loop and ignore bars before t; the firing rule; list hysteresis;
cap bands; near-extreme masks are PIT).

- **Data (read only):** NAS `bars_1m` 2021-10 → 2026-10 (with 04:00–09:30 for pre-market); R5-cal20s
  walk-forward folds (out of sample per day, 2023-01-23 →; blank before, and before 09:45); results
  windows from the 5y panel; quarter-end shares (`out/shares`) for market cap; `agg_1s` for re-timing.
- **Splits:** yearly walk-forward: train ≤ 2023 → test 2024; ≤ 2024 → 2025; ≤ 2025 → 2026-01…10-26.
  Discover on 2024–2025, confirm on 2026. Training: 80k random stock-minutes per month (2M per fit).
  Isotonic recalibration on the last 3 months of each training span. Climatology per 5-min bucket ×
  σ tercile × liquidity tercile.
- **Test points, three kinds** (user, 2026-11-01: random points dilute an edge):
  1. a grid: every stock, every 30 min from a random offset per day (the "all moments" view);
  2. **near the high / low so far** (PIT, usable live): p_t within 0.1σ of today's high (low) so far,
     with today's range ≥ 0.5σ; ~30k sampled per month each;
  3. **around the day's actual high / low** (±5 min of the final turning point, 09:45–15:45): hindsight,
     test-only, shows the ceiling of "if we knew it was the turn".
  Each kind is scored against the base rate at the same moments; results are kept apart.
- **Report:** skill per ladder step (Brier on touch ±0.5σ/60 min, race, volume balance; rank IC of
  lean; top-20 spread) vs climatology, daily-only, the open rule; opportunity map by time of day and by
  pre-registered slice; predictability map (discover vs confirm); recommendations per day; 1 s
  re-timing of new-high / low minutes.
- **Recommendations per typical day** (the user's question): on 2026, every minute, every eligible
  stock. A recommendation = the first minute a (stock, direction, level, h) has calibrated P(touch) ≥
  60%; it can't fire again for the same key until its window has ended (close window: once a day).
  Came true = touched within the window. Per day: count, came true, precision; median, p10, p90 over
  days; per target and per hour; the median size in %. Two counts: **all** and **edge-only** (the base
  rate at that moment < 60%). Three settings: all stocks at d = 0 (the ceiling), the **pipeline**
  (stocks on the d = 15 top-20 lists, d = 0 probabilities), all stocks at d = 15.

### Probe result (2026-11-01; details and CIs: LEARNINGS `tag-ml-intraday`, "Step 0 probe: results")
- **Size of the move: predictable.** Touch ±0.5σ within 60 min, 2026: +3.4% Brier skill from the daily forecast
  alone, +5.4% with the price path, +6.3% with everything at d = 0. Market, VAP, signed flow and results days add
  ≈ 0 at d = 15. Bursts (+17%), results days (+17%) and gaps (+13%) are the confirmed strong slices.
- **Direction: only at d = 0.** Rank IC of lean +0.043 [+0.039, +0.048] and top-20 spread +0.07σ at d = 0;
  ≈ 0 at d = 15 for every input group. **Stage A on 15-min-delayed data fails the go rule (§7.6).** The d = 0
  edge is unverified against bid–ask bounce (targets are measured from the last close); the bounce check
  (targets from the next minute's open) comes first.
- **Recommendations per day (2026, edge-only):** pipeline 1,614 signal keys a day, 68% came true (typical 41%);
  one target, +1% within 60 min, 66 a day on 43 stocks, 65.5% true (typical 33%). Calibration holds at common
  levels and is overconfident at rare ones (+1σ / 15 min).
- **Consequence for the build:** drop the full-market d = 15 ranker as the selector; score at d = 0 on the
  stocks MTap polls live, with the stitched features of §3F. Plan: [live-recommendations-plan.md](live-recommendations-plan.md).
- **Bounce check (P1, 2026-11-01; LEARNINGS "P1 bounce check"): the d = 0 direction edge was bid–ask bounce.**
  Measured from the next minute's open or VWAP, rank IC falls from +0.034 to +0.006 … +0.009 and the top-20 spread
  from +0.41σ to +0.05σ (CI touching 0); size skill (touch +6.2%, up first +4.2%) is unchanged. So no stage of
  this design has a direction edge on minute-bar inputs: Stage A and the d = 0 scorer are **not built**. What
  stays useful is the size side (touch probabilities, ranges), e.g. for target / stop sizing and options later.
  The live recommender sends base-rate scores (`baseline.climatology`) until a new input shows direction.

## 9. MTap → MBin exchange (contract v1.0-draft)

Contract: `MBin/docs/contracts/` (protocol, schemas, fixtures, REVIEW.md). Both sides edit it in
place, per its README; MTap owns RECOMMENDATION, EXIT_RECOMMENDATION and INTRADAY_FEED. **MTap owns the
signal (what, why, how likely, how long); MBin owns the money (budget, size, approval, execution).**
Nothing is final; MTap's answers are in REVIEW items 1–12.

- **RECOMMENDATION (BUY)** only from Stage B (d = 0). Publish when the stock is on the up list, the
  calibrated P(touch +target within the horizon) ≥ max(our threshold, `hints.min_probability`), the
  up-first race is above typical, and the price expiry trigger has not passed. It is sent in every
  `trading_state` (REVIEW item 12: MBin filters, so its outcomes record what was missed).
  - `score` = calibrated P(target before stop), 0–1, comparable across strategies; the expected move
    in % travels in `inputs`.
  - `probability` = `{value, target_pct, horizon}` (horizon `<n>m`, `close`).
  - `target_pct` = the up level with the chosen touch probability; `stop_pct` = the down level with
    20% chance to be touched first; `ttl_ms` until the time expiry trigger, capped at R½.
  - `market_cap` = shares outstanding (latest Massive ticker-overview snapshot) × the previous close;
    `cap_band` with the bands in §1; `sector` / `industry` from MTap's own copy.
  - `strategy_id` = `intraday.pressure`, `strategy_version` = the model's MODEL.md label.
- **EXIT_RECOMMENDATION:** where the down side acts on held positions (v1 has no shorts): target
  reached or price expiry, probability drop, momentum hazard, down pressure, stop echo, session end.
  Level-triggered, re-sent with `revision + 1`, `cleared` to withdraw.
- **Feedback:** every published rec and exit is graded against market data (not only fills);
  RECOMMENDATION_OUTCOME is stored next to it, never used as a training label.
- **In MTap:** the live scorer writes candidates to the internal stream `mtap:int:REC_CANDIDATE`;
  `sourcing-py recommender` (`python/sourcing_py/recommender/`, built 2026-11-01) maps them to the
  contract, validates against the vendored schemas (`schemas/contracts/`, sha256-pinned, checked
  against MBin's) and publishes. Phase 1 only: no EXIT_RECOMMENDATION / POSITIONS_SNAPSHOT yet.

## 10. Challengers and later
- A sequence encoder (PatchTST or a 1-D CNN over the last 120 min) on the same targets, adopted only
  if it beats the GBM with a CI clear of 0 (one model per job, no ensembles).
- Chronos-2 zero-shot as an outside benchmark; QRF as the check on Stage B quantiles.
- Options: implied volatility as input and benchmark.
- Dashboard: probabilities next to every percentile (a separate small change, design in the plan).
