# Intraday model v2: columns, targets and testing

Status: **schema v2 locked 2026-11-06; S1 code + tests done 2026-11-07 (`ml/intraday_ft/`, 39 tests on
synthetic frames); nothing built on real data yet (S2 waits for the OK).** Phase 1 = a sequence model (TFT, or an
equivalent such as PatchTST) on the pocket targets. Phase 2 = the GBM twin, the input ladder and the GBM
overlay, after Phase 1's report. The decisions behind each column are in MODEL-STRATEGY Q40; the earlier
design (Stage A/B GBM, Step 0 probe, P1) is in [intraday-design.md](intraday-design.md) and LEARNINGS
`tag-ml-intraday`. This file is the single dictionary. When `ml/intraday_ft/features.py` exists, its
`FEATURES` table is the source and this file must match it.

**Why v2.** P1 (2026-11-01) showed that minute-bar summaries predict how **big** the next hour's move is,
not which **way** once bid–ask bounce is removed. v2 adds second-level inputs (resistance, time at price),
a sequence model over the whole day, and targets that describe the volume above and below the entry over
time windows instead of a single touch.

## 1. Conventions

| term | meaning | example |
| --- | --- | --- |
| **σ** | The stock's typical daily move: std of the VWAP-to-VWAP log change over the **previous 20 sessions** (today not included) | σ = 2% on a $100 stock → 0.1σ = 0.2% = $0.20 |
| **20 sessions** | The previous 20 trading days before today | for 2026-11-06: 2026-10-09 → 11-05 |
| **σ₁ₘ** | The stock's 1-minute return std for that time of day (15-min bucket), over 20 sessions | σ = 2% → σ₁ₘ ≈ 0.10% midday |
| **normal(x)** | The median of x over the previous 20 sessions, for the same minute of the day where it varies by time | normal volume at 10:30 = 80k |
| **typical day** | The median regular-session volume (minute bars, 09:30–16:00) of the previous 20 sessions | 50M shares |
| **price unit** | `ln(x / prev_vwap) / σ`, clipped to ±10; `prev_vwap` = yesterday's regular-session VWAP | prev VWAP $270, σ 1.5%, price $272.70 → +0.66 |
| **volume unit** | `ln(x / normal(x))`, or a share of the typical day where named "share" | |
| **market unit** | the % log change of each instrument against **its own** reference (never across instruments) | §4 |
| **entry P₀** | The first trade after the decision moment: minute i+1's open (minute tier), the first trade after the decision second (second tier). Only the zero line for targets, **not** a prediction | minute 10:14 closes at $272.70 (ask), 10:15 opens $272.66 → P₀ = $272.66 |
| **decision minute i** | The model stands at the end of minute i, with everything up to and including minute i | |
| **held price** | The last trade price carried forward to every later second until the next trade | trades at :05 $100.01 and :09 $100.03 → :06–:08 hold $100.01 |
| **poll grid** | The seconds a live E*TRADE poll would see. Training samples Massive 1 s bars on the same grid | 1 s from the switch; 2 s before |
| **stock id** | `composite_figi` (else CIK). The ticker is only a label | FI → FISV keeps one id |

**Leakage rules (tests enforce each):**
- changing any bar after minute i never changes an input at i;
- σ, σ₁ₘ, normals and the typical day use prior sessions only;
- targets use only bars after the entry;
- a zig-zag pivot exists only from the minute it is confirmed (no repainting);
- the stock is left out of its own sector / market move;
- whether a stock later delists is never an input;
- daily forecasts are walk-forward out-of-sample only.

## 2. Tiers, universe and dates

| tier | source (training) | source (live) | first date |
| --- | --- | --- | --- |
| **M** minute | `bars_1m` (04:00–20:00 ET; ~84% of the day's volume, the auction prints are in no minute bar) | Massive to t − 15 min + minutes built from E*TRADE polls for the last 15 | 5-year edge (~2021-11) |
| **S** second | `agg_1s` `massive_rest` on the poll grid (every `*.parquet` of a day: `day-D` + `day-D.extra`) | E*TRADE polls from 09:30 | 2021-11-08 (D1 done 2026-11-07) |
| daily forecast (Chunk 0) | `R5~px-cal20s` (09:30), `R5-cal20s` (09:45) walk-forward test predictions | the live daily job | 2023-01-23 |
| SPY | `out/market/spy_1m.parquet`, `spy_daily`, `spy_dividends` (Massive, `sourcing-py market fetch`); 1 s in `agg_1s` (in the `.extra` files 2024-10-21 → 2026-10-16) | E*TRADE SPY | 2021-11-08 |
| SPX daily | `out/market/spx_daily.parquet` from `docs/transient/spx-history.csv` | the same file + E*TRADE SPX | 2005-03-01 |

**Universe per day (point in time):** Massive type CS as of the time (delisted and renamed stocks
included), minus the massive_1s `exclusion()` names (notes, preferreds, funds, rights) and ADRs; prior-20
mean dollar volume ≥ $20M, σ > 0.3%, ≥ 3 prior sessions. The live set (E*TRADE list symbols) is a flag
and a report slice, not a filter.

**The frames hold the day's top 1,000 eligible stocks by prior dollar volume** (`LIVE_N`,
`intraday-ft-build --stocks`). They stand in for the live set, because the E*TRADE lists exist only from
2026-10-12. The whole universe would be ~2.5 GB of float16 sequences a month, while the model is used on
~1,000 live symbols. `live_set` = in that top 1,000.

**Cache** (`/tmp/mtap/intraday_ft/`, env `MTAP_INTRADAY_FT_CACHE`; the NAS is read only):
- the layout is `frames/month=YYYY-MM/` with:
  - `rows.parquet`: stock-days, keys and statics;
  - `dec.parquet`: decisions with every target and `pers_*`;
  - `seq.npy`: stock-days × 390 × 98 float16;
  - `meta.json`: the column order;
- the full 2023-01 → 2026-10 build is ≈ 966k stock-days and ≈ 74 GB;
- `status.txt` holds the live status.

**`bars_1m` folders.** Each `month=YYYY-MM` folder's file is named after its quarter
(`bars-2026-01-01_2026-03-31.parquet`) but holds only that month's rows (checked 2026-11-07). The reader
still cuts to the month's ET days as a guard.

## 3. Inputs

TFT roles: **static** = one value per stock-day; **known** = known in advance for every step, including the
decoder; **observed** = known only up to minute i.

### Chunk 0: the daily forecast

The 19 daily quantiles are near-copies of each other, so they go in as five shape numbers (all in price units
vs yesterday's VWAP). Which forecast: 09:31–09:44 the 09:30 run, 09:45 → close the 09:45 run.

| column | definition | role | example (AAPL; q05 −1.2, q25 −0.3, q50 +0.1, q75 +0.5, q95 +1.8) |
| --- | --- | --- | --- |
| `fc_med` | q50 | known (steps at 09:45) | +0.1 |
| `fc_iqr` | q75 − q25 | known | 0.8 |
| `fc_w90` | q95 − q05 | known | 3.0 |
| `fc_skew` | ((q95 − q50) − (q50 − q05)) / (q95 − q05) | known | (1.7 − 1.3)/3.0 = +0.13 (longer up tail) |
| `fc_pup` | P(today's VWAP > yesterday's), read off the quantile curve | known | 0.55 |
| `fc_stage` | 0930 or 0945: which run the row reads | known | 0930 at 09:40, 0945 at 09:45 |
| `band_px` | percentile of the price now inside the forecast's quantile curve | observed | price at +0.5 → 0.75 |
| `band_vwap` | the same for today's VWAP so far | observed | |

### Chunk 1: the stock's own day

**Price and volume per minute (M tier).**

| column | definition | example |
| --- | --- | --- |
| `m_o m_h m_l m_c m_vw` | minute OHLC and VWAP, price units | |
| `m_rvol` | ln(minute volume ÷ normal for that minute) | 120k vs 80k → ln 1.5 = 0.41 |
| `m_n` | ln(minute trades ÷ normal) | |
| `vol_used` | regular-session volume so far ÷ typical day (share; the user's main volume feature) | 15M of 50M by 10:30 → 0.30 |
| `vol_pace` | volume so far ÷ normal volume by this minute | normal by 10:30 = 12.5M → 1.2. Check: `vol_used` = `vol_pace` × normal share by now = 1.2 × 0.25 = 0.30 |
| `from_high from_low` | price now vs today's high / low, price units | |
| `min_since_high min_since_low` | minutes since the high / low was set | |

Both volume features use minute-bar volume on both sides, so the ~19% of daily volume printed in auctions
(not in any minute bar) doesn't skew them.

**Resistance: the volume it takes to move 1% (S tier; M fallback).**
- Each second with a trade gets its % change since the previous trade and its volume. A second with no
  change goes to the side of the last change (tick rule): volume absorbed without moving the price.
- `res_up_w` = up-side volume (% of a typical day) ÷ total % of the up-moves, over window w ∈ {1, 5, 15, 30
  min, open}. `res_dn_w` is the mirror. `res_bal_w` = ln(res_dn ÷ res_up): > 0 = cheaper to go up.
- `vol_to_up1` / `vol_to_dn1` = today's volume already traded between the price now and ±1%, % of a typical
  day: supply or demand sitting in the way.
- M fallback (no seconds): minute volume ÷ |minute VWAP change|, split by its sign.

| second | price | volume | side |
| --- | --- | --- | --- |
| 10:00:01 | $100.00 | 2k | start |
| 10:00:02 | $100.02 | 3k | up (+0.02%) |
| 10:00:03 | $100.02 | 5k | up (no change, absorbed) |
| 10:00:05 | $100.01 | 1k | down (−0.01%) |

Window example (typical day 50M): up-moves +0.40% on 300k (0.6%) → `res_up` = 1.5% of a day per 1% up;
down-moves −0.20% on 400k (0.8%) → `res_dn` = 4.0; `res_bal` = ln(4.0/1.5) = +0.98. Price $100 with 4M
already traded between $100 and $101 → `vol_to_up1` = 8%.

**Time at price: where the price lingers (S tier).** Every second counts once, for the held price, traded
or not. OHLC can't show this.

| column | definition |
| --- | --- |
| `tpo_poc` | the 0.1σ bin with the most seconds today ("point of control"), price units vs the price now |
| `tpo_lo70 tpo_hi70` | edges of the value area holding 70% of today's seconds (method below) |
| `tpo_lo50 tpo_hi50` | the same at 50% |
| `time_above` | share of today's seconds above the price now |
| `dwell_now` | share of today's seconds in the price-now bin |
| `*_hl30` | the same five with each second weighted by a 30-min half-life (recent time weighs more) |
| `linger_hi linger_lo` | share of today's seconds within 0.25σ of today's high / low (≈ 0 = a spike, large = a plateau) |
| `visits_hi visits_lo` | separate visits there (a visit ends after ≥ 30 s away) |
| `dwell_hi dwell_lo` | mean seconds per visit |
| `accept` | time share ÷ volume share in the price-now bin: > 1 quiet drift, < 1 fast heavy prints |
| `twap` | per minute: time-weighted mean price, price units |
| `linger_pos` | per minute: where in the minute's range the time was spent, 0 = low, 1 = high |
| `twap_gap` | per minute: VWAP − TWAP, price units (> 0 = volume printed above where time was spent) |

**The value area (how the 70% band is found).** A counting rule, not a fit:
1. Count today's seconds into 0.1σ bins.
2. Start the band at the bin with the most seconds (POC).
3. Look at the next bin above the band and the next bin below it, and add the one with more seconds
   (ties: the one nearer the price now).
4. Repeat until the band holds ≥ 70% of the seconds. Its outer edges are `tpo_lo70` / `tpo_hi70`.

70% is the market-profile convention (≈ ±1 std of a normal = 68%); the 50% band is there so the model
can show which one it uses.

Example: price now $100.00, σ = 2% (bin = $0.20), 09:30–10:30 = 3,600 s; 70% = 2,520 s.

| bin (σ vs now) | −0.4 | −0.3 | −0.2 | **−0.1** | 0.0 | +0.1 | +0.2 | +0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price ≈ | $99.20 | $99.40 | $99.60 | $99.80 | $100.00 | $100.20 | $100.40 | $100.60 |
| seconds | 200 | 400 | 700 | **900** | 600 | 450 | 200 | 150 |

| step | compare | added | band holds |
| --- | --- | --- | --- |
| start | POC | −0.1 | 900 |
| 1 | −0.2 (700) vs 0.0 (600) | −0.2 | 1,600 |
| 2 | −0.3 (400) vs 0.0 (600) | 0.0 | 2,200 |
| 3 | −0.3 (400) vs +0.1 (450) | +0.1 | 2,650 ≥ 2,520, stop |

Result:
- `tpo_lo70` = −0.25σ ($99.50), `tpo_hi70` = +0.15σ ($100.30);
- `tpo_poc` = −0.1;
- `time_above` = 800/3,600 = 0.22, `dwell_now` = 600/3,600 = 0.17.

The price is near the top of where it has spent the morning.

Other examples:
- **At the high.** The high is $100.70, with 110 s within $0.50 of it over 3 visits → `linger_hi` 0.03,
  `visits_hi` 3, `dwell_hi` 37 s: a spike.
- **Inside a minute.** 10:31 held $100.00 for 40 s and $100.30 for 20 s:
  - `twap` = $100.10, `linger_pos` = 0.33;
  - with 80% of its volume at $100.30, VWAP = $100.24 and `twap_gap` = +$0.14.

**Zig-zag: the trend without the wiggles (M on minute VWAPs, S on the grid).** A leg ends only when the
counter-move reaches k·σ₁ₘ, with k = 2 (`_k2`) and k = 5 (`_k5`).

| column | definition |
| --- | --- |
| `zz_dir` | +1 up leg, −1 down leg |
| `zz_len` | the leg's size so far, price units (σ) |
| `zz_age` | minutes since the leg started |
| `zz_pull` | current counter-move ÷ threshold, 0…1 (near 1 = about to break) |
| `zz_prev` | the previous leg's size |
| `zz_legs` | legs today |

Example (σ = 2%, σ₁ₘ = 0.10%, k = 2 → threshold 0.20%):
1. The VWAP goes $100.00 → $100.50: an up leg, `zz_len` 0.25.
2. Back to $100.35: `zz_pull` 0.75, still up.
3. Up to $100.60: `zz_pull` 0.
4. Down to $100.38: the counter-move is 0.22% ≥ 0.20%, so the up leg ends at $100.60. That is confirmed at
   this minute, not before.

A $0.05 wiggle on a $300 stock (0.017%) is 0.17σ₁ₘ and never ends a leg.

### Chunk 2: sector and market (observed)

| column | definition | example |
| --- | --- | --- |
| `sec_mv` | cap-weighted mean of the **other** sector members' ln(price ÷ own prev VWAP); caps = shares × previous close, each capped at 25%; members = the live set | MSFT $3.8T +0.6%, NVDA $4.5T +1.2%, ORCL $0.6T −0.8% → (2.28 + 5.40 − 0.48)/8.9 = +0.81% |
| `sec_mv_ew` | the equal-weight twin | (0.6 + 1.2 − 0.8)/3 = +0.33%: the big names, not the breadth |
| `ind_mv` | `sec_mv` over the industry if it has ≥ 5 members, else the sector | |
| `mkt_mv` | cap-weighted move of the whole universe, the stock left out | |
| `resid` | the stock's move − β₆₀ × `spy_mv` | |
| `breadth` | share of the live set above its own prev VWAP | |
| `spy_mv` `spy_vwap_mv` `spy_rvol` | §4 | |

Sector = the 2-digit `industry_code` (TRBC); industry = the full code. Codes that aren't digits ("unknown")
are dropped.

### Chunk 3: the previous session and reference data (static)

| column | definition |
| --- | --- |
| `y_close` | yesterday's close vs yesterday's VWAP, price units |
| `y_post` | yesterday's 20:00 last trade vs VWAP |
| `gap` | today's official open vs yesterday's close |
| `pm_open` | the official open vs the last pre-market trade (exists from 09:31) |
| `pm_vwap` `pm_rvol` | pre-market VWAP (price units) and volume vs normal, cut at 09:15 (`pm555`) for 09:31–09:44 rows and at 09:30 (`pm570`) from 09:45, in training and live |
| `spy_gap` `spx_y` `spx_gap` | §4 |
| `sigma` `log_dvol` `log_mcap` `beta60` | σ, ln mean dollar volume (20 sessions), ln market cap (shares × previous close), β vs SPY over 60 sessions |
| `results` | results-release day flag (EDGAR 8-K 2.02, pit) |
| `cap_band` | categorical: mega ≥ $200B, large $10–200B, mid $2–10B, small $250M–2B, micro < $250M |

### Time and flags

| column | role | definition |
| --- | --- | --- |
| `minute` `to_close` `dow` | known | minute of the session, minutes to 16:00, weekday |
| `normal_rvol` | known | the normal volume share of each minute (the U-shaped curve) |
| `poll_tier` | static | grid of the stock's second tier that day: 1 s, 2 s, 20 s, none |
| `pm_cut` | known | 555 or 570 |
| `bin_cut_at_close` | known (decoder) | the bin was shortened by the close |

### Overlay (static categorical, maskable)

`sector`, `industry`, `symbol` (the stock id). In training each is replaced by `unknown` at random
(p = 0.5 symbol, 0.3 the others; always for stocks with < 60 sessions). At test time every row is scored 4
ways, `none` / `sec` / `ind` / `all`, so the dashboard can switch the overlay on and off without running a model.

## 4. SPY and SPX: separate columns, never compared as prices

SPY is about SPX ÷ 10, but not exactly, and the gap moves:

| date | SPX close | SPY close | SPY ÷ SPX | SPX day | SPY day |
| --- | --- | --- | --- | --- | --- |
| 2026-11-02 | 7,541.40 | 752.10 | 0.09973 | | |
| 2026-11-03 | 7,568.92 | 754.35 | 0.09967 | +0.36% | +0.30% |
| 2026-11-04 | 7,688.62 | 766.81 | 0.09973 | +1.58% | +1.65% |
| 2026-11-05 | 7,353.73 | 733.00 | 0.09968 | −4.36% | −4.41% |

Why the two differ:
- SPY's fee (0.09% a year) and its quarterly dividends lower the ratio over time.
- On each ex-dividend date SPY opens lower by the dividend (2026-09-18: $1.89 ≈ 0.25%) while SPX doesn't.

**Rule:** each instrument is only compared with itself, as a % log change, in its own column.

| column | definition | training | live | input? |
| --- | --- | --- | --- | --- |
| `spy_mv` | ln(SPY now ÷ SPY yesterday's close) | Massive SPY minute / 1 s | E*TRADE SPY | yes, observed |
| `spy_vwap_mv` | ln(SPY now ÷ SPY yesterday's VWAP) | same | same | yes, observed |
| `spy_rvol` | ln(SPY minute volume ÷ normal) | same | same | yes, observed |
| `spy_gap` | ln(SPY open ÷ SPY yesterday's close) + the dividend ÷ yesterday's close on ex-dates | Massive SPY + `/v3/reference/dividends` | same | yes, static |
| `spx_y` | yesterday's SPX close-to-close log change | SPX file | SPX file | yes, static |
| `spx_gap` | ln(SPX open ÷ yesterday's close) | SPX file (opens checked first: index opens often equal the previous close, as in the 2005 rows) | E*TRADE SPX | yes, static |
| `spx_live_mv` | ln(SPX now ÷ yesterday's close) | none (`I:SPX` is 403 on this plan) | E*TRADE SPX | **no**, monitor only |

Examples:
- **Agreement.** 2026-11-06 10:00 with SPY 740.00 and SPX 7,423:
  - `spy_mv` = ln(740/733) = +0.95%;
  - `spx_live_mv` = ln(7,423/7,353.73) = +0.94%.

  A daily check logs `spy_mv − spx_live_mv`; it should be within a few bp except on SPY ex-dates.
- **The wrong way.** SPY × 10 vs SPX gives ln(7,400/7,353.73) = +0.63%, a 0.3% fake move.
- **An ex-dividend day.** 2026-09-18 (real numbers; SPY paid $1.89 ≈ 0.25%):
  - the raw SPY gap is −0.17%;
  - `spy_gap` = −0.17% + 0.25% = **+0.08%**;
  - `spx_gap` = +0.25%. The add-back moves SPY from the wrong sign to near SPX. Over 20 ex-dates (2021-11 →
    2026-11) the median |SPY gap − SPX gap| drops from 41 bp to 16 bp; on ordinary days it is 7 bp (LEARNINGS
    `tag-price`).
- **Massive's `adjusted=true` adjusts splits only, not dividends** (checked 2026-11-06), so the add-back is ours.
- **The SPX file's opens** are real from 2014 on (≤ 0.4% of days equal the previous close); before that many
  opens are copies of the previous close. The model uses 2021-11 on, so `spx_gap` is usable as is.

## 5. Targets: the pocket space over durations

Nothing is predicted at an exact future minute. Every target is an aggregate over a **duration bin** after
the entry. The bins are disjoint, and are cut at the close (flagged).

| bin | b1 | b2 | b3 | b4 | b5 |
| --- | --- | --- | --- | --- | --- |
| minutes after entry | 0–5 | 5–15 | 15–30 | 30–60 | 60 → close |

Minute j's volume is placed at its VWAP: dⱼ = ln(vwⱼ / P₀) / σ, sⱼ = vⱼ / Σv over the bin.

| target (per bin) | definition | reads as |
| --- | --- | --- |
| `bulge` | Σ s·d over d > 0 | volume above the entry × how far above (σ) |
| `sag` | Σ s·\|d\| over d < 0 | the same below |
| `mass_up` | Σ s over d > 0 | share of the bin's volume traded above |
| `cm_up` | bulge ÷ mass_up | concentration: centre of mass of the volume above (σ); masked if mass_up < 5% |
| `cm_dn` | sag ÷ (1 − mass_up) | the same below; masked if < 5% |
| `net` | bulge − sag | the direction readout |
| `rvol` | ln(bin volume ÷ normal for those minutes) | |
| `occ_up` `occ_dn` | share of the bin's seconds held above / below P₀ (S tier) | the time twin |
| `pi_up` `pi_dn` | bulge × (bin volume ÷ normal) ÷ bin minutes; undefined (masked) where the normal volume for those minutes is 0 | **rate**: pocket intensity per minute |

**Row-level targets:**
- `t50_up` / `t50_dn`: the minute when half of the 0–60-min up / down pocket (Σ vⱼ·dⱼ) has traded;
- `t_peak_up` / `t_peak_dn`: the bin with the highest intensity (5 classes);
- `vol_to_up1` / `vol_to_dn1`: volume (share of a typical day) traded until the first ±1% touch within 60
  min, set to the cap 1.0 if it isn't touched;
- the ±0.5σ race within 60 min (secondary, comparable with P1): class 0 = up first, 1 = down first,
  2 = neither or both in the same minute;
- `mass_up60` `bulge60` `sag60` `net60`: the pocket of the whole 60 minutes after the entry (the run-file
  readouts).

**Entry fallback.** If minute i+1 has no trade, P₀ = the held price at i.

**Persistence** (the baseline, `pers_*`) = the pocket of minutes i−14 … i, measured from the open of
minute i−14. It is known at i.

**Decision sampling.** One decision every 5 minutes per stock-day, between minutes 4 and 384. The start
is staggered at random per stock-day, so every minute of the day is covered.

**Quantiles.** Each real target is learnt as P10 / P25 / P50 / P75 / P90. "How much is possible" = the upper
quantiles of bulge and sag.

**Worked example (b1; P₀ = $100.00, σ = 2%, normal b1 volume 800k).**

| minute | VWAP | volume | d | s |
| --- | --- | --- | --- | --- |
| 1 | $100.10 | 200k | +0.05 | 0.20 |
| 2 | $100.30 | 300k | +0.15 | 0.30 |
| 3 | $100.20 | 100k | +0.10 | 0.10 |
| 4 | $99.90 | 250k | −0.05 | 0.25 |
| 5 | $99.80 | 150k | −0.10 | 0.15 |

Result:
- `bulge` 0.065, `sag` 0.0275, `mass_up` 0.60;
- `cm_up` 0.108 (≈ $100.22), `cm_dn` 0.069 (≈ $99.86);
- `net` +0.0375;
- bin volume ÷ normal 1.25 → `pi_up` = 0.065 × 1.25 / 5 = 0.016 per minute;
- 190 of 300 s above $100 → `occ_up` 0.63.

**Rate readouts (from the predicted bin curve).**

| readout | definition | example (predicted `pi_up` b1…b5 = 0.016, 0.010, 0.006, 0.004, 0.002; b1 climatology 0.010) |
| --- | --- | --- |
| build | predicted b1 intensity ÷ climatology | 1.6× normal |
| half-time | `t50` P50 | 8 min |
| fade | slope of intensity b2 → b4 (< 0 = fading, ≥ 0 = still building) | fading |
| span | burst if t50 ≤ 10 min, sustained if ≥ 30 | burst |

## 6. How the model is tested

1. **The real outcomes are computed first.** For every test row, the build computes the realised targets from
   the bars after the entry and stores them with the row.
2. **The model predicts blind.** It sees only the inputs. Its predictions are joined to the realised targets
   afterwards.
3. **The test months are never seen in training.**
   - Fold B (first): train to 2025-12, test 2026-01 → 10.
   - Fold A: train to 2024-12, test 2025.
   - Early stopping uses the last 2 months of the training span.
4. **The scores**, each against the realised targets:
   - Brier for probabilities;
   - pinball loss for quantiles;
   - coverage of P10–P90 (should be 80%);
   - direction: the rank IC of predicted `net` P50 vs realised `net` across stocks at the same minute, and
     the AUC of P(mass_up > 50%).
5. **Baselines, scored against the same realised targets.** Skill = 1 − model loss ÷ baseline loss.
   - **Climatology:** the training-span average for that 5-min bucket × σ tercile × liquidity tercile.
   - **Persistence:** the last 15 minutes' realised pocket, repeated.
   - P1's numbers are the reference: race AUC 0.514, mass_up AUC 0.506 vs a base of 0.504.
6. **Go rule (changed 2026-11-07 by the user, MODEL-STRATEGY Q41).** The direction of the move is **not**
   required. What counts is the **pocket shape**: how much volume sits above and below the entry, how far out
   it concentrates (`cm_up` / `cm_dn`, the high / low centres of mass), and how busy each duration bin is.
   Go = on 2026 (fold B), and holding on 2025 (fold A):
   - **beats climatology:** pinball skill of `cm_up`, `cm_dn`, `bulge`, `sag` and `rvol` clear of 0 in every
     bin, and their P50 rank stocks better than climatology's (rank IC CI above climatology's);
   - **calibrated:** P10–P90 coverage within 75–85% and the P50 within 0.45–0.55 of outcomes, or fixed by a
     recalibration fitted on the validation months.

   The direction numbers stay in the report as a record; they don't decide.
   Was (2026-11-06): direction skill over both baselines clear of 0; size skill alone didn't count.

Example, P(mass_up_60 > 50%) on three rows:

| row | happened | model | (model − real)² | climatology | (clim − real)² |
| --- | --- | --- | --- | --- | --- |
| AAPL 10:30 | 1 | 0.70 | 0.090 | 0.50 | 0.250 |
| MSFT 10:30 | 0 | 0.40 | 0.160 | 0.50 | 0.250 |
| NVDA 10:30 | 1 | 0.45 | 0.303 | 0.50 | 0.250 |
| mean | | | 0.184 | | 0.250 |

Skill = 1 − 0.184/0.250 = +26%. The real report uses ~250k rows with a day-bootstrap CI.

**As built (S1, 2026-11-07; `ml/intraday_ft/tft.py`, `windows.py`, `report.py`).**
- **Model: a thin TFT of our own**, built from pytorch-forecasting's sub-modules (VSNs, GRNs, interpretable
  attention). It is the plan's fallback, used from the start because the library's `TimeSeriesDataSet`
  can't take whole-day encoders with several decisions per day or row-level targets.
  - It runs once per stock-day over the 390 minutes, causally: an LSTM, then attention masked to earlier
    minutes. Its state at minute i is the encoder for decision i, so the day isn't re-read per decision.
  - The decoder has one step per bin, fed with the known inputs: bin minutes, normal volume share,
    minutes to close and the cut flag.
  - The encoder ablations (`--encoder 120` / `60`) re-read only each decision's last N minutes. The
    variable selection is per minute, so it still runs once per day; only the LSTM and the attention run
    per window (a zero minute pads windows that start before 09:30).
  - Row heads give the row quantiles and the class heads (race, `t_peak_*`).
  - Quantiles are ordered by construction: the median ± cumulative softplus.
- **Pre-market cut, per minute.** `pm_vwap` / `pm_rvol` go in as two sequence columns. Minutes 0–14 hold
  the 09:15 cut and minutes 15 on hold the 09:30 cut, so the 09:30 cut never reaches a decision before
  09:45.
- **Standardisation:** each column's median / IQR over the training months, clipped to ±10, NaN → 0.
- **Loss:**
  - the pinball loss per target, divided by that target's training IQR so every target weighs about the
    same;
  - plus 0.2 × the cross-entropy of the class heads;
  - masked targets (`cm_*` below 5%, bins past the close) weigh 0.
- **Overlay masking in training:** sector p = 0.3, industry 0.3, symbol 0.5. A symbol needs ≥ 60 training
  sessions, otherwise it is always `unknown`.
- **Test thinning:** one decision every 15 min (`--test-every 3`) on 25% of the test stock-days
  (`--test-frac 0.25`). That keeps the scoring and the run file at a few million rows.
- **Climatology** = the training span's P10…P90 per cell (5-min bucket × σ tercile × dollar-volume
  tercile). Event rates are per cell.
- **Persistence:**
  - quantile targets: `pers_<target>` repeated as every quantile;
  - events: the training rate given the sign of the past-15-minute `net`.

## 7. Run-file columns

`ml/runs/intraday_ft/<stamp>-ft-B[-noS][-enc120]/fold_B/intraday_ft_pred.parquet`: one row per stock ×
decision minute.

| column | meaning |
| --- | --- |
| `date` `minute` `stock_id` `symbol` `sector` `industry` `cap_band` `live_set` `delisted_later` `poll_tier` `sigma` `log_dvol` | keys and slices |
| `<readout>__{none,sec,ind,all}` | each readout under each overlay setting |
| `real_<target>` | the realised targets, plus `real_pers_*` (persistence) |
| `dec_row` | the row's index into `quantiles.npy` |

**Readouts:**
- `p_mass_up60` = P(mass_up_60 > 50%);
- `p_bulge60` = P(bulge_60 ≥ 0.5σ), `p_sag60` = P(sag_60 ≥ 0.5σ);
- `net60_p10` / `p50` / `p90`;
- `t50_up_p50`, `t50_dn_p50`;
- `p_burst_up`, `p_burst_dn` = P(t50 ≤ 10 min);
- `p_race_up`, `p_race_dn`.

The probabilities are read off the predicted quantiles, linear between them, with 0.05 / 0.95 outside
P10 / P90.

**Next to it:**
- `quantiles.npy`: decisions × 63 targets × 5 quantiles, float16, the `all` overlay. `quantiles.json`
  names the targets.
- `model.pt`, `prep.json` (scaler + vocab), `history.json`.
- At the run level: `config.json`, `report.json`, `report.md` and MODEL.md, written by
  `intraday-ft-report`.

The dashboard's "Intraday pockets" page reads only this file, and its MODEL.md (which records the daily run
label used for Chunk 0).

## Change log

| date | change |
| --- | --- |
| 2026-11-06 | Created: schema v2 locked (Q40). Phase 1 sequence model, Phase 2 GBM. |
| 2026-11-06 | S0 checks: `adjusted=true` doesn't adjust dividends; `spy_gap` example now uses 2026-09-18's real numbers (SPX opened +0.25%, not flat); SPX file opens usable from 2014. The 1 s universe (`--from-bars`) is one list per run range, not per quarter. |
| 2026-11-07 | S0 data done: S tier from 2021-11-08 (D1, D2, D1b `.extra` files), SPY/SPX in `out/market/`. Readers take every `*.parquet` of an `agg_1s` day. |
| 2026-11-07 | S1 code + tests: `intraday-ft-build` / `intraday-ft` / `intraday-ft-report`. As built: frames = the top 1,000 eligible stocks per day (live stand-in), cache layout and size (§2); race classes, vol_to cap 1.0, the 60-min readout targets, entry fallback, persistence, staggered 5-min decisions (§5); a thin TFT of our own, the per-minute pre-market cut, IQR-scaled loss, test thinning, the baselines (§6); actual readout names, `quantiles.npy` (§7). `bars_1m` month folders hold one month in a quarter-named file. |
| 2026-11-07 | S2 check (2026-01): 20,000 stock-days, 1.52M decisions, S tier 100%. Build streams each `bars_1m` month once (WARM = 4 months of summaries before it). The S tier runs per day over `--workers` processes, with the 1 s zig-zag on all of a day's stocks together (sampled at minute ends; tests: equal to the full series and to one row at a time): 3 min a month (was ~6 h). Built months are skipped (`FRAME_VERSION` + settings in `meta.json`). Highest input NaN shares: `res_*_1` 0.18–0.27 (no up or down move in the last minute), `zz_prev_k5` 0.12, the `fc_*` 0.05 (days with no daily forecast). Target NaN: `cm_up/cm_dn` 0.27 (b1) … 0.40 (b5), by design: masked where one side holds < 5% of the bin's volume (~27–31% of rows), plus bins past the close. |
| 2026-11-07 | Zig-zag cost: in the build it is Python per-step overhead (23,400 steps per stock-day, replayed for every day of history), not the work itself; batched over a day's 1,000 stocks it is 1.7 s per day. Live it is incremental: one state update per stock per poll (~75 µs per second for 1,000 stocks), so it stays an input, not an overlay. To keep training and live identical, the live scorer must call the same step code as `zigzag.py`. Its value is measured by a "no zig-zag" ablation. |
| 2026-11-07 | S3 first timing check: loss NaN from step 1. Cause: `pi_up` / `pi_dn` = ±inf on 0.1–0.7% of rows (bins whose minutes have 0 normal volume, mostly b1–b4), and float16 overflow of large finite values; the batch turned masked targets into float max (`nan_to_num`), so masked × 0 = NaN. Fix: masked targets are 0 in the batch, `pi_*` is NaN (masked) when the normal volume is 0, training stops on the first non-finite loss. The built frames stay valid (inf is masked like NaN), no rebuild. Largest finite `pi_*` ≈ 3·10⁴ (2023-01): a heavy tail the IQR-scaled pinball handles; not clipped. |
| 2026-11-07 | S3 second timing check: training and scoring passed, the report failed on undefined realised targets (bins past the close, no trade). Fixes: event rates as float (a bool column with NaN is object dtype), the top/bottom-20 ranking on the rows it keeps. The end-to-end test now puts NaN into the realised targets. Timing (fold B, whole day): ~0.5 s a training step, scoring 25% of the test days ≈ 12 min. |
| 2026-11-07 | Encoder ablations were ~19 s a step (~40× IA1): each window re-ran the variable selection over all 98 inputs. It is per minute, so it now runs once per day and only the LSTM + attention run per window (test: equals the explicit windows). CPU bench at batch size: whole day 0.55 s, enc120 1.42 s, enc60 0.84 s a step, so the ablations keep IA1's training budget (no cut). |
| 2026-11-07 | IA1 (fold B) result: direction none, size real (LEARNINGS "Intraday v2 IA1"). Two readout limits found: `p_bulge60` / `p_sag60` come from `cdf_at` on the 5 quantiles, which can't go below 5% (real rate 1.7%, Brier −7%), so rare-event readouts need a direct head or a fitted tail; `race_auc_up` mixes activity with side (79% of rows = neither), the side-only AUC is 0.508. Both to fix before S5; not yet changed. |
| 2026-11-07 | **Go rule changed by the user (Q41):** direction not required; the pocket shape decides (`cm_up` / `cm_dn`, `bulge` / `sag`, `rvol` per bin: skill vs climatology, a better ranking than climatology, calibration). The report adds a "Pocket shape" table (share of outcomes at or below each quantile, model vs climatology coverage, rank IC of P50 vs climatology's) and the side-only race AUC; IA1's report was rebuilt with it (old copy `/tmp/mtap/IA1-report.v1.md`). |
| 2026-11-07 | **Two readout fixes, no retrain:** (1) `cdf_at` tails past P10 / P90 are exponential (decay fitted to the outer quantile pair), not a flat 5%: on IA1, `p_bulge60` / `p_sag60` Brier skill vs climatology −7.7% / −7.1% → **+2.5% / +2.6%**, AUC 0.86 (mean p 0.019 vs real 0.017). (2) `bulge` / `sag` quantiles below 0.01σ read as 0 (the quantile loss can't sit on the point mass at 0, 22–31% of bins): b1 pinball skill −1.5% / −3.0% → +1.1% / −0.9%, b2–b5 +1.8 … +2.1%. The shape table shows below .. at-or-below shares (ties at 0). The report re-reads `all` from `quantiles.npy`, so older runs score like for like; new runs write the new readouts. DUST picked on fold B, checked on fold A. |
| 2026-11-08 | **Rolling recalibration** (`intraday_ft/recal.py`; report writes `fold_k/recal.parquet` + a "Recalibrated" table): per day, target and quantile the offset from the previous 20 test sessions' errors (past only; the cal20s rule, not the validation months). All 25 shape targets calibrated on both folds after it (18 raw). The report also writes `clim.parquet` / `clim_rates.parquet` (climatology per cell) so the dashboard grades without ml/ code. |
| 2026-11-08 | **Ablations (fold B):** no 1 s tier = same or better shape skill (25/25 pass); a 120-min encoder = same or better (`rvol` +2 pts); 60 min worse (early stop at epoch 9). The S-tier inputs and the whole-day encoder aren't needed for the pocket shape (LEARNINGS "Intraday v2 IA1" §6). |
| 2026-11-08 | Dashboard Intraday → **Pockets model** reads the run files (§7 + the three side files); sector codes are TRBC economic sectors (57 = Technology). |
