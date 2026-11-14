# Intraday model v2: data, columns, targets, training, testing and live use

Status (2026-11-14):
- **Built and tested on real data.**
  - Schema v2 was locked 2026-11-06.
  - S1 code and tests were done 2026-11-07.
  - The S2 frames for 2023-01 → 2026-10 were built 2026-11-07.
- **Trained and scored:** IA1 (fold B, 2026), IA1A (fold A, 2025), and the ablations noS / enc120 / enc60 (fold
  B, 2026-11-08).
- **Read-outs built:** rolling recalibration, Moves (the ±X% filter) and extremes.
- **Not built:** the live scorer (§16).

The phases:
- Phase 1 = this sequence model (a thin TFT of our own) on the pocket targets.
- Phase 2 = the GBM twin, the input ladder and the GBM overlay. Not started; it is the user's call.

What this file covers:
- **§1–§7:** the column dictionary. `ml/intraday_ft/features.py` `FEATURES` (117 columns) is the source, and this
  file must match it. A test checks that every name is here.
- **§8–§19:** the pipeline end to end: data, preprocessing, the frames, training, testing, the read-outs, the
  results, live use, the assumptions and the decision pivots.

Other documents:
- The decisions behind each column: MODEL-STRATEGY Q40.
- The earlier design (Stage A/B GBM, Step 0 probe, P1): [intraday-design.md](intraday-design.md) and LEARNINGS
  `tag-ml-intraday`.
- §20 maps which document holds what.

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
| **volume unit** | `ln((x + 1) / (normal(x) + 1))` (the +1 keeps a zero-volume minute finite), or a share of the typical day where named "share" | 120k vs 80k → ≈ ln 1.5 = 0.41 |
| **market unit** | the % log change of each instrument against **its own** reference (never across instruments) | §4 |
| **entry P₀** | The first trade after the decision moment: minute i+1's open (minute tier), the first trade after the decision second (second tier). Only the zero line for targets, **not** a prediction | minute 10:14 closes at $272.70 (ask), 10:15 opens $272.66 → P₀ = $272.66 |
| **decision minute i** | The model stands at the end of minute i, with everything up to and including minute i | |
| **held price** | The last trade price carried forward to every later second until the next trade | trades at :05 $100.01 and :09 $100.03 → :06–:08 hold $100.01 |
| **poll grid** | The seconds a live E*TRADE poll would see. **As built, training uses Massive's 1 s bars as they are (step 1) on every day.** Live polling is 2 s (MAIN) / 20 s (MIDCAP) today, with a 1 s trial planned (§16). `profile.POLL_STEP` has 2 s / 20 s grids, but no training run uses them | a day with seconds → `poll_tier` 1 |
| **stock id** | `composite_figi`, else CIK, else the ticker. The ticker is only a label. A recycled ticker takes the row in force on the day (§9) | FI → FISV keeps one id |

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
~1,000 live symbols. `live_set` = in that top 1,000. The frames keep only those rows, so `live_set` is True on
every row of a run file: it isn't a usable slice there.

**Cache** (`/tmp/mtap/intraday_ft/`, env `MTAP_INTRADAY_FT_CACHE`; the NAS is read only). The seven runs used
`/tmp/mtap/intraday_ft_w8/` (each run's `config.json` `cache`). macOS clears `/tmp` on a restart, and **that cache
was gone on 2026-11-14**. The run files are kept (§8):
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
- M fallback (no seconds): minute volume ÷ |minute VWAP change|, split by its sign. **Designed, not built**
  (checked 2026-11-14). On a day without seconds the S-tier columns stay NaN, which becomes 0 after
  standardisation, and `poll_tier` = 0. Every training day from 2023-01 has 1 s data, so no run was affected.

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
| `poll_tier` | static | as built: 1 = the day has Massive seconds (the S tier is filled), 0 = none (the S columns fall back or are NaN). The planned 1 s / 2 s / 20 s / none split is for live (§16) |
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
     recalibration. The rule first said a recalibration "fitted on the validation months". As built (2026-11-08)
     it is the rolling 20-session recalibration on past test sessions (§13), the daily model's cal20s rule.

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
- `extremes.parquet` (optional, `intraday-ft-extremes RUN`, no training, ~2 min a fold): per `dec_row` and bin k,
  `hi_bk` / `lo_bk` = ln(max high / P0), ln(min low / P0) over the bin's minutes after the entry, and
  `t_hi_bk` / `t_lo_bk` = their minute counted from the decision (−1 = none). Since 2026-11-08 also `vw_bk` /
  `tw_bk` = ln(VWAP / P0), ln(TWAP / P0) over the bin's minutes (VWAP = the minute VWAPs weighted by volume, TWAP =
  the mean of the held minute closes), `cvw_bk` / `ctw_bk` = the same from the entry to the bin's end (`cvw_b4` =
  the first 60 minutes) and `cl_bk` = ln(the last price at the bin's end / P0). The volume comes back from `m_rvol`
  × `normal_rvol` (a share of the normal day: only its ratios count); the bin VWAP matches the run file's
  `real_net_bk` × σ to 0.01 bp (median), 3 bp at the 99th percentile. Read from the cached frames'
  `m_h` / `m_l` / `m_vw` / `m_c`; a bin holding a minute at the frames' ±10σ clip (a big gap from yesterday's close) is NaN, as is a
  bin without a trade (b2 0.06%, b4 7.8% of IA1 rows). Matched on (stock_id, symbol, day): a when-issued line
  (APTVw) shares its company's stock_id. Checked against `bars_1m` on 5 stocks (exact).

The dashboard's "Intraday pockets" page reads only this file, and its MODEL.md (which records the daily run
label used for Chunk 0).

## 8. Data sources, paths and commands

Everything is read from the NAS (read only from ml/, env `MTAP_NAS`, default `/Volumes/Backups/MTap-Data`) and from
two ml/ files. ml/ imports nothing from `sourcing_py`: where a rule must match sourcing (the `exclusion()` list), it
is copied and kept in step by hand.

| input | path | what it gives |
| --- | --- | --- |
| minute bars | `bars_1m/month=YYYY-MM/` | 04:00–20:00 ET bars: the M tier, the targets, every reference value |
| second bars | `agg_1s/massive_rest/` (`day-D` + `day-D.extra`) | the S tier (held prices on the 1 s grid) |
| market | `out/market/` `spy_1m`, `spy_daily`, `spy_dividends`, `spx_daily` | SPY / SPX columns (§4), `beta60` |
| ticker reference | `out/tickers_cs/tickers_cs-*.jsonl` | type (CS), `composite_figi`, CIK, delisting date: the universe, the stock id, `delisted_later` |
| shares | `out/shares/shares-*.jsonl` | quarter-end shares outstanding: the market cap |
| daily panel | `out/tft/exp5y-r1/panel.parquet` (repo `out/`, `data.PANEL`) | `industry_code` (TRBC, 6 digits), `results_window_pit` (the earnings flag) |
| daily forecasts | `ml/runs/20261029-211059Z-gbm-4438sym~px-cal20s` (09:30), `…-gbm-4438sym-cal20s` (09:45) | Chunk 0 (`data.FORECAST`) |
| E*TRADE feed | `intraday_feed/date=*/source=etrade_watchlist` | only `data.live_symbols()`; the frames don't use it |

**Commands** (ml/pyproject.toml; run from `ml/` with `uv run`):

| command | does | flags (default) |
| --- | --- | --- |
| `intraday-ft-build` | the month frames (§10) | `--from 2023-01 --to 2026-10 --every 5 --stocks 1000 --workers 8`, `--check` (one month + the S2 check), `--no-seconds`, `--force`, `--status-dir`, `--out` (the cache) |
| `intraday-ft` | train, then score the test months blind (§12–§13) | `--folds B`, `--cache`, `--out runs/intraday_ft`, `--no-seconds`, `--encoder N`, `--hidden 32 --heads 4 --dropout 0.1 --lr 1e-3 --batch-days 32 --steps 1000 --max-epochs 15 --patience 3 --valid-days 2000 --valid-every 3 --test-every 3 --test-frac 0.25 --accelerator auto --seed 0`, `--label`, `--about` |
| `intraday-ft-report RUN` | the report, climatology files and recalibration (§13) | `--cache` |
| `intraday-ft-extremes RUN` | `extremes.parquet` (§7) | `--cache` (default: the run's `config.json` cache) |

**The runs** (`ml/runs/intraday_ft/`, 8.0 GB on the Mac):

| label | run | what | epochs (best valid) |
| --- | --- | --- | --- |
| IA0-timing | `20261107-140359-ft-B`, `20261107-141636-ft-B` | 1-epoch timing checks | 1 |
| **IA1** | `20261107-141854-ft-B` | fold B, whole day, M + S tier | 15 (0.907) |
| IA1A | `20261107-162439-ft-A` | fold A, the same | 13 (0.906) |
| IA1-noS | `20261107-181646-ft-B-noS` | fold B, S-tier inputs zeroed | 15 (0.908) |
| IA1-enc120 | `20261107-201958-ft-B-enc120` | fold B, last 120 minutes | 15 (0.904) |
| IA1-enc60 | `20261107-225358-ft-B-enc60` | fold B, last 60 minutes | 9 (0.921) |

Every run's `config.json` says `cache: /tmp/mtap/intraday_ft_w8`. **That cache is gone** (§2), so retraining,
re-scoring or a new `extremes.parquet` needs the frames rebuilt first (≈ 2½ h). The dashboard doesn't need them.

## 9. Reference values and identity (prior sessions only)

All in `build.reference()`, `normals()`, `beta60()` and `data.py`. "Prior" = the stock's earlier sessions, today
left out (`shift(1)`). The window is `N_NORM` = 20 sessions and needs `MIN_SESS` = 15 of them, or the value is NaN.

| value | rule | example |
| --- | --- | --- |
| stock id (`sid`) | `composite_figi`, else `cik:` + CIK, else `tk:` + ticker. A ticker used by two companies takes the row delisted soonest on or after the day, else the last row | FI and FISV share one figi → one id |
| `prev`, `prevclose`, `prevpost` | yesterday's regular-session VWAP, close, after-hours last | |
| daily change r | ln(VWAP / prev VWAP); \|r\| > 0.4 → NaN (a split guard: a split shows as one huge change) | |
| σ (`sig`) | std of r over the prior 20 (≥ 15) | |
| `ndol` | mean dollar volume over the prior 20 | $45M |
| `typical` | median regular-session volume over the prior 20 | |
| `prior` | the count of earlier sessions | |
| `npm555`, `npm570` | median pre-market volume to 09:15 / 09:30 over the prior 20 | |
| normals `nvm`, `nnm` | per minute of the day: the median volume / trade count over the prior 20 (≥ 15) | |
| σ₁ₘ (`sig1m`) | per 15-min bucket (26 of them): √(Σ r² ÷ n) over the prior 20 sessions' 1-min returns, n ≥ 20 | |
| `beta60` | vs the daily SPY VWAP change, over the prior 60 sessions (≥ 40). NaN → 1.0 in Chunk 2's `resid` | |
| market cap | shares from the last quarter-end snapshot before the day × prevclose | 15.4B × $270 = $4.2T |
| cap band | mega ≥ $200B, large ≥ $10B, mid ≥ $2B, small ≥ $250M, micro; unknown if no shares | |
| sector / industry | the TRBC industry code last seen in the panel per stock id; sector = its first 2 digits | 57 = Technology |
| `results` | 1 if the panel's point-in-time earnings window says Earnings_Day or Post_Earnings_Reaction | |
| `delisted_later` | the stock id is delisted after the day (from the ticker file). A report slice only, never an input | |

**Sector and industry are not point in time:** each stock gets its last known code for every day. Codes rarely
change, so this was accepted.

## 10. Universe and the frame build

**Eligibility per day** (`build.eligible`, constants in `data.py`):
- `ndol` ≥ $20M (`MIN_DOLLAR_VOL`), σ > 0.3% (`MIN_SIGMA`), ≥ 3 prior sessions (`MIN_PRIOR`);
- an open today, a previous VWAP, `typical` > 0;
- a common stock: Massive type CS on the day, minus the `exclusion()` names (notes, preferreds, funds, rights). ADRs
  are typed ADRC by Massive, so they are out.

Then the eligible stocks are ranked by `ndol` per day and the top `LIVE_N` = 1,000 kept (the live stand-in, §2).

**One month at a time** (`build_month`):
1. Read the month's `bars_1m` once. Reference values need history, so the build also holds summaries of the
   `WARM` = 4 months before (≥ 61 sessions, enough for the 60 changes of `beta60`).
2. Reference values and eligibility (§9) for the month's stock-days; keep the top 1,000 per day.
3. The per-minute normals, only for the kept rows (the full window would be 20 × stock-days × 390).
4. The inputs per chunk (§11). The S tier runs one day per job over `--workers` processes, and the 1 s zig-zag runs
   over all of a day's stocks together.
5. Decisions: per stock-day an offset `off` = `rng.integers(0, every)` with `rng = default_rng(seed + YYYYMM)`,
   then minutes `4 + off, 9 + off, …` up to 384 (`FIRST_DEC`, `LAST_DEC`). Every minute of the day is covered over
   the stock-days, and the same seed gives the same decisions.
6. The targets (§5) and persistence for every decision.
7. Write `rows.parquet`, `dec.parquet`, `seq.npy` (float16), then `meta.json` last. A month with `meta.json` of
   the same `FRAME_VERSION` (1) and settings is skipped; a month cut short has none and is built again; `--force`
   rebuilds.

**Size and time.** 2023-01 → 2026-10 = 46 months, ≈ 966k stock-days, 83–86 GB on disk; ≈ 2.1 min a month with
`--workers 12` (≈ 2½ h in all; the 2026-01 check took 3 min with 8). The NAS reads are not the bottleneck
(~0.5 s per 1 s day file, ~10 s per month of minute bars).

## 11. Transformations, input by input

The column-by-column definitions are §3. This section gives the rules that sit between the columns.

**Units.** Prices are `ln(x / prev_vwap) / σ`, volumes `ln((x + 1) / (normal(x) + 1))`, market moves % log
changes vs each instrument's own reference (§1, §4). Every input is clipped to ±10 (`CLIP`).

**Chunk 0 (daily forecast).** The 09:30 run's shape (`R5~px-cal20s`, prices only) for minutes 0–14 and the 09:45
run's (`R5-cal20s`) from minute 15. Only walk-forward test predictions exist, so Chunk 0 starts 2023-01-23; the
`fc_*` columns are NaN (then 0) when a stock-day has no forecast (~5% of rows). `band_px` / `band_vwap` place the
price and the running VWAP within the forecast's quantiles.

**Chunk 1 (the stock's day).** The running digests of the day from minute bars, plus the zig-zag pivots. A pivot
counts only from the minute it is confirmed, so later bars can't change it. The live scorer must call the same step
code (`zigzag.py`) so live and training agree.

**Chunk 2 (sector and market), per minute.**
- Each stock's move mv = ln(close so far / prev VWAP) × 100.
- Groups: the sector, the industry, the whole market. Members = the day's live set (in training, the top 1,000).
- Each member gets the **other** members' move (leave one out): cap-weighted with no member above 25%
  (`MKT_CAP_W`; the excess is spread over the rest, repeated; skipped if 25% × n < 1), and equal-weight
  (`sec_mv_ew`). A stock outside the live set gets the full live mean.
- `ind_mv` falls back to the sector when the industry has fewer than 5 other members (`IND_MIN`).
- SPY: `spy_mv` vs its previous close, `spy_vwap_mv` vs its previous VWAP, `spy_rvol`.
- `resid` = mv − β × `spy_mv` (β NaN → 1.0). `breadth` = the share of the live set with mv > 0.

**Chunk 3 (static).** Yesterday's close / after-hours vs its VWAP, today's gap, the pre-market, σ, `log_dvol`,
`log_mcap`, `beta60`, `results`, `spy_gap`, `spx_y`, `spx_gap`. The pre-market comes as two cuts (09:15 and 09:30);
the 09:30 cut reaches decisions only from minute 15 (§6).

**S tier (`profile.py`).** From the held price every second of the session (23,400 s):
- time at price: a profile of 0.1σ bins from −10σ to +10σ (201 bins) around yesterday's VWAP;
- near the day's high / low = within 0.25σ;
- a visit to a price ends after ≥ 30 s away; visits and dwell are recomputed every 5 minutes and carried
  forward (at most 4 minutes old);
- decays use a 30-minute half-life; resistance windows are 1 / 5 / 15 / 30 minutes and the open;
- each value at the end of minute m uses only the seconds before 60 × (m + 1).

`poll_tier` = 1 where the day has Massive seconds, else 0 (then every S column is NaN → 0). **The minute-bar
fallback for the S tier is designed but not built**; every training day from 2023-01 has 1 s data, so it never
mattered in training.

**Targets** (`targets.py`, §5). The entry P₀ = minute i+1's open, else the held price at i; no target when
i + 1 ≥ 390. `cm_*` are masked below 5% of the bin's volume (`CM_MIN`); the race is at ±0.5σ (`RACE`); `vol_to_*`
is capped at 1.0 (`VOL_TO_CAP`). Bins past the close are cut (`bin_cut`) and their targets masked.

## 12. Training

**Folds** (`windows.FOLDS`; the last 2 training months are the validation months for early stopping):

| fold | train | validation | test |
| --- | --- | --- | --- |
| B (first) | 2023-01 → 2025-10 | 2025-11 → 12 | 2026-01 → 10 |
| A | 2023-01 → 2024-10 | 2024-11 → 12 | 2025-01 → 12 |

**Fitted on the training months only:**
- the scaler (`fit_scaler`): 3,000 stock-days sampled evenly over the months; each sequence column and static
  real gets its median and IQR; each target its IQR (for the loss weights), from 20 decisions per sampled day;
- the vocabularies (`fit_vocab`): sector, industry, symbol; a symbol needs ≥ 60 training sessions
  (`MIN_SYMBOL_SESSIONS`), else it is 0 = unknown;
- standardised = (x − median) / IQR, NaN → 0, clipped to ±10.

**Batches.** 32 stock-days with all their decisions (`--batch-days`). Each batch masks the overlays at random:
sector 30%, industry 30%, symbol 50% (`OVERLAY_P`) are set to unknown, so the model also works without them. The
cap band is never masked.

**The network** (`tft.Net`, built from pytorch-forecasting's sub-modules):
- statics: a variable selection network over the static reals and the embeddings (cap band, sector, industry,
  symbol; size 8), giving 4 context vectors (selection, enrichment, the LSTM's initial h and c);
- per minute: variable selection over the 98 sequence inputs, an LSTM, a gated skip, enrichment, causal
  interpretable multi-head attention (each minute sees only earlier minutes), a gated skip, a position-wise GRN;
- decoder: one step per bin, from the known inputs (`bin_minutes`, `bin_normal_share`, `bin_to_close`,
  `bin_cut`) plus a bin embedding, with the decision's state as context;
- heads: per-bin quantiles, row quantiles, and the classes (race 3, `t_peak_up` / `t_peak_dn` 5);
- quantiles = the median ± cumulative softplus steps, so they never cross.

**Loss.** Mean pinball loss over the bin and row quantiles, each target divided by its training IQR, plus
0.2 × the cross-entropy of the class heads (`CLS_W`). Masked targets weigh 0. A non-finite loss stops the run.

| setting | value |
| --- | --- |
| hidden size, attention heads, dropout | 32, 4, 0.1 |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, gradient clip 1.0 |
| epoch | 1,000 batches; at most 15 epochs |
| early stop | after 3 epochs without a validation loss 1e-4 below the best; the best `model.pt` is reloaded |
| validation | 2,000 sampled stock-days of the validation months, every 3rd decision (15 min) |
| device, seed | MPS if available, else CPU; seed 0 |

Time: ~0.45–0.55 s a step on MPS, ~7.5 min an epoch, 2 h 6 min for IA1 with scoring. `pi_*` reaches ≈ 3·10⁴ in a
few batches; the IQR scaling and the gradient clip hold it (not capped).

**Ablations.** `--no-seconds` zeroes the S-tier inputs (the network is the same). `--encoder N` gives each decision
only its last N minutes: the variable selection still runs once per day; the LSTM and attention run per window,
with a zero minute padding windows that start before 09:30.

## 13. Testing, the report and the read-outs

**Scoring** (`tft.score`): the best model predicts every 3rd decision (every 15 min) of 25% of the test stock-days
(seeded), blind. Each decision is predicted four times, with the overlays none / sector / sector + industry / all
(`VARIANTS`); the run file holds each read-out four times (§7), `quantiles.npy` the `all` quantiles.

**Read-outs from quantiles (`cdf_at`).** P(X ≤ v) is linear between the five quantiles. Past P10 / P90 the tails
are exponential, with the decay fitted to the outer pair (s = (P25 − P10) / ln 2.5 below, (P90 − P75) / ln 2.5
above). A flat 5% outside can't say less than 5% for a rare event, which is why it changed (2026-11-07).

**The report** (`report.py`, `intraday-ft-report`):
- baselines on the same rows:
  - climatology = the training span's quantiles and event rates per cell (5-min bucket × σ tercile × dollar-volume
    tercile; up to 2M training decisions);
  - persistence = the last 15 minutes' pocket (§5);
- scores: pinball skill, coverage, the share of outcomes at or below each quantile, rank IC across stocks at a
  minute (≥ 10 stocks), Brier skill and AUC for events, top-20 minus bottom-20;
- CIs: 500 bootstrap draws over days (`N_BOOT`);
- slices: minute bucket (0–15, 15–60, 60–120, 120–240, 240–330, 330–390), cap band, `live_set`, `delisted_later`,
  `poll_tier`;
- `bulge` / `sag` quantiles below 0.01σ read as 0 (`DUST`): they sit on a point mass at 0;
- the go-rule table "Pocket shape" (§6) for `cm_up`, `cm_dn`, `bulge`, `sag`, `rvol` per bin.

**Rolling recalibration** (`recal.py`, `fold_k/recal.parquet`). Per day, target and quantile τ:
- offset = the τ-quantile of (outcome − predicted quantile) over the previous 20 test sessions (`WINDOW`);
- calibrated quantiles = sort(q + offset), clipped to the target's range (shares 0 … 1; `net`, `rvol`, `net60` any
  sign; the rest ≥ 0);
- the first 20 sessions get no offset and are left out of the "Recalibrated" table;
- one offset for every stock at a minute, so the ranking can't change.

Files: `report.json`, `report.md`, MODEL.md (run level); `clim.parquet`, `clim_rates.parquet`, `recal.parquet`
(fold level), so the dashboard grades without ml/ code.

**Extremes** (`intraday-ft-extremes`, §7): the highs, lows, VWAP / TWAP and last price after the entry, from the
cached frames. A bin holding a minute at the ±10σ clip is NaN.

## 14. The Moves read-out (dashboard, no model change)

`dashboard/forecast_dash/pockets_calc.py` (`move_*`), on the page Intraday → Pockets model → Moves. Q42.

**The probability, per bin K and side:**
- the side forms: P(`mass_up` ≥ 5%) up, P(`mass_up` ≤ 95%) down;
- its centre is beyond the target: 1 − P(`cm_up` < ln(1 + X) / σ) up, the same with `cm_dn` and −ln(1 − X) down;
- P = forms × beyond, both read with `cdf_at` off the stored quantiles.

**"By the window's end"** (toggle, on by default): P = 1 − Π(1 − P_k) over b1 … bK, treated as independent; the
calibration corrects what that gets wrong.

**Calibration** (`move_recal`): for each day, the previous 20 sessions' decisions, sorted by P into equal-count bins
of ≥ 150, their hit rates made non-decreasing (pool-adjacent violators), then read off by linear interpolation. It
needs ≥ 1,500 past rows, else P stays raw (`warm` False). **The hit it calibrates on is `reached`** (the centre),
not `touched` (pockets_page.py).

**Outcomes** (`move_outcome`, from the run file and `extremes.parquet`):
- `reached` = the real centre is at or beyond the target (NaN if the bin had no trade);
- `touched` = the best price (the high for up, the low for down) reached the target;
- `best_pct` / `best_min` = that best price and its minute after the decision (the best, not the first touch);
- with the toggle, each is "in any window" and the shown centre is the window with the highest P
  (`move_centre_cum`);
- three row states (commit 7e7ad38, 2026-11-12): happened (`reached`, green), touched only (blue), did not
  (dimmed);
- `best_x` = the best move ÷ X; `captured` = the real centre move ÷ the predicted P50 move.

The list shows the 500 rows with the highest P; the summary, by-day and by-time tables use every row.

## 15. Results so far

Details and CIs are in LEARNINGS (`tag-ml-intraday`); the Scoreboard row in MODEL-STRATEGY is the summary.

| check | result |
| --- | --- |
| IA1, fold B (2026) | **pocket shape passes:** `cm` rank IC +0.27 … +0.30 (climatology +0.12 … +0.17), `rvol` +0.54 … +0.61; skill `cm` +1.4 … +5.0%, `rvol` +18.5 … +23.0%, `bulge` / `sag` b2–b5 +1.8 … +2.1% (b1 `sag` −0.9%). **Direction none:** net60 rank IC +0.000 [−0.005, +0.005] |
| IA1A, fold A (2025) | holds: `cm` rank IC +0.29 … +0.31, `rvol` +0.57 … +0.63, `bulge` / `sag` +0.9 … +3.3% every bin. Calibration level drifts between years |
| rolling recalibration | 25 of 25 shape targets calibrated on both folds (18 raw) |
| ablations, fold B | no S tier: as good or better (25/25); 120-min encoder: as good (`rvol` +2 points); 60 min: worse (23/25, early stop at epoch 9) |
| Moves (±1%) | calibrated P good to a few points up to 0.4–0.5; by the window's end, 0–60 min, P ≥ 0.5: said 56–58%, centre happened 55–56% on both folds |
| Moves export check (2026-11-13/14) | the price moved, but not in the predicted direction: below |

**The export check** (the user's IA1-noS export: +1%, calibrated, P ≥ 0.55, the entry to the close, 2026-08-05 →
10-20):
- **The summary is right as a size forecast.** Up: 613 predictions, said 0.60, centre reached 58%, touched +1% 78%.
  Down: 602, said 0.60, reached 54%, touched −1% 77%. The median best move is ~2.3× the target.
- **It isn't a direction forecast.** In the 500-row list (the top P), the up picks touched −1% almost as often as
  +1%:

  | predicted | touched +1% | touched −1% | both | median end of day | ended on the predicted side |
  | --- | --- | --- | --- | --- | --- |
  | up (245 known) | 82% | 80% | 62% | −0.44% | 46% |
  | down (242 known) | 80% | 86% | 66% | −0.05% | 50% |

  420 decisions are in both lists: the model names stocks that will move a lot, then rates both sides.
- **A higher P:** up improves (P 0.70–0.80: centre reached 90%, n 21); down gets worse (P 0.70–0.80: said 74%,
  reached 42%, touched +1% 95%, n 19). The high-P down rows are a few days (08-06, 09-02, 08-12) and a few stocks
  (NBIS, SNDK, AAOI, BSP, WDC). Small samples; no CI yet.
- **Not the inputs or the calibration.** The raw P is 0.81–0.98 on both sides of the same rows and correlates 0.83
  with the calibrated P. The isotonic map only rescales P; it can't swap up for down. The targets (where the volume
  centres, on each side) are size targets by design (Q41), and the model learnt them.
- **"Touched 78%" is not "wins 78%".** `high_min` / `low_min` are the minutes of the best extremes, not the first
  touch, so the files can't say which side came first. In 33 of the 50 high-P down rows the low came before the
  high.
- The earlier raw b5 read (~19.7k rows, P > 0.5) gave the same shape: touched +1% 69%, −1% 69%, median end
  −0.16%.

## 16. Live prediction (planned, not built)

This is design only (P16 in [live-recommendations-plan.md](live-recommendations-plan.md) §0). Nothing below runs
yet.

**User choices (2026-11-12):**
- the model is **IA1** (over noS);
- the score is the calibrated b5 "touched +X%" read-out ("good enough for me"), with the skill vs climatology shown
  next to it.

**The scorer** (planned `ml/intraday_ft/live.py`; reads contract files and Redis only, no `sourcing_py` imports):
- **symbols:** the E*TRADE watchlist tickers only (MAIN 605 symbols, polled every 2 s; MIDCAP 422, every 20 s);
- **M tier:** Massive minute bars to t − 15 min, stitched with minutes built from the E*TRADE polls;
- **S tier:** held prices from the polls from 09:30; IA1 was trained on the 1 s grid, so 1 s polling is planned
  (a one-session trial first, staggered calls: MAIN 0, MIDCAP 350, POSITIONS 700 ms);
- **Chunk 0:** the morning daily forecast. `mtap-morning` is not deployed (branch `feat/daily-5y-inputs`);
- **Chunk 2:** sector / market moves over the live set (in training, the top 1,000);
- **model:** loads `fold_B/model.pt` + `prep.json` + `quantiles.json` and calls the same `tft` code; writes the
  same quantile columns as `intraday_ft_pred.parquet`, so the dashboard views work unchanged;
- **ladder:** the Moves maths moved into a shared helper inside ml/ (not imported from dashboard/), with the
  rolling 20-session calibration on live graded outcomes, seeded from the stored test runs;
- **calibration of the score:** the score is a touch, but `move_recal` calibrates on the centre. The live ladder
  must calibrate each touch rung on `touched` (centre P mapped to the past touch rate);
- **skill gate:** ≥ 60 test days, ≥ 1,000 events and a CI low end above 0 vs climatology, else the tier says
  "skill unknown" / "no edge";
- **grading:** the extremes after each window closes, vs `price_ref` at the decision time;
- **replay test:** one stored test day through `live.py` must reproduce the run file's quantiles to 1e-4.

If 1 s polling fails, IA1-noS is the fallback. It needs no S tier, and it scored as well as IA1 (§15).

## 17. Assumptions

Each is stated in the code or a test unless marked "not checked".

1. **The top 1,000 eligible stocks by prior dollar volume stand in for the live set.** The E*TRADE lists exist only
   from 2026-10-12. Live predictions cover ~1,000 watchlist names, which aren't the same 1,000.
2. **Past sessions only** for σ, the normals, β, the typical day and the scaler (tests enforce it).
3. **Massive's 1 s held prices equal what a 1 s E*TRADE poll would see.** Not checked; the replay and polling trial
   are the checks. E*TRADE's ChartIQ minute volume is about half of Massive's (LEARNINGS `tag-chartiq`), so
   volumes built from polls may differ too.
4. **Minute bars miss the auction prints** (~16% of the day's volume): the targets and normals use what the minute
   bars hold, live and in training alike.
5. **The daily forecast is out of sample** (walk-forward test predictions only).
6. **Sector and industry are fixed per stock** (the last code seen), not point in time.
7. **Split guard:** a daily change above 40% is a data break, not a move.
8. **Bins are disjoint and cut at the close**; a decision never sees past 16:00.
9. **Dividends:** Massive's `adjusted=true` adjusts splits only; SPY's gap adds the dividend back (§4).
10. **The 25% test sample is representative** of every stock-day (seeded, at random).
11. **The rolling recalibration's 20 sessions are known by the next morning live** (they are graded the same day).
12. **"By the window's end" P assumes independent windows**; the calibration corrects the error.

## 18. Decision pivots

| date | from | to | why | where |
| --- | --- | --- | --- | --- |
| 2026-10-12 | no intraday model | `tft_intraday` on E*TRADE watchlist panels | first try: 1–15 min edge ~2–3% vs climatology, no direction | [intraday-model.md](intraday-model.md) |
| 2026-10-30 | the TFT as predictor | an analog engine (similar past moments, pockets in price × time × volume) | the user's rethink | Q37, [intraday-redesign.md](intraday-redesign.md) |
| 2026-10-31 | | the analog check: fast, no edge | | [analog-pockets.md](analog-pockets.md) |
| 2026-11-01 | analogs as predictor | a GBM ladder (Stage A/B) and a Step 0 probe | test before weeks of building | Q38, [intraday-design.md](intraday-design.md) |
| 2026-11-01 | "d = 0 has direction" | no direction | P1: the edge was bid–ask bounce | LEARNINGS P1 |
| 2026-11-02 → 06 | GBM on summaries | a sequence model on raw 1-min and 1-s series | direction failed on summaries; maybe the sequence holds it | Q39, Q40 |
| 2026-11-06 | single touch targets | pocket targets over five duration bins | describe where volume goes, not one event | Q40, §5 |
| 2026-11-07 | the library's TFT | a thin TFT of our own | `TimeSeriesDataSet` can't do whole-day encoders with many decisions | §6 |
| 2026-11-07 | the whole universe | the top 1,000 per day | ~2.5 GB a month for all, ~1,000 used live | §2 |
| 2026-11-07 | go = direction skill | go = pocket shape (direction not required) | IA1: no direction, real size skill; the user's call | Q41, §6 |
| 2026-11-07 | flat 5% tails, raw `bulge` / `sag` | exponential tails, the 0.01σ dust rule | rare events and point mass at 0 scored wrongly | §13 |
| 2026-11-08 | recalibrate on the validation months | rolling 20 sessions of past test days | the daily model's cal20s rule; usable live | §13 |
| 2026-11-08 | | the S tier and the whole day not needed for the shape | ablations | §15 |
| 2026-11-08 | quantiles only | Moves: P(centre beyond ±X%), isotonic calibration, "by the window's end" | the user's filter question | Q42, §14 |
| 2026-11-12 | centre only | touched and centre shown apart (three states) | price touches 15–20 points more often | 7e7ad38, §14 |
| 2026-11-12 | score not chosen | the calibrated b5 touched +X%, IA1 for live | the user's choice | §16 |
| 2026-11-14 | | export check: high P is size, not direction | the user's 1,215-row report | §15, Q43 |

## 19. Open items and limits

- **Direction:** none found in any arm or fold. A direction target ("+X% before a −stop", first touch) would suit
  Phase 2's GBM; not started.
- **The first touch isn't measured:** `extremes.parquet` has the best extremes' minutes, not the first crossing.
- **The 25% sample:** a re-score of every stock-day (~5 h both folds, no training) would give full counts. It needs
  the frames rebuilt first (≈ 2½ h), since the cache is gone.
- **Fold A for noS + enc120 together** (the proposed cheaper model): not run.
- **The minute-bar fallback for the S tier:** designed, not built.
- **Read-only checks awaiting the user's OK:**
  1. per-P-bucket CIs clustered by day, with and without 08-06, 09-02 and 08-12;
  2. IA1 vs noS vs fold A on the same settings;
  3. whether P(up) − P(down) predicts the direction;
  4. the first-touch race, +1% before a −0.7% stop.
- **Live:** §16, not built; `mtap-morning` not deployed; 1 s polling untried.

## 20. Document map

| document | holds |
| --- | --- |
| this file | the column dictionary and the pipeline end to end |
| [../ml/MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) | the Scoreboard (current choices and results); the Q&A log: Q37–Q43 for intraday |
| [LEARNINGS.md](LEARNINGS.md) `tag-ml-intraday` | every experiment with numbers: v2 design, IA1, Moves, the export check, P1 |
| [live-recommendations-plan.md](live-recommendations-plan.md) | the live path to MBin; §0 = the pending list by owner (P16 = the live scorer) |
| [STATUS.md](STATUS.md) | hosts, schedules, data locations, the dated Changes log |
| [intraday-design.md](intraday-design.md) | the Stage A/B GBM design and the Step 0 probe (before v2) |
| [intraday-redesign.md](intraday-redesign.md), [analog-pockets.md](analog-pockets.md) | the analog engine and the Space page |
| [intraday-model.md](intraday-model.md), [../ml/FEATURES-intraday.md](../ml/FEATURES-intraday.md) | the first `tft_intraday` model (superseded) |
| `ml/runs/intraday_ft/<run>/report.md`, MODEL.md | each run's full report and label |

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
| 2026-11-08 | **Moves tab + `extremes.parquet`** (§7): the dashboard reads P(centre at or beyond ±X%) off the stored `mass_up` / `cm_*` quantiles, calibrated by a rolling 20-session isotonic map; no model change. Extremes written for IA1 and IA1A. |
| 2026-11-08 | `extremes.parquet` adds the prices after the entry: `vw_bk` / `tw_bk` (bin VWAP / TWAP), `cvw_bk` / `ctw_bk` (from the entry to the bin's end), `cl_bk` (last price). Written for all seven runs. Moves list: high / low / VWAP / TWAP / last price in $ and %, the predicted centre of the window with the highest P (toggle on), `max_said` per day. |
| 2026-11-08 | Moves: the bins are exclusive (b4 = 30–60 min only). A toggle (default on) gives P by the window's end: 1 − Π(1 − P_k) over b1..bK, calibrated on the OR of their outcomes; better calibrated than one bin. |
| 2026-11-14 | **The file now covers the whole pipeline** (§8–§20): sources and commands, reference values, the build, the transformations, training, testing, Moves, results, live (planned), assumptions, decision pivots, open items, a document map. Corrected against the code: training uses Massive 1 s as is (step 1), `poll_tier` is 1 / 0, the volume unit has +1, `live_set` is True on every run-file row, the minute-bar S fallback isn't built, the go rule's recalibration is the rolling one. The frame cache in `/tmp` is gone. Export check: high P is size, not direction (§15). |
