# Intraday model: reference

**Last updated: 2026-10-28.**

**Scope.** This page covers two things:
- what the intraday model needs from the daily model, which is ready now (the **handoff**);
- the facts any intraday design has to respect: data, folds, leakage.

**Out of scope.** The intraday model itself (targets, quantile strategy, inputs, training) is
being revised in a separate session. Where this page describes today's intraday code (branch
`intraday-model`), that is the current state, not the plan. The revision may change it.

**Related docs.**
- Field spec of today's intraday code: `ml/FEATURES-intraday.md`.
- ML questions and answers: [ml/MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) (Q5, Q7, Q19, Q20).

## 1. The daily handoff (ready)

**What it is.** One table per daily run, `ml/runs/<daily run>/handoff.parquet` (plus
`handoff.meta.json`), with one row per (symbol, forecast date). Every column is known before
that date opens. It is the only thing the intraday side should read from the daily model.

```bash
cd ml
env -u VIRTUAL_ENV uv run tft-vwap-handoff runs/<daily run>            # out-of-fold history
env -u VIRTUAL_ENV uv run tft-vwap-handoff runs/<prod run> --live      # + live forecasts
```

**Contents** (code and docstring: `ml/tft_vwap/handoff.py`; tests:
`ml/tests/test_handoff.py`). Moves are logs vs `prev_vwap`, the VWAP of the stock's previous
session.

| column | what it is |
| --- | --- |
| `symbol`, `date`, `industry_code`, `prev_vwap` | keys; `date` is the forecast session D |
| `dq05` … `dq95` | the 19 daily quantiles, sorted |
| `d_med` | P50 |
| `d_width`, `d_width50`, `d_width90` | P10–P90, P25–P75, P5–P95 |
| `d_skew`, `d_tail`, `d_p_up` | lean of the band; P5–P95 ÷ P25–P75; share of quantiles above 0 |
| `d_width_rel` | `d_width` ÷ the stock's mean over its previous 20 sessions ("unusual day") |
| `hist_mae`, `hist_cov80`, `hist_bias`, `hist_n` | the daily model's record on the stock over its previous 20 sessions: \|error\| of P50, P10–P90 coverage, mean signed error, and how many sessions |
| `sec_d_med`, `mkt_d_med`, `d_width_rank` | the sector's (others only) and market's mean `d_med`; the width's percentile that day |
| `d_state` | which daily forecast: 0 night, 1 09:30 prices, 2 09:45 full |
| `fold`, `source` | the daily test fold (−1 live); `oof` or `live` |

**The R0 table** (`runs/20261028-030847Z-return-4438sym/handoff.parquet`, built 2026-10-28):
- 3,169,248 rows, 4,176 symbols;
- sessions 2023-01-20 … 2026-10-26, 945 in all;
- 425 MB, built in 9 s.

Checks on it:

| check | value |
| --- | --- |
| mean P10–P90 coverage | 0.79 (target 0.80) |
| median P10–P90 width | 5.1% |
| median tail ratio | 2.7 |
| mean `d_width_rel` | 1.005 |

Rows with missing values:
- **History:** only each stock's first session (`hist_n` = 0).
- **Sector:** the 418,745 rows with industry `unknown` have no sector, so their `sec_d_med` is
  NaN. They are not pooled into a fake sector.

**Why the table is built this way:**
- **Out of fold only.** Each daily test fold forecasts 63 sessions with a model trained only on
  earlier sessions, so the folds together give one honest forecast per (stock, day). On days
  the daily model trained on, its forecast looks better than it will ever be live. An intraday
  model fed those days would learn to over-trust it. The table never contains such a day.
- **Live rows.** `--live` adds the production run's `forecast_<date>.parquet` for dates after the
  last fold, graded from `actuals.parquet` where available. So training history and live serving
  read the same columns.
- **Scale-free.** Every column is a log move, a ratio or a rank, so it compares across stocks and
  across daily models (later ones can be narrower).
- **Series breaks.** `A~1` (the earlier security under ticker A) is folded into `A`. On a shared
  date the current security wins.

**Which daily run.** R0 until R1, R1a or R1c beats it; then rebuild the table from the winner.
Each intraday run should record the daily run it used (`daily_run` in `handoff.meta.json`). A new
daily model means rebuilding the table and re-checking the intraday model.

**Not in the table, on purpose:**
- the daily `actual` for D (the answer);
- dollar levels;
- per-step fields that need the intraday price, such as where the price sits in the band, or the
  rest-of-day VWAP the daily P50 implies. Those are computed on the intraday side from `dq*` and
  `prev_vwap` (see Q19 for candidates).

**Earlier than 2023-01-20.** No out-of-fold daily forecast exists before the first daily test
fold, and none can without training on that day. For 2021-10 … 2023-01 there are two options:
- train on those intraday days without the daily inputs (`has_daily` = 0, so missing doesn't
  read as zero);
- or use them only in the earliest intraday folds.

That choice belongs to the intraday revision.

## 2. Data history for intraday

The daily model trains on 5 years (2021-10-21 on). The intraday model can too: **Massive
1-minute bars for every common stock go back to 2021-10-29**.

| source | resolution | days | where | notes |
| --- | --- | --- | --- | --- |
| **Massive 1-minute bars** | 1 min, `o h l c v vw n` | **2021-10-29 … 2026-10-26** (~1,255 sessions) | NAS `bars_1m/month=YYYY-MM/` (25 GB) | 04:00–19:59 ET; ~5.5k stocks; the regular session is 95% of the bars' volume; `vw` gives exact VWAPs; written by `inputs-backfill --save-bars` |
| Massive REST 1-second bars | 1 s | 2024-10-21 … 2026-10-16 (499) | NAS `agg_1s/…/source=massive_rest/` | ~73% of daily volume; no closing auction |
| Massive WebSocket 1-second bars | 1 s | from 2026-10-14 | NAS `agg_1s/…/source=massive_ws/` | the live feed; same profile as REST |
| E*TRADE watchlist feed | ~2 s / ~21 s | 2025-12 … 2026-01 (MBin, 12) and from 2026-10-12 | `out/tft_intraday/panel`, NAS `intraday_feed/` | what today's intraday panel uses; bid/ask on MAIN only |

What this means for the revision:
- **Minute bars cover the full 5 years.** The fine sources (1-second bars, the E*TRADE feed)
  cover at most 2 years. A design that needs 1-minute resolution or coarser can use all
  5 years; finer resolution limits history to 2 years.
- **Today's intraday panel builder reads only the E*TRADE feed** (12 sessions). Any longer
  history needs a builder from the minute (or second) bars.
- **`r_close`-style whole-day targets** still come from `daily_bars`. The minute and second bars
  leave out the closing auction.
- **Size.** Today's run (12 sessions × 1,055 stocks, 15 s steps) took 39 min and 30 GB of memory.
  5 years at that density is about 400 times more. The step size, the universe and per-day
  sampling of decisions are design choices for the revision.

## 3. Lining up training and test days with the daily model

**Question.** Should the intraday model train on the daily model's training days and test on its
test days, so the test data runs in parallel?

**Answer: align the test days, yes. Align the training days, no.** Every daily forecast the
intraday model sees, on its training days as well as its test days, must come from a daily
model that did not train on that day. The handoff table guarantees that by construction.

**Why align the test days:**
- If intraday test block k is the same 63 sessions as daily test block k, both models are out of
  sample on exactly those days.
- So the combined system (morning daily forecast, then intraday updates) can be graded as it
  would run live, quarter by quarter.
- A good or bad regime shows up in both models side by side.

**The daily test blocks** (R0, `fold` in the handoff):

| fold | sessions | fold | sessions |
| --- | --- | --- | --- |
| 0 | 2023-01-20 … 2023-04-20 | 8 | 2025-01-24 … 2025-04-24 |
| 1 | 2023-04-21 … 2023-07-21 | 9 | 2025-04-25 … 2025-07-25 |
| 2 | 2023-07-24 … 2023-10-19 | 10 | 2025-07-28 … 2025-10-23 |
| 3 | 2023-10-20 … 2024-01-22 | 11 | 2025-10-24 … 2026-01-26 |
| 4 | 2024-01-23 … 2024-04-22 | 12 | 2026-01-27 … 2026-04-27 |
| 5 | 2024-04-23 … 2024-07-23 | 13 | 2026-04-28 … 2026-07-28 |
| 6 | 2024-07-24 … 2024-10-21 | 14 | 2026-07-29 … 2026-10-26 |
| 7 | 2024-10-22 … 2025-01-23 | | |

The daily folds trained on 2021-10-21 up to each block's 21-session early-stop block (fold 0 on
about 1 year, fold 14 on about 4.6 years).

**With 5 years of minute bars**, the intraday side can mirror this exactly:
- test on the same 15 blocks;
- train each fold on the intraday days before its own early-stop block, from 2021-10-29.

The first ~14 months (before 2023-01-20) have no daily inputs (see section 1). How those days
are used is part of the intraday revision.

**Two things that still shift:**
1. Later daily folds, and the live production model, are trained on more data, so their bands
   can be narrower. The handoff's relative columns (`d_width_rel`, `d_width_rank`, `hist_*`)
   mean the same whichever daily model made the forecast.
2. Live, the daily forecast is the production model's. It is still out of sample for the day it
   forecasts, so it matches training in kind. Rebuild the handoff with `--live` and keep the
   pair rule: a new daily model means re-checking the intraday model.

## 4. Today's intraday code (current state, under revision)

- **Branch and package:** `intraday-model`, `ml/tft_intraday/`. One TFT, six targets: the
  forward VWAP over 1/5/15/60 minutes, the rest of the day, and the whole day, as 19 quantiles
  of the log move from the last price.
- **Its one run:** `ml/runs/intraday/20261012-003925Z-intraday-1055sym`, 12 sessions. Skill vs
  climatology:

  | horizon | skill |
  | --- | --- |
  | 1 min | +2.0% |
  | 5 min | +2.9% |
  | 15 min | +2.6% |
  | 60 min | +0.3% (CI spans 0) |
  | rest of day | −17% |
  | close | +5% (CI spans 0) |

  Direction is about 50%.
- **Its daily join:** `daily_join.py` reads one fold of an old 2-year daily run (`daily_fold`)
  and builds 8 of the handoff's columns itself. The revision should read `handoff.parquet`
  instead.

## 5. For the intraday revision session

Ready:
- `tft-vwap-handoff` and R0's `handoff.parquet`;
- the 5-year minute bars on the NAS;
- this page's fold table.

Open, to decide there:
- the intraday targets and quantile strategy;
- the resolution and the history it allows (minute bars: 5 years; second bars or feed:
  2 years);
- how the pre-2023 days without daily inputs are used;
- which per-step daily fields to compute (Q19 lists candidates);
- run size: step, universe, sampling;
- a panel builder from minute or second bars.

Proposed direction for these (2026-10-29, [MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) Q31–Q34; not decided or built):
- a 1-minute grid from `bars_1m` with empty minutes filled, a 60-minute encoder, training decisions every 5 minutes per stock;
- volume relative to the time of day, price moves in the stock's volatility, effort vs result, quiet/normal/burst regime summaries, multi-scale windows;
- the 1-second bars as within-minute summary columns (2 years), an ablation;
- a climatology baseline scaled by each stock's volatility; feature groups screened on HGB first, then TFT;
- a streaming dataset for 5 years (in memory does not fit);
- the handoff rebuilt from the base run; the 09:30/09:45 forecasts cover only the odd folds until the overlays run on all 15.

Results go in [LEARNINGS.md](LEARNINGS.md) under a new `tag-ml-intraday` topic.
