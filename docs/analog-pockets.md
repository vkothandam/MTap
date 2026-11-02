# Analog pockets: matching a moment against past moments of any stock

**Status (2026-10-31).** First check done; the dashboard page is built (Intraday → Space, §6).
This page is the whole record of the experiment: what was asked, what was run, what came
out, and what to try next. It is kept apart from the other docs on purpose, so it can be picked
up and improved on its own. The learning is also logged in
[LEARNINGS.md](LEARNINGS.md) under `tag-ml-intraday`.

Background: [intraday-redesign.md](intraday-redesign.md) proposed an analog engine (predict from
what happened after similar past moments) shown as "pockets" in a price × time × volume space.
This page is the first measured step of that idea.

## 1. The question

In the user's words (2026-10-31):

> Find another symbol that matches the trajectory from the past, based on selectable input
> parameters such as the VWAP difference from the previous day and the open, plus more chosen in
> the dashboard. The exact day doesn't have to match: it can be any day of any stock that has had
> such a trajectory over the past few days, or the past day until today and the point in time.
> Compare the pockets: how stocks with similar inputs behaved, with upside and downside pockets
> in the future. The accuracy of the match can be chosen: how the pockets look for a very
> similar match and for a slightly similar one. How feasible is it? Does it need XGBoost or
> similar training?

Follow-ups in the same session:
- **Pockets are blobs, not bars.** They can be uneven and split in the space, so a uniform bar
  per cell is not a true picture.
- **A short history window is fine:** the last 30–90 sessions, if that keeps it fast.
- **It must be interactive:** pick the stock, the date and the time, and change the parameters.

## 2. Terms

| term | meaning |
| --- | --- |
| **moment** | one stock at one time of one day, e.g. MU on 2026-10-02 at 11:00. The library holds one every 15 minutes, 09:45 … 15:45 |
| **σ** | the stock's typical daily move: the std of the log VWAP-to-VWAP change over the 20 sessions before that day. Every price parameter is in σ units, so a calm and a wild stock compare |
| **parameter** | a number describing the moment so far (table in §3.2) |
| **match** | a past moment whose parameters are all within a tolerance of the query's. Closer matches get more weight (§3.3) |
| **dial** | how close a match must be: the tolerance as a multiple of each parameter's spread. 0.25 = narrow, 0.5 = medium, 1.0 = broad |
| **pocket** | where the later volume traded, on a map of time ahead × price vs now. It is a smoothed density (a blob), and the 50 / 80 / 95% pockets are the smallest regions holding that share of the volume |
| **climatology** | the same pocket built from every moment at that time of day, with no matching. It is the baseline any match must beat |
| **N_eff days** | how many independent days the matches really stand for. Twenty matches all from one market day count as about one day, because that day's mood moved them together |
| **shrinkage** | blend the match result with climatology by its evidence: weight a = N_eff days / (N_eff days + 10). With 1 effective day the result is 91% climatology; with 30 it is 75% match |
| **rest-of-day VWAP** | the VWAP of all trades after now, vs the price now, in σ |
| **bulge / sag** | the highest high / lowest low after now, vs the price now, in σ |

## 3. The experiment

### 3.1 Data

- **Bars:** Massive minute bars on the NAS, `bars_1m/month=2026-06 … 2026-10`, read-only. The
  regular session only (09:30–16:00 ET, 390 minutes), with empty minutes given zero volume and the
  last price.
- **Sessions:** 2026-06-01 … 2026-10-26, 105 sessions. June only warms up σ and normal volume;
  the moments start in late June. 87 sessions have eligible moments.
- **Stocks:** each day, the stocks with ≥ $5M average dollar volume over the prior 20 sessions,
  price ≥ $3, σ > 0.3%, and the last 3 days' moves known. About 2.7k stocks a day; 241,784
  stock-days.
- **Library:** 6,041,054 moments on 87 sessions (2026-06-24 … 2026-10-26; 69 before October),
  about 300 MB in memory, built in 42 s from the NAS.
- **No look-ahead.** σ, normal volume and dollar volume use only the 20 sessions before the day.
  A query only matches moments **on earlier sessions**, never its own day or later. Parameters use
  bars up to the decision minute; only the readings (what followed) use bars after it.

### 3.2 Parameters

| parameter | meaning, at the moment |
| --- | --- |
| `vwap_vs_prev` | today's VWAP so far vs yesterday's VWAP |
| `open_vs_prev` | the open vs yesterday's VWAP |
| `move_since_open` | the price now vs the open |
| `day_1`, `day_2`, `day_3` | yesterday's, and the two days before's, VWAP-to-VWAP move |
| `path_30m` | the price now vs 30 minutes ago |
| `from_low`, `from_high` | the price now vs the day's low / high so far |
| `rvol` | the volume so far vs normal for this time of day (log) |

All prices are logs in σ units. The user's five are `vwap_vs_prev`, `open_vs_prev`,
`move_since_open`, `day_1` and `day_2`.

### 3.3 Matching and pockets

- **Pool:** moments at the **same time of day**, in the last W sessions before the query day
  (W = 30, 60 or 90).
- **Weights:** for each parameter, z = |x − x_query| / (dial × spread), where spread = IQR / 1.349
  over the library. A moment matches only if every z < 1, and its weight is Π(1 − z²) (an
  Epanechnikov product kernel: full weight at an exact match, falling to zero at the edge).
- **Pocket map:** each match's later minutes go on a 36 × 32 grid:
  - x = log₂(minutes ahead + 1), from now to the close;
  - y = that minute's VWAP vs the price now, ±4σ;
  - mass = that minute's volume ÷ the stock's normal daily volume × the match weight.
  The grid is smoothed (Gaussian, 1 cell) and normalised to 1.
- **Pockets:** the 50 / 80 / 95% highest-density regions of the map. Each can be one blob or
  several; the count of separate blobs is reported.
- **Readings** (weighted over the matches): P(rest-of-day VWAP > now), rest-of-day VWAP p10/50/90,
  bulge and sag p10/50/90, and the share of later volume above now.

### 3.4 Scoring

October moments (2026-10-01 … 10-26) at 10:30, 11:00, 11:30, 12:30, 13:30 and 14:30, of stocks
with ≥ $20M a day, 500 drawn at random (`--seed 1`). Each is scored against what it actually did:
- **P(up):** Brier score.
- **Rest-of-day VWAP, bulge, sag:** pinball loss on p10/50/90.
- **Pocket density:** log score, Σ (actual volume share × log predicted density), over the map.
  Higher is better; the difference vs climatology is in nats.
- **Skill** = 1 − loss / climatology's loss, in % (positive = better than climatology).
- **95% CIs:** bootstrap over days (2,000 resamples), because moments on one day move together.

## 4. Results

### 4.1 One example: MU on 2026-10-02 at 11:00

MU opened 1.36σ above yesterday's VWAP and had faded 0.75σ since the open (σ = 2.97%).
Matched on 7 parameters against every 11:00 moment before October:

| | matches | stocks | N_eff days | P(up) | later volume above now | rest-of-day VWAP p50 | bulge p50 | sag p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| narrow (0.25) | 0 | | | | | | | |
| medium (0.5) | 3 | 3 | 1.1 | 96% | 91% | +0.20σ | +1.53σ | −0.45σ |
| broad (1.0) | 192 | 184 | 20.9 | 56% | 55% | +0.04σ | +0.47σ | −0.29σ |
| climatology | 192,662 | | | 49% | 49% | −0.01σ | +0.36σ | −0.38σ |
| **what MU did** | | | | down | **5%** | **−0.16σ** | **+0.08σ** | **−0.34σ** |

The three medium matches were RIO 2026-07-29, GIII 2026-08-03 and DGXX 2026-08-12. They all went
up, but they are one effective day of evidence. Shrunk to climatology they say 54% up.
The broad matches lean slightly up; MU went down, with 95% of its later volume below the 10:59
price ($1,082.86; it traded $1,072–1,085 for the rest of the day, all within ±0.35σ).

(An earlier draft of this page said 26% above now. That was wrong: the rerun and the raw bars
both give 5%.)

Shown as text maps (`@` 50% pocket, `o` 80%, `.` 95%; time ahead runs left → right on a log
scale):

```
broad match: 192 matches, N_eff 20.9 days, later volume above now 55%
 +1.5..+2.0σ |                               .....|
 +1.0..+1.5σ |                       .......ooo@@o|
 +0.5..+1.0σ |                  ........oooooo@@@@|
 +0.0..+0.5σ |         .........ooooooooo@@@@@@@@@|
 ---- now -- |------------------------------------|
 -0.5..+0.0σ |         ..........oooooooooo@@@@@@@|
 -1.0..-0.5σ |                   ........ooooo@@@o|
 -1.5..-1.0σ |                               ..oo.|
                      1m   4m      16m     64m     4h

what MU actually did: later volume above now 5%
 +0.0..+0.5σ |                          o         |
 ---- now -- |------------------------------------|
 -0.5..+0.0σ |              ....o..oooo..o@o@@@@@ |
                      1m   4m      16m     64m     4h
```

### 4.2 History window × dial (your 5 parameters)

500 October queries (`--seed 1`). Every result is shrunk to climatology by its N_eff days. Skill
in %, 95% CI in brackets; bold = the CI is clear of 0.

| window | dial | matches (median) | N_eff days (median) | queries with < 5 matches | P(up) Brier | rest-of-day VWAP | bulge | sag | pocket density (nats) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | narrow | 4 | 1.9 | 54% | +0.6 [−2, +3] | −1.4 [−16, +12] | −1.0 [−21, +18] | −0.6 [−14, +8] | +0.005 [−0.002, +0.011] |
| 30 | medium | 103 | 11.0 | 10% | **−2.3 [−5, −0]** | −1.2 [−14, +12] | −0.9 [−19, +18] | −0.2 [−12, +8] | +0.003 [−0.006, +0.011] |
| 30 | broad | 2,328 | 17.8 | 1% | −1.3 [−4, +1] | −0.9 [−14, +12] | −1.2 [−19, +17] | +0.1 [−13, +9] | +0.006 [−0.002, +0.014] |
| 60 | narrow | 7 | 3.2 | 44% | −1.2 [−3, +1] | −2.6 [−17, +12] | −2.3 [−22, +17] | −0.8 [−16, +9] | −0.001 [−0.011, +0.009] |
| 60 | medium | 200 | 21.0 | 6% | **−2.9 [−5, −1]** | −1.6 [−15, +12] | −1.4 [−20, +19] | −0.6 [−15, +9] | +0.004 [−0.008, +0.016] |
| 60 | broad | 4,476 | 35.3 | 0% | −1.1 [−3, +0] | −0.9 [−15, +13] | −1.2 [−20, +18] | +0.6 [−15, +10] | **+0.011 [+0.001, +0.022]** |
| 90 | narrow | 9 | 4.2 | 37% | −0.8 [−3, +1] | −2.3 [−16, +12] | −1.9 [−23, +17] | −1.3 [−16, +9] | −0.005 [−0.017, +0.006] |
| 90 | medium | 270 | 26.9 | 6% | **−2.8 [−5, −1]** | −1.4 [−15, +13] | −1.5 [−19, +18] | −0.2 [−14, +9] | +0.003 [−0.010, +0.015] |
| 90 | broad | 5,660 | 44.7 | 0% | −1.2 [−3, +1] | −0.9 [−14, +13] | −1.3 [−20, +18] | +0.8 [−14, +10] | +0.009 [+0.000, +0.019] |

With only 3 parameters (`vwap_vs_prev`, `open_vs_prev`, `move_since_open`), even narrow finds
plenty: 767 matches and 39.1 N_eff days at 60 sessions. Brier and pinball skills are again ≈ 0
(−1.3% to −0.4%, every CI spans 0); pocket density is +0.002 to +0.013 nats, clear of 0 for
broad at every window and for medium at 60–90.

**Speed:** match 2–5 ms, match + pockets 3–7 ms (5 parameters); up to 47 ms when broad finds
about 40k matches (3 parameters, 90 sessions).

**The first draw said more** (a different 500 queries, same months; 2026-10-31 before the
rerun): medium P(up) Brier −0.1% to −0.6% (CI spanning 0) and pocket density +0.012 to +0.020
nats with the CI clear of 0 at every window for medium and broad. Two draws of 500 queries
disagree by about as much as the effects themselves, so read every number here as noise-level.

**Without shrinkage** (the very first run, Jul–Sep library, 400 queries), medium matches were
clearly **worse** than climatology: Brier −11% [−16, −5], pocket density −0.098 nats. The cause
is noise from a handful of matches standing for one or two days. Shrinking by N_eff days removes
most of that, and is now always on; the rerun's medium −2 to −3% Brier is what is left.

### 4.3 A trained model on the same inputs

HistGradientBoosting, trained on 1.5M moments before October, tested on all 843,405 October
moments of stocks with ≥ $20M a day (baseline: the up-rate and quantiles by time of day):

| inputs | P(up) Brier skill | rest-of-day VWAP pinball skill |
| --- | --- | --- |
| your 5 + time of day | −0.5% [−1.5, +0.2] | +0.5% [−8.8, +8.8] |
| all 10 + time of day | −0.8% [−1.8, +0.0] | +1.3% [−7.4, +9.7] |

Training does not find what matching misses: **there is no edge in these inputs** for direction
or the rest-of-day VWAP.

### 4.4 Are the pockets honest? (coverage)

500 October queries, 60 sessions, your 5 parameters. The share of the **actual** later volume
that landed inside each predicted pocket (on target = the pocket's nominal share), and the
density log score (higher is better):

| smoothing | pockets from | inside 50% | inside 80% | inside 95% | the 80% pocket covers | log score |
| --- | --- | --- | --- | --- | --- | --- |
| 0.7 cell | climatology | 57.1% [54, 60] | 84.6% [82, 87] | 98.3% [98, 99] | 6.7% of the map | −4.536 |
| 0.7 cell | medium + shrink | 55.0% [52, 58] | 82.5% [80, 85] | 96.9% [96, 98] | 6.2% | −4.527 |
| 0.7 cell | broad + shrink | 55.9% [53, 59] | 83.0% [80, 86] | 97.4% [97, 98] | 6.3% | −4.520 |
| 1.5 cell | climatology | 61.8% [59, 64] | 88.0% [86, 90] | 98.6% [98, 99] | 9.1% | −4.772 |
| 1.5 cell | medium + shrink | 60.7% [58, 63] | 86.5% [84, 88] | 98.1% [98, 98] | 8.6% | −4.758 |
| 1.5 cell | broad + shrink | 61.0% [58, 64] | 86.8% [85, 89] | 98.3% [98, 99] | 8.7% | −4.756 |

- The pockets are **a little too wide**: more volume falls inside than they claim (55–62% in the
  50% pocket). Less smoothing is closer; the page uses 0.7 cell.
- Matched pockets are **tighter than climatology** (6.2% vs 6.7% of the map) **and closer to
  their nominal share**. That is the same small gain as the density score.
- **Blobs.** The median 50% pocket is one blob. Sparse matches split it (MU medium: 3 blobs),
  and the 95% pocket often splits into several (the 11:00 climatology: 4): a near-term blob
  around now and a later one towards the close. So blobs, not bars, are the right picture.
  (The first draw's "often 2–3 blobs" was from the MU example, not the median.)

## 5. Verdict

- **Feasible and fast.** A 90-session library is about 6M moments and builds in under a minute.
  A match takes milliseconds, so the page can re-match on every slider move.
- **No training is needed** for the engine. It is weighted counting. A trained model (HGB) on the
  same inputs gains nothing either.
- **No predictive edge yet** on these inputs, for direction, the rest-of-day VWAP, the bulge or
  the sag. The pinball CIs all span 0; the medium dial's P(up) is even 2–3% worse than
  climatology in the rerun.
- **The pocket shape is at most slightly better than climatology:** broad +0.006 to +0.011
  nats (CI clear of 0 at 60 sessions), tighter pockets, closer coverage. The first draw showed
  +0.012 to +0.020. It is about where the volume goes, not which way the price goes, and it is
  near the noise level.
- **The narrow dial needs a short parameter list.** With 5 parameters and 30–90 sessions, a third
  to a half of queries find fewer than 5 matches. Medium and broad with 60–90 sessions are the
  useful defaults for the page; broad is the better-scored of the two.
- **N_eff days, not the match count, is the honest evidence.** Matches cluster on market-wide
  days.

## 6. The dashboard view (Intraday → Space)

Built from the same engine in a standalone package, `space/` (`mtap_space`; numpy, pandas,
pyarrow; no ml/ or sourcing imports).

- **Where:** Dashboard → Intraday → View: **Space (pockets)**. The other view, **Model runs**,
  keeps the six `tft_intraday` tabs.
- **Sidebar:**
  - library;
  - the moment: stock, date, time;
  - parameters, with your 5 as the default, and per-parameter tolerance;
  - two dials: close (0.5) and loose (1.0);
  - history: 30 / 60 / 90 sessions;
  - pool filters.
- **3-D pocket space (three.js):** close match, loose match, climatology and what the stock
  actually did, side by side on one canvas with one shared camera.
  - Height = volume density (√ by default, or linear); colour = pocket (50 / 80 / 95%, grey
    outside), blue above now and red below (blue/red rather than green/red, so colour-blind
    readers can tell them apart); rings at the pocket edges with floor shadows; a wall at now.
  - Hover shows the cell (time window, $ band, share, region); a click picks the cell for the
    Compare tab's picked-cell table.
  - The camera stays where you left it when a control changes: the scene is rebuilt in place,
    not re-sent.
  - Built as a `st.components.v2` component: `dashboard/js/src/space3d.js`, bundled with esbuild
    into `dashboard/forecast_dash/space3d/space3d.js` (about 0.58 MB, three.js 0.186.1, MIT).
- **Evidence table:** matches, stocks, N_eff days, P(up), rest-of-day VWAP, bulge, sag and share
  above now. For past dates it adds the actual coverage and the log score vs climatology.
- **Tabs:** Compare (2-D contours, match − climatology), Matches (the closest 50; pick one to see
  its own pocket), Dial curve, Day path.

How to build and open it: §8.

## 7. Improve later

Ideas, roughly in order of promise:
1. **More informative inputs.** The current ones are all about the stock's own price path; none
   predicts. Candidates:
   - the market's and sector's move so far (breadth, the index vs its VWAP);
   - results day / news;
   - the daily model's forecast for today (R5-cal20s P50 and width), once it runs live;
   - the pre-market;
   - 1-second order flow (signed volume).
2. **Learned similarity.** Weight parameters by how much they matter (forest proximity, or a
   metric learned on the readings) instead of equal tolerances.
3. **Longer history.** 5 years of `bars_1m` exist. Test whether a 1-year window, or a
   regime-matched one, beats 60 sessions.
4. **Calibrate the pockets.** They hold slightly more than nominal: tune the smoothing per time
   of day, or recalibrate the levels on recent days.
5. **Score it as a ranking.** Rank October stocks by match P(up) at 10:00 and check the top and
   bottom deciles' realised moves. That is the momentum-list goal in
   [forecasting-design-brief.md](forecasting-design-brief.md).
6. **Live.** Today's moment would come from the live feed, after a parity check against Massive
   minute bars.

## 8. How to rerun

**The check** (read-only, about 10 minutes; needs the NAS mounted at `/Volumes/Backups`):

```
cd ml
uv run python scripts_space_analog_check.py                       # all four parts
uv run python scripts_space_analog_check.py --parts example       # just the MU example
uv run python scripts_space_analog_check.py --parts windows,coverage --queries 300 --seed 2
```

- `--parts`: example, windows, hgb, coverage.
- `--queries`: October queries per run (default 500).
- `--seed`: the query draw.
- The months and the query window are constants at the top of the script.

**The page:** see `space/README.md` and the dashboard README (Intraday → Space).

## 9. Data notes found on the way

- **`bars_1m` layout.** Each `month=YYYY-MM/` folder holds that month's bars only, even though
  its file is named after the whole quarter (e.g. `month=2026-08/bars-2026-07-01_2026-09-30.parquet`
  holds August). One month is about 27M rows, 21–22 sessions, all tickers, 04:00–19:59 ET.
- **Time zone.** `t` is UTC ms. Converting with a fixed −4 h is right only in EDT; from
  2026-11-01 (and every winter) it is −5 h. The rerun script and `space/` convert per UTC date
  with the New York time zone. (Jun–Oct 26 is all EDT, so this check was not affected.)
- **Volume:** the daily VWAP rebuilt from minute bars excludes the auction prints in the daily
  bar's volume (about 19%; see `volume` in [ml/FEATURES.md](../ml/FEATURES.md)). That does not matter
  here, because both sides use minute bars.
