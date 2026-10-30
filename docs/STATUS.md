# MTap status and reference

The one page that says what runs where, when, on what data, with which secrets and limits.
The details are in the linked docs.

**Last updated: 2026-10-29 18:45 PT.**

**Keep it current.** Any change to any of these updates this file, in the same change:

- a job, host or schedule;
- a data path or data set;
- where a secret lives;
- an external limit;
- a standing rule.

Add a line to [Changes](#changes) at the bottom. Deferred work goes in [TODO.md](TODO.md),
not here.

## What MTap is

MTap sources market data for the ML work.

- **Sources:** Massive (REST and WebSocket), the E*TRADE website, SEC EDGAR and Yahoo
  news.
- **Where the output goes:** files on the NAS, then local DuckDB stores, then the model's
  panel (`out/tft/panel.parquet`).
- **How the code is split:** Python and Node share no code, only the output contract in
  `schemas/` ([README.md](../README.md)).

## Repo map

| folder | what | doc |
| --- | --- | --- |
| `schemas/` | the output contract: JSON schemas and conventions | [conventions.md](../schemas/conventions.md) |
| `python/` | `sourcing-py`: Massive sources, the intraday store, massive-ws, massive-ext, E*TRADE fundamentals and EDGAR (DuckDB), news | [python/README.md](../python/README.md) |
| `node/` | the E*TRADE watchlist collector (puppeteer) and Node sources | [node/README.md](../node/README.md), [etrade README](../node/src/services/etrade/README.md) |
| `shared/config/` | source registry; `settings.toml` (gitignored, has secrets) | |
| `deploy/container/` | the generic Docker target: images, compose, `jobs.sh` | [README](../deploy/container/README.md) |
| `deploy/synology/` | the NAS wrapper: host.env, DSM task.sh, push.sh; **the runbook** | [README](../deploy/synology/README.md) |
| `scripts/schedule/`, `scripts/storage/` | the macOS (iMac) target: launchd, pmset, SMB mount | [scripts/README.md](../scripts/README.md), [IMAC.md](../scripts/IMAC.md) |
| `nas/massive1s/` | Rust 1-second REST backfill, runs on the NAS | [nas/README.md](../nas/README.md) |
| `ml/` | TFT trainer, report, predict, grade (to move to its own repo) | [ml/README.md](../ml/README.md), [MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) |
| `dashboard/` | Streamlit forecast dashboard (reads run files only) | [dashboard/README.md](../dashboard/README.md) |

## Hosts

| host | role | how it runs |
| --- | --- | --- |
| **NAS**: Synology DS925+, 192.168.1.100 | **all scheduled sourcing, from Mon 2026-10-26** (production config since 2026-10-25) | DSM Task Scheduler (root) → `deploy/synology/task.sh JOB` → `deploy/container/jobs.sh` |
| **This Mac** (laptop) | development; owns the DuckDB stores; ML training; dashboard | by hand. Its Docker only builds and tests images, never runs jobs |
| **iMac** | ran the sourcing until Fri 2026-10-23 | stopped at cutover (10-25) and kept as the fallback; retired after a stable week ([nas-migration.md](nas-migration.md#decommissioning-the-imac)) |

## The weekday schedule (PT)

| time | NAS | notes |
| --- | --- | --- |
| 06:00 | power on (DSM power schedule) | |
| 06:10 | `mtap-start` → `start` | Redis, store, E*TRADE collector; massive-ws (connects 09:00 ET = 06:00 PT, ends after 16:21 ET) |
| 06:40 | `mtap-daily` → `daily` | Massive grouped daily and extended hours for every missed session (14 days, at most 10) |
| 06:14 | `mtap-morning` → `morning` (**not deployed yet**, branch `feat/daily-5y-inputs`) | no jobs lock, pid file. `forecast-night` if no done claim; 09:29:30 ET `premarket precheck` → shortlist; 09:30 `premarket prices` (+ `--open` to 09:32), `export-next`, predict px, screener; 09:45 `premarket full` (Massive), `export-next`, predict full, screener |
| 13:15 | `mtap-evening` → `evening` | stop the collector, flush, intraday summary, daily fill, wait for massive-ws, compact; done by about 13:45 |
| 22:10 | power off | |
| 17:25 | `mtap-night` → `night` (**not deployed yet**) | `forecast-night --date <next session>`: Massive EOD + extended hours of day t, derive-features, universe, xids, export-tft, TFT night forecast. Claim file `MTap-State/forecast/claim-D.json` makes it run once across NAS and Mac; refused before 20:20 ET |
| every 5 min | `mtap-queue` → `queue` | `watch` (in market hours, an email when a heartbeat is over 10 min old, and when all recover), then the job named in `MTap-State/run-next` (how the Mac runs `status` or a one-off job) |

- **Not scheduled:**
  - `massive1s` (run by hand in DSM);
  - `massive_tickers_cs` (by hand);
  - the DuckDB load on this Mac (by hand; TODO.md).
- **Status from the Mac:** `deploy/synology/push.sh status`, or read
  `MTap-State/logs/jobs.log` over ssh.

## Data

**On the NAS, `/volume2/Backups/MTap-Data`** (`/Volumes/Backups/MTap-Data` on a Mac):

| data | path | from | written by |
| --- | --- | --- | --- |
| daily bars | `out/daily_summary/daily_summary-D.jsonl` | Massive grouped daily (adjusted) | `daily`; rolling 5 years (first file 2021-10-21) |
| pre-market / after-hours | `out/extended_hours/extended_hours-D.jsonl` | Massive 1-min aggregates | `daily` (`massive-ext gaps`) |
| common-stock universe | `out/tickers_cs/` | Massive reference tickers | by hand |
| E*TRADE watchlists | `intraday_feed/date=D/source=etrade_watchlist/hHH-*.parquet` | E*TRADE website, every ~2 s and ~20 s | collector → Redis → store, hourly |
| Massive second bars, live | `agg_1s/date=D/source=massive_ws/`, compacted to `day-D.parquet` | Massive WebSocket `A.*`, 15 min delayed | massive-ws; `evening` compacts |
| Massive second bars, history | `agg_1s/date=D/source=massive_rest/day-D.parquet` | Massive REST 1-second bars | `nas/massive1s` (2-year run deployed 2026-10-19) |
| news | `news/` | Yahoo raw HTML; a read-only copy of news.duckdb | by hand |
| treasury yields, inflation | `out/treasury_yields/`, `out/inflation/` (vintage files) | Massive economy endpoints | `daily` (`daily-capture.sh`); yields lag 1 session, CPI used from M+2 |
| shares outstanding | `out/shares/shares-D.jsonl` (quarter ends) | Massive reference | `daily`, `inputs-backfill` |
| minute bars (panel stocks) | `bars_1m/month=YYYY-MM/*.parquet` (US Eastern month) | Massive 1-min aggregates | `inputs-backfill` (`--save-bars`, `MTAP_BARS_UNIVERSE`) |
| morning live files (audit only, never training) | `out/premarket_live/{universe,xids,shortlist,prices,premarket}-D.*` | Massive snapshot + minute aggs, E*TRADE watchlist (Redis), ChartIQ prices | `night`, `morning` |
| screener lists | `out/screener/screener-D.parquet`, `-top20.csv`, `outcomes.parquet` | `ml/screener` | `morning` (when ml/ is in the job container), or the Mac |
| DuckDB backups | `backup/etrade/` | | `daily-sourcing.sh --ingest-only` on this Mac |

**Elsewhere:**

- **Trained model runs** (`ml/runs/<id>/`: checkpoints, config, test predictions; not in git):
  copied to `/volume2/Backups/PProjects/MTap/ml/runs` (`/Volumes/Backups/PProjects/MTap/ml/runs`)
  by `scripts/storage/backup-runs.sh` after each run.

- **NAS state** (`/volume2/Backups/MTap-State`): segments not yet uploaded, heartbeats, the
  E*TRADE Chrome profile, logs. It is local to the containers; never open it over SMB.
- **NAS trial data:** `/volume2/Backups/MTap-Trial`, from the 2026-10-24 trial. It can be
  deleted.
- **This Mac's local stores** (never on SMB: DuckDB needs file locks):
  - `state/etrade/fundamentals.duckdb`: symbols, daily_bars, fundamentals, SEC filings,
    EDGAR, splits;
  - `state/news/news.duckdb`;
  - `out/tft/panel.parquet`.

## Secrets (where they live, never their values)

| secret | NAS | iMac | this Mac |
| --- | --- | --- | --- |
| `MASSIVE_API_KEY` | `~/.mtap.env`, `~/.massive1s.env` (chmod 600) | repo `.env` | `shared/config/settings.toml` |
| E*TRADE login (`ET_*`), changed 2026-10-24 | `~/.mtap.env` | repo `.env`: **needs the new password** | |
| E*TRADE device trust and session | `MTap-State/etrade/chrome-profile` | `<state>/etrade/chrome-profile`, `session.json` | `state/etrade/session.json` (fundamentals) |
| E*TRADE bearer token (live, short-lived) | `MTap-State/etrade/bearer.json` (mode 600), written by the collector; read only by `premarket prices`, which never logs in and stops on the first 401/403/429 | | |
| noVNC password (`SE_VNC_PASSWORD`) | `~/.mtap.env` | | |
| SMS gateway (`SMS_EMAIL*`) | not set: no texted OTP link | `.env`, if set | |
| NAS SMB password | | login Keychain (`mount-nas.sh --save-password`) | |

## External services and limits

- **Massive:**
  - Paid plan since 2026-10-14, with 5 years of rolling history.
  - **One WebSocket connection per key.**
  - REST is not throttled (`MASSIVE_RATE_LIMIT_PER_MIN=0`, 16 workers).
  - A day's grouped bars appear only after midnight ET.
  - History comes back split-adjusted and is corrected late.
  - Details: [massive facts in LEARNINGS.md](LEARNINGS.md).
- **E*TRADE:**
  - Website login with puppeteer, behind Akamai Bot Manager.
  - On the NAS it is headed Google Chrome 155. The iMac ran headless Chrome for Testing 127.
  - A new device needs a texted code; one session at a time.
  - Details: [nas-migration.md](nas-migration.md#etrade-on-the-nas).
- **SEC EDGAR:** at most 10 requests/s (we use 8), with a User-Agent that names us.
- **Yahoo:** needs a browser User-Agent; otherwise it returns 429.

## Standing rules

- **One host at a time** for the Massive WebSocket and the E*TRADE session. Stop one
  before starting the other (cutover, rollback, any test).
- **Secrets** are never in the repo or on the share.
- **DuckDB** is never opened over SMB.
- **The iMac's code** (`scripts/schedule`, `scripts/storage`) is not moved or edited.
  Changes to the core are additive and env-gated, with today's behaviour as the default.
- **Jobs run on the NAS.** The Mac's Docker only builds and tests.
- **Where findings go:**
  - analyses: [LEARNINGS.md](LEARNINGS.md), with tags;
  - ML questions: [ml/MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md);
  - deferred work: [TODO.md](TODO.md).

## Current state

- **2026-10-24:**
  - The NAS trial passed steps 0–9, including a password-only login with no code (results
    in the runbook and LEARNINGS.md `tag-sourcing-host`).
  - The DSM weekday tasks were created and checked; `massive1s` was unticked.
- **2026-10-25:**
  - The overnight trial containers came back at boot and failed 70 logins (Chrome's saved
    login doubled the typed one). Fixed: the saved password was deleted and saving is off,
    and the collector logged in by itself at 11:42.
  - Recovery fixes: a log-off no longer loops, and a failed recovery exits, so the restart
    loop starts a fresh collector (the 10-24 idle gap is closed).
  - **The NAS cutover is done (12:30):** the new image was built and tested on the trial
    (a new container removed the old lock and reused the session). Then came `stop`, the
    production push, `selftest`, `ws-test`, and `daily` (10-23 written). Everything is
    stopped until the 06:10 `mtap-start`.
  - **Open:** `setup-schedule.sh stop` on the iMac (before Monday 06:00), and the new
    E*TRADE password in the iMac's `.env`.
- **Next:** Monday 10-26, the first live day on the NAS, with the runbook's Monday checks.

- **2026-10-27:** the daily model's 5-year retrain and the same-morning forecasts are built on
  `feat/daily-5y-inputs` (ml/MODEL-STRATEGY.md Q11–Q16): new panel inputs, the `morning` and
  `night` jobs, the screener and the dashboard Watchlist page. **Not deployed:** the NAS push,
  the collector restart outside market hours, S1 (one live morning) and S7 (inference time on
  the NAS) come first, then the 5-year load and the runs R0–R8.

## Model performance

The daily model's research runs on the 5-year panel, as of **2026-10-29 18:45 PT**. Choices and
verdicts: [ml/MODEL-STRATEGY.md Scoreboard](../ml/MODEL-STRATEGY.md). Details and CIs:
[LEARNINGS.md](LEARNINGS.md) `tag-ml-daily`. Each table scores every model on the same
(stock, day) rows. Production (`20261005-025909Z`, 2-year) stays live until all models are
evaluated.

**Goal (user, 10-29):** one model for every forecast at and after 09:30, with full pre-market. The
night forecast is dropped. Models are ranked first by how well the quantiles hit their probabilities
(mean/max |q err|, grade, ECE, coverage, bias), then by pinball loss.

**Candidate single model: R5 + rolling recal** (`ml/runs/20261029-211059Z-gbm-4438sym-cal20s`
for 09:45, `…~px-cal20s` for 09:30). One HGB for 09:30 and 09:45; each day's quantiles are shifted
by the error of the 20 sessions before it, per pre-market × width group (`tft-vwap-recal
--sessions 20`). Folds 3–13, all 1,265,370 rows every run has:

| model | forecast | pinball | mean / max \|q err\| | ECE | P5–P95 / P10–P90 | bias | grade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **R5-cal20s** | 09:45 | 0.006691 | **0.18 / 0.31** | 1.16 | 89.5 / 79.5 | +0.06 | excellent |
| **R5~px-cal20s** | 09:30 | 0.006710 | **0.18 / 0.31** | 1.16 | 89.5 / 79.5 | +0.06 | excellent |
| R5-cal (4 folds) | 09:45 | 0.006675 | 0.57 / 0.94 | 1.37 | 90.5 / 80.7 | −0.53 | good |
| R5 | 09:45 | 0.006674 | 0.79 / 1.26 | 1.27 | 88.5 / 78.4 | −0.64 | good |
| R7f-cal | 09:45 | 0.006921 | 0.35 / 0.54 | 1.00 | 90.0 / 80.1 | +0.33 | excellent |
| R6p-cal | 09:30 | 0.007003 | 0.47 / 0.89 | 1.34 | 89.9 / 79.9 | +0.45 | excellent |

R5 beats R7f-cal by 3.5% and R6p-cal by 4.4% in pinball (CIs clear, every fold). Rolling recal is
good or excellent in 13 of 14 folds; the 4-fold recal is poor in 3, because R5's error drifts
between quarters. R8 (screener on R5~px-cal) is kept: top 20 range 12.5% vs 11.0% for ATR, CI
clear. **Production switch: the user's call.** MODEL-STRATEGY Q36.

**Earlier calibration table** (6 odd folds 3–13, 716,862 rows with full pre-market). `-cal` = walk-forward
recalibration (`tft-vwap-recal --by pm,width --window 4`): each quantile moves by the error seen on
the 4 test blocks before it, per pre-market × width group.

| model | forecast | pinball | mean / max \|q err\| | ECE | P5–P95 / P10–P90 | bias | grade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R5-cal20s | 09:45 | 0.007877 | 0.17 / 0.30 | 1.03 | 89.6 / 79.6 | +0.12 | excellent |
| R5 | 09:45 | 0.007860 | 0.84 / 1.25 | 1.16 | 88.5 / 78.4 | −0.69 | good |
| **R7f-cal** | 09:45 | 0.007901 | **0.10 / 0.22** | 0.97 | 90.0 / 80.0 | +0.05 | excellent |
| R7f | 09:45 | 0.007902 | 0.78 / 1.08 | 1.17 | 89.4 / 79.3 | −0.72 | good |
| R6f-cal | 09:45 | 0.007908 | 0.16 / 0.37 | 1.31 | 90.0 / 80.1 | +0.15 | excellent |
| R4f-cal | 09:45 | 0.007950 | 0.87 / 1.55 | 1.45 | 90.8 / 81.5 | −0.57 | good |
| R6p-cal | 09:30 | 0.007967 | 0.16 / 0.40 | 1.30 | 90.0 / 80.0 | +0.16 | excellent |
| R4p-cal | 09:30 | 0.007971 | 0.89 / 1.94 | 1.59 | 91.0 / 82.1 | +0.70 | good |

Recalibration takes out most of the calibration error at no cost in pinball. On the pre-market
rows alone R7f-cal is a little better calibrated than R5-cal20s, but 0.3% worse in pinball and
0.74 pts on the rows without pre-market. Details: MODEL-STRATEGY Q35, Q36.

**Night forecast** (made the evening before, no pre-market; dropped 10-29, kept for reference). Test folds 1–14, 2.95M rows, 882 days:

| model | what | skill | vs R0 (95% CI) | P5–P95 | P10–P90 | direction | big-move direction | rank IC | bias | mean \|q err\| | grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 5-year baseline | 26.6% | — | 88.7% | 78.7% | 59.0% | 65.6% | 0.30 | −0.25 | 0.40 | excellent |
| R1 | + market, macro, shape, after-hours | 25.3% | +1.72% (+1.12 … +2.39) | 87.5% | 76.1% | 58.7% | 65.5% | 0.30 | −0.75 | 1.41 | fair |
| R1c | + session shape, after-hours (TFT base) | 27.2% | −0.79% (−1.10 … −0.49) | 88.6% | 78.6% | 59.6% | 66.7% | 0.31 | +0.14 | 0.57 | good |
| **R7** | HGB per quantile on R1c's inputs, no TFT | **27.9%** | **−1.81% (−2.12 … −1.47)** | 88.8% | 78.7% | 60.1% | 67.3% | 0.32 | −0.68 | 0.78 | good |
| R6c | event-band overlay on R0 | 26.6% | +0.02% (+0.00 … +0.05) | 88.8% | 78.8% | 59.0% | 65.6% | 0.30 | −0.26 | 0.39 | excellent |
| R1a | R1 − macro (folds 1–6 only) | 23.6% | +2.97% (+1.51 … +4.50) | 86.5% | 75.4% | 57.4% | | | −1.94 | 2.29 | poor |
| rule | yesterday's close vs its VWAP | | | | | 61.3%* | 67.7%* | 0.33* | | | |

**Pre-market bake-off** (night vs 09:30 vs 09:45). 7 odd folds, rows with full pre-market data,
812k rows, 441 days (more active stocks, so skill differs from the table above):

| model | forecast | approach | skill | vs R0 (95% CI) | P5–P95 | P10–P90 | direction | big-move direction | rank IC | bias | mean \|q err\| | grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | night | baseline | 26.8% | — | 88.0% | 77.9% | 60.1% | 66.4% | 0.31 | −1.33 | 1.46 | fair |
| R1c | night | base | 27.6% | −1.07% (−1.43 … −0.75) | 87.9% | 77.8% | 60.7% | 67.6% | 0.33 | −0.64 | 0.82 | good |
| R7 | night | HGB, no TFT | 28.6% | −2.75% (−3.15 … −2.37) | 87.6% | 77.1% | 61.3% | 68.3% | 0.34 | −1.19 | 1.41 | fair |
| R4~night | night | A: one masked TFT | 27.4% | −0.83% (−1.36 … −0.45) | 86.9% | 76.7% | 60.4% | 67.0% | 0.32 | −1.55 | 1.79 | fair |
| R4~px | 09:30 | A | 39.2% | −16.9% (−17.7 … −15.6) | 89.0% | 79.5% | 70.1% | 80.5% | 0.53 | −1.25 | 1.35 | fair |
| R4p | 09:30 | B: dedicated TFT | 39.6% | −17.5% (−18.4 … −16.1) | 88.0% | 77.4% | 69.9% | 80.3% | 0.53 | −0.87 | 1.18 | fair |
| **R6p** | **09:30** | C: overlay on R1c | **39.8%** | −17.8% (−18.8 … −16.6) | 90.8% | 81.0% | 70.8% | 81.1% | 0.54 | −0.91 | 0.99 | good |
| R4 | 09:45 | A | 39.3% | −17.1% (−17.9 … −15.8) | 88.8% | 79.2% | 70.1% | 80.5% | 0.53 | −1.36 | 1.47 | fair |
| R4f | 09:45 | B | 39.9% | −17.9% (−18.8 … −16.6) | 89.8% | 79.9% | 70.3% | 80.7% | 0.53 | −1.00 | 1.09 | fair |
| **R6f** | **09:45** | C | **40.3%** | −18.4% (−19.4 … −17.2) | 90.8% | 81.1% | 70.8% | 81.1% | 0.54 | −0.92 | 1.00 | good |
| **R7f** | **09:45** | HGB + full pre-market, no TFT | **40.3%** | −18.3% (−19.5 … −17.2) | 89.3% | 79.2% | 70.5% | 81.0% | 0.53 | −0.71 | 0.77 | good |
| rule | 09:30 | open vs previous VWAP | | | | | 70.7% | 81.0% | 0.53 | | | |
| rule | night | yesterday's close vs its VWAP | | | | | 61.3% | 67.7% | 0.33 | | | |

\* on the pre-market rows, not the folds 1–14 rows. R7's and R7f's figures are on the 808k of the 812k rows
every model has; the other models' figures move by 0.01 at most on them.

**Leaders per forecast:**
- **night:** R7, 1.3% better than R1c on 3.17M rows (folds 0–14, CI clear), on earnings and other
  days;
- **09:30:** R6p, 0.35% better than R4p and 1.05% better than R4~px;
- **09:45:** R6f and R7f tie (R7f 0.06% better, CI spans 0, no fold clear). Both beat R4f (0.55%,
  0.76%) and R4 (1.55%, 1.6%). On a tie fewer models wins, so R7f leads: one model instead of R1c +
  an overlay. R6f's bands are closer to 90% (P5–P95 90.8% vs 89.3%).

All of these CIs are clear of 0 except R7f vs R6f. R5 (one HGB for 09:30 and 09:45) now beats all
of them; see the first table.

**What it says:**
- **Pre-market is the big gain.** Skill goes from 27% to 40%, the loss drops 17–18% and direction
  goes from 60% to 70%. Which of the approaches (A, B or C) is used is worth only 0.3–1.5% more.
- **Direction: no model beats the one-input rule.** "Today moves the way the open moved vs the
  previous VWAP" gets 70.7%. At night, "the way yesterday closed vs its VWAP" gets 61.3%, above
  every night model. The models gain in move size and honest bands, not in calling up or down.
- **Calibration is good to fair everywhere.** Every model is a little pessimistic: the actual
  lands slightly above the forecast. The overlays (R6p, R6f) and R7f are the best calibrated in
  the morning.
- **Plain gradient boosting beats the TFT on these inputs.** R7 (minutes on the CPU, no GPU) beats
  R1c at night, most in the early folds where the TFT has the least training data; R7f ties the
  TFT + overlay at 09:45. The TFT's advantage, the sequence of past days, adds little over lags 1,
  2, 5 and 20 of the move.

### Legend

| term | meaning | good is |
| --- | --- | --- |
| pinball loss | the quantile loss over the 19 quantiles (P5…P95) of the next-session VWAP move; scores the whole forecast, centre and spread | lower |
| skill | 1 − pinball / naive pinball, where naive = every quantile at a 0% move | higher, see bands |
| vs R0 (CI) | paired pinball change against R0 on the same rows; 95% bootstrap CI over days. −1% = 1% lower loss | below 0, CI clear of 0 |
| P5–P95, P10–P90 | share of actual moves inside the band | 90% and 80% |
| direction | how often the median (P50) had the sign of the actual move right | above 50% and above the rule |
| big-move direction | the same on moves bigger than 2% | as above |
| rank IC | per day, the Spearman correlation between the median forecast and the actual move across stocks, averaged; does it order the stocks right | above 0.05 is useful |
| bias | 0.5 − mean PIT, in points; above 0 optimistic (actual below the forecast), below 0 pessimistic | near 0 |
| mean \|q err\| | per quantile, \|share of actuals below it − its level\| in points (P5 should have 5% below), averaged over the 19 | near 0 |
| rule | a one-input forecast: direction = the sign of that input | the bar a model must beat |
| A / B / C | the bake-off approaches: one masked TFT for all three forecasts / a dedicated TFT per forecast / gradient-boosted overlays on R1c | |

**Calibration grade** (from mean |q err|; proposed 10-29, in the dashboard): ≤ 0.5 pts excellent,
≤ 1.0 good, ≤ 2.0 fair, above 2.0 poor. The older `calibrated` label (|bias z| < 2) is not used:
with millions of rows every model fails it.

**Skill bands** (MODEL-STRATEGY Q28):

| skill | night forecast | with same-day pre-market |
| --- | --- | --- |
| below 0% | broken: worse than "no move" | broken |
| 0–15% | weak | weak |
| 15–25% | fair (production is 22.9%) | weak |
| 25–30% | good (R0, R1c) | fair |
| 30–35% | very good | fair |
| 35–45% | suspect: check for leakage | expected (R4p, R6p, R6f, R7f) |

### Comparing models in the dashboard

The forecast dashboard's **Compare runs** tab shows this comparison for any runs (MODEL-STRATEGY
Q30):

1. `cd dashboard && uv run streamlit run forecast_dash/app.py`, then go to the cross-day section.
2. Open **Compare runs** and pick the runs. The first one picked is the reference for `vs_first`. Runs show by their label (R0, R1c,
   R6f, …) from the run's `MODEL.md`, which every training command writes; an ⓘ per run opens its card.
3. Keep **Only rows present in every chosen run** ticked.

Each run gets one row: pinball, `vs_first`, skill, direction, big-move direction, rank IC, median
error, both coverages, `q_err_mean`/`q_err_max`, the grade, bias and ECE. The tab also shows a
reliability chart and the calls per run. The morning states are separate runs: R4's `…~px` and
`…~night`, and R6p/R6f as `…-overlay-px`/`…-overlay-full`.

It does not show:
- a CI on the differences (`tft-vwap-compare` gives that);
- the one-input rules;
- a pre-market-rows filter. Picking R0 or R1c with the morning runs scores all 1.48M odd-fold
  rows, so the morning skill reads ~35% there instead of ~40%.

The tables above come from `ml/scripts_compare_vs_r0.py night|bake`, `ml/scripts_direction.py` and
`ml/scripts_direction_naive.py`.

## Docs index

| doc | what |
| --- | --- |
| [nas-migration.md](nas-migration.md) | why and how the sourcing moved to the NAS; E*TRADE on the NAS; decommissioning the iMac; a cloud VM later |
| [deploy/synology/README.md](../deploy/synology/README.md) | NAS runbook: one-time setup, trial, cutover, rollback, Monday checks |
| [LEARNINGS.md](LEARNINGS.md) | what the data checks found, by tagged topic |
| [intraday-model.md](intraday-model.md) | the intraday model: data, the daily forecast as input, folds aligned with the daily model |
| [TODO.md](TODO.md) | deferred work |
| [database-schema.md](database-schema.md) | DuckDB ER diagrams (generated; CI checks them) |
| [stock-prediction-plan.md](stock-prediction-plan.md), [next-session-parquet-tft.md](next-session-parquet-tft.md) | ML plans |
| [../ml/MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) | **Scoreboard** (current model choices and results), then the ML decisions Q&A |
| [../ml/GUIDE.md](../ml/GUIDE.md) | ML terminology, model architecture, metrics, decisions made without asking, shortcomings, how to interpret reports |

## Changes

- **2026-10-24:**
  - Created.
  - NAS trial steps 0–8 passed.
  - The NAS collector uses Google Chrome 155 with its own UA and client hints, after
    Akamai refused Chrome for Testing.
  - The E*TRADE password changed.
  - Trial step 9 (password-only login) passed.
  - The DSM weekday tasks were created.
- **2026-10-25:**
  - The collector exits when its session recovery fails; a log-off no longer loops.
  - Chrome's saved E*TRADE password on the NAS was deleted and saving turned off.
  - `run-etrade.sh` removes a Chrome profile lock left by another container.
  - The cutover starts with a NAS `stop`.
  - `push.sh` (production) fixed; the NAS runs production: MTap-Data, roles `etrade massive-ws`.
  - DSM updated to 7.4.1-90080 (Compose still v2.20.1; tasks unchanged); updates notify only.
  - Alerts: `watch` in every queue tick, mailed by DSM on a non-zero task exit.
  - 2026-10-26: Redis `--maxmemory` 2 GB to 4 GB in compose.yml (4000 entries of up to ~490 KB is ~1.95 GB); applies at the next `start`, no build needed.
- 2026-10-27 (5y-morning): `etrade premarket precheck|xids|prices|full` and `forecast-night` (claim file
  `MTap-State/forecast/claim-D.json`, earliest 20:20 ET). NAS jobs `morning` (DSM mtap-morning 06:14 PT, no jobs
  lock, pid file) and `night` (mtap-night 17:25 PT) replace `premarket`. Settings: `[premarket]` live_source,
  shortlist_size, move_pct, vol_ratio, full_workers; `[forecast]`. Needs a container-local DuckDB, ml/ + model
  in the job container (MTAP_PREDICT_CMD, MTAP_FORECAST_RUN_DIR); see deploy/synology/README.md.
- 2026-10-27 (5y retrain, not deployed): panel schema 4 gains `volume`, `vol_10d_mean` and the two
  pre-market groups (`pm_px_known`, `pm_full_known`); `[premarket] cutoff` 09:30, `min_dollar_volume`,
  `min_price`; `etrade universe`; massive-ext stores `pm_last/pm_last_t/pm_high/pm_low` per cutoff;
  `--bars-universe` (env `MTAP_BARS_UNIVERSE` in `inputs-backfill`) and ET-month partitions; `[screener]`
  settings; env `MTAP_SCREENER_CMD`, `MTAP_SCREENER_MODEL` for the morning job.
- 2026-10-27: the panel, the liquid universe and the market/sector index are common stocks (CS) only;
  ADRs dropped (`export-tft --issue-types CS,DR` brings them back). NAS `inputs-backfill` started
  17:25 (`MTAP_BACKFILL_FROM=2021-10-29` in host.env); image rebuilt 17:21.
- 2026-10-28: the daily 5-year baseline R0 (`ml/runs/20261028-030847Z-return-4438sym`) finished: skill 26.6%,
  6.2% better than production on shared days (LEARNINGS `tag-ml-daily`). This Mac's DuckDB now has the yields,
  inflation (`ingest-reference`) and the refetched extended-hours VWAPs (`ingest-eod --extended-only`), so
  `macro_daily` and the after-hours inputs are filled (they were NULL → 0 in R0's panel); backup
  `state/etrade/fundamentals.pre-r1-inputs.duckdb`. A missing CPI month (2025-10) carries the previous one
  (`CPI_CARRY` 2). R1 panel `out/tft/exp5y-r1/panel.parquet`; R1 training on this Mac
  (`logs/r1.log`). Trained runs are copied to the NAS `PProjects/MTap/ml/runs/` by
  `scripts/storage/backup-runs.sh` (queued after R1 with `--after PID`).
- 2026-10-28 (later): R1a (R1 − macro) and R1c (R1 − macro − market) queued after R1 on this Mac
  (`logs/r1-variants.sh`, log `logs/r1-variants.log`), each backed up to the NAS when done.
  `tft-vwap-handoff` builds the daily → intraday table; R0's is
  `ml/runs/20261028-030847Z-return-4438sym/handoff.parquet` (3.17M rows, 2023-01-20 … 2026-10-26).
  [intraday-model.md](intraday-model.md) documents it, the 5-year `bars_1m` history and the fold
  alignment; the intraday model itself is revised separately.
- 2026-10-28 16:32: R1a paused after fold 6 at the user's request (worse than R0), backed up; R1c
  `ml/runs/20261028-233305Z-return-4438sym` started from the same queue. Nothing is queued after R1c:
  the next runs wait for the input review (ml/MODEL-STRATEGY.md Q21).
- 2026-10-28 (evening): `tft-vwap-event-overlay` (R6c) built and run on R0 in seconds: no gain (LEARNINGS, ml/MODEL-STRATEGY.md
  Q24). Output `ml/runs/20261028-030847Z-return-4438sym-event*` (not backed up; regenerated on demand). The planned
  release-calendar source (BLS, FOMC) would run on the NAS in `daily`; not built.
- 2026-10-28 17:45: R4 queued after R1c on this Mac (`logs/r4-queue.sh`, log `logs/r4-queue.log`), as the user asked:
  base = R1c only if it beats R0 with a CI clear of 0 (Q21 rule 1), else R0; R4 = base + pre-market `optional` on folds
  1,3,…,13; then R6p/R6f overlays on the base, the night/09:30/09:45 comparisons, and the NAS backup.
- 2026-10-28 19:20: R4 queue restarted (pid 72122) with the user's tie rule: base = R1c also when it ties overall
  but is clearly better on earnings days and clearly worse in no fold (Q21 rule 1). Release-calendar overlay waits
  for R4's result.
- 2026-10-28 19:50: overnight queue after R4 on this Mac (`logs/overnight-queue.sh`, log `logs/overnight-queue.log`,
  run ids in `logs/overnight-runs.txt`): R4p and R4f (pre-market prices / prices+full always on, odd folds), R7 (HGB,
  all folds) and R7f (HGB + pre-market, odd folds), each compared with its rivals and backed up to the NAS. The NAS
  stays on overnight for the backups (power-off skipped once). The remote keeps `main` only: feature branches were
  deleted from origin, and this branch's work stays local until the merge.
- 2026-10-28 21:08: R1c finished: 0.58% better than R0 with a CI clear of 0, so it is the base (Q21 rule 1,
  outright). R4 `ml/runs/20261029-040858Z-return-4438sym` started on it from `logs/r4-queue.sh`. R4, R4p and R4f
  still include the constant `issue_type` input (the panel is common stocks only); see 22:30.
- 2026-10-28 22:15: docs. [ml/GUIDE.md](../ml/GUIDE.md) rewritten and checked against the code and run files (terms,
  report columns, runs, decisions made without asking, shortcomings, corrections). ml/MODEL-STRATEGY.md has a
  **Scoreboard** at the top (current model choices and results). Fold counts as built (15 folds, 7 odd) fixed in
  FEATURES, README and MODEL-STRATEGY.
- 2026-10-28 22:30: TFT training drops constant static inputs (`drop_constant_statics`, ml/MODEL-STRATEGY.md Q27):
  `issue_type` is gone from R4p, R4f and every later run; R4 (already running) keeps it. Old runs reload unchanged.
- 2026-10-29: R4 done 00:06 (`ml/runs/20261029-040858Z-return-4438sym`); R6p/R6f overlays and the comparisons done
  01:58; R4p done 04:30 (`20261029-090243Z`); R4f started 05:20 (`20261029-122046Z`). Results: LEARNINGS `tag-ml-daily`,
  ml/MODEL-STRATEGY.md Scoreboard. `logs/overnight-queue.log` is ~10.7 GB of progress-bar output.
- 2026-10-29 09:05: R4f done 08:17 (second at 09:45, behind R6f); R7 (HGB) running since 08:34, R7f queued.
  New [Model performance](#model-performance) section: per-model skill, vs R0, coverage, direction, rank IC,
  calibration grade, the legend and skill bands. Dashboard Compare runs table gains `vs_first`, `direction`,
  `direction_big`, `rank_ic`, `median_abs_err`, `coverage80`, `q_err_mean`/`q_err_max` and `grade`.
- 2026-10-29 09:45: every `ml/runs/<run>/MODEL.md` model card (label, forecast, summary, purpose, result) from
  `ml/scripts_model_cards.py`; the dashboard shows runs as "R1c · <dir>", the card in an ⓘ, and a models table in
  Compare runs. ETA: R7 ~10:00 PT, R7f ~11:00 PT.
- 2026-10-29 10:30: intraday design questions answered in ml/MODEL-STRATEGY.md Q31–Q34 (handoff from the base run,
  even steps, volume/price regime features, 1-minute grid with second-level summaries); proposals linked from
  [intraday-model.md](intraday-model.md) §5. Docs only, nothing built.
- 2026-10-29 11:40: overnight queue finished 10:52 PT; nothing running or queued. R7 (HGB, night) done: 1.3% better
  than R1c (CI clear), the night leader. R7f (HGB + full pre-market) done: ties R6f at 09:45 (CI spans 0) and leads on
  fewer models. [Model performance](#model-performance), Scoreboard, LEARNINGS `tag-ml-daily`, GUIDE §8 updated.
  Every trainer (`tft-vwap`, `tft-vwap-gbm`, both overlays, `tft-intraday`, `screener-train`) now writes the run's
  `MODEL.md` itself (`ml/model_card/`, `--label/--about/--purpose`); the dashboard's Intraday run picker shows labels
  and an ⓘ. Progress bars off in training logs (`enable_progress_bar=False`).
- 2026-10-29 12:10: local cleanup, ~77 GB freed: the training logs `logs/{r0-resume,r1,r1-variants,r4-queue,overnight-queue}.log`
  (67 GB, nearly all progress bars) replaced by `logs/*.digest.txt` (timestamps, comparisons, errors); the merged
  `.claude/worktrees` and their branches (9.6 GB); the 10-04/05 smoke and first runs and two run logs (on the NAS).
  `logs/` is gitignored. Repo work merged to `main` and pushed (remote holds `main` only).
- 2026-10-29 14:30: direction set by the user: one model for 09:30 and after, with full
  pre-market; the night forecast is dropped; quantile calibration ranks first. New
  `tft-vwap-recal` (walk-forward recalibration; R7f mean |q err| 0.78 → 0.10). `tft-vwap-gbm` gains
  `--premarket optional`, `--open`, `--vol` and `--tail-iter`. R5 queued (`ml/runs/queue-r5.sh`, 14:10 PT):
  R5 → recal → calibration tables → R8 → NAS backup. [Model performance](#model-performance),
  MODEL-STRATEGY Q35.
- 2026-10-29 18:45: R5 done (`ml/runs/20261029-211059Z-gbm-4438sym`, + `~px`; queue ended 18:26 PT,
  NAS backup done): 3.5% better than R7f-cal at 09:45, 4.4% better than R6p-cal at 09:30. New
  `tft-vwap-recal --sessions N` (rolling recal on the N sessions before each day); R5-cal20s and
  R5~px-cal20s written: mean |q err| 0.18 pts, excellent, good or excellent in 13 of 14 folds.
  R8 screener kept. Proposal: R5 + rolling 20-session recal as the single 09:30+ model; the
  production switch waits for the user. [Model performance](#model-performance), MODEL-STRATEGY Q36.
