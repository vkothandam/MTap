# MTap status and reference

The one page that says what runs where, when, on what data, with which secrets and limits.
The details are in the linked docs.

**Last updated: 2026-10-25.**

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
| 13:15 | `mtap-evening` → `evening` | stop the collector, flush, intraday summary, daily fill, wait for massive-ws, compact; done by about 13:45 |
| 22:10 | power off | |
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
| DuckDB backups | `backup/etrade/` | | `daily-sourcing.sh --ingest-only` on this Mac |

**Elsewhere:**

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

## Docs index

| doc | what |
| --- | --- |
| [nas-migration.md](nas-migration.md) | why and how the sourcing moved to the NAS; E*TRADE on the NAS; decommissioning the iMac; a cloud VM later |
| [deploy/synology/README.md](../deploy/synology/README.md) | NAS runbook: one-time setup, trial, cutover, rollback, Monday checks |
| [LEARNINGS.md](LEARNINGS.md) | what the data checks found, by tagged topic |
| [TODO.md](TODO.md) | deferred work |
| [database-schema.md](database-schema.md) | DuckDB ER diagrams (generated; CI checks them) |
| [stock-prediction-plan.md](stock-prediction-plan.md), [next-session-parquet-tft.md](next-session-parquet-tft.md) | ML plans |

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
