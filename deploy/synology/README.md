# deploy/synology: MTap's sourcing on the NAS

Runbook for the Synology DS925+ at 192.168.1.100. The containers and jobs are generic
([deploy/container](../container/README.md)); this folder only adds what the NAS needs:

| file | what it is |
| --- | --- |
| `host.env` | the NAS's paths, owner (1026:100), roles, `OTP_HOST`, `TZ`; no secrets |
| `task.sh JOB` | what DSM Task Scheduler runs as root. It waits for Docker, reads host.env and appends to `MTap-State/logs/jobs.log` |
| `push.sh` | on the Mac: copies the code to `MTap-App` over ssh. `--trial` points the next jobs at `MTap-Trial`; `status` prints `status.txt` |

**NAS paths:**

| what | path |
| --- | --- |
| code | `/volume2/Backups/MTap-App`, written only by push.sh |
| state | `/volume2/Backups/MTap-State`: segments, heartbeats, the Chrome profile, logs |
| data | `/volume2/Backups/MTap-Data` (production); `/volume2/Backups/MTap-Trial` (the trial) |
| secrets | `/var/services/homes/vijayan.synology/.mtap.env`, chmod 600 |

The Mac never opens `MTap-State` over SMB, and no DuckDB is used here.

## One-time setup

1. **Secrets file.** Copy the values from the iMac's `.env`:
   ```bash
   ssh vijayan.synology@192.168.1.100
   umask 077
   cp ~/.massive1s.env ~/.mtap.env          # MASSIVE_API_KEY, the same key
   vi ~/.mtap.env                            # add the lines below
   chmod 600 ~/.mtap.env
   ```
   - **Lines to add:**
     - `ET_USERNAME`, `ET_PASSWORD`
     - `SMS_EMAIL`, `SMS_EMAIL_PASSWORD`, `SMS_PHONE_NUMBER`, `SMS_CARRIER`
     - `OTP_PHONE_PRIORITY`, if the iMac sets it
     - `SE_VNC_PASSWORD=<a password for the noVNC page>`
   - **Quoting:** put a value with `$`, `#` or spaces in single quotes.
   - **Missing VNC password:** if `SE_VNC_PASSWORD` is not set, the noVNC password is the
     image default, `secret`.
2. **DSM tasks.** Control Panel > Task Scheduler > Create.
   - **Settings for every task:**
     - user **root**
     - Run command: `bash /volume2/Backups/MTap-App/deploy/synology/task.sh <JOB>`
   - **Manual tasks:** Triggered Task > User-defined script, event Boot-up, **Enabled
     unticked**, so they run only when you press Run (the same as `massive1s`):
     - `mtap-build`: `build`
     - `mtap-status`: `status`
     - `mtap-stop`: `stop`
   - **The queue (recommended for the trial):** Scheduled Task > User-defined script.
     - Daily, first run 00:00, repeat every 5 minutes, last run 23:55
     - Name `mtap-queue`, command `task.sh queue`
     - It runs the job named in `MTap-State/run-next`, then deletes the file. An empty
       queue does nothing.
     - The Mac (Claude) can queue a step over ssh, so you don't have to press Run for
       each one:
       ```bash
       ssh vijayan.synology@192.168.1.100 'echo start > /volume2/Backups/MTap-State/run-next'
       ```
     - Only build, start, daily, evening, stop, status, ws-test, selftest, morning, night and
       inputs-backfill are accepted, with no arguments.
   - **The weekday schedule:** Scheduled Task, Mon–Fri. Create these now with **Enabled
     unticked**, and tick Enabled at cutover:
     - `mtap-start` 06:10: `start`
     - `mtap-daily` 06:40: `daily`
     - `mtap-evening` 13:15: `evening`
     - `mtap-morning` 06:14: `morning` (09:14 ET; the DSM times are PT). Runs
       forecast-night if no host has a done claim, waits for 09:29:30 ET and runs the
       pre-check, then prices at 09:30 and the open until 09:32, export-next, `predict --state px`
       and `screener-score`; at 09:45 the Massive full pre-market, export-next, `predict
       --state full`, `screener-score`. Every step is time-boxed. It takes NO jobs lock (it
       waits on the clock) and `MTap-State/morning.pid` keeps a second run out. The E*TRADE
       part only reads `MTap-State/etrade/bearer.json`, never logs in and stops on the first
       401/403/429. Output: `MTap-Data/out/premarket_live/` and `MTap-State/logs/morning-<date>.log`.
     - `mtap-night` 17:25: `night` (20:25 ET): `forecast-night` for the next session. The claim
       file `MTap-State/forecast/claim-<date>.json` makes it run once across hosts (a claim
       with a heartbeat under 30 min old is respected; a stale one is taken over).
     - **Setup the morning and night jobs need (S1):**
       - A container-local DuckDB copy (never on SMB): the jobs read/write the DuckDB from a
         local folder in the job container, refreshed from the NAS Parquet by the daily job
         (not automated yet; copy it in as a setup step).
       - `premarket xids` (run by the night job) writes `out/premarket_live/xids.csv` so the
         NAS needs no DuckDB for ChartIQ ids.
       - The ML project and the trained model must be reachable from the job container. Set
         `MTAP_PREDICT_CMD` (default `uv run --project ml tft-vwap-predict`) and
         `MTAP_FORECAST_RUN_DIR` (the run directory) in the env file, and for the screener
         `MTAP_SCREENER_CMD` (default `uv run --project ml screener-score`) and
         `MTAP_SCREENER_MODEL` (`runs/screener/<id>`). The 09:30 pass reads
         `<run>~px/forecast_D_subset.parquet`, the 09:45 pass `<run>/forecast_D.parquet`.
         The image does not yet carry ml/; S7 decides whether these run on the NAS or the Mac.
   - On the **Schedule** tab, set "Run on the following days" to Mon–Fri and "First run
     time" to the time above. DSM's defaults are every day at 00:00, when the NAS is off. On
     2026-10-24 the first three tasks kept those defaults. To check, read
     `/usr/syno/etc/synoschedule.d/root/<id>.task`: `run hour`, `run min`, and
     `week=0111110`, where the first digit is Sunday.
   - **Alerts by email:** on `mtap-queue`, `mtap-start`, `mtap-daily` and `mtap-evening`,
     Settings tab: tick "Send run details by email" (your address) and "Send run details
     only when the script terminates abnormally". This needs Control Panel > Notification >
     Email set up (Gmail) **and enabled**. On 2026-10-25 it was set up but not enabled, so
     the first alert sent nothing; "Send a test email" stays greyed out until it is enabled
     and applied.
   - **Task output (optional):** Task Scheduler > Settings > save output results to a
     folder, so Action > View Result shows each run's output.
     - A failed job mails its `jobs.log` tail.
     - The queue's `watch` mails once when a heartbeat is over 10 min old in market hours,
       and once when all are fresh again. A login waiting for a code shows up this way.
3. **`massive1s` task:** untick Enabled. It is a Boot-up task and fires at every boot; its
   task.log shows it failing at 06:01 on 10-20 and 10-21 because Docker was not up yet.
4. **Power schedule** (Control Panel > Hardware & Power > Power Schedule):
   - the NAS starts at about 06:00; keep that;
   - it must not shut down before 13:45 on weekdays.
5. **Firewall:** if the DSM firewall is on, allow ports 7900 (noVNC) and 8787 (the OTP
   page) from the LAN.

## The 5-year inputs backfill (one-off)

`inputs-backfill` loads the data for the 5-year daily model. Every step is resumable and
idempotent, so after a failure or a stop, queue it again.

1. Put the first date with minute aggs (spike S5) in `host.env`: `MTAP_BACKFILL_FROM=YYYY-MM-DD`.
   To keep minute bars only for the panel's stocks (~25–65 GB instead of up to 145 GB), also set
   `MTAP_BARS_UNIVERSE=/data/out/tft/panel.parquet` (a panel copied to the share); unset, every
   ticker's bars are kept. Check the free space on volume2 first.
2. After `push.sh`, queue `build` once (the new sources need the new image).
3. Queue it **after the 13:15 `evening` job** (all jobs share one lock; the queue skips
   a run while another job holds it and takes the file on a later 5-minute tick):
   ```bash
   ssh vijayan.synology@192.168.1.100 'echo inputs-backfill > /volume2/Backups/MTap-State/run-next'
   ```
4. It copies `out/extended_hours` to `MTap-Data/backup/extended_hours-pre-vwap` (once), runs
   `massive-ext fetch --chunk quarter --allow-failures 50 --save-bars /data/bars_1m`, refreshes
   `tickers_cs`, fetches the quarter-end share snapshots, then the economy series.
   The log is `MTap-State/logs/inputs-backfill-<date>.log`.

## Collector restart for the token file

The collector now also writes the E*TRADE bearer token to `MTap-State/etrade/bearer.json`
(mode 600, atomic) whenever it changes; the `morning` job reads it. After `push.sh` and
`build`, the collector needs **one restart outside market hours** (queue `stop`, then `start`,
before 06:10 PT or after 13:45 PT) to pick the change up. Until then `morning` finds no
token file and prices come from the watchlist only. The token is a secret: it stays in `MTap-State`,
never in the repo or the data share.

## Trial (the weekend before cutover)

There is no market at the weekend, so the trial checks the login, the polling, the
connections and the file flow, not market data.

The iMac is untouched until cutover. The iMac runs its jobs only on weekdays, so the
NAS's Massive connection does not collide with it.

| step | who | what | pass when |
| --- | --- | --- | --- |
| 0 | Claude | `deploy/synology/push.sh --trial` | it prints the VERSION and `override: MTAP_DATA_DIR=…/MTap-Trial MTAP_ROLES=etrade` |
| 1 | you | the one-time setup above | |
| 2 | you or the queue | `build`. The first build takes several minutes | `status.txt` lists `mtap-py` and `mtap-collector`; jobs.log says `images built` |
| 3 | queue | `daily`: fetches the last 10 sessions into MTap-Trial | Claude diffs them against MTap-Data's files for the same dates: row counts plus content |
| 4 | queue, then you | `start`. Open `http://192.168.1.100:7900` and watch the login. Enter the code in the page and tick "save device", or use the texted link | `collector-<date>.log` shows polling; `status.txt` shows fresh collector and store heartbeats |
| 5 | queue | after about 15 min, `stop` then `start` | the login completes with **no code**: the saved device holds |
| 6 | queue | `ws-test` | the log shows the connection and auth succeeding (no bars at the weekend) |
| 7 | queue | `evening` | each trial `intraday_feed/` file has the row count the store logged; no segments left |
| 8 | Claude | `status` and the logs | no unexplained errors |
| 9 | queue, you watching | the next day, `start` then `stop` | a **fresh login** (the session from step 4 has expired) asks for no code |

The trial results go into [docs/LEARNINGS.md](../../docs/LEARNINGS.md) (`tag-sourcing-host`).

**Results on 2026-10-24:** steps 0–9 passed (step 9 ran the same evening).

- **Step 4, the first try:** E*TRADE's bot manager refused puppeteer's Chrome for Testing.
  Google Chrome 155 then passed. Only the code was entered by hand.
- **Step 5:** the restart re-used the saved session (`serving` with no login at all). That
  is why step 9 exists: it checks a login with the password only.
- **Step 9:**
  - At 21:20, `start` reused the saved session again. Then a log-off by hand in the
    window ended it (401s). The collector's recovery failed, and it went idle (see below).
  - After `stop` and `start` at 21:30: `Login successful`, `No phone verification needed`,
    and polling resumed. The profile holds the device trust.
- **Found in step 9:** when the collector's session recovery fails, it stops polling but the
  process stays up, so supervisord does not restart it. The fix is to queue `stop` then
  `start`. The iMac code behaves the same way.
- **Step 3:** the rows match MTap-Data. The differences come from when each was fetched
  (Massive's split adjustments and late corrections).
- **Steps 6 and 7:** `ws-test` subscribed, survived a Massive restart (1012, then 503) and
  re-subscribed. `evening`'s summary counted 144,844 rows, the same as the store's two
  segments.

**If a step fails:**

1. Fix it on the Mac.
2. Run `push.sh --trial`.
3. Queue `build` if code changed.
4. Rerun the step.

**After the trial, before cutover:**

- **Segments:** `MTap-State/segments` and `segments_massive` must be empty. Otherwise
  trial rows would upload to MTap-Data.
- **Folders:** MTap-Trial can be deleted; the Chrome profile in MTap-State stays.

## Cutover (Sunday evening)

**Before you start, check these:**

- [x] Trial step 9 passed (a login with no code), 2026-10-24 21:30.
- [x] `mtap-start`, `mtap-daily` and `mtap-evening` exist (one-time setup, step 2) with
  Mon–Fri and their times, not DSM's defaults of every day at 00:00. Checked on
  2026-10-24.
- [x] `massive1s` is not Enabled (one-time setup, step 3). Only DSM shows this; its
  database is root-only.
- [x] `MTap-State/segments` and `segments_massive` are empty (2026-10-25 12:00).

**Steps:**

0. **NAS:** queue `stop`. Containers left running at the 22:10 power-off come back at boot
   (`restart: unless-stopped`); on 10-25 that gave 70 failed logins at 06:02.
1. **iMac:** run `scripts/schedule/setup-schedule.sh stop` (it asks for sudo, for pmset).
   - **What it does:** it stops the collectors and Massive, removes every `com.mtap.*`
     agent (Redis and the store too), and cancels the pmset wake/sleep schedule.
   - **Check:**
     - `launchctl list | grep com.mtap` prints nothing;
     - `pmset -g sched` shows no repeat;
     - `pgrep -fl 'serve etrade|massive-ws|redis-server'` prints nothing.
   - **What stays, for a rollback:** the iMac's code, `.env` and Chrome profile.
     - Put the **new E*TRADE password** in the iMac's `.env` (changed on 2026-10-24), or a
       rollback cannot log in.
2. **Mac:** run `deploy/synology/push.sh` (no `--trial`). With no override, the jobs use
   MTap-Data and host.env's roles (`etrade massive-ws`).
3. **NAS:** queue `build` if code changed since the trial build. On 10-25 it did: the
   stale-lock removal in `run-etrade.sh` and the retype in `Authentication.typeInto`.
4. **NAS:** tick Enabled on `mtap-start`, `mtap-daily` and `mtap-evening`. Leave
   `mtap-queue` on: it is how the Mac runs `status` or a one-off job.
5. **NAS:** queue `daily` once. It fills any session the iMac missed (10-23 on 10-25).
6. **Monday 06:10:** be ready to open noVNC.
   - The secrets file has no `SMS_EMAIL`, so a code request sends no texted link: only
     E*TRADE's own text arrives.
   - Add `SMS_EMAIL` and `SMS_EMAIL_PASSWORD` to `~/.mtap.env` to get the link.

**Done on 2026-10-25:** steps 0, 2, 3 and 5 (on the NAS); `selftest`, `ws-test` and `status`
passed, and `daily` wrote 10-23. The weekday tasks were already enabled (step 4). Open:
step 1, on the iMac.

- `push.sh` without `--trial` had never run before: under `set -e` it exited silently before
  copying, so the first production `daily` still read MTap-Trial. Fixed; check `VERSION` and
  that `host.override.env` is gone after a push.

Retiring the iMac for good, after a stable week: see
[docs/nas-migration.md](../../docs/nas-migration.md#decommissioning-the-imac).

## Rollback (any day)

1. **NAS:** run `stop`, and untick Enabled on the three weekday tasks. Only one Massive
   connection and one E*TRADE session may run.
2. **iMac:** run `scripts/schedule/setup-schedule.sh prod --with-daily`, then
   `scripts/schedule/mtap-schedule.sh start`.

## Monday checks (Claude, read-only over ssh)

| time (PT) | check |
| --- | --- |
| 06:10 | containers up; the login done without a code (noVNC if it asks) |
| from 06:30 | `MTap-Data/agg_1s/date=<today>/source=massive_ws/` 5-min files; from ~07:00 `intraday_feed/` hourly files |
| 06:40 | `out/daily_summary` and `out/extended_hours` for the previous session |
| evening | `day-<today>.parquet` compacted; the summary in `daily-summary.log`; no late error in `status.txt` |

Then compare against a typical iMac day: rows per hour, bars per hour, disconnects.

```bash
deploy/synology/push.sh status        # status.txt
ssh vijayan.synology@192.168.1.100 'tail -50 /volume2/Backups/MTap-State/logs/jobs.log'
```

## Notes

- **Docker access:** only root can use Docker on DSM, and ssh as `vijayan.synology` has no
  sudo without a password. `status.txt` and the logs are therefore how the Mac sees the
  containers.
- **The code folder runs as root:** DSM runs `MTap-App` as root (as it does `massive1s`'s
  run.sh). Anyone who can write the `Backups` share can change what root runs.
- **Owners:**
  - the py containers run as 1026:100;
  - the collector image's user is remapped to 1026:100 at build time;
  - so every file in MTap-State and MTap-Data has the NAS user as owner.
- **Software:** DSM 7.3 ships Docker 24.0.2 and Compose v2.20. The Dockerfiles avoid
  BuildKit-only features.
