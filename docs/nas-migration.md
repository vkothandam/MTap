# Moving the sourcing from the iMac to the NAS

Status on 2026-10-24 (evening):

- **Trial on the NAS:** steps 0–9 passed (runbook), including a password-only login with
  no code. Results in [LEARNINGS.md](LEARNINGS.md) (`tag-sourcing-host`).
- **Done:** the three weekday DSM tasks (Mon–Fri 06:10, 06:40, 13:15, checked in their
  task files); `massive1s` unticked.
- **Still open before cutover:** the cutover checklist in the runbook (stop the iMac,
  `push.sh` without `--trial`, queue `daily`).
- **Then:** cutover on Sunday 10-25. The NAS polls from the open on Mon 10-26.
- **Runbook:** [deploy/synology/README.md](../deploy/synology/README.md).
- **The generic container target:** [deploy/container/README.md](../deploy/container/README.md).

## Why

The iMac runs all the sourcing under launchd: the E*TRADE feed with Redis and the store, the
Massive per-second WebSocket, the daily Massive REST fetches and the evening compact. Every
output has to reach the NAS over an SMB mount that the iMac re-establishes after each wake.

**What happened on 2026-10-21.** At first it looked as if the iMac had produced nothing that
day. The NAS showed otherwise later:

- **E*TRADE feed:** the iMac collected the whole day. Its hourly files start at 13:21Z
  (06:21 PT), but all of h09–h14 reached the NAS together at 12:28 PT.
- **Massive:** the per-second day file is full size (224 MB, a normal day).
- **Daily fetch:** the 06:40 job's file for 10-20 was also written at 12:28.

So nothing was lost. Collection worked; the uploads to the share were blocked for about six
hours, and the store and the Massive collector caught up once the share came back, as
designed. The iMac's local logs (not reachable from this Mac) would say why the share was
unavailable.

**What moving to the NAS fixes.**

- **Storage:** the data is on the NAS's own disk, so there is no SMB mount, no remount, and
  no upload over the LAN.
- **Mac-specific moving parts are gone:** no wake/sleep, no caffeinate, no Keychain-backed
  mount, no logged-in GUI session.
- **One always-on box:** it already runs `massive1s` from DSM Task Scheduler and Docker.

**The iMac's jobs today** (launchd, PT, weekdays):

| time | job | what it does |
| --- | --- | --- |
| always | `com.mtap.redis`, `com.mtap.store` | Redis (`scripts/schedule/redis-mtap.conf`); `sourcing-py intraday store --follow` |
| 06:15 | pmset | wakes the Mac |
| 06:16 | `com.mtap.collector.startup` | `mtap-schedule.sh start`: mount the NAS, start massive-ws (`MTAP_MASSIVE_WS=1`), validate, `storage check`, kick Redis and the store, supervise `serve etrade` |
| 06:40 | `com.mtap.sourcing.daily` | `daily-sourcing.sh`, capture only (`MTAP_DAILY_CAPTURE_ONLY=1`) |
| 13:15 | `com.mtap.collector.shutdown` | `mtap-schedule.sh sleep`: stop, flush the store, the day's summary, `daily-sourcing.sh --fill-only`, wait ≤20 min for massive-ws, its status and compact, `pmset sleepnow` |

**The same jobs on the NAS** (DSM Task Scheduler, weekdays):

| time | DSM task | jobs.sh |
| --- | --- | --- |
| 06:10 | `mtap-start` | `start`: redis, store and etrade (role etrade); massive-ws (role massive-ws) |
| 06:40 | `mtap-daily` | `daily`: `bin/daily-capture.sh`, the capture branch of daily-sourcing.sh |
| 13:15 | `mtap-evening` | `evening`: the sleep job without the sleep |

## Decisions

- **Keep the Python and Node code; run it in containers.**
  - The work is network-bound: the live WebSocket averages about 470 bars/s, and the daily
    REST jobs make tens of calls.
  - The evening compact takes about 1 min and up to 4 GB on the iMac. On the NAS expect
    2–4 min; it has 31 GB of RAM.
  - The one CPU-heavy job, the one-off 5-year extended-hours backfill, might run 1.5–3×
    slower on the NAS. That is an estimate, not a measurement, and it is fine for a job that
    runs once.
- **Rust: only for a hot spot that a measurement finds** (for example, compact).
  - Porting the Massive Python would be about 3–4k lines of Rust plus the ~40 tests. Most of
    it would be new code: the WebSocket client, crash-safe segment rolling, compact.
  - Rust has no exchange calendar, so Python would still have to supply one.
  - `massive1s` paid off because it moved billions of bars. The daily jobs wait on the
    network.
- **Bun: no.** `node/` has no Massive code to reuse, so it would be the same full rewrite. JS
  has no good Parquet writer (`node/` has it stubbed out), and the DuckDB, pyarrow and
  calendar pieces would all have to be rebuilt.
- **The E*TRADE feed moves too.** The user asked for the whole migration, and MBin is not
  running, so there is no external Redis consumer to keep on the iMac.
- **Selenium Standalone Chrome as the desktop for the collector.** It provides Xvfb, noVNC
  and supervisord, and its Google Chrome is the collector's browser (see below).

## The rule: one core, thin host targets

```
python/ node/ schemas/ shared/ nas/massive1s   the core: host-agnostic, same code everywhere
deploy/container/                             generic target: any Docker host (NAS now, a cloud VM later)
deploy/synology/                              the NAS's wrapper: host.env, task.sh (DSM), push.sh (Mac)
scripts/schedule/ scripts/storage/            the macOS (iMac) target: launchd, pmset, SMB mount
```

**Rules that keep the iMac working:**

1. **The macOS scripts stay where they are.** `scripts/schedule` and `scripts/storage` are
   not moved or edited: the iMac's installed launchd plists point at those paths.
2. **Core changes are additive and keep their defaults.** This migration made three:
   - **`OTP_BIND_HOST`** ([constants.ts](../node/src/services/etrade/constants.ts)): the
     address the OTP page listens on. It defaults to `OTP_HOST`, so the iMac is unchanged.
     In a container it is `0.0.0.0`, while the texted link keeps the host's LAN address. A
     unit test covers both cases.
   - **The torch CPU wheel on Linux only** (`[tool.uv.sources]` in
     [pyproject.toml](../python/pyproject.toml), with a `sys_platform == 'linux'` marker).
     This keeps mtap-py at 2.9 GB on the NAS instead of several GB more of CUDA. The macOS
     resolution in uv.lock is unchanged (diffed): torch 2.13.0 on the Mac, 2.14.1+cpu in
     the containers.
   - **The browser identity when headed** ([Browser.ts](../node/src/services/etrade/Browser.ts)).
     Headless (the iMac) is unchanged: the user agent is overridden to drop "Headless".
     Headed (the NAS) keeps Chrome's own user agent and client hints: no override, and no
     stealth `user-agent-override` evasion. See "E*TRADE on the NAS".
3. **Host differences are environment only.** The core already allowed this:
   - caffeinate runs only on macOS;
   - `mount-nas.sh` runs only when `MTAP_NAS_URL` is set;
   - the upload is a copy plus a row-count check, which also works when both roots are on
     one disk;
   - the roots, `REDIS_URL`, `HEADLESS`, `CHROME_PROFILE_DIR` and `PUPPETEER_EXECUTABLE_PATH`
     all come from env.
4. **Fixing a macOS assumption.** If the trial finds another one, fix it the same way: a new
   env setting whose default is today's behaviour.
5. **The Massive key and the E*TRADE login run on one machine at a time.** The key allows
   one WebSocket connection, and a second E*TRADE session would interfere with the first.
   Cutover and rollback always stop one side before starting the other.

## What the E*TRADE feed captures

The collector logs in to the E*TRADE website with puppeteer:

- It uses the stealth plugin and a persistent Chrome profile.
- It intercepts the watchlist page's requests to pick up their auth headers.
- It then calls the watchlist JSON API directly.

**The data:**

- **Watchlists:** the main watchlist every ~2 s and the midcap list every ~20 s.
- **Fields:** price, cumulative volume, day open, previous close, bid/ask and their sizes,
  and 10-day volume.
- **Quotes:** real-time, unlike Massive's 15-minute delay.
- **Volume:** about 57 MB a day, in hourly parquet files under
  `intraday_feed/date=D/source=etrade_watchlist/`. Columns:
  [schemas/intraday_feed.md](../schemas/intraday_feed.md).

**The path:** Node publishes to Redis (`mtap:evt:INTRADAY_FEED`). The store reads the stream
into hourly segments in STATE and uploads each closed hour to DATA.

## E*TRADE on the NAS

- **Display:** the collector runs headed (`HEADLESS=false`) on the container's Xvfb display.
  noVNC shows it at `http://192.168.1.100:7900`, behind `SE_VNC_PASSWORD`.
- **The browser:** the selenium image's Google Chrome 155.0.8059.79
  (`PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome` in compose.yml), driven by puppeteer
  22.15. Puppeteer's Chrome for Testing 127 is still installed but is not used: E*TRADE
  refused it (next bullets). The selenium server does not run.
  - The selenium image's `selenium.conf` is replaced: its selenium-standalone program shuts
    supervisord down when it exits.
  - Its browser-cleanup program, which kills long-lived Chrome processes, is removed.
- **Restart loop:** [run-etrade.sh](../deploy/container/collector/run-etrade.sh) has the
  iMac's backoff. It waits 15 s, doubles up to 10 min, and goes back to 15 s after a run
  longer than 10 min, so a failing login is not retried in a tight loop.
- **The first login needs a code.** Either:
  - enter it in the E*TRADE page through noVNC and tick "save device"; or
  - use the texted link to the collector's OTP page, `http://192.168.1.100:8787/auth/otp`.
    `Authentication.ts` ticks `#saveDevice` itself when the code comes through that page.
- **After that:** the profile (`MTap-State/etrade/chrome-profile`) holds the device trust,
  so later logins need only the password. The trial checks this with a restart.
- **The iMac's profile cannot be reused.** Chrome encrypts profile cookies with an
  OS-specific key, so Linux Chrome cannot read a Mac profile.
- **The trust can lapse.** E*TRADE may drop it from time to time; then it is one more code,
  by noVNC or SMS.
- **E*TRADE sees a new device.** The browser reports Linux (`X11; Linux x86_64`).
- **Bot detection (2026-10-24 trial).** E*TRADE runs Akamai Bot Manager (the `_abck` and
  `bm_sz` cookies). About ten logins with Chrome for Testing 127 stayed on the login page
  with "We're unable to process your request". A login typed by hand in the same noVNC
  window failed too: Akamai scores the browser, not just the typing. A probe of what the
  pages can read (no E*TRADE contact) found:

  | | iMac (headless, macOS) | NAS before | NAS now (Chrome 155, headed) |
  | --- | --- | --- | --- |
  | client hints (`navigator.userAgentData`) | empty | empty | Google Chrome 155, Linux |
  | window: outer vs inner 1280×800 | 1200×955 | 945×1060 (outer smaller than inner) | 1280×885 |
  | WebGL | none (`--disable-gpu`) | none | none |

  - The empty client hints came from `page.setUserAgent`, which drops them.
  - Skipping it alone was worse: the stealth plugin then claimed Windows on a Linux browser.
  - The iMac logs in with the same tells, so its aged profile and macOS probably carry it.
    The fix gives the NAS a consistent real-browser identity instead.
- **One collector at a time:** a NAS login can end the iMac's E*TRADE session. After
  cutover the iMac does not run, and a rollback re-logs in on the iMac.

## The NAS

| fact | value |
| --- | --- |
| model and OS | Synology DS925+, DSM 7.3.2 (86009) |
| CPU, RAM, disk | AMD Ryzen V1500B (8 threads), 31 GB (27 free), 7.5 TB free on volume2 |
| Docker | Container Manager: Docker 24.0.2, Compose v2.20. The Dockerfiles avoid BuildKit-only features |
| Docker access | the socket is root-only, so jobs run from DSM tasks as root. ssh as `vijayan.synology` (1026:100) cannot run docker; the Mac reads `MTap-State/logs/status.txt` and the logs instead |
| tools | GNU find, stat, tar and flock; no `rg` |
| copying code | rsync over ssh is refused unless DSM's rsync service is on, so push.sh streams a tar over ssh into `MTap-App.new` and swaps it in |
| power | the NAS starts at ~06:00 every day (06:01 on 10-24). The power schedule must keep it on until 13:45 on weekdays |
| `massive1s` task | a Boot-up task that fires at every boot. Its task.log shows it failing at 06:01 on 10-20 and 10-21 because Docker was not up yet. Untick Enabled |
| share | `/volume2/Backups` (`/Volumes/Backups` on a Mac). It uses ACLs; files written by the containers belong to 1026:100, as SMB writes do |

**Paths:**

| what | path |
| --- | --- |
| code | `/volume2/Backups/MTap-App` |
| state | `/volume2/Backups/MTap-State` |
| data | `/volume2/Backups/MTap-Data`, and `MTap-Trial` for the trial (with its own `.mtap-data-root` and a copy of `out/tickers_cs`) |
| secrets | `/var/services/homes/vijayan.synology/.mtap.env`, chmod 600 |

**Secrets file contents:**

- `MASSIVE_API_KEY`
- the `ET_*` and `SMS_*` lines
- `SE_VNC_PASSWORD`

It is never in the repo or on the share. compose reads it, so a value with `$` goes in single
quotes.

## Versions (container vs Mac)

| | Mac / iMac | containers |
| --- | --- | --- |
| Python | 3.11.15 (`python/.venv`) | 3.11.17 (python:3.11-slim-bookworm), uv 0.11.7 |
| torch | 2.13.0 | 2.14.1+cpu |
| Node | 20+ | 20 (node:20-bookworm-slim's binary) |
| Chrome | puppeteer's Chrome for Testing 127.0.6533.88, headless | Google Chrome 155.0.8059.79, headed |
| Redis | Homebrew, `redis-mtap.conf` | redis:7-alpine, the same settings on the compose network |
| image sizes | | on the NAS (`docker image ls`): mtap-py 2.92 GB, mtap-collector 3.02 GB |

## Trial, cutover, rollback

These steps are in the runbook, [deploy/synology/README.md](../deploy/synology/README.md):

- **Trial:**
  - one-time setup, then build;
  - REST `daily` into MTap-Trial, diffed against MTap-Data;
  - the E*TRADE login with a restart that must not ask for a code;
  - `ws-test`, `evening`, `status`.
- **Cutover:** `setup-schedule.sh stop` on the iMac, `push.sh` without `--trial`, and enable
  the three weekday DSM tasks.
- **Rollback:** NAS `stop` first, then `setup-schedule.sh prod --with-daily` and
  `mtap-schedule.sh start` on the iMac.
- **Monday checks:**
  - containers up and the login done by 06:10;
  - the 5-min Massive files from 06:30;
  - the hourly E*TRADE files from ~07:00;
  - the daily files at 06:40;
  - compact and summary in the evening;
  - then a comparison with a typical iMac day.

The trial's results go into [LEARNINGS.md](LEARNINGS.md).

## Risks and things to check

- **Massive request rate.** The containers have no `settings.toml` (it holds the key and is
  never copied), so host.env sets `MASSIVE_RATE_LIMIT_PER_MIN=0` (no cap, the paid plan).
  This Mac's `settings.toml` still says 5. Check the iMac's value before cutover; if it is
  not 0, set host.env to match.
- **The trial login may end the iMac's E*TRADE session.** The iMac re-logs in at 06:16 on
  Monday only after a rollback; that login may then ask for a code.
- **Segments before cutover.** `MTap-State/segments` and `segments_massive` must be empty,
  or the trial's rows upload into MTap-Data.
- **Secrets quoting.** compose expands `$` in unquoted and double-quoted values; a password
  with `$`, `#` or spaces must be in single quotes, or the login fails.
- **The NAS power schedule.** A shutdown before the evening job ends loses the compact and
  the summary for that day (the raw files are already on disk).
- **The first NAS build** pulls about 2 GB of images and takes several minutes; do it during
  the trial, not on Monday morning.

## Running without the iMac

After cutover nothing the NAS runs needs the iMac:

| need | where it is on the NAS |
| --- | --- |
| code | `MTap-App`, copied by push.sh from any Mac with the repo |
| images | built on the NAS (`build`) |
| secrets | `~/.mtap.env` |
| E*TRADE device trust | `MTap-State/etrade/chrome-profile` |
| schedule | DSM tasks; power on 06:00, off 22:10 |
| data and state | `MTap-Data`, `MTap-State` on volume2 |
| Redis | a container, memory only |

**What still needs a person or another machine:**

- **An E*TRADE code** when the device trust lapses: noVNC or the OTP page from any device
  on the LAN.
- **Code changes:** push.sh from a Mac, then `build`.
- **The DuckDB load** (`ingest-eod` → `derive-features` → `export-tft`) already runs on
  the Mac that owns the DuckDB. It reads `MTap-Data` and never ran on the iMac.

**Not proven by the weekend trial** (no market): the volume of a full trading day, a
massive-ws window from 09:00 to 16:21 ET, and the evening compact on real data. The Monday
checks in the runbook cover them.

## Decommissioning the iMac

**Stage 1, at cutover (Sunday 10-25): stop it, keep it ready.** These are the runbook's
cutover step 1:

- `setup-schedule.sh stop`: no agents and no wake schedule.
- The code, `.env` (with the new E*TRADE password) and Chrome profile stay, so a rollback
  is two commands.

**Stage 2, after a stable week on the NAS: retire it.** First decide whether the iMac stays
as the fallback (docs/TODO.md). If it goes, on the iMac:

1. **Check:** `scripts/schedule/setup-schedule.sh status` shows no `com.mtap.*` agents and
   no power schedule.
2. **Secrets:** delete the repo's `.env`. It holds the Massive key, the E*TRADE login and the
   SMS account.
3. **E*TRADE session:** delete `<MTAP_STATE_ROOT>/etrade/`. It holds the Chrome profile
   with a live session, and `session.json`.
4. **NAS password:** delete the SMB Keychain entry that `mount-nas.sh --save-password`
   wrote:
   ```bash
   security delete-internet-password -s 192.168.1.100 -a <nas user> -r "smb "
   ```
5. **Homebrew Redis:** `brew services stop redis` if it runs (setup-schedule.sh uses it in
   place of `com.mtap.redis` when present). Optionally `brew uninstall redis`.
6. **The repo clone:** delete it, or keep it without `.env` for reference.
7. **MBin:** its `com.trading.*` agents are not MTap's. If `--disable-mbin` moved them,
   they are in `~/Library/LaunchAgents/disabled-mbin/`.
8. **E*TRADE:** optionally remove the iMac from the recognised devices (Security
   settings).
9. **Docs:**
   - mark `scripts/schedule` as retired, or as the fallback, in
     [STATUS.md](STATUS.md) and [scripts/README.md](../scripts/README.md);
   - tick the TODO item.

There is nothing to rotate as long as the iMac is wiped or stays at home. Rotate the
Massive key and the E*TRADE password if it leaves the house.

## Later: a cloud VM

The containers and jobs.sh carry over unchanged. A VM needs:

- **Disk:** a persistent disk (a POSIX filesystem) for DATA and STATE, not a bucket mounted
  with gcsfuse. The store and the WebSocket collector append, fsync and rename files, which
  gcsfuse does not support. Copy finished files to a bucket afterwards if wanted.
- **Secrets:** Secret Manager, written to a chmod-600 file at boot.
- **Access to noVNC and the OTP page:** Tailscale or IAP. Never open them to the internet.
  `OTP_HOST` is then the address the phone can reach.
- **Scheduling:** the VM's own scheduler (cron or systemd timers) for start, daily and
  evening; the VM's zone or `TZ` sets the times.
- **E*TRADE:** expect more login challenges, since a datacenter IP looks less like a home
  browser.
- **One key, one session:** stop the NAS before the VM starts.
