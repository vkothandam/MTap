# scripts/: setup, schedule and storage (sourcing only)

| script | what it does |
| --- | --- |
| `setup/bootstrap.sh` | new checkout: checks tools, `uv sync`, `npm ci`, puppeteer Chrome, creates `.env` / `settings.toml` from the examples, mounts the NAS, `storage init` |
| `schedule/setup-schedule.sh` | installs the LaunchAgents and the wake/sleep schedule (`services`, `test`, `prod`, `massive`, `status`, `stop`) |
| `schedule/mtap-schedule.sh` | what the agents run: `start`, `stop`, `sleep`, `status`, `validate`, `run-store`, `run-massive` |
| `schedule/daily-sourcing.sh` | the daily chain: Massive bars for every missed session, then `ingest-eod` … `export-tft` (only with `prod --with-daily`; `MTAP_DAILY_CAPTURE_ONLY=1` stops after the Massive bars) |
| `schedule/redis-mtap.conf` | Redis for the feed: localhost only, no persistence, 2 GB cap |
| `storage/mount-nas.sh` | mounts `MTAP_NAS_URL` using the password in the login Keychain |

Everything reads the repo-root `.env`. Paths come from the scripts' location, so the repo
can live anywhere. ML jobs are never started from here.

## A weekday under `prod`

All times are the Mac's local time, which must be **Pacific**.

| time (PT) | what happens |
| --- | --- |
| always | `com.mtap.redis` and `com.mtap.store` (the storage consumer) are KeepAlive agents |
| 06:15 | `pmset` wakes the Mac |
| 06:16 | `com.mtap.collector.startup` runs `mtap-schedule.sh start`. It mounts the NAS, runs `storage check`, makes sure Redis and the store are up, then starts the collector (`serve etrade`) under a restart loop holding `caffeinate` |
| 06:30 – 13:00 | the collector polls both watchlists. The store writes every row to an hourly segment under `MTAP_STATE_ROOT/segments/` and uploads each closed hour to `MTAP_DATA_ROOT/intraday_feed/` |
| 06:40 | (`--with-daily` only) `daily-sourcing.sh` for the previous session |
| 13:15 | `com.mtap.collector.shutdown` runs `mtap-schedule.sh sleep`. It stops the collector, flushes the store to the NAS, appends the day's summary to `logs/daily-summary.log`, then puts the Mac to sleep |
| 13:30 | `pmset` sleeps the Mac even if the shutdown job failed |

**Logs**, all under `MTAP_STATE_ROOT/logs/`, kept 14 days:
- `schedule.log`: start, stop and sleep;
- `collector-<date>.log`: the Node collector;
- `launchd-<label>.log`: each agent's own output;
- `daily-summary.log`: rows and gaps per watchlist, one block per day.

**Heartbeats:**
- `MTAP_STATE_ROOT/etrade/collector.heartbeat`: the last poll;
- `MTAP_STATE_ROOT/intraday/store.heartbeat`: rows, open segments, Redis lag, pending uploads.

## Checks

```bash
./scripts/schedule/setup-schedule.sh status   # power schedule, agents, collector, Redis, store, recent log
cd python && uv run sourcing-py intraday status
cd python && uv run sourcing-py intraday summary --date 2026-10-14   # rows per watchlist, gaps > 5 min
cd python && uv run sourcing-py storage check
redis-cli XLEN mtap:evt:INTRADAY_FEED          # stays around 4000 (trimmed, ~2 h); the store acks within seconds
```

If an E*TRADE login needs a one-time code, the collector texts a link to the OTP page on
`OTP_HOST:OTP_PORT`. Set `OTP_HOST` to the Mac's LAN IP so the link opens on a phone.

## Moving to a new Mac (or an iMac that ran MBin's schedule)

1. **Prerequisites:**
   - `brew install node redis` and the uv installer (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
   - Set the time zone to Pacific.
   - The user must stay logged in: the agents run in the GUI session. Let the Mac sleep; don't log out or shut down.
2. **Code:** `git clone` (or `git pull`), then run `./scripts/setup/bootstrap.sh`. It creates `.env` and `settings.toml`.
3. **Fill in `.env`:**
   - `ET_USERNAME`, `ET_PASSWORD`, the SMS settings, and `OTP_HOST` (the LAN IP);
   - `MTAP_STATE_ROOT=~/mtap-state`;
   - `MTAP_DATA_ROOT=/Volumes/<share>/mtap` and `MTAP_NAS_URL=smb://<user>@<nas>/<share>`.
4. **NAS password:** connect once in Finder (Go > Connect to Server, tick "Remember this password"). Then rerun `bootstrap.sh`, which mounts the share and runs `storage init`.
5. **State from the old machine** (optional), copied into `MTAP_STATE_ROOT`:
   - `etrade/session.json`, to skip the first OTP;
   - `etrade/fundamentals.duckdb`;
   - `news/news.duckdb`.
6. **MBin must not run alongside.** It wakes the Mac and scrapes the same E*TRADE session:
   ```bash
   launchctl list | grep -Ei 'trading|mbin'     # com.trading.* agents
   ls ~/Library/LaunchAgents/                    # com.trading.*.plist
   pmset -g sched                                # MBin's repeating wake/sleep
   crontab -l                                    # any cron jobs
   pgrep -fl 'MBin|server.js'                    # a running MBin server
   ```
   - Stop any running MBin server.
   - Set `FEATURE_SCRAPING=false` in MBin's own env.
   - `setup-schedule.sh prod --disable-mbin` unloads the `com.trading.*` agents and moves their plists to `~/Library/LaunchAgents/disabled-mbin/`. MBin's checkout is not touched.
   - `pmset repeat` is replaced by MTap's schedule.
7. **Validate:** `./scripts/schedule/mtap-schedule.sh validate` should print `environment OK`.
8. **Dry run:**
   - Run `./scripts/schedule/setup-schedule.sh test --disable-mbin`, then `pmset sleepnow`. The Mac wakes in 2 minutes, collects from minute 3 to minute 6, flushes and sleeps.
   - Then wake it and run `setup-schedule.sh status`. Check the end of `schedule.log`, the collector log, `daily-summary.log`, and the NAS `intraday_feed/` folder.
   - Outside market hours, E*TRADE returns stale quotes, but the full path still runs.
9. **Production:** `./scripts/schedule/setup-schedule.sh prod` (add `--with-daily` to run the daily chain on this Mac). Check that `pmset -g sched` shows `wake MTWRF 06:15` and `sleep MTWRF 13:40`.
10. **Massive per-second bars** (optional): see [IMAC.md](IMAC.md).

To undo: `setup-schedule.sh stop` removes MTap's agents and cancels the power schedule. To
bring MBin back, move its plists back from `disabled-mbin/` and load them.
