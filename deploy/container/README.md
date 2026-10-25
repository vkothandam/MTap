# deploy/container: MTap's sourcing on any Docker host

The same containers run the sourcing on the Synology NAS today and could run it on a cloud VM
later. Nothing in here is NAS-specific: a host supplies one env file, a secrets file and two
folders, then calls `jobs.sh`. The NAS wrapper is [deploy/synology](../synology/README.md).
The design and the reasons for it are in [docs/nas-migration.md](../../docs/nas-migration.md).

The iMac's launchd scripts (`scripts/schedule`) are a separate target and do not use anything
here.

```
deploy/container/
  Dockerfile.py         python:3.11-slim + uv; `uv sync --frozen --no-dev` (torch is the CPU wheel on Linux)
  Dockerfile.collector  selenium/standalone-chrome: the desktop (Xvfb, noVNC :7900, supervisord)
                        and the browser (its Google Chrome) + Node 20 + node/ (npm ci)
  collector/            the collector's supervisord programs and its restart loop
  compose.yml           redis, store, etrade, massive-ws, job
  bin/daily-capture.sh  the Massive daily fetches (the capture branch of daily-sourcing.sh)
  jobs.sh               build | start | daily | evening | stop | status | ws-test | selftest | watch | run | queue
```

## What runs

| service | image | what it does | restart |
| --- | --- | --- | --- |
| `redis` | redis:7-alpine | the bus between the feed and the store; on the compose network only | unless-stopped |
| `store` | mtap-py | `sourcing-py intraday store --follow`: Redis → hourly segments in STATE → `DATA/intraday_feed/` | unless-stopped |
| `etrade` | mtap-collector | `serve etrade`: logs in to E*TRADE in a headed Chrome and polls both watchlists | unless-stopped |
| `massive-ws` | mtap-py | `massive-ws follow --until-close`: per-second bars → `DATA/agg_1s/`; exits 0 after 16:21 ET | on-failure |
| `job` | mtap-py | one-off `sourcing-py` commands from jobs.sh | none |

The roles a host runs at `start` come from `MTAP_ROLES`:

- `etrade` starts redis, store and etrade;
- `massive-ws` starts the per-second collector. One connection per Massive key, so only one
  host may have this role.

The daily REST fetches run when the host schedules `daily`.

## The host contract

jobs.sh reads the env file named by `MTAP_ENV_FILE`. compose also passes that file to every
container, so it uses plain `KEY=VALUE` lines.

| variable | meaning |
| --- | --- |
| `MTAP_DATA_DIR` | the data folder (mounted at `/data`); must hold `.mtap-data-root` |
| `MTAP_STATE_DIR` | the state folder (mounted at `/state`): segments, heartbeats, the Chrome profile, logs. Local disk: the containers write it all day |
| `MTAP_SECRETS_FILE` | chmod 600; `MASSIVE_API_KEY`, the `ET_*` and `SMS_*` lines from `.env.example`, `SE_VNC_PASSWORD` |
| `MTAP_UID`, `MTAP_GID` | the owner of both folders. The py containers run as it, and the collector image's user is remapped to it at build time |
| `MTAP_ROLES` | `etrade`, `massive-ws`, both or neither |
| `MTAP_NOVNC_PORT` | the noVNC port on the host (default 7900) |
| container settings | `TZ`, `OTP_HOST` (the host's address, put in the texted link), `OTP_PORT`, `FEATURE_SCRAPING`, `INTRADAY_FEEDS`, `MASSIVE_RATE_LIMIT_PER_MIN`, … as in `.env.example` |

`MTAP_ENV_OVERRIDE` can name a second file that jobs.sh reads afterwards. It is for
host-level values only, such as a trial data folder; the containers still get
`MTAP_ENV_FILE`.

Secrets file values are read by compose, which expands `$` inside double quotes and
unquoted values. Put a value that contains `$`, `#` or spaces in single quotes.

The containers set their own `MTAP_DATA_ROOT=/data`, `MTAP_STATE_ROOT=/state`,
`REDIS_URL=redis://redis:6379` and `BUS_TRANSPORT=redis`. The collector also sets
`HEADLESS=false`, `OTP_BIND_HOST=0.0.0.0` and `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome`
(E*TRADE's bot manager refused puppeteer's older Chrome for Testing; docs/nas-migration.md).

## Jobs

```bash
MTAP_ENV_FILE=/path/host.env deploy/container/jobs.sh start
```

| job | what it does |
| --- | --- |
| `build` | builds both images, pulls redis. After every code change |
| `start` | checks the data marker, `storage check`, `up -d` the roles' services |
| `daily` | `bin/daily-capture.sh`: grouped daily bars for every missed session of the last 14 days, `--retry-failed`, then `massive-ext gaps`. Log: `STATE/logs/daily-sourcing-<date>.log` |
| `evening` | stop etrade; stop store (SIGTERM closes and uploads its segments); `intraday upload --drain`; stop redis; `intraday summary`; `daily` (the fill); wait up to 20 min for massive-ws to end its window; `massive-ws status` and `compact`; delete logs older than 14 days |
| `stop` | stops every container |
| `status` | writes `STATE/logs/status.txt` |
| `ws-test` | `massive-ws follow --no-window --duration 120`; refused while the massive-ws service runs |
| `selftest` | what a py container sees: its user, Python's path, where `sourcing_py` resolves, `sourcing-py --help`. A first check on a new host |
| `run ARGS` | `sourcing-py ARGS` in a one-off container |
| `watch` | in market hours (Mon–Fri, 09:30–16:10 ET; massive-ws to 16:15), a role's heartbeat older than `MTAP_WATCH_STALE_MIN` (10) minutes is a problem. It exits 1 once per incident (a new problem, or all clear again) with the reason in `STATE/logs/alert.txt`; `STATE/watch.state` holds the open problems. Reads files only |
| `queue` | runs `watch`, then the job named in `STATE/run-next`, then deletes the file. The name must be one of build, start, daily, evening, stop, status, ws-test, selftest, with no arguments. It exits non-zero if the watch alerts or the job fails |

`status.txt` covers the containers, heartbeats, Redis, the newest files, local segments,
today's errors, the collector's supervisor output and the `jobs.log` tail.

- **Alerts:** a host mails a non-zero exit. On the NAS, `task.sh` prints the reason
  (`alert.txt` and the end of `jobs.log`) to the task's own output, which DSM mails
  (deploy/synology/README.md, one-time setup step 2). On another host, mail the exit of
  `jobs.sh queue` from cron.
- **Lock:** one job runs at a time. A job waits up to 30 minutes for the lock; `queue` does
  not wait.
- **status.txt:** every job except `run` ends by rewriting it, so you can read the state
  without Docker access.

**Logs**, all under `STATE/logs/` and kept 14 days:

- `jobs.log`: the jobs (the host wrapper appends to it)
- `collector-<date>.log`: Node
- `store-<date>.log`
- `massive-<date>.log`
- `daily-sourcing-<date>.log`
- `daily-summary.log`
- `status.txt`

The collector container's own supervisor lines are in `docker logs mtap-etrade-1`.

## The E*TRADE login

The browser runs headed on the container's virtual display. Open
`http://<host>:7900` and enter the `SE_VNC_PASSWORD` to watch it.

When E*TRADE asks for a code, use either route:

- type it into the E*TRADE page in that window and tick "save device";
- or use the texted link, which opens the collector's own OTP page on `OTP_HOST:OTP_PORT`.

The Chrome profile is kept in `STATE/etrade/chrome-profile`. Once the device is saved, later
logins skip the code. Copying a profile from a Mac does not carry the device trust over.
Chrome encrypts the profile's cookies with an OS-specific key, so Linux Chrome cannot read
a Mac profile's cookies, and the first login on a new host needs the code.

## Building and testing on a Mac

Build and test only; real runs happen on the host.

```bash
docker build --platform linux/amd64 -f deploy/container/Dockerfile.py -t mtap-py .
docker build --platform linux/amd64 -f deploy/container/Dockerfile.collector -t mtap-collector .
```

The repo-root `.dockerignore` keeps the context to the core. It leaves out `.env`,
`settings.toml`, venvs and `node_modules`.

## Another host (a cloud VM)

Copy `deploy/synology` as a pattern:

- an env file with the host's paths, uid and address;
- a wrapper that waits for Docker and calls jobs.sh;
- the host's scheduler (cron or systemd timers) for start, daily and evening.

See the cloud notes in [docs/nas-migration.md](../../docs/nas-migration.md#later-a-cloud-vm).
