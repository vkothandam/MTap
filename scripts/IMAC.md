# iMac: adding the Massive per-second feed

The Massive collector (`com.mtap.massive`) is a separate process. It has its own launchd
agent, lock, heartbeat and folders, and it does not use Redis. Installing, restarting or
removing it never touches the E*TRADE collector, Redis or the storage consumer, so it can
be added in the middle of a trading session.

## Pick it up (safe while E*TRADE is collecting)

```bash
cd <repo> && git pull && (cd python && uv sync)      # uv sync only adds websockets
# .env: add these two lines
#   MASSIVE_API_KEY=<key>
#   MTAP_DAILY_CAPTURE_ONLY=1
./scripts/schedule/setup-schedule.sh massive         # installs com.mtap.massive only
./python/.venv/bin/sourcing-py massive-ws status     # expect: collecting, delay_s ~900
```

- Don't rerun `setup-schedule.sh prod` during the session: it restarts the storage consumer.
- After the close, run `prod --with-daily` once. It moves the pmset sleep backstop from 13:30 to 13:40 and leaves the Massive agent as it is. Until you do, the 13:30 backstop still falls after Massive's 13:21 window end.
- The collector connects only from 09:00 to 16:21 ET on trading days. Outside that window it sits idle.
- A key allows only one WebSocket connection, so no other machine may run `massive-ws follow`.
- Undo: `setup-schedule.sh massive --remove`.

## A weekday (Pacific time)

| time | what runs |
| --- | --- |
| 06:15 | wake |
| 06:16–13:15 | E*TRADE watchlist collector (unchanged) |
| 06:15–13:21 | Massive per-second bars for all stocks (15 min delayed, so the 13:00 close arrives about 13:15) |
| 06:40 | daily bars for every session missed in the last 14 days. Capture-only mode stops there |
| 13:15 | sleep job: stops E*TRADE, flushes, fills missed daily bars, waits up to 20 min for Massive's last upload, then sleeps |
| 13:40 | pmset backstop sleep |

## Storage

- **Local:** `<MTAP_STATE_ROOT>/segments_massive/`. A file closes every 5 minutes, is uploaded, and is deleted once the NAS copy's row count matches. Normally under 0.5 GB.
- **NAS:** `<MTAP_DATA_ROOT>/agg_1s/date=YYYY-MM-DD/source=massive_ws/*.parquet`.
- **NAS down:** files stay local. If more than 10 GB is waiting, or less than 20 GB of disk is free, the collector disconnects. It reconnects once the files are uploaded. Nothing is deleted before it is uploaded.
- **Size estimate:** about 38 bytes per bar, measured pre-open. That is roughly 0.2–0.45 GB per regular-session hour on the NAS, 1.5–3 GB per day, and 1.3–2 GB per hour of network. Measure after the first day with `du -sh <MTAP_DATA_ROOT>/agg_1s/date=<day>`.

## Checks

```bash
./scripts/schedule/setup-schedule.sh status   # "Massive per-second collector" section
tail -20 <MTAP_LOGS>/schedule.log             # the sleep job logs "Massive collector idle, all uploaded"
tail <MTAP_LOGS>/launchd-com.mtap.massive.log
```

In `massive-ws status`, each heartbeat field means:

| field | meaning |
| --- | --- |
| `state` | `collecting` / `reconnecting` / `paused` (local guard) / `idle` (outside the window) |
| `delay_s` | about 900 on the delayed plan |
| `disconnects` | bars are lost while disconnected; the feed has no replay |
| `last_status` | `max_connections` means another machine uses the key; `auth_failed` means a bad key (the collector retries every 10 min) |
| `pending_upload` | files not yet on the NAS |
