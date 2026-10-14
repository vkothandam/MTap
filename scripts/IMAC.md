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
| 13:15 | sleep job: stops E*TRADE, flushes, fills missed daily bars, waits up to 20 min for Massive's last upload, combines its files into one per day, then sleeps |
| 13:40 | pmset backstop sleep |

## Storage

- **Local:** `<MTAP_STATE_ROOT>/segments_massive/`. A file closes every 5 minutes, is uploaded, and is deleted once the NAS copy's row count matches. Normally under 0.5 GB.
- **NAS:** `<MTAP_DATA_ROOT>/agg_1s/date=YYYY-MM-DD/source=massive_ws/*.parquet`. The evening sleep job combines each day's 5-minute files into one `day-YYYY-MM-DD.parquet`, with the same data, about 40% smaller. This takes about a minute and up to 4 GB of memory, and is safe to interrupt. By hand: `sourcing-py massive-ws compact [--date D]`.
- **NAS down:** files stay local. If more than 10 GB is waiting, or less than 20 GB of disk is free, the collector disconnects. It reconnects once the files are uploaded. Nothing is deleted before it is uploaded.
- **Size, measured on 2026-10-14 (09:35–11:00 ET):**
  - 13,031 symbols, about 1.7 M bars an hour;
  - 5-minute files take about 58 MB an hour;
  - the combined day file is about 58% of that: roughly 240 MB a day, or about 60 GB a year.

## Checks

```bash
./scripts/schedule/setup-schedule.sh status   # "Massive per-second collector" section
tail -20 <MTAP_LOGS>/schedule.log             # the sleep job logs "Massive collector idle, all uploaded"
tail <MTAP_LOGS>/launchd-com.mtap.massive.log
tail <MTAP_LOGS>/daily-summary.log            # per day: bars, symbols, MB, longest market-wide gap
```

In `massive-ws status`, each heartbeat field means:

| field | meaning |
| --- | --- |
| `state` | `collecting` / `reconnecting` / `paused` (local guard) / `idle` (outside the window) |
| `delay_s` | about 900 on the delayed plan |
| `disconnects` | bars are lost while disconnected; the feed has no replay |
| `last_status` | `max_connections` means another machine uses the key; `auth_failed` means a bad key (the collector retries every 10 min) |
| `pending_upload` | files not yet on the NAS |
