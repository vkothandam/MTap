# iMac: the Massive per-second feed

The Massive collector is a separate process from the E*TRADE collector:
- its own restart loop, lock, heartbeat and folders;
- no Redis, no storage consumer.

The 06:16 start job starts it next to E*TRADE. It ends by itself once its window (09:00–16:21 ET) is over, after uploading its last file. Starting or stopping it never touches E*TRADE, and the reverse.

A key allows only one WebSocket connection, so only this iMac runs it (`MTAP_MASSIVE_WS=1` in `.env`).

## 2026-10-15: one-off 5-year backfill (tonight, for the iMac agent)

Adds 5 years of daily bars, and pre-market / after-hours prices and volumes for every
common stock (`sourcing-py massive-ext`), to `<MTAP_DATA_ROOT>/out/`. REST only: no
WebSocket, so it never clashes with the per-second collector. Run it **after the 13:15 PT
sleep job** (wake the iMac first), and let it finish before the 06:16 PT start.

Measured on 2026-10-15: 64 random stocks × 5 years took 20 s with 16 workers (about 48 MB/s
down). The whole universe (about 8,200 stocks, 1.25 billion minute bars, about 120 GB
downloaded and not kept) takes about **45 minutes**, or 1.5–3 hours on a slower link.
Disk: about 1.7 GB of temporary staging under `<MTAP_STATE_ROOT>/massive_ext/`, removed at
the end; about 2.4 GB (daily bars) + 2–2.5 GB (extended hours) added on the NAS.

The plan keeps 5 years on a **rolling** basis: this changes every day. **Today (2026-10-18) the
first date served is 2021-10-21** (2021-10-18 was refused). Use the dates below; adjust
the day-of-run if you wait longer.

```bash
cd <repo> && git pull && (cd python && uv sync)
# paid plan: no client-side throttle, 16 parallel requests
grep -q '^MASSIVE_RATE_LIMIT_PER_MIN=' .env || echo 'MASSIVE_RATE_LIMIT_PER_MIN=0' >> .env
grep -q '^MASSIVE_WORKERS=' .env || echo 'MASSIVE_WORKERS=16' >> .env
cd python
uv run sourcing-py storage check                                  # the NAS must be mounted
caffeinate -dims uv run sourcing-py run massive_grouped_daily --fromdate 2021-10-21 --todate 2024-08-15
uv run sourcing-py run massive_grouped_daily --retry-failed       # holidays print "no data", not a failure
uv run sourcing-py run massive_tickers_cs                         # common stocks, listed and delisted
caffeinate -dims uv run sourcing-py massive-ext fetch --fromdate 2021-10-21 --todate 2026-10-17
#   prints "N/8224 fetched" every 250 stocks, then "done: ... date file(s) written"
```

- **Interrupted, or "N ticker(s) failed":** rerun the same `massive-ext fetch` command. Stocks already fetched are skipped; nothing is written to `out/extended_hours/` until every stock is in.
- **2026-10-15 itself** comes with the normal 06:40 run (`massive-ext gaps`), as every later session will.
- **Check:** `ls <MTAP_DATA_ROOT>/out/extended_hours | wc -l` is about 1,255, and `ls <MTAP_DATA_ROOT>/out/daily_summary | head -1` is `daily_summary-2021-10-21.jsonl`.
- The DuckDB load (`ingest-eod` → `derive-features` → `export-tft`) runs on the other Mac, which reads these files from the NAS.

## Update of 2026-10-14: switch to the per-session collector (for the iMac agent)

Today the iMac still runs the old always-on agent `com.mtap.massive`. Massive is already installed: do **not** redo any install step.

**Do now. This is safe mid-session and restarts nothing:**

```bash
cd <repo> && git pull && (cd python && uv sync)       # no new packages
grep -q '^MTAP_MASSIVE_WS=1' .env || echo 'MTAP_MASSIVE_WS=1' >> .env
./scripts/schedule/mtap-schedule.sh status | sed -n '/Massive/,/^$/p'
#   expect: "the old always-on com.mtap.massive agent", then state: collecting
```

**Do not run today:**
- `setup-schedule.sh` with any argument;
- `mtap-schedule.sh start|stop|start-massive|stop-massive`;
- `massive-ws follow`;
- `launchctl kickstart`/`bootout`.

Any of these can restart a collector mid-session.

**What then happens on its own:**
- **Today 13:15 PT, the sleep job (new code):**
  1. stops E*TRADE and flushes the store;
  2. waits for the old Massive agent to finish its window (13:21) and upload;
  3. combines today's files into `day-2026-10-14.parquet`;
  4. **removes the old `com.mtap.massive` agent**;
  5. puts the Mac to sleep.
- **Tomorrow 06:16 PT, the start job:** starts Massive (new code, with the fractional-volume columns) and then E*TRADE, as two independent processes.

**Check tomorrow after 06:20 PT:**

```bash
grep -E "removed the old com.mtap.massive|Massive collector idle" <MTAP_LOGS>/schedule.log | tail -2
./scripts/schedule/mtap-schedule.sh status | sed -n '/Massive/,/^$/p'
#   expect: "running (supervisor <pid>, ...)", then state: collecting, delay_s ~900
launchctl list | grep com.mtap.massive || echo "old agent gone"
```

After 06:35 PT, the newest NAS file should have the new column (expect `True`):

```bash
./python/.venv/bin/python -c "from sourcing_py.common import config; import pyarrow.parquet as pq; \
fs = sorted((config.data_root() / 'agg_1s').glob('date=*/source=massive_ws/*.parquet'), key=lambda p: p.stat().st_mtime); \
print(fs[-1].name, 'volume_i_dec' in pq.read_schema(fs[-1]).names)"
```

**If the old agent is still listed tomorrow** (the sleep job timed out waiting): run `./scripts/schedule/setup-schedule.sh prod --with-daily` after 13:21 PT. That needs `sudo`, so it is a manual step.

## Manual, once, after a close (needs sudo)

`./scripts/schedule/setup-schedule.sh prod --with-daily`, adding `--disable-mbin` if it asks. This does three things:
- moves the pmset sleep backstop from 13:30 to 13:40;
- removes the old Massive agent if it is still there;
- restarts the storage consumer, so run it only after the close.

Skip it if `pmset -g sched` already shows a 13:40 sleep.

## A weekday (Pacific time)

| time | what runs |
| --- | --- |
| 06:15 | wake |
| 06:16 | start job: mounts the NAS, starts Massive, then E*TRADE |
| 06:16–13:15 | E*TRADE watchlist collector |
| 06:16–13:21 | Massive per-second bars for all stocks (15 min delayed, so the 13:00 close arrives about 13:15); then it uploads and exits |
| 06:40 | daily bars for every session missed in the last 14 days. Capture-only mode stops there |
| 13:15 | sleep job: stops E*TRADE, flushes, fills missed daily bars, waits up to 20 min for Massive to exit, combines its files into one per day, then sleeps |
| 13:40 | pmset backstop sleep (13:30 until the manual step above) |

## Code updates

Each session starts fresh, so a `git pull` takes effect at the next 06:16 start. Don't restart anything to apply it.

## By hand

- `mtap-schedule.sh start-massive`: start Massive now. Inside the window it collects. Outside it, it uploads any leftover local files and exits.
- `mtap-schedule.sh stop-massive`: stop it. It closes and uploads its open file first, and bars are lost until it is started again.
- **Turn it off on this machine:** remove `MTAP_MASSIVE_WS=1` from `.env`, then run `stop-massive`.

## Storage

- **Local:** `<MTAP_STATE_ROOT>/segments_massive/`. A file closes every 5 minutes, is uploaded, and is deleted once the NAS copy's row count matches. Normally under 0.5 GB.
- **NAS:** `<MTAP_DATA_ROOT>/agg_1s/date=YYYY-MM-DD/source=massive_ws/*.parquet`.
  - The evening sleep job combines each day's 5-minute files into one `day-YYYY-MM-DD.parquet`: the same data, about 40% smaller.
  - It takes about a minute and up to 4 GB of memory, and is safe to interrupt.
  - By hand: `sourcing-py massive-ws compact [--date D]`.
- **Rule:** data files live only on the NAS; DuckDB files stay local (python/README.md, "One central store"). Deferred storage work: [docs/TODO.md](../docs/TODO.md).
- **NAS down:** files stay local and upload at the next start.
  - If more than 10 GB is waiting, or less than 20 GB of disk is free, the collector disconnects until uploads catch up.
  - Nothing is deleted before it is uploaded.
- **Size, measured on 2026-10-14 (09:35–11:00 ET):**
  - 13,031 symbols, about 1.7 M bars an hour;
  - 5-minute files take about 58 MB an hour;
  - the day file is about 58% of that: roughly 240 MB a day, or about 60 GB a year.

## Checks

```bash
./scripts/schedule/setup-schedule.sh status   # "Massive per-second collector" section
tail -20 <MTAP_LOGS>/schedule.log             # "massive: started", "massive: window over for today"
tail <MTAP_LOGS>/massive-$(date +%F).log      # the collector's own log
tail <MTAP_LOGS>/daily-summary.log            # per day: bars, symbols, MB, longest market-wide gap
```

In `massive-ws status`, each heartbeat field means:

| field | meaning |
| --- | --- |
| `state` | `collecting` / `reconnecting` / `paused` (local guard) / `idle`; `— stopped` once it has exited |
| `delay_s` | about 900 on the delayed plan |
| `disconnects` | bars are lost while disconnected; the feed has no replay |
| `last_status` | `max_connections` means another machine uses the key; `auth_failed` means a bad key (it retries every 10 min) |
| `pending_upload` | files not yet on the NAS |
