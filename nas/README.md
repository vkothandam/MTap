# nas/: tools that run on the Synology NAS

`massive1s` fetches Massive 1-second bars for every listed common stock and writes one
file per session to `MTap-Data/agg_1s/date=D/source=massive_rest/day-D.parquet` (columns:
[schemas/intraday_feed.md](../schemas/intraday_feed.md#sourcemassive_rest-the-rest-backfill-nasmassive1s)).
It runs on the NAS itself (DS925+), so the data goes straight to its disk with nothing
copied over the LAN.

```
nas/
  README.md         this runbook
  deploy.sh         on the Mac: prepare the inputs, copy the source to the share, show status
  massive1s/        the Rust tool (Cargo.toml, src/, tests/)
    run.sh          the one command the NAS runs
```

## Running it

On the Mac (the share mounted at `/Volumes/Backups`):

```bash
./nas/deploy.sh 2026-09-01 2026-09-30
```

This writes the stock list and the session windows to `MTap-Data/massive_1s/inputs/`, copies
`nas/massive1s/` to `MTap-Data/nas/massive1s/`, and writes `MTap-Data/massive_1s/run.args`.
Check the stock count and exclusions it prints (the full list is `inputs/universe.csv`).

On the NAS: **DSM > Control Panel > Task Scheduler > `massive1s` > Run.** That is all.

Watch from the Mac:

```bash
./nas/deploy.sh status
```

- **Stop:** DSM > Container Manager > Container > `massive1s` > Stop (or `sudo docker stop massive1s`).
- **Resume:** press Run again. Finished quarters are skipped and staged stocks are not
  fetched again.
- **Days already on disk are not fetched again.** A session that already has its
  `day-D.parquet` when its quarter's job starts is left out of the job, so a range can
  overlap an earlier run (the 2-year run skips the 2026-09 test month).
- **Failures:** a stock that fails (after 3 retries) is listed in
  `jobs/<quarter>/failures.jsonl`, and its quarter is not finalized. Press Run again to retry
  only those stocks. To give up on them, finalize anyway by running
  `massive1s finalize --data /data <quarter>` in the container.
- **Fetch a finished quarter again:** delete `MTap-Data/massive_1s/jobs/<quarter>/`, then Run.

If a Run seems to do nothing, `deploy.sh status` shows `logs/task.log`: what the task
printed, including Docker errors and the output of a container that exited at once.

The first Run pulls the `rust:1-alpine` image and compiles the tool, which takes several
minutes. Later Runs compile only if the source changed.

## What it does

1. Splits the range into calendar quarters (jobs) under `MTap-Data/massive_1s/jobs/<d0>_<d1>/`.
2. **For each stock**, makes one range call for the quarter, `/v2/aggs/ticker/T/range/1/second/d0/d1`.
   - Pages through `next_url`, 50k bars per page.
   - Keeps only the bars from 09:00:00 to session close + 6:59 (New York time). The endpoint
     returns 04:00–19:59 every day.
   - Stages the bars as `staging/<T>.parquet`.
3. **Once every stock in the quarter is staged**, writes one sorted `day-D.parquet` per session.
   - zstd 19; the row count is checked before the file is renamed into place.
   - Then deletes the staging files and writes `done`.
   - This runs while the next quarter is being fetched.

The log is `MTap-Data/massive_1s/logs/run-<time>-<from>_<to>.log`. Each job also writes:

- a progress line every 500 stocks
- per-job totals: calls, bars, GB downloaded, peak RAM, CPU seconds
- per-day file sizes

`progress.json` in each job folder is what `deploy.sh status` shows.

Inputs come from Python on the Mac (`sourcing-py massive-1s prepare`), because the
exchange calendar and the ticker list live there:

- **universe.csv:** the stocks listed today, from the newest `out/tickers_cs` file. Notes,
  preferreds, depositary shares, funds and rights are excluded, even though Massive types
  them as CS.
- **windows.csv:** `[09:00, close + 7 min)` per session, with half days ending at 13:07.

## One-time setup

1. **Container Manager:** install it from DSM Package Center.
2. **Key file**, in your NAS home folder (never in the share or the repo).
   - The home service must be on: Control Panel > User & Group > Advanced > Enable user home service.
   - SSH must be on: Control Panel > Terminal & SNMP > Enable SSH.
   - Then, from the Mac:
     ```bash
     ssh vijayan.synology@192.168.1.100
     read -rsp 'Massive key: ' K; printf 'MASSIVE_API_KEY=%s\n' "$K" > ~/.massive1s.env; unset K
     chmod 600 ~/.massive1s.env
     ls /volume2/Backups/MTap-Data/.mtap-data-root    # the share path the task uses
     exit
     ```
   - `run.sh` refuses to start unless the file is mode 600.
   - On this NAS the `Backups` share is on **volume2** (`/volume2/Backups`), not volume1.
     If a NAS path is ever in doubt, `ls -d /volume*/Backups` on the NAS shows it; the
     Mac's `/Volumes/Backups` is only the Mac's view of the share.
3. **The task:** Control Panel > Task Scheduler > Create > Triggered Task > User-defined script.
   - **General:** name `massive1s`, user **root** (needed to start docker; the tool itself
     runs as your user), event Boot-up, and **untick Enabled** so it runs only when you
     press Run.
   - **Task Settings > Run command:**
     ```
     bash /volume2/Backups/MTap-Data/nas/massive1s/run.sh
     ```

The NAS user is taken from the Mac's SMB mount (`vijayan.synology`). If that differs from
your DSM login, set `MTAP_NAS_USER` in `.env`. `MASSIVE1S_WORKERS` in `.env` sets the number
of concurrent requests (default 32).

## Plan and sizes

1. One month first (2026-09). Measure bars, calls, time, GB, RAM/CPU and bytes per bar,
   compare against a DuckDB copy, and check REST against the live feed on 14–16 Oct.
   Measured on 2026-09 (21 sessions, 5,216 stocks): the fetch took 120 s with 0 failures
   and 6,587 calls; 133M bars kept; 2.03 GB downloaded; 2.24 GB on disk (16.8 B/bar,
   ~105 MB per day). Finalize took 552 s and is the slow step.
2. Then 2 years: `./nas/deploy.sh 2024-10-21 2026-10-16`. From the month above: about 3.2
   billion bars, ~48 GB downloaded, ~54 GB on disk, a few hours in total (finalize-bound).
3. Then 5 years, with an earlier start date. The paid plan's 5-year edge rolls forward daily
   (2021-10-21 on 2026-10-18); a date before it fails with HTTP 403.

## Developing

The tests use a local mock server, so they make no Massive calls. To run them on the Mac
(Docker only builds and tests here; real runs happen on the NAS):

```bash
docker run --rm -v "$PWD/nas/massive1s":/src -v ~/.cache/massive1s:/cache \
  -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target -w /src rust:1-alpine cargo test
```

`python/tests/test_massive_1s.py` checks the Python inputs, and checks that a day file
written by this tool has the exact `AGG_1S_SCHEMA`. The fixture is
`python/tests/fixtures/agg_1s/massive_rest_sample.parquet`; to rewrite it, run the test
above with `-e MASSIVE1S_FIXTURE_OUT=/fixture/massive_rest_sample.parquet` and the fixture
folder mounted at `/fixture`.
