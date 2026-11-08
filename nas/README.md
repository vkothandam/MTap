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
- **Add stocks to days that already have a file (`--extra`):** `massive1s fetch --extra` fetches only the
  universe's tickers that are in none of the quarter's day files, and writes `day-D.extra.parquet` next to
  `day-D.parquet` (never touching it). Jobs go in `massive_1s/jobs-extra/`; finalize one by hand with
  `massive1s finalize --extra --data /data <quarter>`. Readers of a day must take every `*.parquet` in
  `date=D/source=massive_rest/`.
- **A second run's inputs (`--name`):** `sourcing-py massive-1s prepare ... --name N` writes
  `massive_1s/inputs-N/` instead of `inputs/` (which a running or resumable job reads), and
  `massive1s fetch --inputs .../inputs-N` uses it.

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
  - With `--from-bars` (`./nas/deploy.sh FROM TO --from-bars`): instead the tickers that traded in
    `bars_1m` over the range, under the ticker they traded as, typed CS in any `tickers_cs` file (listed
    or not), minus the same names and ADRs, plus SPY. One list covers the whole range; a stock that
    did not trade in a quarter costs one empty call there. The column `listed` = 0 marks a stock no
    longer listed under that ticker (delisted or renamed); the fetcher reads only `ticker` and `kept`.
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
   **Deployed 2026-11-06 ~21:55 PT on the NAS** (the NAS kept on overnight):
   - `./nas/deploy.sh 2021-11-08 2024-10-18 --from-bars`: 741 sessions, 6,882 stocks (of 7,457 traded; 2,642 no
     longer listed under that ticker; excluded: 465 ADRs, 67 notes, 32 preferreds, 11 funds).
   - 2021-11-08 was the first date the 5-year edge allowed (2021-11-05 returned 403 on 2026-11-07 UTC). A resume
     days later can get 403s on the oldest sessions: finalize that quarter anyway (see Failures above).
   - About 77 GB and ~5.5 h, finalize-bound (613 s a month on the NAS); the 2022-03 trial days are skipped.
   - The universe is the stocks that traded then (`./nas/deploy.sh 2021-11-01 2024-10-18 --from-bars`, built
     2026-11-06): today's list drops delisted stocks, and renamed ones are stored under their old ticker
     (LEARNINGS `tag-survivorship`). One list for the range, not one per quarter (the fetcher takes one list).
   - SPY is added by name.
   - The stocks missing from step 2's range go into separate `day-D.extra.parquet` files.
4. **Queued behind step 3 on 2026-11-06 23:23 PT** (one-off, no DSM task): a `nohup setsid` shell started
   over ssh as the NAS user waits for D1's process to exit, then runs the binary directly on the NAS host
   (`massive_1s/bin/massive1s`, built for x86_64 in Mac Docker; no docker or root needed). Log
   `massive_1s/logs/queue-d2-d1b.log`; each run's own log lines go there too.
   - **D2:** `--inputs inputs-d2`, 2026-10-19 → 2026-11-06 (15 sessions, today's list 5,207 + SPY). Those days
     had only `source=massive_ws`; `massive_rest` is then the one training source.
   - **D1b:** `--inputs inputs-d1b --extra`, 2024-10-21 → 2026-10-16 (499 sessions). Universe = traded in
     `bars_1m` then (`--from-bars`: 6,297 kept, 1,079 no longer listed under that ticker, 437 ADRs out); each
     quarter fetches only the ones missing from its day files.
   - Delete `massive_1s/bin/` and `inputs-d2/`, `inputs-d1b/` after both finish and are checked.
   - **Done 2026-11-07, 0 failures:** D1 03:21 PT (4.02B bars, 62.3 GB, 15.5 B/bar, 5 h 26 min; a quarter fetches in
     ~6 min and finalizes in 22–34 min), D2 03:31 (107M bars, 1.80 GB), D1b 03:46 (184.9M bars, 2.83 GB in 499 extra
     files). Per-quarter numbers in LEARNINGS "1-second backfill, full runs".
   - **The one-month trial (2022-03) passed on 2026-11-06:**
     - 5,730 stocks, 0 failed;
     - fetch 142 s, finalize 613 s;
     - 2.40 GB on disk (15.7 B/bar);
     - 1 s = 96% of minute volume;
     - delisted stocks present.

     The same finalize took 92 s on the Mac (zstd-19, 14 threads), and zstd-9 cuts its CPU about 4× for 5% more disk.
     Numbers in LEARNINGS (2022-03 trial).

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
