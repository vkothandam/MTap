# sourcing-py

Python sourcers for MTap. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
cd python
uv sync --extra dev
```

## Run a source

```bash
uv run sourcing-py run example_api
# output -> ./out/example_api/dt=YYYY-MM-DD/example_api-<run_id>.jsonl
```

## Storage roots and the intraday feed

```bash
uv run sourcing-py storage check          # both roots, NAS marker, free space, unuploaded segments
uv run sourcing-py storage init           # create the folders and DATA_ROOT/.mtap-data-root
uv run sourcing-py intraday store --follow   # storage consumer: Redis INTRADAY_FEED -> segments -> NAS
uv run sourcing-py intraday upload --drain   # one-off upload of closed segments (skips if the store runs)
uv run sourcing-py intraday summary [--date D] [--gap-min 5]   # rows and RTH gaps per watchlist
uv run sourcing-py intraday status        # store heartbeat, collector heartbeat, local segments
```

`MTAP_DATA_ROOT` (NAS, large write-once parquet) and `MTAP_STATE_ROOT` (local: DuckDB,
session, logs, unuploaded segments) come from env, then the repo `.env`, then `[storage]`
in settings.toml. Unset, everything stays where it was: the repo root and `./state`. The
feed contract is [schemas/intraday_feed.md](../schemas/intraday_feed.md); the schedule that
runs the store is in [scripts/README.md](../scripts/README.md).

**One central store.** With `MTAP_DATA_ROOT` set, every data file (`out/` datasets,
`agg_1s`, `intraday_feed`) lives on the NAS and is the only copy: the iMac fetches into
it and this Mac reads from it. Files are written to a hidden `.<name>.tmp` and renamed,
so a reader on the other Mac never sees half a file, and nothing is written while the
share is not mounted. DuckDB database files are **never** opened on the NAS (SMB has no
file locks): each stays local to the one Mac that writes it, as a rebuildable index over
the NAS files, and `daily-sourcing.sh` copies `fundamentals.duckdb` to
`DATA_ROOT/backup/etrade/`. A Mac that kept files in `./out` before moves them up once
with `scripts/storage/reconcile-out.sh`. Storing the bars twice (NAS files + DuckDB) is
a known cost; see [docs/TODO.md](../docs/TODO.md).

## Test (includes schema conformance)

```bash
uv run pytest
```

## Adding a source

Create `sourcing_py/sources/<name>/source.py` with a `Source` subclass that
implements `fetch()` and `extract()`. Register it in `../shared/config/sources.json`.
The base class handles the writer, provenance fields, and file naming per
`../schemas/conventions.md`.
