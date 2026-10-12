"""`sourcing-py intraday ingest-mbin --root DIR`: backfill MBin's watchlist CSVs as feed rows.

MBin wrote one CSV per symbol per tracking session:
`<root>/<date>/<SYM>_<HH-MM-SS>.csv` (the MAIN watchlist, ~2 s polls) and
`<root>/<date>/MC_<SYM>_<HH-MM-SS>.csv` (MidCap, ~20 s). Each row is one poll; `sysdate`
is the poll time in UTC. This turns them into canonical intraday rows
(schemas/intraday_feed.md), exactly as `EtradeWatchlistFeed.ts` maps a live poll, so the
panel export reads backfill and live days the same way.

MBin is only ever read. One file per day is written atomically:
`DATA_ROOT/intraday_feed/date=<D>/source=etrade_watchlist/mbin.parquet`.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..utils import trading_calendar
from . import feed
from .upload import nas_dir

log = logging.getLogger(__name__)

SOURCE = "etrade_watchlist"
FILE_NAME = "mbin.parquet"

# The CSV headers have a space after every comma, and the layout changed over time: the
# 2025-12-15 files have no "Day's Gain %", and their MidCap files carry the MAIN header
# over MidCap rows. So a file's columns come from its header when the header is as wide
# as its rows, else from the known layout of that width; columns are then looked up by
# name, like the live adapter does.
MAIN_COLUMNS = (
    "Symbol", "Volume", "Ask", "Bid", "Ask Size", "Bid Size", "Qty", "Open", "Previous Close",
    "Change %", "10 Day Volume", "Last Trade", "Day's Gain %", "VWAP", "sysdate",
)
MC_COLUMNS = (
    "Symbol", "Last Price", "Change $", "Change %", "Price Paid", "Date Acquired",
    "Change since Added %", "Volume", "VWAP", "sysdate",
)
LAYOUTS = {
    len(MAIN_COLUMNS): MAIN_COLUMNS,
    len(MAIN_COLUMNS) - 1: tuple(c for c in MAIN_COLUMNS if c != "Day's Gain %"),
    len(MC_COLUMNS): MC_COLUMNS,
}

# canonical field -> CSV header, per watchlist (the live adapter's FIELD_HEADERS). MBin's
# VWAP column is dropped: it resets whenever the scraper restarted.
MAIN_FIELDS = {
    "price": "Last Trade", "volume_cum": "Volume", "day_open": "Open",
    "prev_close": "Previous Close", "bid": "Bid", "ask": "Ask", "bid_size": "Bid Size",
    "ask_size": "Ask Size", "vol_10d_src": "10 Day Volume",
}
MC_FIELDS = {"price": "Last Price", "volume_cum": "Volume"}
ZERO_IS_NULL = {"price", "day_open", "prev_close", "bid", "ask"}

# Same names and ids as node/src/services/etrade/constants.ts WATCHLISTS.
WATCHLISTS = {
    "main": {"watchlist": "Experimental", "watchlist_id": "215565674306"},
    "mc": {"watchlist": "MidCap", "watchlist_id": "215584661306"},
}


def _num(header: str, zero_is_null: bool) -> str:
    """SQL for a lenient number (the live adapter's parseNum): strip , % $ and spaces."""
    v = f"TRY_CAST(NULLIF(regexp_replace(\"{header}\", '[,%$\\s]', '', 'g'), '') AS DOUBLE)"
    return f"NULLIF({v}, 0)" if zero_is_null else v


def split_files(day_dir: Path) -> tuple[list[Path], list[Path]]:
    """(MAIN files, MidCap files). By prefix, not a glob: `[!M]*` would drop MSFT."""
    files = sorted(p for p in day_dir.glob("*.csv") if p.is_file())
    mc = [p for p in files if p.name.startswith("MC_")]
    main = [p for p in files if not p.name.startswith("MC_")]
    return main, mc


def layout(path: Path) -> tuple[str, ...] | None:
    """The column names of one CSV (None for a file with no data rows)."""
    with path.open() as f:
        head = tuple(c.strip() for c in f.readline().split(","))
        first = f.readline()
    if not first.strip():
        return None
    width = len(first.split(","))
    return head if len(head) == width else LAYOUTS.get(width)


def by_layout(files: list[Path]) -> dict[tuple[str, ...], list[Path]]:
    groups: dict[tuple[str, ...], list[Path]] = {}
    for p in files:
        cols = layout(p)
        if cols is None:
            continue
        groups.setdefault(cols, []).append(p)
    return groups


def _read(con: duckdb.DuckDBPyConnection, files: list[Path], fields: dict[str, str],
          watchlist: dict, day: date) -> pa.Table:
    groups = {cols: fs for cols, fs in by_layout(files).items() if "Symbol" in cols and "sysdate" in cols}
    if not groups:
        return feed.ARROW_SCHEMA.empty_table()
    table = pa.concat_tables([_read_group(con, fs, list(cols), fields, watchlist, day)
                              for cols, fs in groups.items()])
    if len(groups) > 1:  # the same poll in files of two header versions
        con.register("merged", table)
        table = con.execute("SELECT * FROM merged QUALIFY row_number() OVER (PARTITION BY symbol, ts_utc) = 1").arrow()
        table = (table.read_all() if isinstance(table, pa.RecordBatchReader) else table).cast(feed.ARROW_SCHEMA)
        con.unregister("merged")
    table = table.sort_by([("ts_utc", "ascending"), ("symbol", "ascending")])
    # feed_seq: the poll's index in the day (all symbols of one poll share its time).
    polls = pc.unique(table.column("ts_utc"))
    seq = pc.index_in(table.column("ts_utc"), value_set=polls).cast(pa.int64())
    return table.set_column(table.schema.get_field_index("feed_seq"), "feed_seq", seq)


def _read_group(con: duckdb.DuckDBPyConnection, files: list[Path], columns: list[str],
                fields: dict[str, str], watchlist: dict, day: date) -> pa.Table:
    meta = json.dumps({"origin": "mbin", **watchlist}, sort_keys=True)
    selects = []
    for f in feed.ARROW_SCHEMA:
        name = f.name
        if name == "symbol":
            selects.append("trim(\"Symbol\") AS symbol")
        elif name == "ts_utc":
            selects.append("ts AS ts_utc")
        elif name == "source":
            selects.append(f"'{SOURCE}' AS source")
        elif name == "kind":
            selects.append("'snapshot' AS kind")
        elif name == "feed_seq":
            selects.append("NULL::BIGINT AS feed_seq")  # numbered after all groups are read
        elif name in fields and fields[name] in columns:
            selects.append(f"{_num(fields[name], name in ZERO_IS_NULL)} AS {name}")
        elif name == "source_meta":
            selects.append(f"'{meta}' AS source_meta")
        elif name == "trades_i":
            selects.append("NULL::BIGINT AS trades_i")
        elif name == "stream_id":
            selects.append("NULL::VARCHAR AS stream_id")
        else:
            selects.append(f"NULL::DOUBLE AS {name}")
    names = ", ".join("'" + c.replace("'", "''") + "': 'VARCHAR'" for c in columns)
    sql = f"""
        WITH raw AS (
            SELECT * REPLACE (trim("Symbol") AS "Symbol"),
                   TRY_CAST(trim(sysdate) AS TIMESTAMPTZ) AS ts
            FROM read_csv($files, header = false, skip = 1, delim = ',', quote = '"',
                          columns = {{{names}}}, null_padding = true, ignore_errors = true)
        ),
        day AS (
            SELECT * FROM raw
            WHERE ts IS NOT NULL AND "Symbol" <> ''
              AND CAST(timezone('America/New_York', ts) AS DATE) = $day
            QUALIFY row_number() OVER (PARTITION BY "Symbol", ts) = 1
        )
        SELECT {", ".join(selects)} FROM day ORDER BY ts, symbol
    """
    table = con.execute(sql, {"files": [str(p) for p in files], "day": day}).arrow()
    if isinstance(table, pa.RecordBatchReader):
        table = table.read_all()
    return table.cast(feed.ARROW_SCHEMA)


def read_day(day_dir: Path, day: date) -> pa.Table:
    """Both watchlists of one MBin date folder as canonical feed rows (sorted by time)."""
    main, mc = split_files(day_dir)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    try:
        tables = [
            _read(con, main, MAIN_FIELDS, WATCHLISTS["main"], day),
            _read(con, mc, MC_FIELDS, WATCHLISTS["mc"], day),
        ]
    finally:
        con.close()
    return pa.concat_tables(tables)


def out_path(day: date) -> Path:
    return nas_dir() / f"date={day.isoformat()}" / f"source={SOURCE}" / FILE_NAME


def write_day(table: pa.Table, day: date) -> Path:
    """Atomic: written next to the target, then renamed over it."""
    path = out_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def ingest(root: Path, days: list[date] | None = None) -> list[dict]:
    """Ingest each trading-day folder under `root` (or only `days`). Returns one summary
    per day: rows, symbols and polls per watchlist."""
    root = root.resolve()
    target = nas_dir().resolve()
    if target == root or root in target.parents:
        raise ValueError(f"refusing to write inside the source tree {root}")
    if days is None:
        days = sorted(date.fromisoformat(p.name) for p in root.iterdir()
                      if p.is_dir() and _is_date(p.name))
    out = []
    for day in days:
        day_dir = root / day.isoformat()
        if not trading_calendar.is_trading_day(day):
            log.info("%s: not a trading day, skipped", day)
            continue
        if not day_dir.is_dir():
            log.warning("%s: no folder %s", day, day_dir)
            continue
        table = read_day(day_dir, day)
        if table.num_rows == 0:
            log.warning("%s: no rows", day)
            continue
        path = write_day(table, day)
        stats = {"date": day.isoformat(), "rows": table.num_rows, "path": str(path)}
        metas = table.column("source_meta").to_pylist()
        for key, wl in WATCHLISTS.items():
            sub = table.filter(pa.array([wl["watchlist"] in m for m in metas]))
            stats[key] = {"rows": sub.num_rows,
                          "symbols": len(set(sub.column("symbol").to_pylist())),
                          "polls": len(set(sub.column("ts_utc").to_pylist()))}
        log.info("%s: %s", day, stats)
        out.append(stats)
    return out


def _is_date(name: str) -> bool:
    try:
        date.fromisoformat(name)
        return True
    except ValueError:
        return False

