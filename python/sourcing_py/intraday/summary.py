"""`sourcing-py intraday summary --date D`: what a day's feed captured.

Reads the day's segments wherever they are (NAS partition, closed local parquet, and
still-open local .arrows), and prints rows per watchlist and every gap over
`--gap-min` minutes (default 5) in regular trading hours. The shutdown script logs this.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from ..common import config
from . import feed
from .upload import nas_dir

RTH_OPEN, RTH_CLOSE = time(9, 30), time(16, 0)


def day_files(day: date) -> list[Path]:
    d = f"date={day.isoformat()}"
    files = sorted((nas_dir() / d).glob("source=*/*.parquet"))
    files += sorted((config.state_root() / "segments").glob(f"*/{d}/*.parquet"))
    files += sorted((config.state_root() / "segments").glob(f"*/{d}/*.arrows"))
    return files


def _read(path: Path) -> pa.Table:
    if path.suffix == ".parquet":
        return pq.read_table(path)
    batches = []
    try:
        with pa.OSFile(str(path), "rb") as f:
            reader = ipc.open_stream(f)
            for batch in reader:
                batches.append(batch)
    except (pa.ArrowInvalid, OSError):
        pass  # being written right now; take the complete batches
    return pa.Table.from_batches(batches, schema=feed.ARROW_SCHEMA)


def load_day(day: date) -> pa.Table:
    tables = [_read(p) for p in day_files(day)]
    tables = [t for t in tables if t.num_rows]
    if not tables:
        return feed.ARROW_SCHEMA.empty_table()
    return pa.concat_tables(tables)


def summarize(table: pa.Table, day: date, gap_min: float = 5) -> dict:
    """Per watchlist: polls, rows, symbols, first/last poll (NY) and RTH gaps."""
    open_ny = datetime.combine(day, RTH_OPEN, feed.NY)
    close_ny = datetime.combine(day, RTH_CLOSE, feed.NY)
    meta = table.column("source_meta").to_pylist()
    ts = table.column("ts_utc").to_pylist()
    syms = table.column("symbol").to_pylist()
    sources = table.column("source").to_pylist()
    lists: dict[str, dict] = {}
    for m, t, s, src in zip(meta, ts, syms, sources):
        name = (json.loads(m).get("watchlist") if m else None) or src
        w = lists.setdefault(name, {"rows": 0, "polls": set(), "symbols": set()})
        w["rows"] += 1
        w["polls"].add(t)
        w["symbols"].add(s)
    out = {}
    for name, w in sorted(lists.items()):
        polls = sorted(w["polls"])
        rth = [p for p in polls if open_ny <= p <= close_ny]
        gaps = []
        edges = [open_ny, *rth, close_ny]
        for a, b in zip(edges, edges[1:]):
            if b - a > timedelta(minutes=gap_min):
                gaps.append((a.astimezone(feed.NY).strftime("%H:%M:%S"), b.astimezone(feed.NY).strftime("%H:%M:%S"),
                             round((b - a).total_seconds() / 60, 1)))
        out[name] = {
            "rows": w["rows"],
            "polls": len(polls),
            "symbols": len(w["symbols"]),
            "first": polls[0].astimezone(feed.NY).strftime("%H:%M:%S") if polls else None,
            "last": polls[-1].astimezone(feed.NY).strftime("%H:%M:%S") if polls else None,
            "rth_polls": len(rth),
            "gaps": gaps,
        }
    return out


def print_summary(day: date, gap_min: float = 5) -> int:
    table = load_day(day)
    files = day_files(day)
    print(f"intraday feed {day}: {table.num_rows:,} rows in {len(files)} file(s)")
    if not table.num_rows:
        print("  nothing captured")
        return 1
    span = pc.min_max(table.column("ts_utc"))
    lo, hi = span["min"].as_py(), span["max"].as_py()
    print(f"  span {lo.astimezone(feed.NY):%H:%M:%S} .. {hi.astimezone(feed.NY):%H:%M:%S} ET")
    for name, w in summarize(table, day, gap_min).items():
        print(f"  {name}: {w['rows']:,} rows, {w['polls']:,} polls ({w['rth_polls']:,} in RTH), "
              f"{w['symbols']} symbols, {w['first']} .. {w['last']} ET")
        if not w["gaps"]:
            print(f"    no RTH gaps over {gap_min:g} min")
        for a, b, mins in w["gaps"]:
            print(f"    gap {a} .. {b} ET ({mins:g} min)")
    return 0
