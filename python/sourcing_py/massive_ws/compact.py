"""End-of-day combine: one parquet file per day on the NAS instead of ~90 five-minute files.

The collector uploads one file per 5 minutes of bar time. After the window closes the
evening sleep job runs `massive-ws compact`, which for each day folder

    DATA_ROOT/agg_1s/date=D/source=massive_ws/*.parquet  ->  day-D.parquet

reads every file (including an earlier day-D.parquet), drops exact duplicate bars
(symbol, ts_utc), sorts by (symbol, ts_utc) and writes one zstd-19 file: same columns and
values, about 40% smaller (longer runs per symbol). The new file is written locally,
copied to a .tmp name on the NAS, its row count checked, renamed into place, and only
then are the inputs deleted. An interrupted run leaves inputs that duplicate the day
file; the next run drops the duplicates, so it is safe to rerun at any time.

A file that lands while this runs is not in the listing and is merged next time.
Today's folder is combined only once the collector's window has closed.
Memory: about 100 B per bar, plus the sort (a full day, ~12 M bars, peaks near 3 GB).
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..common import config, storage
from ..intraday.feed import NY
from ..intraday.store import conform
from .collector import _hhmm, nas_dir
from .rows import AGG_1S_SCHEMA, SORT_BY, SOURCE

log = logging.getLogger("massive_ws.compact")

LEVEL = 19
ROW_GROUP = 1_000_000


def day_dir(day: str) -> Path:
    return nas_dir() / f"date={day}" / f"source={SOURCE}"


def due_days(today: date, *, window_closed: bool) -> list[str]:
    """Day folders with more than one file (or one that is not yet a day file)."""
    out = []
    for d in sorted(nas_dir().glob(f"date=*/source={SOURCE}")):
        day = d.parent.name.removeprefix("date=")
        if day > today.isoformat() or (day == today.isoformat() and not window_closed):
            continue
        files = list(d.glob("*.parquet"))
        if len(files) > 1 or (files and files[0].name != f"day-{day}.parquet"):
            out.append(day)
    return out


def _dedup_sorted(t: pa.Table) -> pa.Table:
    n = t.num_rows
    if n < 2:
        return t
    sym, ts = t["symbol"].combine_chunks(), t["ts_utc"].combine_chunks()
    same = pc.and_(pc.equal(sym.slice(1), sym.slice(0, n - 1)), pc.equal(ts.slice(1), ts.slice(0, n - 1)))
    keep = np.concatenate([[True], ~same.to_numpy(zero_copy_only=False)])
    return t if keep.all() else t.filter(pa.array(keep))


def market_gap(t: pa.Table, day: str) -> tuple[int, str | None]:
    """Longest run of seconds in the regular session with no bar at all (a disconnect leaves one).

    Some bar arrives every second of the regular session across ~13k symbols, so a gap of
    more than a couple of seconds means the feed was not being received."""
    from ..utils.trading_calendar import is_trading_day, session_close_utc, session_open_utc

    d = date.fromisoformat(day)
    if not is_trading_day(d):
        return 0, None
    lo, hi = int(session_open_utc(d).timestamp()), int(session_close_utc(d).timestamp())
    s = np.unique(t["ts_utc"].combine_chunks().cast(pa.int64()).to_numpy() // 1000)
    s = s[(s >= lo) & (s < hi)]
    if len(s) == 0:
        return 0, None
    edges = np.concatenate([[lo - 1], s, [hi]])
    gaps = np.diff(edges) - 1
    i = int(gaps.argmax())
    if gaps[i] <= 0:
        return 0, None
    start = datetime.fromtimestamp(int(edges[i]) + 1, NY).strftime("%H:%M:%S")
    return int(gaps[i]), start


def compact_day(day: str, *, work: Path) -> dict | None:
    """Combine one day folder into day-D.parquet. Returns a summary, or None if nothing to do."""
    folder = day_dir(day)
    inputs = sorted(folder.glob("*.parquet"))
    target = folder / f"day-{day}.parquet"
    if not inputs or inputs == [target]:
        return None
    size_in = sum(p.stat().st_size for p in inputs)
    tables = [conform(pq.read_table(p), AGG_1S_SCHEMA) for p in inputs]  # v1 files: dec volumes null
    rows_in = sum(t.num_rows for t in tables)
    t = pa.concat_tables(tables).sort_by([(c, "ascending") for c in SORT_BY])
    del tables
    t = _dedup_sorted(t)

    work.mkdir(parents=True, exist_ok=True)
    local = work / target.name
    pq.write_table(t, local, compression="zstd", compression_level=LEVEL, row_group_size=ROW_GROUP)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(local, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    if pq.read_metadata(tmp).num_rows != t.num_rows:
        tmp.unlink(missing_ok=True)
        local.unlink(missing_ok=True)
        raise OSError(f"row-count mismatch writing {target}")
    os.replace(tmp, target)
    local.unlink()
    for p in inputs:
        if p != target:
            p.unlink()

    gap_s, gap_at = market_gap(t, day)
    return {"date": day, "files": len(inputs), "rows": t.num_rows, "duplicates": rows_in - t.num_rows,
            "symbols": pc.count_distinct(t["symbol"]).as_py(),
            "mb_before": round(size_in / 1e6, 1), "mb_after": round(target.stat().st_size / 1e6, 1),
            "longest_gap_s": gap_s, "gap_at_et": gap_at}


def compact(*, day: str | None = None, now: datetime | None = None) -> tuple[list[dict], list[str]]:
    """Combine every due day (or just `day`). Returns (summaries, errors)."""
    ready, reason = storage.data_root_ready()
    if not ready:
        return [], [f"data root not ready: {reason}"]
    now = (now or datetime.now(NY)).astimezone(NY)
    closed = now.time() >= _hhmm(config.massive_ws_config()["stop_et"])
    days = [day] if day else due_days(now.date(), window_closed=closed)
    work = config.state_root() / "massive" / "compact"
    done, errors = [], []
    for d in days:
        try:
            summary = compact_day(d, work=work)
        except Exception as exc:  # one bad day must not stop the others
            log.exception("compacting %s failed", d)
            errors.append(f"{d}: {type(exc).__name__}: {exc}")
            continue
        if summary:
            done.append(summary)
    return done, errors
