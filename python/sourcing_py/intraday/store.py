"""`sourcing-py intraday store --follow`: the durable storage consumer.

Reads every INTRADAY_FEED batch from Redis (consumer group `storage`) and keeps every
row, at full resolution, in hourly segments:

1. Rows are appended to an open local segment, an Arrow IPC stream file
   `STATE_ROOT/segments/<source>/date=<D>/h<HH>-<opened>.arrows`, and fsynced. Only
   then are the stream entries acked, so an acked row is always on disk. (A parquet file
   is unreadable until it is closed; an IPC stream is readable up to its last batch.)
2. A segment closes when the New York hour changes, after `idle_close_s` without rows,
   or on shutdown. Closing converts it to zstd parquet next to it and checks the row
   count; a crash leaves the .arrows file, which the next start recovers the same way.
3. The uploader thread (upload.py) moves closed parquet to the NAS and deletes the
   local copy once the NAS copy reads back with the same row count.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from ..common import config
from . import feed
from .upload import Uploader

log = logging.getLogger("intraday.store")

EVENT = "INTRADAY_FEED"
GROUP = "storage"
CONSUMER = "storage-1"  # stable, so a restart re-reads its own unacked entries


def segments_root() -> Path:
    return config.state_root() / "segments"


def arrows_to_parquet(arrows: Path) -> Path | None:
    """Convert a (possibly truncated) IPC stream segment to zstd parquet; delete it.

    Returns the parquet path, or None if the segment held no complete batch.
    """
    batches: list[pa.RecordBatch] = []
    try:
        with pa.OSFile(str(arrows), "rb") as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batches.append(reader.read_next_batch())
                except StopIteration:
                    break
    except (pa.ArrowInvalid, OSError) as exc:
        # A crash mid-write truncates the last batch; everything before it is intact.
        log.warning("segment %s truncated (%s); keeping %d complete batch(es)", arrows, exc, len(batches))
    if not batches:
        arrows.unlink(missing_ok=True)
        return None
    table = pa.Table.from_batches(batches, schema=feed.ARROW_SCHEMA)
    out = arrows.with_suffix(".parquet")
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    if pq.read_metadata(tmp).num_rows != table.num_rows:
        raise OSError(f"row-count mismatch converting {arrows}")
    os.replace(tmp, out)
    arrows.unlink()
    return out


def recover(root: Path) -> list[Path]:
    """Close segments a crashed run left open."""
    done = []
    for arrows in sorted(root.rglob("*.arrows")):
        out = arrows_to_parquet(arrows)
        log.info("recovered %s -> %s", arrows.name, out.name if out else "(empty, removed)")
        if out:
            done.append(out)
    return done


@dataclass
class Segment:
    """One open hourly segment of one source."""

    source: str
    date: str
    hour: int
    path: Path
    rows: int = 0
    last_write: float = field(default_factory=time.monotonic)
    _sink: pa.OSFile | None = None
    _writer: ipc.RecordBatchStreamWriter | None = None

    @classmethod
    def open(cls, root: Path, source: str, date: str, hour: int) -> "Segment":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = root / source / f"date={date}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"h{hour:02d}-{stamp}.arrows"
        n = 1
        while path.exists() or path.with_suffix(".parquet").exists():
            path = d / f"h{hour:02d}-{stamp}-{n}.arrows"
            n += 1
        seg = cls(source, date, hour, path)
        seg._sink = pa.OSFile(str(path), "wb")
        seg._writer = ipc.new_stream(seg._sink, feed.ARROW_SCHEMA)
        return seg

    @property
    def key(self) -> tuple[str, int]:
        return self.date, self.hour

    def write(self, table: pa.Table) -> None:
        assert self._writer is not None and self._sink is not None
        self._writer.write_table(table)
        self._sink.flush()
        os.fsync(self._sink.fileno())
        self.rows += table.num_rows
        self.last_write = time.monotonic()

    def close(self) -> Path | None:
        if self._writer is not None:
            self._writer.close()
            self._sink.close()
            self._writer = self._sink = None
        return arrows_to_parquet(self.path)


class Store:
    """Buffers stream entries, writes them to segments, acks after fsync."""

    def __init__(self, consumer, *, root: Path | None = None, cfg: dict | None = None, on_close=None):
        self.consumer = consumer
        self.on_close = on_close  # called after a segment becomes parquet (wakes the uploader)
        self.root = root or segments_root()
        self.cfg = cfg or config.intraday_config()
        self.open: dict[str, Segment] = {}  # source -> its open segment
        self._buf: dict[tuple[str, str, int], list[pa.Table]] = {}
        self._buf_rows = 0
        self._buf_since: float | None = None
        self._ids: list[str] = []
        self._ack_retry: list[str] = []  # written to disk, ack failed (Redis down)
        self.stats = {
            "entries": 0, "rows": 0, "bad_entries": 0, "trimmed_entries": 0,
            "segments_closed": 0, "last_ts_utc": None,
        }

    # -- intake ------------------------------------------------------------------------
    def add(self, entry_id: str, payload: dict) -> None:
        self._ids.append(entry_id)  # acked after the flush, good or bad
        if payload.get("_trimmed"):
            # Trimmed from the stream (MAXLEN) before we wrote it: the data is gone.
            self.stats["trimmed_entries"] += 1
            return
        rows = payload.get("rows")
        if payload.get("schema_version") != feed.SCHEMA_VERSION or not isinstance(rows, list):
            self.stats["bad_entries"] += 1
            log.warning("skipping entry %s: not an intraday_feed v1 batch", entry_id)
            return
        if not rows:
            return
        try:
            ts = feed.parse_ts(payload.get("ts_utc") or rows[0]["ts_utc"])
            source = payload.get("source") or rows[0]["source"]
            # Built per entry, so one bad entry is skipped here instead of failing the
            # whole flush (and the restart re-reading it forever).
            table = feed.to_table(feed.rows_to_columns(rows, entry_id))
        except (KeyError, TypeError, ValueError, pa.ArrowException) as exc:
            self.stats["bad_entries"] += 1
            log.warning("skipping malformed entry %s: %s", entry_id, exc)
            return
        date, hour = feed.partition(ts)
        self._buf.setdefault((source, date, hour), []).append(table)
        self._buf_rows += table.num_rows
        self._buf_since = self._buf_since or time.monotonic()
        self.stats["entries"] += 1
        self.stats["last_ts_utc"] = ts.isoformat()

    def due(self) -> bool:
        if self._ack_retry:
            return True
        if not self._ids:
            return False
        if self._buf_rows >= self.cfg["flush_rows"]:
            return True
        return self._buf_since is None or time.monotonic() - self._buf_since >= self.cfg["flush_s"]

    # -- segments ----------------------------------------------------------------------
    def flush(self) -> None:
        """Write buffered rows to their segments (fsynced), then ack the entries."""
        for (source, date, hour), tables in sorted(self._buf.items(), key=lambda kv: kv[0]):
            seg = self.open.get(source)
            if seg is not None and seg.key != (date, hour):
                self._close(source)
                seg = None
            if seg is None:
                seg = self.open[source] = Segment.open(self.root, source, date, hour)
                log.info("opened segment %s", seg.path)
            table = pa.concat_tables(tables)
            seg.write(table)
            self.stats["rows"] += table.num_rows
        ids = self._ack_retry + self._ids
        self.discard_buffer()
        try:
            self.consumer.ack(ids)
            self._ack_retry = []
        except Exception as exc:  # Redis went away: the rows are on disk, ack them later
            self._ack_retry = ids
            log.error("ack of %d entries failed (%s); retrying on the next flush", len(ids), exc)

    def discard_buffer(self) -> None:
        """Drop unflushed rows without acking them; Redis redelivers them."""
        self._buf.clear()
        self._ids = []
        self._buf_rows = 0
        self._buf_since = None

    def _close(self, source: str) -> None:
        seg = self.open.pop(source)
        out = seg.close()
        self.stats["segments_closed"] += 1
        log.info("closed segment %s (%d rows)", out.name if out else seg.path.name, seg.rows)
        if out and self.on_close:
            self.on_close(out)

    def close_idle(self) -> None:
        idle = self.cfg["idle_close_s"]
        for source, seg in list(self.open.items()):
            if time.monotonic() - seg.last_write >= idle:
                self._close(source)

    def close_all(self) -> None:
        if self._ids or self._ack_retry:
            self.flush()
        for source in list(self.open):
            self._close(source)

    def heartbeat(self, path: Path, extra: dict) -> None:
        body = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            **self.stats,
            "open_segments": {s: {"path": str(g.path), "rows": g.rows} for s, g in self.open.items()},
            **extra,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=2, default=str))
        os.replace(tmp, path)


def _singleton_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit(f"another `intraday store` is running (lock {path})")
    return fh


def follow(*, stop: threading.Event | None = None, consumer=None, upload: bool = True) -> int:
    """Run until SIGTERM/SIGINT (or `stop` is set). Closes and uploads on the way out."""
    cfg = config.intraday_config()
    state = config.state_root()
    lock = _singleton_lock(state / "intraday" / "store.lock")
    stop = stop or threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:  # not the main thread (tests)
            pass

    root = segments_root()
    recover(root)
    if consumer is None:
        from ..common.bus import StreamConsumer

        consumer = StreamConsumer(EVENT, GROUP, CONSUMER, cfg["redis_url"])
    uploader = Uploader(root, retry_s=cfg["upload_retry_s"]) if upload else None
    store = Store(consumer, root=root, cfg=cfg, on_close=(lambda _p: uploader.kick()) if uploader else None)
    if uploader:
        uploader.start()

    hb_path = state / "intraday" / "store.heartbeat"
    last_hb = 0.0
    redis_down_since: float | None = None
    log.info("storage consumer up: %s group=%s segments=%s", EVENT, GROUP, root)
    try:
        while not stop.is_set():  # at boot Redis may come up after us
            try:
                consumer.ensure_group()
                for entry_id, payload in consumer.pending():
                    store.add(entry_id, payload)
                if store.stats["trimmed_entries"]:
                    log.error(
                        "%d pending entries were trimmed from the stream before they were stored "
                        "(the store was down longer than the stream window); that data is lost",
                        store.stats["trimmed_entries"],
                    )
                break
            except Exception as exc:
                store.discard_buffer()  # pending() is re-read from the start next time
                if redis_down_since is None:
                    redis_down_since = time.monotonic()
                    log.error("redis not reachable: %s (retrying)", exc)
                stop.wait(2)
        while not stop.is_set():
            try:
                entries = consumer.read(count=50, block_ms=1000)
                if redis_down_since is not None:
                    log.warning("redis back after %.0f s", time.monotonic() - redis_down_since)
                    redis_down_since = None
            except Exception as exc:  # redis.ConnectionError and friends
                if "NOGROUP" in str(exc):
                    # Redis restarted (it keeps nothing on disk): the stream and group are
                    # gone. Re-create them and carry on; the collector re-creates the stream.
                    log.warning("consumer group missing (Redis restarted?); re-creating it")
                    try:
                        consumer.ensure_group()
                        continue
                    except Exception:
                        pass
                if redis_down_since is None:
                    redis_down_since = time.monotonic()
                    log.error("redis read failed: %s (retrying)", exc)
                stop.wait(2)
                continue
            for entry_id, payload in entries:
                store.add(entry_id, payload)
            if store.due():
                store.flush()
            store.close_idle()
            if time.monotonic() - last_hb >= 10:
                last_hb = time.monotonic()
                extra = {"pending_upload": uploader.pending_summary() if uploader else None}
                try:
                    extra["group"] = consumer.lag()
                except Exception:
                    pass
                store.heartbeat(hb_path, extra)
    finally:
        log.info("shutting down: closing open segments")
        try:
            store.close_all()
        finally:
            if uploader:
                uploader.stop(drain=True)
            store.heartbeat(hb_path, {"stopped": True})
            lock.close()
    return 0
