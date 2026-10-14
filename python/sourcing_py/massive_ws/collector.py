"""`sourcing-py massive-ws follow`: Massive per-second aggregates for all US stocks.

A process of its own (started each session by mtap-schedule.sh), apart from the E*TRADE collector and
its Redis/storage pipeline; it shares only the NAS mount. Every bar is kept:

1. The session thread (client.py) queues raw frames. This loop parses them into the
   agg_1s columns and every `flush_s` appends them to the open local segment
   `STATE_ROOT/segments_massive/massive_ws/date=<D>/h<HH>-m<MM>-<opened>.arrows`, fsynced.
2. Segments roll every `roll_min` minutes of bar time. Closing one sorts it by
   (symbol, ts_utc) into zstd parquet (intraday/store.py arrows_to_parquet).
3. The uploader thread (intraday/upload.py) moves closed parquet to
   `DATA_ROOT/agg_1s/date=<D>/source=massive_ws/` and deletes the local copy once the
   NAS copy has the same row count. So the local footprint is a few segments.
4. Guard: while unuploaded segments exceed `max_local_gb` or the disk has less than
   `min_free_gb` free, the collector disconnects (bars are lost, nothing on disk is
   deleted) and resumes once the uploads catch up.

It connects only inside the New York window [start_et, stop_et) on trading days, and
holds `caffeinate -i` while it does. Started before the window it waits; with
`until_close` it exits once the day's window is over, after closing and uploading its
last segment. Bars missed while disconnected are not replayed.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from datetime import time as dtime
from pathlib import Path

from ..common import config
from ..intraday.feed import NY
from ..intraday.store import Segment, _singleton_lock, recover
from ..intraday.upload import Uploader, pending
from .client import FeedError, Session, _connect
from .rows import AGG_1S_SCHEMA, SORT_BY, SOURCE, Columns

log = logging.getLogger("massive_ws")

QUEUE_FRAMES = 200_000  # about a minute at the open; the loop normally keeps it near 0


def segments_root() -> Path:
    return config.state_root() / "segments_massive"


def nas_dir() -> Path:
    return config.data_root() / "agg_1s"


def heartbeat_path() -> Path:
    return config.state_root() / "massive" / "collector.heartbeat"


def lock_path() -> Path:
    return config.state_root() / "massive" / "collector.lock"


def _hhmm(value: str) -> dtime:
    h, m = value.split(":")
    return dtime(int(h), int(m))


def in_window(now: datetime, cfg: dict) -> bool:
    """Inside [start_et, stop_et) New York time on an exchange trading day."""
    from ..utils.trading_calendar import is_trading_day

    local = now.astimezone(NY)
    if not (_hhmm(cfg["start_et"]) <= local.time() < _hhmm(cfg["stop_et"])):
        return False
    return is_trading_day(local.date())


def day_over(now: datetime, cfg: dict) -> bool:
    """Nothing more to collect today: past stop_et New York time, or not a trading day."""
    from ..utils.trading_calendar import is_trading_day

    local = now.astimezone(NY)
    return local.time() >= _hhmm(cfg["stop_et"]) or not is_trading_day(local.date())


def slot(ts_ms: int, roll_min: int) -> tuple[str, int, str]:
    """(NY date, NY hour, file label) of the segment a bar starting at ts_ms belongs to."""
    local = datetime.fromtimestamp(ts_ms / 1000, UTC).astimezone(NY)
    minute = local.minute - local.minute % roll_min
    return local.date().isoformat(), local.hour, f"h{local.hour:02d}-m{minute:02d}"


class Collector:
    def __init__(self, cfg: dict, *, root: Path, key: str, connect=_connect, uploader=None):
        self.cfg, self.root, self.key = cfg, root, key
        self._connect = connect
        self.uploader = uploader
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self.cols = Columns()
        self.session: Session | None = None
        self.seg: Segment | None = None
        self.seg_key: tuple[str, str] | None = None
        self.paused = False  # local guard tripped
        self.next_connect = 0.0
        self.backoff = 1.0
        self._caffeinate: subprocess.Popen | None = None
        self._last_flush = time.monotonic()
        self.stats = {
            "frames": 0, "bars": 0, "bad_messages": 0, "dropped_frames": 0, "disconnects": 0,
            "segments_closed": 0, "last_bar_utc": None, "last_status": None, "last_error": None,
            "connected_since": None,
        }
        self._rate = (time.monotonic(), 0)  # for bars/s between heartbeats

    # -- the session -------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.session is not None and self.session.is_alive()

    def connect(self) -> None:
        self.session = Session(self.cfg["ws_url"], self.key, self.cfg["channels"], self.q,
                               connect=self._connect)
        self.session.start()
        self._hold_awake(True)

    def disconnect(self) -> None:
        if self.session is not None:
            self.session.stop()
            self._session_ended(expected=True)
        self._hold_awake(False)

    def _session_ended(self, *, expected: bool) -> None:
        s, self.session = self.session, None
        self.stats["dropped_frames"] += s.dropped
        self.stats["connected_since"] = None
        if expected:
            return
        self.stats["disconnects"] += 1
        err = s.error
        self.stats["last_error"] = f"{type(err).__name__}: {err}" if err else "closed by server"
        if isinstance(err, FeedError) and err.status == "auth_failed":
            wait = 600.0  # a bad key will not fix itself; don't hammer the server
        elif isinstance(err, FeedError) and err.status == "max_connections":
            wait = 60.0  # another client holds this key's stream
        else:
            if s.subscribed_at and time.monotonic() - s.subscribed_at > 60:
                self.backoff = 1.0
            wait, self.backoff = self.backoff, min(self.backoff * 2, 60.0)
        self.next_connect = time.monotonic() + wait
        log.warning("disconnected (%s); reconnecting in %.0f s", self.stats["last_error"], wait)

    def _hold_awake(self, on: bool) -> None:
        if on and self._caffeinate is None and sys.platform == "darwin" and shutil.which("caffeinate"):
            self._caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        elif not on and self._caffeinate is not None:
            self._caffeinate.terminate()
            self._caffeinate = None

    # -- intake ------------------------------------------------------------------------
    def drain(self, max_frames: int = 5000, wait_s: float = 0.2) -> int:
        """Parse queued frames into the column buffer. Returns frames handled."""
        n = 0
        try:
            raw = self.q.get(timeout=wait_s)
        except queue.Empty:
            return 0
        while True:
            self._parse(raw)
            n += 1
            if n >= max_frames:
                return n
            try:
                raw = self.q.get_nowait()
            except queue.Empty:
                return n

    def _parse(self, raw: str | bytes) -> None:
        self.stats["frames"] += 1
        try:
            msgs = json.loads(raw)
        except ValueError:
            self.stats["bad_messages"] += 1
            return
        for msg in msgs if isinstance(msgs, list) else [msgs]:
            ev = msg.get("ev") if isinstance(msg, dict) else None
            if ev == "A":
                try:
                    self.cols.add(msg)
                    self.stats["bars"] += 1
                except (KeyError, TypeError, ValueError):
                    self.stats["bad_messages"] += 1
            elif ev == "status":
                self.stats["last_status"] = f"{msg.get('status')}: {msg.get('message')}"
                log.info("status %s", self.stats["last_status"])
            else:
                self.stats["bad_messages"] += 1

    # -- segments ----------------------------------------------------------------------
    def due(self) -> bool:
        if not self.cols.n:
            return False
        return (self.cols.n >= self.cfg["flush_rows"]
                or time.monotonic() - self._last_flush >= self.cfg["flush_s"])

    def flush(self) -> None:
        """Append the buffered bars to their segment (fsynced), rolling it if needed."""
        self._last_flush = time.monotonic()
        if not self.cols.n:
            return
        date, hour, label = slot(self.cols.max_s, self.cfg["roll_min"])
        if self.seg is not None and self.seg_key != (date, label):
            self.close_segment()
        if self.seg is None:
            self.seg = Segment.open(self.root, SOURCE, date, hour, label=label,
                                    schema=AGG_1S_SCHEMA, sort_by=SORT_BY)
            self.seg_key = (date, label)
        self.seg.write(self.cols.table())
        self.stats["last_bar_utc"] = datetime.fromtimestamp(self.cols.max_s / 1000, UTC).isoformat()
        self.cols.reset()

    def close_segment(self) -> None:
        if self.seg is None:
            return
        seg, self.seg, self.seg_key = self.seg, None, None
        out = seg.close()
        self.stats["segments_closed"] += 1
        log.info("closed segment %s (%d rows)", out.name if out else seg.path.name, seg.rows)
        if out and self.uploader:
            self.uploader.kick()

    # -- the local guard ---------------------------------------------------------------
    def check_guard(self) -> None:
        local = pending(self.root)["bytes"] if self.root.exists() else 0
        free = shutil.disk_usage(self.root if self.root.exists() else config.state_root()).free
        cap, floor = self.cfg["max_local_gb"] * 1e9, self.cfg["min_free_gb"] * 1e9
        if not self.paused and (local > cap or free < floor):
            self.paused = True
            log.error("local guard: %.1f GB unuploaded, %.1f GB free; disconnecting until the "
                      "uploads catch up (is the NAS down?)", local / 1e9, free / 1e9)
        elif self.paused and local < 0.8 * cap and free >= floor:
            self.paused = False
            log.warning("local guard cleared (%.1f GB unuploaded); resuming", local / 1e9)

    # -- the loop ----------------------------------------------------------------------
    def tick(self, *, want: bool) -> None:
        """One pass: (dis)connect as wanted, take in frames, flush."""
        if self.session is not None and not self.session.is_alive():
            self._session_ended(expected=False)
        if want and not self.paused:
            if self.session is None and time.monotonic() >= self.next_connect:
                self.connect()
        elif self.session is not None or self._caffeinate is not None:
            self.disconnect()
        if self.session is not None and self.session.state == "subscribed" and not self.stats["connected_since"]:
            self.stats["connected_since"] = datetime.now(UTC).isoformat()
        self.drain()
        if self.due():
            self.flush()
        stopping = not want or self.paused  # (a reconnect keeps the segment open)
        if stopping and self.session is None and self.q.empty() and (self.cols.n or self.seg is not None):
            self.flush()
            self.close_segment()

    def state(self, want: bool) -> str:
        if self.paused:
            return "paused"
        if self.connected:
            return "collecting" if self.session.state == "subscribed" else "connecting"
        if want:
            return "reconnecting"
        return "idle" if self.seg is None else "closing"

    def heartbeat(self, path: Path, want: bool, extra: dict | None = None) -> None:
        now = time.monotonic()
        t0, bars0 = self._rate
        self._rate = (now, self.stats["bars"])
        last = self.stats["last_bar_utc"]
        body = {
            "updated_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "state": self.state(want),
            "window": {"start_et": self.cfg["start_et"], "stop_et": self.cfg["stop_et"], "open": want},
            **self.stats,
            "dropped_frames": self.stats["dropped_frames"] + (self.session.dropped if self.session else 0),
            "bars_per_s": round((self.stats["bars"] - bars0) / max(now - t0, 1e-9), 1),
            "delay_s": round(time.time() - datetime.fromisoformat(last).timestamp(), 1) if last else None,
            "queue_frames": self.q.qsize(),
            "open_segment": {"path": str(self.seg.path), "rows": self.seg.rows} if self.seg else None,
            "pending_upload": self.uploader.pending_summary() if self.uploader else pending(self.root),
            **(extra or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=2, default=str))
        os.replace(tmp, path)


def follow(*, stop: threading.Event | None = None, upload: bool = True, window: bool = True,
           duration_s: float | None = None, until_close: bool = False, connect=_connect) -> int:
    """Run until SIGTERM/SIGINT, `stop`, `duration_s`, or (`until_close`) the end of the
    day's window. Closes and uploads on the way out."""
    cfg = config.massive_ws_config()
    key = config.massive_config()["api_key"]
    lock = _singleton_lock(lock_path(), "massive-ws follow")
    stop = stop or threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:  # not the main thread (tests)
            pass

    root = segments_root()
    recover(root, schema=AGG_1S_SCHEMA, sort_by=SORT_BY)
    uploader = Uploader(root, retry_s=cfg["upload_retry_s"], nas=nas_dir()) if upload else None
    if uploader:
        uploader.start()
    c = Collector(cfg, root=root, key=key, connect=connect, uploader=uploader)
    hb = heartbeat_path()
    started = time.monotonic()
    last_hb = 0.0
    log.info("massive-ws collector up: %s %s, window %s-%s ET%s, segments %s",
             cfg["ws_url"], cfg["channels"], cfg["start_et"], cfg["stop_et"],
             "" if window else " (ignored)", root)
    try:
        while not stop.is_set():
            if duration_s is not None and time.monotonic() - started >= duration_s:
                break
            now = datetime.now(UTC)
            if until_close and day_over(now, cfg):
                log.info("the window is over for today; stopping")
                break
            want = not window or in_window(now, cfg)
            c.tick(want=want)
            if time.monotonic() - last_hb >= 10:
                last_hb = time.monotonic()
                c.check_guard()
                c.heartbeat(hb, want)
    finally:
        log.info("shutting down: closing the open segment")
        try:
            c.disconnect()
            while c.drain(wait_s=0):
                pass
            c.flush()
            c.close_segment()
        finally:
            if uploader:
                uploader.stop(drain=True)
            c.heartbeat(hb, False, {"stopped": True})
            lock.close()
    return 0
