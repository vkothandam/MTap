"""Move closed feed segments from the local state root to the NAS.

Local:  STATE_ROOT/segments/<source>/date=<D>/<name>.parquet
NAS:    DATA_ROOT/intraday_feed/date=<D>/source=<source>/<name>.parquet

Each file is copied to a .tmp name, read back (the row count must match), renamed into
place, and only then deleted locally. Nothing is written while the data root is not
ready (share unmounted, marker missing); the segments wait locally and the uploader
retries every `upload_retry_s`.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from ..common import config, storage
from . import feed

log = logging.getLogger("intraday.upload")


def nas_dir() -> Path:
    return config.data_root() / "intraday_feed"


def destination(local: Path, root: Path) -> Path:
    """The NAS path for a local segment under `root` (<source>/date=<D>/<name>)."""
    source, date_dir = local.relative_to(root).parts[:2]
    return nas_dir() / date_dir / f"source={source}" / local.name


def upload_one(local: Path, root: Path) -> Path:
    """Copy, verify, rename, delete local. Returns the NAS path."""
    rows = pq.read_metadata(local).num_rows
    dest = destination(local, root)
    if dest.exists():
        if pq.read_metadata(dest).num_rows == rows:
            # Uploaded before a crash, local copy not yet deleted.
            local.unlink()
            return dest
        n = 1
        while dest.with_name(f"{dest.stem}-dup{n}.parquet").exists():
            n += 1
        dest = dest.with_name(f"{dest.stem}-dup{n}.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    shutil.copyfile(local, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    if pq.read_metadata(tmp).num_rows != rows:
        tmp.unlink(missing_ok=True)
        raise OSError(f"row-count mismatch uploading {local}")
    os.replace(tmp, dest)
    local.unlink()
    return dest


def closed_segments(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def upload_all(root: Path) -> tuple[int, int, str | None]:
    """One pass: (uploaded, failed, reason-if-not-ready)."""
    ready, reason = storage.data_root_ready()
    if not ready:
        return 0, 0, reason
    done = failed = 0
    for local in closed_segments(root):
        try:
            dest = upload_one(local, root)
            done += 1
            log.info("uploaded %s -> %s", local.name, dest)
        except OSError as exc:
            failed += 1
            log.error("upload of %s failed: %s", local, exc)
    _prune_empty_dirs(root)
    return done, failed, None


def _prune_empty_dirs(root: Path) -> None:
    """Remove emptied date=<D> folders of past days (today's may be about to get a segment)."""
    today = f"date={datetime.now(feed.NY).date().isoformat()}"
    for d in root.glob("*/date=*"):
        if d.is_dir() and d.name != today:
            try:
                d.rmdir()
            except OSError:
                pass


def pending(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".parquet", ".arrows")]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


class Uploader(threading.Thread):
    """Background uploader. `kick()` after a segment closes; retries on a timer."""

    def __init__(self, root: Path, *, retry_s: float = 300):
        super().__init__(name="intraday-uploader", daemon=True)
        self.root = root
        self.retry_s = retry_s
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._last_reason: str | None = None
        self._lock = threading.Lock()

    def kick(self) -> None:
        self._wake.set()

    def _pass(self) -> None:
        with self._lock:
            _done, _failed, reason = upload_all(self.root)
        if reason != self._last_reason:
            if reason:
                log.warning("not uploading: %s (segments stay local, retry every %.0f s)", reason, self.retry_s)
            elif self._last_reason:
                log.warning("data root ready again; uploading")
            self._last_reason = reason

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                self._pass()
            except Exception:  # never let the thread die; the next pass retries
                log.exception("upload pass failed")
            self._wake.wait(self.retry_s)
            self._wake.clear()

    def stop(self, *, drain: bool = True) -> None:
        self._halt.set()
        self._wake.set()
        if self.is_alive():
            self.join(timeout=30)
        if self.is_alive():
            # Stuck in NAS I/O (a stale SMB mount after sleep). Don't wait on it: the
            # segments are safe locally and the next start uploads them.
            log.error("uploader still busy after 30 s (NAS unresponsive?); leaving segments local")
            return
        if drain:
            try:
                self._pass()
            except Exception:
                log.exception("final upload pass failed")

    def pending_summary(self) -> dict:
        return {**pending(self.root), "blocked": self._last_reason}
