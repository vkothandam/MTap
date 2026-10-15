"""Move closed feed segments from the local state root to the NAS.

Local:  STATE_ROOT/segments/<source>/date=<D>/<name>.parquet
NAS:    DATA_ROOT/intraday_feed/date=<D>/source=<source>/<name>.parquet

Each file is copied to a .tmp name, read back (the row count must match), renamed into
place, read back again with the share re-checked, and only then deleted locally. Any
failure leaves the local file where it is.

Nothing is written while the data root is not ready (share unmounted, marker missing).
When the share is not mounted and MTAP_NAS_URL is set, a pass first runs
scripts/storage/mount-nas.sh. The data root is re-checked before every file, so a share
that drops mid-pass stops the pass instead of failing (or writing) file after file.

Every segment a pass could not move is logged in state/failures/upload_<root name>.jsonl
(upload_segments for the E*TRADE feed, upload_segments_massive for Massive), one line per
file: {"file", "error", "attempts", "first_at", "at"}. Each pass retries the logged files
first and rewrites the log with what is still local, so it drains as uploads succeed.
The uploader thread runs a pass every `upload_retry_s` and after each segment closes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from ..common import config, failures, storage
from . import feed

log = logging.getLogger("intraday.upload")

MOUNT_TIMEOUT_S = 300  # mount-nas.sh makes up to 3 attempts of ~100 s each


def nas_dir() -> Path:
    return config.data_root() / "intraday_feed"


def destination(local: Path, root: Path, nas: Path | None = None) -> Path:
    """The NAS path for a local segment under `root` (<source>/date=<D>/<name>).

    `nas` is the NAS folder (default DATA_ROOT/intraday_feed)."""
    source, date_dir = local.relative_to(root).parts[:2]
    return (nas or nas_dir()) / date_dir / f"source={source}" / local.name


def upload_one(local: Path, root: Path, nas: Path | None = None) -> Path:
    """Copy, verify, rename, verify, delete local. Returns the NAS path."""
    rows = pq.read_metadata(local).num_rows
    dest = destination(local, root, nas)
    if dest.exists():
        if pq.read_metadata(dest).num_rows == rows:
            # Uploaded before a crash, local copy not yet deleted.
            _delete_local(local, dest, rows)
            return dest
        n = 1
        while dest.with_name(f"{dest.stem}-dup{n}.parquet").exists():
            n += 1
        dest = dest.with_name(f"{dest.stem}-dup{n}.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    try:
        shutil.copyfile(local, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        if pq.read_metadata(tmp).num_rows != rows:
            raise OSError(f"row-count mismatch uploading {local}")
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass  # the share is gone; the next upload overwrites the .tmp
        raise
    _delete_local(local, dest, rows)
    return dest


def _delete_local(local: Path, dest: Path, rows: int) -> None:
    """Delete the local segment only once the NAS copy is complete and really on the NAS."""
    ready, reason = storage.data_root_ready()
    if not ready:
        # The share dropped after the copy: `dest` may be on the local disk under /Volumes.
        raise OSError(f"data root lost before deleting the local copy: {reason}")
    if pq.read_metadata(dest).num_rows != rows:
        raise OSError(f"row-count mismatch reading back {dest}")
    local.unlink()


def closed_segments(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def remount() -> tuple[bool, str]:
    """Run scripts/storage/mount-nas.sh. (mounted, last line of its output)."""
    script = config.repo_root() / "scripts" / "storage" / "mount-nas.sh"
    try:
        res = subprocess.run([str(script)], capture_output=True, text=True, timeout=MOUNT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    out = (res.stdout + res.stderr).strip().splitlines()
    return res.returncode == 0, out[-1] if out else f"exit {res.returncode}"


def _ready(*, mount: bool) -> tuple[bool, str]:
    ready, reason = storage.data_root_ready()
    if ready or not mount or not config.storage_config()["nas_url"] or "not mounted" not in reason:
        return ready, reason
    log.warning("%s; remounting", reason)
    ok, out = remount()
    log.log(logging.INFO if ok else logging.ERROR, "mount-nas.sh: %s", out)
    return storage.data_root_ready()


def _relpath(local: Path, root: Path) -> str:
    return local.relative_to(root).as_posix()


def failure_log(root: Path) -> Path:
    """One log per segments root: state/failures/upload_<root name>.jsonl."""
    return failures.log_path(f"upload_{root.name}")


def logged_failures(root: Path) -> dict[str, dict]:
    """The failure log of `root`, keyed by segment path relative to it."""
    path = failure_log(root)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entry = json.loads(line)
            out[entry["file"]] = entry
    return out


def _note_failure(logged: dict[str, dict], rel: str, error: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    prev = logged.get(rel, {})
    logged[rel] = {"file": rel, "error": error, "attempts": prev.get("attempts", 0) + 1,
                   "first_at": prev.get("first_at", now), "at": now}


def upload_all(root: Path, nas: Path | None = None, *, mount: bool = True) -> tuple[int, int, str | None]:
    """One pass: (uploaded, failed, reason-if-not-ready). Logged failures go first."""
    logged = logged_failures(root)
    local = closed_segments(root)
    queue = sorted(local, key=lambda p: (_relpath(p, root) not in logged, p))
    logged = {k: v for k, v in logged.items() if (root / k).exists()}
    done = failed = 0
    reason = None
    try:
        ready, why = _ready(mount=mount)
        for seg in queue:
            if ready:
                ready, why = storage.data_root_ready()  # the share can drop mid-pass
            if not ready:
                reason = why
                _note_failure(logged, _relpath(seg, root), f"not uploaded: {why}")
                continue
            try:
                dest = upload_one(seg, root, nas)
            except Exception as exc:  # pyarrow raises more than OSError on a half-written file
                failed += 1
                _note_failure(logged, _relpath(seg, root), str(exc))
                log.error("upload of %s failed: %s (kept locally, retried next pass)", seg, exc)
                continue
            done += 1
            logged.pop(_relpath(seg, root), None)
            log.info("uploaded %s -> %s", seg.name, dest)
    finally:
        failures.rewrite(f"upload_{root.name}", list(logged.values()))
    _prune_empty_dirs(root)
    return done, failed, reason


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

    def __init__(self, root: Path, *, retry_s: float = 300, nas: Path | None = None):
        super().__init__(name="intraday-uploader", daemon=True)
        self.root = root
        self.nas = nas
        self.retry_s = retry_s
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._last_reason: str | None = None
        self._lock = threading.Lock()

    def kick(self) -> None:
        self._wake.set()

    def _pass(self, *, mount: bool = True) -> None:
        with self._lock:
            _done, _failed, reason = upload_all(self.root, self.nas, mount=mount)
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
                # No remount on the way out: it can take minutes, and the next start retries.
                self._pass(mount=False)
            except Exception:
                log.exception("final upload pass failed")

    def pending_summary(self) -> dict:
        return {**pending(self.root), "blocked": self._last_reason,
                "logged_failures": len(logged_failures(self.root))}
