"""CLI for the Massive per-second collector.

Dispatched from `sourcing_py.__main__` when the first arg is `massive-ws`:

    sourcing-py massive-ws follow [--until-close] [--duration S] [--no-window] [--no-upload]
    sourcing-py massive-ws status
    sourcing-py massive-ws upload        (when the collector is not running)
    sourcing-py massive-ws wait-idle [--timeout S]   (the old always-on agent)
    sourcing-py massive-ws compact [--date D]        (the evening shutdown: one file per day)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime

from ..intraday.cli import _parse

_USAGE = (
    "usage:\n"
    "  sourcing-py massive-ws follow [--until-close] [--duration S] [--no-window] [--no-upload]\n"
    "      (--until-close: mtap-schedule.sh start-massive; the rest for a sample run)\n"
    "  sourcing-py massive-ws status   (collector heartbeat, local segments)\n"
    "  sourcing-py massive-ws upload   (upload closed local segments now)\n"
    "  sourcing-py massive-ws wait-idle [--timeout 1200]   (old always-on agent: exit 0 once idle and uploaded)\n"
    "  sourcing-py massive-ws compact [--date D]   (combine each day's NAS files into day-D.parquet)"
)


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _heartbeat() -> tuple[dict | None, float | None]:
    from .collector import heartbeat_path

    hb = heartbeat_path()
    if not hb.exists():
        return None, None
    body = json.loads(hb.read_text())
    age = (datetime.now().astimezone() - datetime.fromisoformat(body["updated_at"])).total_seconds()
    return body, age


def _cmd_follow(params: dict) -> int:
    _logging()
    from .collector import follow, nas_dir, segments_root

    duration = float(params["duration"]) if params.get("duration") not in (None, True) else None
    started = time.time()
    rc = follow(upload=not params.get("no-upload"), window=not params.get("no-window"), duration_s=duration,
                until_close=bool(params.get("until-close")))
    if duration is not None:
        import pyarrow.parquet as pq

        # This run's files, still local or already on the NAS.
        files = [p for root in (segments_root(), nas_dir()) if root.exists()
                 for p in root.rglob("*.parquet") if p.stat().st_mtime >= started]
        rows = sum(pq.read_metadata(f).num_rows for f in files)
        size = sum(f.stat().st_size for f in files)
        body, _ = _heartbeat()
        print(f"sample: {duration:.0f} s, {body and body.get('bars')} bars received, "
              f"{rows} rows in {len(files)} parquet file(s), {size / 1e6:.1f} MB"
              + (f", {size / rows:.1f} B/row, {rows / duration * 3600 / 1e6:.2f} M rows/h, "
                 f"{size / duration * 3600 / 1e6:.0f} MB/h" if rows else ""))
    return rc


def _cmd_status(_params: dict) -> int:
    from ..intraday.upload import pending
    from .collector import segments_root

    body, age = _heartbeat()
    if body is None:
        print("no massive-ws heartbeat (collector never ran here)")
    else:
        print(f"massive-ws heartbeat {age:.0f} s ago (pid {body.get('pid')}): {body.get('state')}"
              f"{' — stopped' if body.get('stopped') else ''}")
        for key in ("window", "connected_since", "bars", "bars_per_s", "last_bar_utc", "delay_s",
                    "disconnects", "last_error", "last_status", "dropped_frames", "bad_messages",
                    "segments_closed", "open_segment", "pending_upload"):
            if body.get(key) is not None:
                print(f"  {key}: {body[key]}")
    left = pending(segments_root())
    print(f"local segments: {left['files']} file(s), {left['bytes'] / 1e6:.1f} MB")
    return 0


def _cmd_upload(_params: dict) -> int:
    _logging()
    from ..intraday.store import _singleton_lock, recover
    from ..intraday.upload import pending, upload_all
    from .collector import lock_path, nas_dir, segments_root
    from .rows import AGG_1S_SCHEMA, SORT_BY

    root = segments_root()
    if not root.exists():
        print("no local segments")
        return 0
    try:
        lock = _singleton_lock(lock_path(), "massive-ws follow")
    except SystemExit:
        print("the collector is running; its own uploader is draining segments")
        return 0
    try:
        recover(root, schema=AGG_1S_SCHEMA, sort_by=SORT_BY)
        done, failed, reason = upload_all(root, nas_dir())
    finally:
        lock.close()
    left = pending(root)
    if reason:
        print(f"not uploading: {reason}")
    print(f"uploaded {done}, failed {failed}; {left['files']} file(s), {left['bytes'] / 1e6:.1f} MB still local")
    return 1 if (failed or reason) else 0


def _cmd_wait_idle(params: dict) -> int:
    """Exit 0 once the collector is idle (window closed) with nothing left to upload.

    For the old always-on com.mtap.massive agent; a per-session collector simply exits."""
    timeout = float(params.get("timeout", 1200))
    deadline = time.monotonic() + timeout
    while True:
        body, age = _heartbeat()
        if body is None:
            print("no massive-ws heartbeat; nothing to wait for")
            return 0
        if age > 120 and not body.get("stopped"):
            print(f"massive-ws heartbeat is {age:.0f} s old (collector not running?); not waiting")
            return 0
        files = (body.get("pending_upload") or {}).get("files", 0)
        quiet = body.get("stopped") or body.get("state") == "idle"
        if quiet and files == 0:
            print(f"massive-ws idle, nothing left to upload ({body.get('bars')} bars today)")
            return 0
        if time.monotonic() >= deadline:
            print(f"massive-ws not idle after {timeout:.0f} s: state {body.get('state')}, "
                  f"{files} file(s) not uploaded ({(body.get('pending_upload') or {}).get('blocked')})")
            return 1
        time.sleep(5)


def _cmd_compact(params: dict) -> int:
    _logging()
    from .compact import compact

    day = params.get("date") if params.get("date") not in (None, True) else None
    done, errors = compact(day=day)
    for s in done:
        dups = f", {s['duplicates']} duplicates dropped" if s["duplicates"] else ""
        gap = f"longest market-wide gap {s['longest_gap_s']} s" + (f" at {s['gap_at_et']} ET" if s["gap_at_et"] else "")
        print(f"massive-ws {s['date']}: {s['files']} files -> day-{s['date']}.parquet, {s['rows']} bars, "
              f"{s['symbols']} symbols, {s['mb_before']} -> {s['mb_after']} MB{dups}; {gap}")
    for e in errors:
        print(f"massive-ws compact FAILED {e}")
    if not done and not errors:
        print("massive-ws compact: nothing to combine")
    return 1 if errors else 0


def main(argv: list[str]) -> int:
    """argv is everything AFTER the `massive-ws` verb."""
    if not argv:
        print(_USAGE, file=sys.stderr)
        return 2
    sub, rest = argv[0], argv[1:]
    try:
        params = _parse(rest)
    except ValueError as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2
    commands = {"follow": _cmd_follow, "status": _cmd_status, "upload": _cmd_upload,
                "wait-idle": _cmd_wait_idle, "compact": _cmd_compact}
    if sub not in commands:
        print(f"unknown massive-ws subcommand {sub!r}\n{_USAGE}", file=sys.stderr)
        return 2
    return commands[sub](params)
