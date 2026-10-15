"""CLI for the intraday feed subcommands.

Dispatched from `sourcing_py.__main__` when the first arg is `intraday`:

    sourcing-py intraday store --follow       (launchd KeepAlive; Redis -> segments -> NAS)
    sourcing-py intraday upload --drain       (when the store is not running: close + upload leftovers)
    sourcing-py intraday summary [--date D] [--gap-min 5]
    sourcing-py intraday status               (store heartbeat + pending uploads)
    sourcing-py intraday ingest-mbin --root DIR [--dates D1,D2]   (backfill MBin CSVs)
    sourcing-py intraday export-tft [--dates D1,D2] [--step-s 15]  (the intraday ML panel)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime

from ..common import config
from . import feed

_USAGE = (
    "usage:\n"
    "  sourcing-py intraday store --follow [--no-upload]   (Redis INTRADAY_FEED -> segments -> NAS)\n"
    "  sourcing-py intraday upload --drain   (upload closed local segments now)\n"
    "  sourcing-py intraday summary [--date D] [--gap-min 5]   (rows per watchlist, RTH gaps)\n"
    "  sourcing-py intraday status   (store heartbeat, pending uploads)\n"
    "  sourcing-py intraday ingest-mbin --root DIR [--dates D1,D2]   (MBin watchlist CSVs -> feed parquet)\n"
    "  sourcing-py intraday export-tft [--dates D1,D2] [--step-s 15]   (feed -> out/tft_intraday/panel)"
)


def _parse(args: list[str]) -> dict:
    params: dict = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise ValueError(f"expected --key value or --flag, got {token!r}")
        key = token[2:]
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            params[key] = args[i + 1]
            i += 2
        else:
            params[key] = True
            i += 1
    return params


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _cmd_store(params: dict) -> int:
    if not params.get("follow"):
        print(f"error: store needs --follow\n{_USAGE}", file=sys.stderr)
        return 2
    _logging()
    from .store import follow

    return follow(upload=not params.get("no-upload"))


def _cmd_upload(params: dict) -> int:
    _logging()
    from .store import _singleton_lock, recover, segments_root
    from .upload import pending, upload_all

    root = segments_root()
    if not root.exists():
        print("no local segments")
        return 0
    try:
        lock = _singleton_lock(config.state_root() / "intraday" / "store.lock")
    except SystemExit:
        print("the store is running; its own uploader is draining segments")
        return 0
    try:
        recover(root)  # safe: holding the store lock, so nothing is writing a segment
        done, failed, reason = upload_all(root)
    finally:
        lock.close()
    left = pending(root)
    if reason:
        print(f"not uploading: {reason}")
    print(f"uploaded {done}, failed {failed}; {left['files']} file(s), {left['bytes'] / 1e6:.1f} MB still local")
    return 1 if (failed or reason) else 0


def _cmd_summary(params: dict) -> int:
    from .summary import print_summary

    day = date.fromisoformat(params["date"]) if params.get("date") else datetime.now(feed.NY).date()
    return print_summary(day, float(params.get("gap-min", 5)))


def _cmd_status(_params: dict) -> int:
    from .upload import logged_failures, pending

    hb = config.state_root() / "intraday" / "store.heartbeat"
    if hb.exists():
        body = json.loads(hb.read_text())
        age = (datetime.now().astimezone() - datetime.fromisoformat(body["updated_at"])).total_seconds()
        print(f"store heartbeat {age:.0f} s ago (pid {body.get('pid')}){' — stopped' if body.get('stopped') else ''}")
        for key in ("entries", "rows", "bad_entries", "segments_closed", "last_ts_utc", "group", "open_segments"):
            if key in body:
                print(f"  {key}: {body[key]}")
    else:
        print(f"no store heartbeat at {hb}")
    collector = config.state_root() / "etrade" / "collector.heartbeat"
    if collector.exists():
        body = json.loads(collector.read_text())
        print(f"collector heartbeat {body.get('updated_at')} (pid {body.get('pid')})")
        for name, w in (body.get("watchlists") or {}).items():
            print(f"  {name}: {w.get('polls')} polls, {w.get('rows')} rows, last {w.get('last_poll_utc')}")
    left = pending(config.state_root() / "segments")
    print(f"local segments: {left['files']} file(s), {left['bytes'] / 1e6:.1f} MB")
    for e in logged_failures(config.state_root() / "segments").values():
        print(f"  not uploaded: {e['file']} ({e['attempts']} attempt(s) since {e['first_at']}): {e['error']}")
    return 0


def _dates(params: dict) -> list[date] | None:
    raw = params.get("dates")
    if not raw or raw is True:
        return None
    return [date.fromisoformat(d.strip()) for d in str(raw).split(",") if d.strip()]


def _cmd_ingest_mbin(params: dict) -> int:
    if not params.get("root") or params["root"] is True:
        print(f"error: ingest-mbin needs --root DIR\n{_USAGE}", file=sys.stderr)
        return 2
    _logging()
    from pathlib import Path

    from .mbin import ingest

    done = ingest(Path(params["root"]).expanduser(), _dates(params))
    for d in done:
        print(f"{d['date']}: {d['rows']} rows "
              + ", ".join(f"{k} {v['symbols']} symbols / {v['polls']} polls" for k, v in d.items()
                          if isinstance(v, dict)))
    return 0 if done else 1


def _cmd_export_tft(params: dict) -> int:
    _logging()
    from .export import export

    meta = export(_dates(params), step_s=int(params.get("step-s", 15)))
    print(json.dumps({k: meta[k] for k in ("panel_dir", "dates", "rows")}, indent=2))
    return 0


def main(argv: list[str]) -> int:
    """argv is everything AFTER the `intraday` verb."""
    if not argv:
        print(_USAGE, file=sys.stderr)
        return 2
    sub, rest = argv[0], argv[1:]
    try:
        params = _parse(rest)
    except ValueError as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2
    if sub == "store":
        return _cmd_store(params)
    if sub == "upload":
        return _cmd_upload(params)
    if sub == "summary":
        return _cmd_summary(params)
    if sub == "status":
        return _cmd_status(params)
    if sub == "ingest-mbin":
        return _cmd_ingest_mbin(params)
    if sub == "export-tft":
        return _cmd_export_tft(params)
    print(f"unknown intraday subcommand {sub!r}\n{_USAGE}", file=sys.stderr)
    return 2
