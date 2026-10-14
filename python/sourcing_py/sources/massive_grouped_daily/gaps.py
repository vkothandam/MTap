"""`sourcing-py daily-gaps`: the sessions that have no daily_summary file yet.

Prints one date per line (oldest first): every exchange session in the last `--days`
calendar days, strictly before today in New York, without an
`out/daily_summary/daily_summary-<date>.jsonl`. Today is never listed: its bar is not
final until after-hours trading ends. scripts/schedule/daily-sourcing.sh fetches each.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def missing_sessions(directory: Path, today: date, *, days: int = 14, cap: int = 10) -> list[date]:
    """Sessions in [today - days, today) lacking a file; the newest `cap` of them."""
    from ...etrade.symbols import _file_date
    from ...utils.trading_calendar import sessions

    have = {d for p in directory.glob("daily_summary-*.jsonl") if (d := _file_date(p))}
    yesterday = today - timedelta(days=1)
    gaps = [d for d in sessions(today - timedelta(days=days), yesterday) if d not in have]
    return gaps[-cap:] if cap > 0 else []


def main(argv: list[str]) -> int:
    from ...etrade.symbols import daily_summary_dir
    from ...intraday.cli import _parse

    try:
        params = _parse(argv)
        days, cap = int(params.get("days", 14)), int(params.get("max", 10))
    except ValueError as exc:
        print(f"error: {exc}\nusage: sourcing-py daily-gaps [--days 14] [--max 10]", file=sys.stderr)
        return 2
    for d in missing_sessions(daily_summary_dir(), datetime.now(NY).date(), days=days, cap=cap):
        print(d.isoformat())
    return 0
