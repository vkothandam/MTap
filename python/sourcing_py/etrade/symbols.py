"""Phase 1 — build the master `symbols` table from recent daily_summary files.

Unions the distinct `ticker` values across a recent window of
`out/daily_summary/daily_summary-<date>.jsonl` files and upserts them into the
DuckDB `symbols` table, tracking first_seen / last_seen / days_seen. No auth needed.

The merge is idempotent: first_seen/last_seen widen (min/max) and days_seen takes the
max across runs, so re-running the same window leaves the table unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path

from ..common import config
from . import db

_FILE_RE = re.compile(r"daily_summary-(\d{4}-\d{2}-\d{2})\.jsonl$")


def daily_summary_dir() -> Path:
    return config.out_root() / "daily_summary"


def _file_date(path: Path) -> date | None:
    m = _FILE_RE.search(path.name)
    return date.fromisoformat(m.group(1)) if m else None


def select_files(
    *,
    days: int | None = 90,
    fromdate: str | None = None,
    todate: str | None = None,
    directory: Path | None = None,
) -> list[Path]:
    """Resolve the daily_summary files in the requested window, oldest → newest.

    fromdate/todate (inclusive) take precedence; otherwise the most recent `days`
    dated files are used.
    """
    directory = directory or daily_summary_dir()
    dated = sorted(
        ((d, p) for p in directory.glob("daily_summary-*.jsonl") if (d := _file_date(p))),
        key=lambda t: t[0],
    )
    if fromdate or todate:
        lo = date.fromisoformat(fromdate) if fromdate else date.min
        hi = date.fromisoformat(todate) if todate else date.max
        return [p for d, p in dated if lo <= d <= hi]
    if days is not None:
        dated = dated[-days:]
    return [p for _, p in dated]


def tickers_in(path: Path) -> set[str]:
    """Distinct, non-empty tickers in one daily_summary file."""
    out: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line).get("ticker")
        if t:
            out.add(t)
    return out


def _aggregate(files: Iterable[Path]) -> dict[str, dict]:
    """Per-symbol {first_seen, last_seen, days_seen} across the given files."""
    stats: dict[str, dict] = {}
    for path in files:
        d = _file_date(path)
        for ticker in tickers_in(path):
            s = stats.get(ticker)
            if s is None:
                stats[ticker] = {"first_seen": d, "last_seen": d, "days_seen": 1}
            else:
                s["first_seen"] = min(s["first_seen"], d)
                s["last_seen"] = max(s["last_seen"], d)
                s["days_seen"] += 1
    return stats


def _upsert(con, stats: dict[str, dict]) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        (sym, s["first_seen"], s["last_seen"], s["days_seen"], now)
        for sym, s in stats.items()
    ]
    con.executemany(
        """
        INSERT INTO symbols (symbol, first_seen, last_seen, days_seen, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (symbol) DO UPDATE SET
            first_seen = least(symbols.first_seen, excluded.first_seen),
            last_seen  = greatest(symbols.last_seen, excluded.last_seen),
            days_seen  = greatest(symbols.days_seen, excluded.days_seen),
            updated_at = excluded.updated_at
        """,
        rows,
    )


def build_symbols(
    *,
    days: int | None = 90,
    fromdate: str | None = None,
    todate: str | None = None,
    directory: Path | None = None,
    db_path: Path | None = None,
) -> dict:
    """Run Phase 1. Returns a summary dict (also useful for the CLI printout)."""
    files = select_files(days=days, fromdate=fromdate, todate=todate, directory=directory)
    if not files:
        raise FileNotFoundError(
            f"no daily_summary-*.jsonl files found in {directory or daily_summary_dir()}"
        )
    stats = _aggregate(files)
    with db.connect(db_path) as con:
        _upsert(con, stats)
        total = con.execute("SELECT count(*) FROM symbols").fetchone()[0]
    return {
        "files": len(files),
        "window": (str(_file_date(files[0])), str(_file_date(files[-1]))),
        "symbols_in_window": len(stats),
        "symbols_total": total,
    }
