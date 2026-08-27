"""Per-source failure log for reattempting failed days.

One JSONL file per source under state/failures/<source>.jsonl, each line:
    {"date": "2024-11-05", "error": "...", "at": "<iso ts>"}

`--retry-failed` reads the distinct dates back, reruns them, and rewrites the file
with only the days that failed again — so the log drains as days succeed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def log_path(source: str) -> Path:
    return config.repo_root() / "state" / "failures" / f"{source}.jsonl"


def record(source: str, date_iso: str, error: str) -> None:
    """Append one failure entry."""
    path = log_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"date": date_iso, "error": error, "at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_dates(source: str) -> list[str]:
    """Distinct failed dates, in first-seen order."""
    path = log_path(source)
    if not path.exists():
        return []
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seen.setdefault(json.loads(line)["date"], None)
    return list(seen)


def rewrite(source: str, entries: list[dict]) -> None:
    """Replace the log with `entries` (used after a retry pass). Deletes it if empty."""
    path = log_path(source)
    if not entries:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# --- Symbol-keyed variant (E*TRADE fundamentals resume) ------------------------
# Same JSONL log shape, but keyed by ticker symbol instead of a trading date, so
# a per-symbol scrape can log and reattempt individual symbols.


def symbol_log_path(scope: str) -> Path:
    return config.repo_root() / "state" / scope / "failures.jsonl"


def record_symbol(scope: str, symbol: str, error: str) -> None:
    """Append one failed-symbol entry under state/<scope>/failures.jsonl."""
    path = symbol_log_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"symbol": symbol, "error": error, "at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_symbols(scope: str) -> list[str]:
    """Distinct failed symbols, in first-seen order."""
    path = symbol_log_path(scope)
    if not path.exists():
        return []
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seen.setdefault(json.loads(line)["symbol"], None)
    return list(seen)


def rewrite_symbols(scope: str, entries: list[dict]) -> None:
    """Replace the symbol log with `entries` (after a retry pass). Deletes it if empty."""
    path = symbol_log_path(scope)
    if not entries:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
