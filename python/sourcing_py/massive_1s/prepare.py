"""The two input files for the NAS 1-second-bar fetcher (nas/massive1s).

    DATA_ROOT/massive_1s/inputs/universe.csv   ticker,name,exchange,kept,reason
    DATA_ROOT/massive_1s/inputs/windows.csv    date,start_utc_ms,end_utc_ms

The universe is the common stocks listed today, from the newest tickers_cs file, minus the
notes, preferreds, depositary shares, funds and rights that Massive also types as CS. LP
common units (ET, BIP) stay. Each window is [09:00 New York, session close + 7 min), the
live WebSocket feed's span; the close comes from the XNYS calendar, so half days end at
13:07. Python owns these because the calendar and ticker list already live here; the Rust
fetcher only reads plain CSV.
"""

from __future__ import annotations

import csv
import io
import os
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..common import config, storage
from ..massive_ext.extended import latest_tickers
from ..utils.trading_calendar import session_close_utc, sessions

NY = ZoneInfo("America/New_York")
WINDOW_START = time(9, 0)
AFTER_CLOSE = timedelta(minutes=7)

_NOT_EQUITY = (
    ("note", re.compile(r"\bnotes?\b|\bdue \d{4}\b|debenture|subordinated", re.IGNORECASE)),
    ("preferred", re.compile(r"preferred|depositary|\bpfd\b", re.IGNORECASE)),
    ("fund", re.compile(r"\bfund\b", re.IGNORECASE)),
    ("rights", re.compile(r"\brights?\b", re.IGNORECASE)),
)


def exclusion(name: str | None) -> str | None:
    """Why a CS-typed name is not a common stock, or None to keep it."""
    for reason, pat in _NOT_EQUITY:
        if pat.search(name or ""):
            return reason
    return None


def stock_list(rows: list[dict]) -> list[dict]:
    """Listed common stocks, sorted by ticker, each marked kept or with its exclusion."""
    out: dict[str, dict] = {}
    for r in rows:
        if not r.get("active"):
            continue
        reason = exclusion(r.get("name"))
        out.setdefault(r["ticker"], {
            "ticker": r["ticker"], "name": r.get("name") or "", "exchange": r.get("primary_exchange") or "",
            "kept": 0 if reason else 1, "reason": reason or "",
        })
    return [out[t] for t in sorted(out)]


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def windows(d0: date, d1: date) -> list[tuple[str, int, int]]:
    """(date, start_ms, end_ms) per session in [d0, d1]: [09:00 New York, close + 7 min)."""
    return [
        (d.isoformat(), _ms(datetime.combine(d, WINDOW_START, NY)), _ms(session_close_utc(d) + AFTER_CLOSE))
        for d in sessions(d0, d1)
    ]


def inputs_dir() -> Path:
    return config.data_root() / "massive_1s" / "inputs"


def _write_csv(path: Path, header: list[str], rows: list) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(buf.getvalue())
    os.replace(tmp, path)


def prepare(d0: date, d1: date) -> dict:
    storage.require_data_root()
    stocks = stock_list(latest_tickers())
    wins = windows(d0, d1)
    if not wins:
        raise ValueError(f"no sessions in {d0} .. {d1}")
    out = inputs_dir()
    out.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "name", "exchange", "kept", "reason"]
    _write_csv(out / "universe.csv", cols, [[s[c] for c in cols] for s in stocks])
    _write_csv(out / "windows.csv", ["date", "start_utc_ms", "end_utc_ms"], wins)
    kept = [s for s in stocks if s["kept"]]
    half = sum(1 for _, s, e in wins if e - s < 7 * 3600 * 1000)
    return {
        "dir": out, "listed": len(stocks), "kept": len(kept),
        "by_exchange": Counter(s["exchange"] or "?" for s in kept).most_common(),
        "excluded": Counter(s["reason"] for s in stocks if not s["kept"]).most_common(),
        "sessions": len(wins), "half_days": half, "first": wins[0][0], "last": wins[-1][0],
    }
