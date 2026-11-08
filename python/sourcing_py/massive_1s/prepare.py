"""The two input files for the NAS 1-second-bar fetcher (nas/massive1s).

    DATA_ROOT/massive_1s/inputs/universe.csv   ticker,name,exchange,kept,reason,listed
    DATA_ROOT/massive_1s/inputs/windows.csv    date,start_utc_ms,end_utc_ms
    (inputs-NAME/ with --name NAME)

The universe is the common stocks listed today, from the newest tickers_cs file, minus the
notes, preferreds, depositary shares, funds and rights that Massive also types as CS. LP
common units (ET, BIP) stay. With --from-bars it is instead the tickers that traded in
bars_1m over the range, under the ticker they traded as (Massive keeps a renamed stock's
history under its old ticker, and today's list has no delisted stocks), typed CS in any
tickers_cs file, minus the same names and ADRs, plus SPY. One list covers the whole range:
a ticker that traded in one quarter only costs one empty call in the others. `listed` = 0
for a stock no longer listed under that ticker; the Rust fetcher reads only ticker and kept.
Both lists get SPY. --name N writes to inputs-N/ instead of inputs/,
for a run queued behind one that is still reading inputs/ (nas/README.md, Queue).
Each window is [09:00 New York, session close + 7 min), the
live WebSocket feed's span; the close comes from the XNYS calendar, so half days end at
13:07. Python owns these because the calendar and ticker list already live here; the Rust
fetcher only reads plain CSV.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..common import config, storage
from ..massive_ext.extended import latest_tickers
from ..utils.trading_calendar import session_close_utc, sessions

NY = ZoneInfo("America/New_York")
MARKET = ("SPY",)  # not a CS, so in no tickers_cs file; added by name
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
        if not r.get("active") or r.get("type") not in (None, "CS"):  # ADRC is not in the 1s universe
            continue
        reason = exclusion(r.get("name"))
        out.setdefault(r["ticker"], {
            "ticker": r["ticker"], "name": r.get("name") or "", "exchange": r.get("primary_exchange") or "",
            "kept": 0 if reason else 1, "reason": reason or "",
        })
    return [out[t] for t in sorted(out)]


def all_tickers() -> dict[str, list[dict]]:
    """Every tickers_cs row by ticker, newest file first, so a ticker that was renamed away
    or delisted since an older file is still found. Older files have no `type` (all CS)."""
    files = sorted((config.out_root() / "tickers_cs").glob("tickers_cs-*.jsonl"), reverse=True)
    if not files:
        raise FileNotFoundError("no tickers_cs file: run `sourcing-py run massive_tickers_cs` first")
    out: dict[str, list[dict]] = {}
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["ticker"], []).append(r)
    return out


def traded(d0: date, d1: date) -> set[str]:
    """Tickers with any minute bar in bars_1m on a New York date in [d0, d1]."""
    lo = _ms(datetime.combine(d0, time(0), NY))
    hi = _ms(datetime.combine(d1 + timedelta(days=1), time(0), NY))
    out: set[str] = set()
    found = False
    m = date(d0.year, d0.month, 1)
    while m <= d1:
        nxt = date(m.year + m.month // 12, m.month % 12 + 1, 1)
        whole = d0 <= m and nxt - timedelta(days=1) <= d1
        # a month folder holds that New York month only, whatever its file names say, so a
        # whole month needs only the ticker column (~0.1 MB over SMB instead of ~30 MB with t)
        for f in sorted((config.data_root() / "bars_1m" / f"month={m:%Y-%m}").glob("*.parquet")):
            found = True
            if whole:
                t = pq.read_table(f, columns=["ticker"])
            else:
                t = pq.read_table(f, columns=["ticker", "t"], filters=[("t", ">=", lo), ("t", "<", hi)])
            out.update(pc.unique(t["ticker"]).to_pylist())
        m = nxt
    if not found:
        raise FileNotFoundError(f"no bars_1m files for {d0:%Y-%m} .. {d1:%Y-%m} under {config.data_root()}")
    return out


def stock_list_from_bars(names: set[str], rows: dict[str, list[dict]], add: list[str] = ()) -> list[dict]:
    """The stocks that traded (`names`), listed or not, under the ticker they traded as.
    Kept when a tickers_cs row types it CS and its name passes exclusion(); ADRs (ADRC) and
    tickers with no tickers_cs row are marked, not kept. `add` (e.g. SPY) is kept as is."""
    out: dict[str, dict] = {}
    for t in sorted(names):
        rs = rows.get(t, [])
        cs = [r for r in rs if r.get("type") in (None, "CS")]
        if cs:
            r = next((r for r in cs if not exclusion(r.get("name"))), cs[0])
            reason = exclusion(r.get("name")) or ""
        else:
            r = rs[0] if rs else {}
            reason = "adr" if rs else "no tickers_cs row"
        out[t] = {"ticker": t, "name": r.get("name") or "", "exchange": r.get("primary_exchange") or "",
                  "kept": 0 if reason else 1, "reason": reason,
                  "listed": int(any(x.get("active") for x in rs))}
    for t in add:
        out[t] = {"ticker": t, "name": "(market, added by name)", "exchange": "", "kept": 1, "reason": "", "listed": 1}
    return [out[t] for t in sorted(out)]


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def windows(d0: date, d1: date) -> list[tuple[str, int, int]]:
    """(date, start_ms, end_ms) per session in [d0, d1]: [09:00 New York, close + 7 min)."""
    return [
        (d.isoformat(), _ms(datetime.combine(d, WINDOW_START, NY)), _ms(session_close_utc(d) + AFTER_CLOSE))
        for d in sessions(d0, d1)
    ]


def inputs_dir(name: str | None = None) -> Path:
    if name is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(f"bad --name {name!r}: lower-case letters, digits, - and _ only")
    return config.data_root() / "massive_1s" / (f"inputs-{name}" if name else "inputs")


def _write_csv(path: Path, header: list[str], rows: list) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(buf.getvalue())
    os.replace(tmp, path)


def prepare(d0: date, d1: date, *, from_bars: bool = False, name: str | None = None) -> dict:
    out = inputs_dir(name)
    storage.require_data_root()
    wins = windows(d0, d1)
    if not wins:
        raise ValueError(f"no sessions in {d0} .. {d1}")
    if from_bars:
        stocks = stock_list_from_bars(traded(d0, d1), all_tickers(), add=MARKET)
    else:
        stocks = [{**s, "listed": 1} for s in stock_list(latest_tickers())] + stock_list_from_bars(set(), {}, add=MARKET)
        stocks.sort(key=lambda s: s["ticker"])
    out.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "name", "exchange", "kept", "reason", "listed"]
    _write_csv(out / "universe.csv", cols, [[s[c] for c in cols] for s in stocks])
    _write_csv(out / "windows.csv", ["date", "start_utc_ms", "end_utc_ms"], wins)
    kept = [s for s in stocks if s["kept"]]
    half = sum(1 for _, s, e in wins if e - s < 7 * 3600 * 1000)
    return {
        "dir": out, "listed": len(stocks), "kept": len(kept), "from_bars": from_bars,
        "not_listed_now": sum(1 for s in kept if not s["listed"]),
        "by_exchange": Counter(s["exchange"] or "?" for s in kept).most_common(),
        "excluded": Counter(s["reason"] for s in stocks if not s["kept"]).most_common(),
        "sessions": len(wins), "half_days": half, "first": wins[0][0], "last": wins[-1][0],
    }
