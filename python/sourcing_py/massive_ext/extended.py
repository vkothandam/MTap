"""Pre-market and after-hours prices and volumes per common stock per day.

Source: Massive 1-minute bars, one range call per ticker (paged by `next_url`, 50k bars a
page, about 50 sessions), split by bar start time in New York:

    pre-market   04:00–09:29     regular   09:30 – close (16:00, 13:00 on half days)
    after-hours  close – 19:59   (the close minute counts as after-hours)

Checked 2026-10-15 against open-close and grouped daily (python/scripts/
check_extended_hours.py): the first pre-market open is Massive's `preMarket`, the last
after-hours close its `afterHours`. Daily `volume` (grouped `v`) already covers the
whole day; it exceeds pre + regular + after-hours by the auction and other prints that
are in no minute bar (about 19%), so these volumes never replace it.

Output: out/extended_hours/extended_hours-<date>.jsonl, one row per ticker with bars that
day. A run stages one file per ticker under state/massive_ext/jobs/<from>_<to>/, so an
interrupted run resumes and a failed ticker is fetched again by rerunning the same
command; once every ticker is done the per-date files are written (merged into any
existing file for the date) and the job folder is removed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from ..common import config, http, ratelimit, schema, writer

NY = ZoneInfo("America/New_York")
SOURCE = "massive_extended_hours"
DATASET = "extended_hours"
OPEN = time(9, 30)


# --- the per-day summary -------------------------------------------------------


@lru_cache(maxsize=4096)
def regular_close(d: date) -> time:
    """16:00, or 13:00 on half days (exchange calendar); 16:00 for a non-session."""
    from ..utils.trading_calendar import session_close_utc

    try:
        return session_close_utc(d).astimezone(NY).time()
    except Exception:  # noqa: BLE001 — not a session (bars on a holiday are rare)
        return time(16)


class DaySummary:
    """Accumulates one ticker-day from bars in time order."""

    __slots__ = ("bars", "post_last", "post_v", "pre_first", "pre_last", "pre_v", "reg_v")

    def __init__(self) -> None:
        self.pre_first = self.pre_last = self.post_last = None
        self.pre_v = self.reg_v = self.post_v = 0.0
        self.bars = 0

    def add(self, t: int, open_ms: int, close_ms: int, bar: dict) -> None:
        self.bars += 1
        v = bar.get("v") or 0.0
        if t < open_ms:
            if self.pre_first is None:
                self.pre_first = bar.get("o")
            self.pre_last = bar.get("c")
            self.pre_v += v
        elif t < close_ms:
            self.reg_v += v
        else:
            self.post_last = bar.get("c")
            self.post_v += v

    def record(self, ticker: str, d: date, fetched_at: str) -> dict:
        return {
            "ticker": ticker, "date": d.isoformat(),
            "pre_market_price": self.pre_first, "pre_market_last": self.pre_last,
            # volumes carry fractional shares; round off the float-sum noise
            "pre_market_volume": round(self.pre_v, 6), "regular_volume": round(self.reg_v, 6),
            "after_hours_price": self.post_last, "after_hours_volume": round(self.post_v, 6),
            "minute_bars": self.bars, "fetched_at": fetched_at,
        }


def _ms(d: date, t: time) -> int:
    return int(datetime.combine(d, t, NY).timestamp() * 1000)


def summarize(ticker: str, pages, fetched_at: str) -> list[dict]:
    """Per-day records from an iterable of bar pages (each a list of Massive 1-min bars).

    The New York day and its open/close cut-offs are worked out once per day, not per bar
    (a per-bar timezone conversion made this the backfill's bottleneck)."""
    days: dict[date, DaySummary] = {}
    acc = None
    start = end = open_ms = close_ms = 0
    for bars in pages:
        for bar in bars:
            t = bar["t"]
            if not start <= t < end:
                d = datetime.fromtimestamp(t / 1000, NY).date()
                start, end = _ms(d, time(0)), _ms(d + timedelta(days=1), time(0))
                open_ms, close_ms = _ms(d, OPEN), _ms(d, regular_close(d))
                acc = days.get(d)
                if acc is None:
                    acc = days[d] = DaySummary()
            acc.add(t, open_ms, close_ms, bar)
    return [days[d].record(ticker, d, fetched_at) for d in sorted(days)]


# --- fetching ----------------------------------------------------------------------


def _pages(client: httpx.Client, cfg: dict, ticker: str, d0: date, d1: date):
    url = f"{cfg['base_url']}/v2/aggs/ticker/{quote(ticker, safe='.')}/range/1/minute/{d0}/{d1}"
    params: dict | None = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": cfg["api_key"]}
    rpm = cfg["rate_limit_per_min"]
    while url:
        if rpm:
            ratelimit.throttle("massive", 60.0 / rpm)
        page = http.get_json(client, url, params=params)
        yield page.get("results") or []
        nxt = page.get("next_url")
        # passing params= would replace next_url's cursor query, so merge the key in instead
        url, params = (str(httpx.URL(nxt).copy_merge_params({"apiKey": cfg["api_key"]})), None) if nxt else (None, None)


def fetch_ticker(client: httpx.Client, cfg: dict, ticker: str, d0: date, d1: date) -> list[dict]:
    fetched_at = datetime.now(UTC).isoformat()
    return summarize(ticker, _pages(client, cfg, ticker, d0, d1), fetched_at)


def latest_tickers() -> list[dict]:
    """The rows of the newest tickers_cs file (common stocks, listed and delisted)."""
    files = sorted((config.out_root() / "tickers_cs").glob("tickers_cs-*.jsonl"))
    if not files:
        raise FileNotFoundError("no tickers_cs file: run `sourcing-py run massive_tickers_cs` first")
    return [json.loads(line) for line in files[-1].read_text().splitlines() if line.strip()]


def universe(d0: date) -> list[str]:
    """Common stocks listed now or delisted on/after d0, from the newest tickers_cs file."""
    keep: dict[str, None] = {}
    for r in latest_tickers():
        gone = r.get("delisted_utc")
        if r.get("active") or (gone and gone[:10] >= d0.isoformat()):
            keep.setdefault(r["ticker"], None)
    return sorted(keep)


def refresh_tickers(today: date, *, max_age_days: int = 7) -> Path | None:
    """Re-fetch the tickers_cs file when there is none or the newest is older than
    max_age_days (new listings join the universe weekly). Returns the new file, if any."""
    files = sorted((config.out_root() / "tickers_cs").glob("tickers_cs-*.jsonl"))
    if files and (today - date.fromisoformat(files[-1].name[len("tickers_cs-"):-6])).days <= max_age_days:
        return None
    from ..sources.massive_tickers_cs.source import MassiveTickersCsSource

    return MassiveTickersCsSource().run(params={})


def job_dir(d0: date, d1: date) -> Path:
    return config.state_root() / "massive_ext" / "jobs" / f"{d0}_{d1}"


def _staged(path: Path) -> str:
    return path.name.removesuffix(".jsonl").replace("%2F", "/")


def _write_atomic(path: Path, lines: list[str]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(lines))
    os.replace(tmp, path)


def fetch(d0: date, d1: date, *, tickers: list[str] | None = None, workers: int | None = None,
          only_dates: set[date] | None = None, keep: bool = False) -> int:
    """Fetch every ticker for [d0, d1], then write the per-date files. 0 when complete."""
    cfg = config.massive_config()
    workers = workers or cfg["workers"]
    targets = tickers or universe(d0)
    job = job_dir(d0, d1)
    staging = job / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    done = {_staged(p) for p in staging.glob("*.jsonl")}
    todo = [t for t in targets if t not in done]
    print(f"massive-ext {d0}..{d1}: {len(targets)} tickers, {len(done)} already staged, "
          f"{len(todo)} to fetch, {workers} workers", flush=True)

    failed: dict[str, str] = {}
    ok = processed = 0
    with http.get_client(timeout=120) as client, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = iter(todo)
        inflight = {}

        def submit_next() -> None:
            for t in pending:
                inflight[pool.submit(fetch_ticker, client, cfg, t, d0, d1)] = t
                return

        for _ in range(workers):
            submit_next()
        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                t = inflight.pop(fut)
                processed += 1
                try:
                    rows = fut.result()
                except Exception as exc:  # noqa: BLE001 — per-ticker failure: log, keep going
                    failed[t] = f"{type(exc).__name__}: {exc}"
                else:
                    _write_atomic(staging / f"{t.replace('/', '%2F')}.jsonl",
                                  [json.dumps(r) + "\n" for r in rows])
                    ok += 1
                if processed % 250 == 0:
                    print(f"  {processed}/{len(todo)} fetched ({len(failed)} failed)", flush=True)
                submit_next()

    _write_atomic(job / "failures.jsonl", [json.dumps({"ticker": t, "error": e}) + "\n" for t, e in failed.items()])
    if failed:
        for t, e in list(failed.items())[:5]:
            print(f"  {t}: {_mask(e)[:200]}", file=sys.stderr)
        print(f"{len(failed)} ticker(s) failed (state: {job}); rerun the same command to retry them",
              file=sys.stderr)
        return 1
    written = finalize(job, only_dates=only_dates)
    print(f"done: {ok} fetched now, {written} date file(s) written")
    if not keep:
        shutil.rmtree(job)
    return 0


def _mask(s: str) -> str:
    import re

    return re.sub(r"apiKey=[A-Za-z0-9]+", "apiKey=***", s)


# --- per-date files ---------------------------------------------------------------

_COLUMNS = {
    "ticker": "VARCHAR", "date": "DATE", "pre_market_price": "DOUBLE", "pre_market_last": "DOUBLE",
    "pre_market_volume": "DOUBLE", "regular_volume": "DOUBLE", "after_hours_price": "DOUBLE",
    "after_hours_volume": "DOUBLE", "minute_bars": "BIGINT", "fetched_at": "VARCHAR",
}


def out_path(d: date) -> Path:
    return config.out_root() / DATASET / f"{DATASET}-{d.isoformat()}.jsonl"


def finalize(job: Path, *, only_dates: set[date] | None = None) -> int:
    """Regroup the staged per-ticker rows into one file per date; return files written."""
    import duckdb

    files = [p for p in (job / "staging").glob("*.jsonl") if p.stat().st_size]
    if not files:
        return 0
    con = duckdb.connect()
    cols = ", ".join(f"'{k}': '{v}'" for k, v in _COLUMNS.items())
    con.execute(f"CREATE TABLE s AS SELECT * FROM read_json(?, format='newline_delimited', columns={{{cols}}})",
                [[str(p) for p in files]])
    dates = [r[0] for r in con.execute("SELECT DISTINCT date FROM s ORDER BY 1").fetchall()]
    if only_dates is not None:
        dates = [d for d in dates if d in only_dates]
    spec = schema.load_schema(config.load_source(SOURCE).schema_path)
    names = list(_COLUMNS)
    for d in dates:
        rows = [dict(zip(names, r)) for r in con.execute(f"SELECT {', '.join(names)} FROM s WHERE date = ? ORDER BY ticker", [d]).fetchall()]
        for r in rows:
            r["date"] = d.isoformat()
        rows = _merge_existing(out_path(d), rows)
        schema.validate_records(rows, spec)
        writer.write(rows, source=SOURCE, dataset=DATASET, fmt="jsonl", dt=d)
    return len(dates)


def _merge_existing(path: Path, rows: list[dict]) -> list[dict]:
    """New rows win per ticker; tickers only in the existing file are kept."""
    if not path.exists():
        return rows
    new = {r["ticker"] for r in rows}
    kept = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["ticker"] not in new:
                kept.append({k: v for k, v in r.items() if not k.startswith("_")})
    return sorted(rows + kept, key=lambda r: r["ticker"])


# --- incremental: the sessions still missing ------------------------------------------


def missing_sessions(today: date, *, days: int = 14, cap: int = 10) -> list[date]:
    """Sessions in [today - days, today) that have a daily_summary file but no
    extended_hours file (the newest `cap`). Today is never listed."""
    from ..etrade.symbols import _file_date, daily_summary_dir
    from ..utils.trading_calendar import sessions

    have_daily = {d for p in daily_summary_dir().glob("daily_summary-*.jsonl") if (d := _file_date(p))}
    have_ext = {date.fromisoformat(p.name[len(DATASET) + 1:-6])
                for p in (config.out_root() / DATASET).glob(f"{DATASET}-*.jsonl")}
    gaps = [d for d in sessions(today - timedelta(days=days), today - timedelta(days=1))
            if d in have_daily and d not in have_ext]
    return gaps[-cap:] if cap > 0 else []
