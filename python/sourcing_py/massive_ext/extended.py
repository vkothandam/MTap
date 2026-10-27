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


# Pre-market cut-offs, minutes before the 09:30 open. A cut-off C counts the bars that
# START before C (a bar stamped 09:24 is in the 09:25 sums). They are fixed offsets from the
# open, so they are right on every day, DST included.
CUTOFFS = {"0800": 90, "0900": 30, "0915": 15, "0925": 5, "0930": 0}
_CUT_MS = {k: m * 60_000 for k, m in CUTOFFS.items()}


def _vw(bar: dict) -> float | None:
    """The bar's own VWAP, else its typical price (h+l+c)/3, else its close."""
    vw = bar.get("vw")
    if vw:
        return vw
    h, lo, c = bar.get("h"), bar.get("l"), bar.get("c")
    if h is not None and lo is not None and c is not None:
        return (h + lo + c) / 3
    return c


def _tp(bar: dict) -> float | None:
    h, lo, c = bar.get("h"), bar.get("l"), bar.get("c")
    if h is not None and lo is not None and c is not None:
        return (h + lo + c) / 3
    return c


class DaySummary:
    """Accumulates one ticker-day from bars in time order."""

    __slots__ = ("bars", "cut", "extreme", "last", "post_last", "post_v", "post_vwv", "pre_first",
                 "pre_last", "pre_v", "reg_v", "reg_vwv")

    def __init__(self) -> None:
        self.pre_first = self.pre_last = self.post_last = None
        self.pre_v = self.reg_v = self.post_v = 0.0
        self.reg_vwv = self.post_vwv = 0.0
        # per cut-off: [sum vw*v, sum typical*v, sum v]
        self.cut = {k: [0.0, 0.0, 0.0] for k in CUTOFFS}
        # per cut-off, from the bars that traded (v > 0): the last one's close and start time
        # (epoch ms), and the highest high / lowest low
        self.last = {k: [None, None] for k in CUTOFFS}
        self.extreme = {k: [None, None] for k in CUTOFFS}
        self.bars = 0

    def add(self, t: int, open_ms: int, close_ms: int, bar: dict) -> None:
        self.bars += 1
        v = bar.get("v") or 0.0
        if t < open_ms:
            if self.pre_first is None:
                self.pre_first = bar.get("o")
            self.pre_last = bar.get("c")
            self.pre_v += v
            if v:
                vw, tp = _vw(bar), _tp(bar)
                for k, off in _CUT_MS.items():
                    if t < open_ms - off:
                        acc = self.cut[k]
                        acc[0] += (vw or 0.0) * v
                        acc[1] += (tp or 0.0) * v
                        acc[2] += v
                        self.last[k][0], self.last[k][1] = bar.get("c"), t
                        hi, lo = bar.get("h"), bar.get("l")
                        ex = self.extreme[k]
                        if hi is not None and (ex[0] is None or hi > ex[0]):
                            ex[0] = hi
                        if lo is not None and (ex[1] is None or lo < ex[1]):
                            ex[1] = lo
        elif t < close_ms:
            self.reg_v += v
            self.reg_vwv += (_vw(bar) or 0.0) * v
        else:
            self.post_last = bar.get("c")
            self.post_v += v
            self.post_vwv += (_vw(bar) or 0.0) * v

    def record(self, ticker: str, d: date, fetched_at: str) -> dict:
        r = {
            "ticker": ticker, "date": d.isoformat(),
            "pre_market_price": self.pre_first, "pre_market_last": self.pre_last,
            # volumes carry fractional shares; round off the float-sum noise
            "pre_market_volume": round(self.pre_v, 6), "regular_volume": round(self.reg_v, 6),
            "after_hours_price": self.post_last, "after_hours_volume": round(self.post_v, 6),
            "minute_bars": self.bars, "fetched_at": fetched_at,
            "regular_vwap": self.reg_vwv / self.reg_v if self.reg_v else None,
            "after_hours_vwap": self.post_vwv / self.post_v if self.post_v else None,
        }
        for k, (vwv, tpv, v) in self.cut.items():
            r[f"pre_market_vwap_{k}"] = vwv / v if v else None
            r[f"pre_market_tpvwap_{k}"] = tpv / v if v else None
            r[f"pre_market_volume_{k}"] = round(v, 6)
            r[f"pm_last_{k}"], r[f"pm_last_t_{k}"] = self.last[k]
            r[f"pm_high_{k}"], r[f"pm_low_{k}"] = self.extreme[k]
        return r


def _ms(d: date, t: time) -> int:
    return int(datetime.combine(d, t, NY).timestamp() * 1000)


def summarize(ticker: str, pages, fetched_at: str, sink: list | None = None) -> list[dict]:
    """Per-day records from an iterable of bar pages (each a list of Massive 1-min bars).

    The New York day and its open/close cut-offs are worked out once per day, not per bar
    (a per-bar timezone conversion made this the backfill's bottleneck)."""
    days: dict[date, DaySummary] = {}
    acc = None
    start = end = open_ms = close_ms = 0
    for bars in pages:
        if sink is not None:
            sink.extend((ticker, b["t"], b.get("o"), b.get("h"), b.get("l"), b.get("c"),
                         b.get("v"), b.get("vw"), b.get("n")) for b in bars)
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


def fetch_ticker(client: httpx.Client, cfg: dict, ticker: str, d0: date, d1: date,
                 sink: list | None = None) -> list[dict]:
    fetched_at = datetime.now(UTC).isoformat()
    return summarize(ticker, _pages(client, cfg, ticker, d0, d1), fetched_at, sink)


def latest_tickers() -> list[dict]:
    """The rows of the newest tickers_cs file (common stocks, listed and delisted)."""
    files = sorted((config.out_root() / "tickers_cs").glob("tickers_cs-*.jsonl"))
    if not files:
        raise FileNotFoundError("no tickers_cs file: run `sourcing-py run massive_tickers_cs` first")
    return [json.loads(line) for line in files[-1].read_text().splitlines() if line.strip()]


def universe(d0: date) -> list[str]:
    """Common stocks and ADRs (types CS and ADRC) listed now or delisted on/after d0, from the newest tickers_cs file."""
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


def quarters(d0: date, d1: date) -> list[tuple[date, date]]:
    """[d0, d1] cut at calendar-quarter boundaries (each piece is its own resumable job)."""
    out = []
    cur = d0
    while cur <= d1:
        qm = (cur.month - 1) // 3 * 3 + 3  # last month of cur's quarter
        end = date(cur.year + (qm == 12), qm % 12 + 1, 1) - timedelta(days=1)
        out.append((cur, min(end, d1)))
        cur = end + timedelta(days=1)
    return out


def fetch(d0: date, d1: date, *, tickers: list[str] | None = None, workers: int | None = None,
          only_dates: set[date] | None = None, keep: bool = False, allow_failures: int = 0,
          chunk: str | None = None, save_bars: Path | str | None = None,
          bars_universe: set[str] | None = None) -> int:
    """Fetch every ticker for [d0, d1], then write the per-date files. 0 when complete.

    chunk="quarter" runs each calendar quarter as its own job (so a backfill resumes at the
    quarter it stopped in). allow_failures=N finalizes even when up to N tickers failed
    (failures.jsonl and the job folder are left; rerun to retry them). save_bars=DIR also
    keeps the raw minute bars as DIR/month=YYYY-MM/*.parquet (month in New York time),
    only for bars_universe when given (the panel's symbols; the other tickers still get their
    summaries)."""
    if chunk not in (None, "quarter"):
        raise ValueError(f"unknown --chunk {chunk!r} (only: quarter)")
    pieces = quarters(d0, d1) if chunk == "quarter" else [(d0, d1)]
    rc = 0
    for q0, q1 in pieces:
        rc = max(rc, _fetch_one(q0, q1, tickers=tickers, workers=workers, only_dates=only_dates,
                                keep=keep, allow_failures=allow_failures, save_bars=save_bars,
                                bars_universe=bars_universe))
    return rc


_BAR_COLS = ("ticker", "t", "o", "h", "l", "c", "v", "vw", "n")


def _write_bars_stage(path: Path, rows: list[tuple]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = list(zip(*rows)) if rows else [[] for _ in _BAR_COLS]
    types = [pa.string(), pa.int64(), pa.float64(), pa.float64(), pa.float64(), pa.float64(),
             pa.float64(), pa.float64(), pa.int64()]
    table = pa.table({n: pa.array(list(c), type=ty) for n, c, ty in zip(_BAR_COLS, cols, types)})
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _publish_bars(job: Path, dest: Path, d0: date, d1: date) -> int:
    """Combine the staged per-ticker bars into one parquet file per New York month under
    dest/month=YYYY-MM/ (written next to the target, then moved). epoch_ms is a naive UTC
    timestamp: timezone('UTC', ..) makes it an instant, then New York local time gives the month. Returns files written."""
    import duckdb

    files = [str(p) for p in (job / "bars").glob("*.parquet") if p.stat().st_size]
    if not files:
        return 0
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("CREATE TABLE m AS SELECT *, strftime(timezone('America/New_York', timezone('UTC', epoch_ms(t))), '%Y-%m') AS mo "
                "FROM read_parquet(?)", [files])
    n = 0
    for (mo,) in con.execute("SELECT DISTINCT mo FROM m ORDER BY 1").fetchall():
        d = dest / f"month={mo}"
        d.mkdir(parents=True, exist_ok=True)
        final = d / f"bars-{d0}_{d1}.parquet"
        tmp = d / f".{final.name}.tmp"
        con.execute(f"COPY (SELECT {', '.join(_BAR_COLS)} FROM m WHERE mo = ? ORDER BY ticker, t) "
                    f"TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)", [mo])
        os.replace(tmp, final)
        n += 1
    return n


def _fetch_one(d0: date, d1: date, *, tickers, workers, only_dates, keep, allow_failures, save_bars,
               bars_universe=None) -> int:
    cfg = config.massive_config()
    workers = workers or cfg["workers"]
    targets = tickers or universe(d0)
    job = job_dir(d0, d1)
    staging = job / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    if save_bars:
        (job / "bars").mkdir(exist_ok=True)
    done = {_staged(p) for p in staging.glob("*.jsonl")}
    todo = [t for t in targets if t not in done]
    print(f"massive-ext {d0}..{d1}: {len(targets)} tickers, {len(done)} already staged, "
          f"{len(todo)} to fetch, {workers} workers", flush=True)

    failed: dict[str, str] = {}
    ok = processed = 0

    def work(client, t):
        if not save_bars or (bars_universe is not None and t not in bars_universe):
            return fetch_ticker(client, cfg, t, d0, d1), None
        sink: list = []
        return fetch_ticker(client, cfg, t, d0, d1, sink), sink

    with http.get_client(timeout=120) as client, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = iter(todo)
        inflight = {}

        def submit_next() -> None:
            for t in pending:
                inflight[pool.submit(work, client, t)] = t
                return

        for _ in range(workers):
            submit_next()
        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                t = inflight.pop(fut)
                processed += 1
                try:
                    rows, sink = fut.result()
                except Exception as exc:  # noqa: BLE001 — per-ticker failure: log, keep going
                    failed[t] = f"{type(exc).__name__}: {exc}"
                else:
                    name = t.replace("/", "%2F")
                    if sink is not None:  # bars first: the .jsonl marks the ticker as staged
                        _write_bars_stage(job / "bars" / f"{name}.parquet", sink)
                    _write_atomic(staging / f"{name}.jsonl", [json.dumps(r) + "\n" for r in rows])
                    ok += 1
                if processed % 250 == 0:
                    print(f"  {processed}/{len(todo)} fetched ({len(failed)} failed)", flush=True)
                submit_next()

    _write_atomic(job / "failures.jsonl", [json.dumps({"ticker": t, "error": e}) + "\n" for t, e in failed.items()])
    if failed:
        for t, e in list(failed.items())[:5]:
            print(f"  {t}: {_mask(e)[:200]}", file=sys.stderr)
        if len(failed) > allow_failures:
            print(f"{len(failed)} ticker(s) failed (state: {job}); rerun the same command to retry them",
                  file=sys.stderr)
            return 1
        print(f"{len(failed)} ticker(s) failed, within --allow-failures {allow_failures}: finalizing "
              f"anyway (see {job / 'failures.jsonl'})", file=sys.stderr)
    written = finalize(job, only_dates=only_dates)
    bars = _publish_bars(job, Path(save_bars), d0, d1) if save_bars else 0
    print(f"done: {ok} fetched now, {written} date file(s) written" + (f", {bars} bars file(s)" if save_bars else ""))
    if not keep and not failed:
        shutil.rmtree(job)
    return 0


def panel_symbols(path: Path | str) -> set[str]:
    """The distinct symbols of a panel.parquet (--bars-universe)."""
    import pyarrow.parquet as pq

    return set(pq.read_table(path, columns=["symbol"]).column("symbol").to_pylist())


def _mask(s: str) -> str:
    import re

    return re.sub(r"apiKey=[A-Za-z0-9]+", "apiKey=***", s)


# --- per-date files ---------------------------------------------------------------

_COLUMNS = {
    "ticker": "VARCHAR", "date": "DATE", "pre_market_price": "DOUBLE", "pre_market_last": "DOUBLE",
    "pre_market_volume": "DOUBLE", "regular_volume": "DOUBLE", "after_hours_price": "DOUBLE",
    "after_hours_volume": "DOUBLE", "minute_bars": "BIGINT", "fetched_at": "VARCHAR",
    # VWAPs (added 2026-10): minute-bar sums; see DaySummary
    "regular_vwap": "DOUBLE", "after_hours_vwap": "DOUBLE",
    **{f"pre_market_{kind}_{k}": "DOUBLE" for k in CUTOFFS for kind in ("vwap", "tpvwap", "volume")},
    # last pre-market trade (close of the last traded bar before the cut-off; its start time in
    # epoch ms) and the pre-market high / low (added 2026-10-27, panel inputs pm_last / pm_range)
    **{f"{kind}_{k}": ty for k in CUTOFFS for kind, ty in
       (("pm_last", "DOUBLE"), ("pm_last_t", "BIGINT"), ("pm_high", "DOUBLE"), ("pm_low", "DOUBLE"))},
}
NEW_COLUMNS = [c for c in _COLUMNS if c in ("regular_vwap", "after_hours_vwap")
               or c.startswith(("pre_market_vwap_", "pre_market_tpvwap_", "pre_market_volume_", "pm_"))]
NEW_COLUMN_TYPES = {c: _COLUMNS[c] for c in NEW_COLUMNS}


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
