"""Live pre-market from E*TRADE ChartIQ (`sourcing-py etrade premarket --date D`).

For every symbol with a stored XID (the panel's universe), one `getChartiqFeed` Minute call
(extended hours) and one Day call give:

    pm_tpvwap            sum(typical * volume) / sum(volume) over the bars with DT in
                         [04:00 ET, cutoff) on D, typical = (high + low + close) / 3
    pm_volume            sum(volume) of those bars
    vol_10d_mean_source  mean volume of the 10 Day bars before D (this source's own volume)

Written to out/premarket_live/premarket-D.jsonl (+ a summary json). Never part of the
history files.

Probe facts (2026-10-27, delayed token): data.chartData.series[] = {DT (UTC ISO), Open, High,
Low, Close, Volume}; bars are 86.4 s buckets whatever `dataInterval` says; only the FIRST
symbol in `elements` keeps Volume (one symbol per call); `dateStart` is ignored for short
windows (about 2 sessions come back), so the bars are filtered by DT here; the Day period
returns DT "YYYY-MM-DD".

The job only READS the bearer token the collector mirrors to state/etrade/bearer.json (or
session.json). It never logs in. At most 4 calls run at once. The first 401/403/429 stops the
E*TRADE part of the run (no retry); with --source auto the remaining symbols are then fetched
from Massive minute aggregates (same definitions, 15 min delayed feed).

Each record carries `data_until` and `complete` (data_until >= cutoff). A delayed feed read
before cutoff + delay (Massive, or a delayed E*TRADE token) gives complete = false; export-next
then leaves pm_known = 0 for it rather than passing a partial window as the full pre-market.
Re-run with --source massive after cutoff + feed_delay_min to fill those.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

import httpx

from ..common import config, schema, writer
from ..common.errors import ConfigError
from .client import _ORIGIN, _bearer_token

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
PM_START = dtime(4, 0)
AUTH_STATUSES = (401, 403, 429)
PERIOD_DAYS = 10


class AuthStop(Exception):
    """E*TRADE answered 401/403/429 (or there is no token): stop, no retry, no login."""


# -- token ------------------------------------------------------------------------------
def load_token(token_path: Path | None = None, session_path: Path | None = None) -> str:
    """The Authorization header value ("Bearer ..."), from bearer.json, else session.json."""
    cfg_token = token_path or config.premarket_config()["token_path"]
    try:
        data = json.loads(Path(cfg_token).read_text())
        tok = data.get("token")
        if tok:
            return tok if tok.startswith("Bearer ") else f"Bearer {tok}"
    except (OSError, ValueError):
        pass
    sess = session_path or config.etrade_config()["session_path"]
    try:
        tok = _bearer_token(json.loads(Path(sess).read_text()))
    except (OSError, ValueError):
        tok = None
    if tok:
        return f"Bearer {tok}"
    raise AuthStop(f"no E*TRADE bearer token in {cfg_token} or {sess}")


# -- parsing and arithmetic (pure) ------------------------------------------------------
@dataclass
class Bar:
    dt: datetime  # UTC for minute bars; midnight UTC of the session date for Day bars
    high: float
    low: float
    close: float
    volume: float


def _parse_dt(raw: str) -> datetime:
    if len(raw) == 10:
        return datetime.fromisoformat(raw).replace(tzinfo=UTC)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def parse_series(resp: dict) -> list[Bar]:
    series = (((resp or {}).get("data") or {}).get("chartData") or {}).get("series") or []
    out = []
    for r in series:
        if not r.get("DT") or any(r.get(k) is None for k in ("High", "Low", "Close")):
            continue
        out.append(Bar(_parse_dt(r["DT"]), float(r["High"]), float(r["Low"]), float(r["Close"]),
                       float(r.get("Volume") or 0.0)))
    return sorted(out, key=lambda b: b.dt)


def parse_cutoff(cutoff: str) -> dtime:
    h, m = cutoff.split(":")
    return dtime(int(h), int(m))


def window(d: date, cutoff: str) -> tuple[datetime, datetime]:
    """[04:00 ET, cutoff ET) of day d, in UTC."""
    return (datetime.combine(d, PM_START, NY).astimezone(UTC),
            datetime.combine(d, parse_cutoff(cutoff), NY).astimezone(UTC))


def premarket_stats(bars: list[Bar], d: date, cutoff: str) -> dict:
    lo, hi = window(d, cutoff)
    sel = [b for b in bars if lo <= b.dt < hi]
    vol = sum(b.volume for b in sel)
    tp = sum((b.high + b.low + b.close) / 3.0 * b.volume for b in sel)
    return {
        "pm_tpvwap": (tp / vol) if vol > 0 else None,
        "pm_volume": vol,
        "n_bars": len(sel),
        "last_bar_dt": max((b.dt for b in sel), default=None),
    }


def mean_volume_before(day_bars: list[Bar], d: date, n: int = PERIOD_DAYS) -> float | None:
    prior = [b.volume for b in day_bars if b.dt.date() < d][-n:]
    return mean(prior) if prior else None


# -- universe ---------------------------------------------------------------------------
def read_xids_file(path: Path) -> dict[str, int]:
    """{"AAPL": 123, ...} JSON, or lines "SYMBOL,XID"."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        return {str(k): int(v) for k, v in data.items()}
    except ValueError:
        out = {}
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                out[parts[0]] = int(parts[1])
        return out


def panel_symbols(panel: Path | None = None) -> list[str] | None:
    panel = panel or config.out_root() / "tft" / "panel.parquet"
    if not Path(panel).exists():
        return None
    import duckdb

    con = duckdb.connect()
    try:
        return [r[0] for r in con.execute(f"SELECT DISTINCT symbol FROM read_parquet('{panel}')").fetchall()]
    finally:
        con.close()


def load_universe(symbols: list[str] | None, xids_file: Path | None, db_path: Path | None = None,
                  panel: Path | None = None) -> dict[str, int]:
    """symbol -> XID. Explicit --symbols, else the newest panel's symbols, else every symbol
    with an XID. XIDs come from --xids-file, else the local DuckDB (opened read-only)."""
    wanted = symbols or panel_symbols(panel)
    if xids_file:
        xids = read_xids_file(xids_file)
    else:
        path = db_path or config.etrade_config()["db_path"]
        if not Path(path).exists():
            raise ConfigError(f"{path} is not available here; pass --xids-file SYMBOL,XID lines")
        import duckdb

        con = duckdb.connect(str(path), read_only=True)
        try:
            xids = {s: int(x) for s, x in con.execute("SELECT symbol, xid FROM symbols WHERE xid IS NOT NULL").fetchall()}
        finally:
            con.close()
    if wanted is None:
        return xids
    return {s: xids[s] for s in wanted if s in xids}


# -- E*TRADE fetch ----------------------------------------------------------------------
class Stats:
    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.lock = threading.Lock()

    def add(self, seconds: float) -> None:
        with self.lock:
            self.latencies.append(seconds)


class EtradeChart:
    def __init__(self, token: str, base_url: str, stop: threading.Event, stats: Stats,
                 client: httpx.Client | None = None) -> None:
        self.url = f"{base_url.rstrip('/')}/getChartiqFeed"
        self.headers = {"accept": "application/json", "content-type": "application/json",
                        "origin": _ORIGIN, "referer": f"{_ORIGIN}/", "authorization": token}
        self.stop, self.stats = stop, stats
        self.client = client or httpx.Client(timeout=30)

    def _post(self, body: dict) -> dict:
        if self.stop.is_set():
            raise AuthStop("stopped after an earlier auth failure")
        t0 = time.monotonic()
        resp = self.client.post(self.url, json=body, headers=self.headers)
        self.stats.add(time.monotonic() - t0)
        if resp.status_code in AUTH_STATUSES:
            self.stop.set()
            raise AuthStop(f"E*TRADE answered HTTP {resp.status_code}")
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _elements(xid: int) -> list[dict]:
        return [{"type": "price", "symbol": str(xid)}, {"type": "Volume", "symbol": str(xid)}]

    def minute_bars(self, xid: int, d: date, now: datetime) -> list[Bar]:
        start, _ = window(d, "09:25")
        return parse_series(self._post({
            "elements": self._elements(xid), "dataPeriod": "Minute", "dataInterval": 1,
            "extendedHours": True, "dateStart": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "dateEnd": now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }))

    def day_bars(self, xid: int, d: date) -> list[Bar]:
        return parse_series(self._post({
            "elements": self._elements(xid), "dataPeriod": "Day", "dataInterval": 1,
            "extendedHours": False,
            "dateStart": (d - timedelta(days=40)).isoformat() + "T00:00:00.000Z",
            "dateEnd": d.isoformat() + "T23:59:59.000Z",
        }))


# -- Massive fallback -------------------------------------------------------------------
def massive_symbol(client: httpx.Client, cfg: dict, symbol: str, d: date) -> tuple[list[Bar], list[Bar]]:
    def aggs(mult_span: str, frm: str, to: str) -> list[Bar]:
        url = f"{cfg['base_url']}/v2/aggs/ticker/{symbol}/range/{mult_span}/{frm}/{to}"
        r = client.get(url, params={"adjusted": "true", "limit": 50000, "sort": "asc", "apiKey": cfg["api_key"]})
        r.raise_for_status()
        return [Bar(datetime.fromtimestamp(x["t"] / 1000, UTC), x["h"], x["l"], x["c"], x.get("v") or 0.0)
                for x in r.json().get("results") or []]

    minute = aggs("1/minute", d.isoformat(), d.isoformat())
    day = aggs("1/day", (d - timedelta(days=30)).isoformat(), (d - timedelta(days=1)).isoformat())
    return minute, day


# -- the run ----------------------------------------------------------------------------
def _record(symbol: str, xid: int | None, d: date, cutoff: str, minute: list[Bar], day: list[Bar],
            source: str, fetched_at: datetime) -> dict:
    st = premarket_stats(minute, d, cutoff)
    newest = max((b.dt for b in minute), default=None)
    return {
        "symbol": symbol, "xid": xid, "date": d.isoformat(), "cutoff": cutoff,
        "pm_tpvwap": st["pm_tpvwap"], "pm_volume": st["pm_volume"],
        "vol_10d_mean_source": mean_volume_before(day, d), "n_bars": st["n_bars"],
        "last_bar_dt": st["last_bar_dt"].strftime("%Y-%m-%dT%H:%M:%SZ") if st["last_bar_dt"] else None,
        "source": source, "fetched_at": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_newest": newest,  # newest bar of the whole response; dropped before writing
    }


def mark_complete(records: list[dict], d: date, cutoff: str, now: datetime, feed_delay_min: int) -> dict:
    """Set `data_until` / `complete` on every record, per source. A delayed feed run before
    cutoff + delay holds only part of the window, which must not pass as the full pre-market.
    E*TRADE: data_until = the median newest bar over the symbols (live token = about now,
    delayed token = now - 15 min). Massive: now - feed_delay_min. complete = data_until >= cutoff."""
    _, hi = window(d, cutoff)
    until: dict[str, datetime | None] = {}
    for src in {r["source"] for r in records}:
        if src == "massive":
            until[src] = now - timedelta(minutes=feed_delay_min)
        else:
            newest = sorted(r["_newest"] for r in records if r["source"] == src and r["_newest"])
            until[src] = newest[len(newest) // 2] if newest else None
    for r in records:
        r.pop("_newest")
        u = until[r["source"]]
        r["data_until"] = u.strftime("%Y-%m-%dT%H:%M:%SZ") if u else None
        r["complete"] = bool(u and u >= hi)
    return {s: (u.strftime("%Y-%m-%dT%H:%M:%SZ") if u else None) for s, u in until.items()}


def output_path(d: date) -> Path:
    return config.out_root() / "premarket_live" / f"premarket-{d.isoformat()}.jsonl"


def write_records(records: list[dict], d: date) -> Path:
    """Provenance + atomic write (tmp + rename) to premarket_live/premarket-D.jsonl."""
    run_id = writer._run_id()
    enriched = writer._with_provenance(records, "etrade_premarket", run_id)
    path = output_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in enriched))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def run_premarket(d: date, universe: dict[str, int], *, source: str = "auto", cutoff: str | None = None,
                  token: str | None = None, client: httpx.Client | None = None,
                  now: datetime | None = None) -> dict:
    """Fetch and write. source: etrade | massive | auto (E*TRADE, Massive for what an auth
    failure left). Returns the summary dict (includes `path`)."""
    pcfg = config.premarket_config()
    cutoff = cutoff or pcfg["cutoff"]
    workers = pcfg["max_concurrency"]
    now = now or datetime.now(UTC)
    stop, stats = threading.Event(), Stats()
    records: dict[str, dict] = {}
    errors: dict[str, str] = {}
    auth_error: str | None = None

    if source in ("etrade", "auto"):
        try:
            tok = token or load_token()
            chart = EtradeChart(tok, config.etrade_config()["base_url"], stop, stats, client)

            def one(item):
                sym, xid = item
                try:
                    minute = chart.minute_bars(xid, d, now)
                    day = chart.day_bars(xid, d)
                    return sym, _record(sym, xid, d, cutoff, minute, day, "etrade_chartiq",
                                        datetime.now(UTC)), None
                except AuthStop:
                    return sym, None, "auth"
                except Exception as exc:  # per-symbol failure: log, keep going
                    return sym, None, f"{type(exc).__name__}: {exc}"

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for sym, rec, err in pool.map(one, universe.items()):
                    if rec:
                        records[sym] = rec
                    elif err and err != "auth":
                        errors[sym] = err
            if stop.is_set():
                auth_error = "E*TRADE answered 401/403/429; stopped without retry"
        except AuthStop as exc:
            auth_error = str(exc)
    use_massive = source == "massive" or (source == "auto" and auth_error is not None)
    if use_massive:
        mcfg = config.massive_config()
        todo = [(s, x) for s, x in universe.items() if s not in records]
        mc = client or httpx.Client(timeout=30)
        def one_m(item):
            sym, xid = item
            try:
                minute, day = massive_symbol(mc, mcfg, sym, d)
                return sym, _record(sym, xid, d, cutoff, minute, day, "massive", datetime.now(UTC)), None
            except Exception as exc:
                return sym, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=mcfg["workers"]) as pool:
            for sym, rec, err in pool.map(one_m, todo):
                if rec:
                    records[sym] = rec
                elif err:
                    errors[sym] = err

    out = list(records.values())
    data_until = mark_complete(out, d, cutoff, now, pcfg["feed_delay_min"])
    if out:
        schema.validate_records(out, schema.load_schema(config.repo_root() / "schemas" / "etrade_premarket.schema.json"))
    path = write_records(out, d) if out else None
    lags = [(now - _parse_dt(r["last_bar_dt"])).total_seconds() for r in out
            if r["last_bar_dt"] and r["source"] == "etrade_chartiq"]
    lat = sorted(stats.latencies)
    summary = {
        "date": d.isoformat(), "cutoff": cutoff, "symbols": len(universe), "written": len(out),
        "by_source": {s: sum(1 for r in out if r["source"] == s) for s in {r["source"] for r in out}},
        "with_bars": sum(1 for r in out if r["n_bars"]), "errors": len(errors),
        "data_until": data_until, "complete": sum(1 for r in out if r["complete"]),
        "auth_error": auth_error, "fallback_used": use_massive and auth_error is not None,
        "latency_s": ({"calls": len(lat), "mean": round(mean(lat), 3), "p95": round(lat[int(0.95 * (len(lat) - 1))], 3),
                       "max": round(lat[-1], 3)} if lat else None),
        "newest_bar_lag_s": round(min(lags), 1) if lags else None,
        "path": str(path) if path else None,
    }
    if path:
        path.with_name(f"premarket-{d.isoformat()}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary
