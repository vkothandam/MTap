"""Morning live PRICES (`sourcing-py etrade premarket prices --date D --symbols-file F [--open]`).

This module is the one small live-price adapter behind `[premarket] live_source`. The morning's
volume, VWAP and range come from Massive (premarket_full.py); the shortlist from precheck.py.
Nothing here computes a volume or a VWAP, and ChartIQ volume is never read.

Per symbol:
    last   the watchlist bid/ask mid from the collector's Redis stream (before the open `price`
           is only the previous close, so it is not used until `day_open` shows the stock has
           opened), else ChartIQ's newest price bucket before the cutoff
    open   (--open) the watchlist `day_open`, else ChartIQ's first regular bucket (>= 09:30 ET)

Written atomically to out/premarket_live/prices-D.jsonl, records
{symbol, date, last, last_at, open, open_at, source, as_of} (schemas/premarket_prices.schema.json).
A --open pass keeps the last prices of the file already written and fills the opens.

ChartIQ facts (probe 2026-10-27): data.chartData.series[] = {DT (UTC ISO), Open, High, Low, Close};
buckets are 86.4 s whatever `dataInterval` says; about 2 sessions come back whatever `dateStart`
says, so bars are filtered by DT here; with several symbols in `elements` the first one sits at the
top level of each bar and the rest are nested under their XID (to confirm in S1: the nested shape,
which is read defensively as an object with Open/Close or a bare number).

The job only READS the bearer token the collector mirrors to state/etrade/bearer.json (or
session.json). It never logs in. The first 401/403/429 stops the ChartIQ part of the run (no retry).
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from ..common import config, schema
from ..common.errors import ConfigError
from .client import _ORIGIN, _bearer_token

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
AUTH_STATUSES = (401, 403, 429)
BATCH = 20  # symbols per ChartIQ price call


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



def xids_path() -> Path:
    return config.out_root() / "premarket_live" / "xids.csv"


def write_xids(db_path: Path | None = None) -> tuple[Path, int]:
    """symbol,XID lines from the DuckDB, for hosts without one (the NAS). Run by the night job."""
    xids = load_universe(None, None, db_path, panel=Path("/nonexistent"))
    path = xids_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text("".join(f"{s},{x}\n" for s, x in sorted(xids.items())))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path, len(xids)


# -- parsing (pure) ---------------------------------------------------------------------
@dataclass
class PBar:
    dt: datetime  # UTC
    open: float | None
    close: float | None


def _parse_dt(raw: str) -> datetime:
    if len(raw) == 10:
        return datetime.fromisoformat(raw).replace(tzinfo=UTC)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def parse_cutoff(cutoff: str) -> dtime:
    h, m = cutoff.split(":")
    return dtime(int(h), int(m))


def cutoff_utc(d: date, cutoff: str) -> datetime:
    return datetime.combine(d, parse_cutoff(cutoff), NY).astimezone(UTC)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _field(obj, *names) -> float | None:
    """Open/Close from a nested symbol object (any key case) or a bare number (the close)."""
    if isinstance(obj, dict):
        low = {str(k).lower(): v for k, v in obj.items()}
        return _num(next((low[n.lower()] for n in names if n.lower() in low), None))
    return _num(obj) if names[0] == "Close" else None


def parse_prices(resp: dict, xids: list[int]) -> dict[int, list[PBar]]:
    """{xid: bars sorted by time} from a (batched) price response. xids[0] is the first symbol
    of `elements` (top level of each bar); the others are nested under str(xid)."""
    series = (((resp or {}).get("data") or {}).get("chartData") or {}).get("series") or []
    out: dict[int, list[PBar]] = {x: [] for x in xids}
    for r in series:
        if not r.get("DT"):
            continue
        dt = _parse_dt(r["DT"])
        out[xids[0]].append(PBar(dt, _num(r.get("Open")), _num(r.get("Close"))))
        for x in xids[1:]:
            nested = r.get(str(x))
            if nested is not None:
                out[x].append(PBar(dt, _field(nested, "Open"), _field(nested, "Close")))
    return {x: sorted((b for b in bars if b.close is not None or b.open is not None), key=lambda b: b.dt)
            for x, bars in out.items()}


def last_before(bars: list[PBar], hi: datetime, now: datetime) -> PBar | None:
    """The newest bucket with a close that started before the cutoff (and not in the future)."""
    sel = [b for b in bars if b.close is not None and b.dt < hi and b.dt <= now]
    return sel[-1] if sel else None


def first_regular(bars: list[PBar], d: date, now: datetime) -> PBar | None:
    """The first bucket at or after 09:30 ET on d that has an open."""
    lo = datetime.combine(d, dtime(9, 30), NY).astimezone(UTC)
    sel = [b for b in bars if b.open is not None and b.dt >= lo and b.dt <= now]
    return sel[0] if sel else None



# -- E*TRADE ChartIQ (price-only) -------------------------------------------------------
class EtradeChart:
    def __init__(self, token: str, base_url: str, stop: threading.Event,
                 client: httpx.Client | None = None) -> None:
        self.url = f"{base_url.rstrip('/')}/getChartiqFeed"
        self.headers = {"accept": "application/json", "content-type": "application/json",
                        "origin": _ORIGIN, "referer": f"{_ORIGIN}/", "authorization": token}
        self.stop = stop
        self.client = client or httpx.Client(timeout=30)

    def _post(self, body: dict) -> dict:
        if self.stop.is_set():
            raise AuthStop("stopped after an earlier auth failure")
        resp = self.client.post(self.url, json=body, headers=self.headers)
        if resp.status_code in AUTH_STATUSES:
            self.stop.set()
            raise AuthStop(f"E*TRADE answered HTTP {resp.status_code}")
        resp.raise_for_status()
        return resp.json()

    def prices(self, xids: list[int], d: date, now: datetime) -> dict[int, list[PBar]]:
        """Price buckets of up to BATCH symbols in one call (no volume element)."""
        start = datetime.combine(d, dtime(4, 0), NY).astimezone(UTC)
        return parse_prices(self._post({
            "elements": [{"type": "price", "symbol": str(x)} for x in xids],
            "dataPeriod": "Minute", "dataInterval": 1, "extendedHours": True,
            "dateStart": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "dateEnd": now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }), xids)


# -- the live adapter -------------------------------------------------------------------
def _iso(dt: datetime | str | None) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def watchlist_price(row: dict | None) -> tuple[float, str | None] | None:
    """(last, time) from a latest-watchlist row: the bid/ask mid, else `price` once the stock
    has opened (before that `price` is the previous close). None when neither is live."""
    if not row:
        return None
    bid, ask = _num(row.get("bid")), _num(row.get("ask"))
    if bid and ask and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0, row.get("bid_at") or row.get("ts_utc")
    if _num(row.get("day_open")) and _num(row.get("price")):
        return float(row["price"]), row.get("price_at") or row.get("ts_utc")
    return None


def watchlist_open(row: dict | None) -> tuple[float, str | None] | None:
    if row and _num(row.get("day_open")):
        return float(row["day_open"]), row.get("day_open_at") or row.get("ts_utc")
    return None


class EtradeLive:
    """The `etrade` live source: watchlist rows first, ChartIQ price buckets for the rest."""

    name = "etrade"

    def __init__(self, rows: dict[str, dict], chart: EtradeChart | None, xids: dict[str, int],
                 workers: int = 4) -> None:
        self.rows, self.chart, self.xids, self.workers = rows, chart, xids, workers
        self.auth_error: str | None = None
        self.errors: dict[str, str] = {}

    def _chartiq(self, symbols: list[str], d: date, now: datetime) -> dict[str, list[PBar]]:
        if self.chart is None:
            return {}
        known = [s for s in symbols if s in self.xids]
        batches = [known[i:i + BATCH] for i in range(0, len(known), BATCH)]
        bars: dict[str, list[PBar]] = {}

        def one(batch):
            xs = [self.xids[s] for s in batch]
            try:
                return batch, self.chart.prices(xs, d, now), None
            except AuthStop as exc:
                return batch, None, ("auth", str(exc))
            except Exception as exc:  # one batch failing must not stop the others
                return batch, None, ("error", f"{type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for batch, res, err in pool.map(one, batches):
                if err and err[0] == "auth":
                    self.auth_error = err[1]
                elif err:
                    for s in batch:
                        self.errors[s] = err[1]
                else:
                    for s in batch:
                        bars[s] = res.get(self.xids[s], [])
        return bars

    def last_prices(self, symbols: list[str], d: date, cutoff: str, now: datetime) -> dict[str, dict]:
        """{symbol: {last, last_at, source}} for the symbols with a price."""
        out: dict[str, dict] = {}
        for s in symbols:
            wp = watchlist_price(self.rows.get(s))
            if wp:
                out[s] = {"last": wp[0], "last_at": wp[1], "source": "watchlist"}
        rest = [s for s in symbols if s not in out]
        hi = cutoff_utc(d, cutoff)
        for s, bars in self._chartiq(rest, d, now).items():
            b = last_before(bars, hi, now)
            if b:
                out[s] = {"last": b.close, "last_at": _iso(b.dt), "source": "etrade_chartiq"}
        return out

    def opens(self, symbols: list[str], d: date, now: datetime) -> dict[str, dict]:
        """{symbol: {open, open_at, source}} for the symbols that have opened."""
        out: dict[str, dict] = {}
        for s in symbols:
            wo = watchlist_open(self.rows.get(s))
            if wo:
                out[s] = {"open": wo[0], "open_at": wo[1], "source": "watchlist"}
        rest = [s for s in symbols if s not in out]
        for s, bars in self._chartiq(rest, d, now).items():
            b = first_regular(bars, d, now)
            if b:
                out[s] = {"open": b.open, "open_at": _iso(b.dt), "source": "etrade_chartiq"}
        return out


LIVE_SOURCES = {"etrade": EtradeLive}  # live Massive joins here later


# -- the run ----------------------------------------------------------------------------
def read_symbols_file(path: Path) -> list[str]:
    """One symbol per line, or a CSV whose first column is the symbol (header `symbol`)."""
    out = []
    for line in Path(path).read_text().splitlines():
        sym = line.split(",")[0].strip()
        if sym and sym.lower() != "symbol":
            out.append(sym)
    return out


def output_path(d: date) -> Path:
    return config.out_root() / "premarket_live" / f"prices-{d.isoformat()}.jsonl"


def read_existing(d: date) -> dict[str, dict]:
    path = output_path(d)
    if not path.exists():
        return {}
    return {r["symbol"]: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def write_records(records: list[dict], d: date) -> Path:
    """Atomic write (tmp + rename) to premarket_live/prices-D.jsonl."""
    path = output_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def run_prices(d: date, symbols: list[str], *, open_pass: bool = False, source: str | None = None,
               rows: dict[str, dict] | None = None, token: str | None = None,
               client: httpx.Client | None = None, now: datetime | None = None,
               xids: dict[str, int] | None = None) -> dict:
    """Fetch and write prices-D.jsonl. `rows` = the latest watchlist rows (read from Redis by
    the caller; {} when Redis is down). Returns a summary (includes `path`, `missing_open`)."""
    pcfg = config.premarket_config()
    source = source or pcfg["live_source"]
    if source not in LIVE_SOURCES:
        raise ConfigError(f"[premarket] live_source {source!r} is not one of {sorted(LIVE_SOURCES)}")
    now = now or datetime.now(UTC)
    rows = rows or {}
    auth_error = None
    chart = None
    need_chart = [s for s in symbols if not (watchlist_open(rows.get(s)) if open_pass else watchlist_price(rows.get(s)))]
    if need_chart:
        try:
            xmap = xids if xids is not None else load_universe(
                need_chart, xids_path() if xids_path().exists() else None)
            chart = EtradeChart(token or load_token(), config.etrade_config()["base_url"], threading.Event(), client)
        except AuthStop as exc:
            auth_error = str(exc)
            xmap = {}
        except ConfigError as exc:
            auth_error = f"no XIDs: {exc}"
            xmap = {}
    else:
        xmap = {}
    live = LIVE_SOURCES[source](rows, chart, xmap, pcfg["max_concurrency"])
    prev = read_existing(d)
    recs: dict[str, dict] = {}
    as_of = _iso(now)
    cutoff = pcfg["cutoff"]
    if open_pass:
        opens = live.opens(symbols, d, now)
        for s in symbols:
            r = dict(prev.get(s) or {"symbol": s, "date": d.isoformat(), "last": None, "last_at": None,
                                      "open": None, "open_at": None, "source": "none"})
            if s in opens and r.get("open") is None:
                r["open"], r["open_at"] = opens[s]["open"], opens[s]["open_at"]
                if r["last"] is None:
                    r["source"] = opens[s]["source"]
            r["as_of"] = as_of
            recs[s] = r
    else:
        lasts = live.last_prices(symbols, d, cutoff, now)
        for s in symbols:
            got = lasts.get(s) or {}
            old = prev.get(s) or {}
            recs[s] = {"symbol": s, "date": d.isoformat(), "last": got.get("last"), "last_at": got.get("last_at"),
                       "open": old.get("open"), "open_at": old.get("open_at"),
                       "source": got.get("source") or "none", "as_of": as_of}
    out = list(recs.values())
    schema.validate_records(out, schema.load_schema(config.repo_root() / "schemas" / "premarket_prices.schema.json"))
    path = write_records(out, d)
    auth_error = auth_error or live.auth_error
    by_source: dict[str, int] = {}
    for r in out:
        if r["last"] is not None:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    summary = {
        "date": d.isoformat(), "open_pass": open_pass, "symbols": len(symbols),
        "with_last": sum(1 for r in out if r["last"] is not None),
        "with_open": sum(1 for r in out if r["open"] is not None),
        "missing_open": sum(1 for r in out if r["open"] is None),
        "by_source": by_source, "errors": len(live.errors), "auth_error": auth_error, "path": str(path),
    }
    return summary
