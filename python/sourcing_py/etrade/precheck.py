"""The 09:29:30 pre-check (`sourcing-py etrade premarket precheck --date D`): a RANKING only.

It never feeds the model. It picks which stocks get the early 09:30 price forecast:

    shortlist = the top `[premarket] shortlist_size` by score
                + every stock with |move so far| >= move_pct % or volume ratio >= vol_ratio

Inputs (each optional; the check degrades, it does not fail):
    Massive snapshot   one GET /v2/snapshot/locale/us/markets/stocks/tickers (15 min delayed):
                       day{o,h,l,c,v,vw}, min{av,c,t}, prevDay{c,v,vw}, todaysChangePerc
    watchlist rows     the newest INTRADAY_FEED rows per symbol, read from the collector's Redis
                       stream (live bid/ask mid and volume_cum, about 1k symbols)
    night quantiles    the night forecast parquet (symbol, q05..q95 as written by tft-vwap-predict)
    vol_10d_mean       panel.next.parquet (mean Massive daily volume of the last 10 sessions)

Features per stock: move so far vs the previous VWAP, volume so far / vol_10d_mean, night P5-P95
width, |night median|. Score = weighted mean of the available features' percentile ranks. With no
snapshot and no watchlist the night quantiles alone rank the shortlist.

Redis read (no consumer group, so nothing is acked or disturbed): XREVRANGE on
`mtap:evt:INTRADAY_FEED` (common.bus.stream_key), newest first. Each entry's `data` field is the
batch envelope {rows: [...]} (schemas/intraday_feed.md); fields that are null are left out of a row,
so per symbol each field takes its newest non-null value, with bid and ask taken together.

Writes premarket_live/shortlist-D.csv (symbol, score, reasons).
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import date
from pathlib import Path

import httpx

from ..common import config, http
from ..common.bus import stream_key

EVENT = "INTRADAY_FEED"
WEIGHTS = {"move": 0.35, "vol": 0.25, "width": 0.25, "median": 0.15}
_PAIR = ("bid", "ask")


# -- watchlist rows from Redis ------------------------------------------------------------
def merge_rows(entries: list[dict]) -> dict[str, dict]:
    """Newest-first batch envelopes -> {symbol: merged row}. Newest non-null value per field;
    bid and ask come together from the newest row that has both."""
    out: dict[str, dict] = {}
    for env in entries:
        for row in env.get("rows") or []:
            sym = row.get("symbol")
            if not sym:
                continue
            cur = out.setdefault(sym, {})
            for k, v in row.items():
                if v is None or k in _PAIR:
                    continue
                cur.setdefault(k, v)
            if "bid" not in cur and row.get("bid") is not None and row.get("ask") is not None:
                cur["bid"], cur["ask"] = row["bid"], row["ask"]
    return out


def read_watchlist(url: str | None = None, *, client=None, count: int = 300) -> dict[str, dict]:
    """The latest watchlist rows from the collector's Redis stream; {} when Redis is unreachable."""
    try:
        if client is None:
            import redis

            client = redis.Redis.from_url(url or config.intraday_config()["redis_url"],
                                          decode_responses=True, socket_timeout=5)
        entries = []
        for _id, fields in client.xrevrange(stream_key(EVENT), count=count):
            try:
                env = json.loads(fields.get("data") or "")
            except ValueError:
                continue
            if isinstance(env, dict):
                entries.append(env)
        return merge_rows(entries)
    except Exception:  # Redis down / no stream: the pre-check runs without it
        return {}


# -- Massive snapshot ---------------------------------------------------------------------
def fetch_snapshot(client: httpx.Client | None = None) -> dict[str, dict]:
    """{ticker: snapshot record} from the one full-market call."""
    cfg = config.massive_config()
    url = f"{cfg['base_url']}/v2/snapshot/locale/us/markets/stocks/tickers"
    if client is not None:
        page = http.get_json(client, url, params={"apiKey": cfg["api_key"]})
    else:
        with http.get_client(timeout=60) as c:
            page = http.get_json(c, url, params={"apiKey": cfg["api_key"]})
    return {t["ticker"]: t for t in page.get("tickers") or [] if t.get("ticker")}


def _pos(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 else None


def snapshot_features(rec: dict) -> dict:
    """price / volume so far / previous VWAP from one snapshot record.
    (To confirm in S1: what `day` and `min` hold before the open; both are read defensively.)"""
    day, mn, prev = rec.get("day") or {}, rec.get("min") or {}, rec.get("prevDay") or {}
    return {
        "price": _pos(mn.get("c")) or _pos(day.get("c")),
        "volume": max(_pos(day.get("v")) or 0.0, _pos(mn.get("av")) or 0.0) or None,
        "prev_vwap": _pos(prev.get("vw")) or _pos(prev.get("c")),
        "change_pct": rec.get("todaysChangePerc") if isinstance(rec.get("todaysChangePerc"), (int, float)) else None,
    }


def watchlist_features(row: dict) -> dict:
    bid, ask = _pos(row.get("bid")), _pos(row.get("ask"))
    price = (bid + ask) / 2.0 if bid and ask and ask >= bid else (_pos(row.get("price")) if row.get("day_open") else None)
    return {"price": price, "volume": _pos(row.get("volume_cum")), "prev_close": _pos(row.get("prev_close"))}


# -- scoring ------------------------------------------------------------------------------
def percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    """Rank / (n - 1) in [0, 1]; ties share the average rank."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 0.5}
    out, i = {}, 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        for k in range(i, j + 1):
            out[items[k][0]] = ((i + j) / 2.0) / (n - 1)
        i = j + 1
    return out


def build_features(symbols: list[str], snapshot: dict[str, dict], rows: dict[str, dict],
                   quantiles: dict[str, dict], mean10: dict[str, float]) -> dict[str, dict]:
    """{symbol: {move, vol_ratio, width, median}} (None where unknown). Live watchlist values
    win over the delayed snapshot where both exist."""
    feats: dict[str, dict] = {}
    for s in symbols:
        snap = snapshot_features(snapshot[s]) if s in snapshot else {}
        wl = watchlist_features(rows[s]) if s in rows else {}
        price = wl.get("price") or snap.get("price")
        prev = snap.get("prev_vwap") or wl.get("prev_close")
        move = price / prev - 1.0 if price and prev else None
        if move is None and snap.get("change_pct") is not None:
            move = snap["change_pct"] / 100.0
        vol = wl.get("volume") or snap.get("volume")
        m10 = mean10.get(s)
        q = quantiles.get(s) or {}
        lo, hi, med = q.get("q05"), q.get("q95"), q.get("q50")
        feats[s] = {
            "move": move,
            "vol_ratio": vol / m10 if vol and m10 and m10 > 0 else None,
            "width": hi - lo if lo is not None and hi is not None else None,
            "median": abs(med) if med is not None else None,
        }
    return feats


def score(feats: dict[str, dict]) -> dict[str, float]:
    """Weighted mean of the percentile ranks of the features each stock has."""
    cols = {
        "move": {s: abs(f["move"]) for s, f in feats.items() if f["move"] is not None},
        "vol": {s: math.log1p(f["vol_ratio"]) for s, f in feats.items() if f["vol_ratio"] is not None},
        "width": {s: f["width"] for s, f in feats.items() if f["width"] is not None},
        "median": {s: f["median"] for s, f in feats.items() if f["median"] is not None},
    }
    ranks = {k: percentile_ranks(v) for k, v in cols.items() if v}
    out = {}
    for s in feats:
        num = den = 0.0
        for k, r in ranks.items():
            if s in r:
                num += WEIGHTS[k] * r[s]
                den += WEIGHTS[k]
        out[s] = num / den if den else 0.0
    return out


def shortlist(feats: dict[str, dict], scores: dict[str, float], *, size: int, move_pct: float,
              vol_ratio: float) -> list[tuple[str, float, str]]:
    """[(symbol, score, reasons)] best first: the top `size` plus every threshold crossing."""
    ordered = sorted(scores, key=lambda s: (-scores[s], s))
    top = set(ordered[:size])
    out = []
    for s in ordered:
        f, why = feats[s], []
        if s in top:
            why.append("top")
        if f["move"] is not None and abs(f["move"]) * 100.0 >= move_pct:
            why.append("move")
        if f["vol_ratio"] is not None and f["vol_ratio"] >= vol_ratio:
            why.append("volume")
        if why:
            out.append((s, scores[s], "+".join(why)))
    return out


# -- inputs from files --------------------------------------------------------------------
def read_night_quantiles(path: Path | None) -> dict[str, dict]:
    """{symbol: {q05, q50, q95}} from the night forecast parquet (qNN fractional-move columns)."""
    if not path or not Path(path).exists():
        return {}
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    need = [c for c in ("symbol", "q05", "q50", "q95") if c in t.column_names]
    cols = {c: t.column(c).to_pylist() for c in need}
    return {s: {c: cols[c][i] for c in need if c != "symbol"} for i, s in enumerate(cols.get("symbol", []))}


def read_panel_next(path: Path | None) -> tuple[list[str], dict[str, float]]:
    """(symbols, {symbol: vol_10d_mean}) from panel.next.parquet."""
    if not path or not Path(path).exists():
        return [], {}
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    syms = t.column("symbol").to_pylist()
    if "vol_10d_mean" not in t.column_names:
        return syms, {}
    return syms, {s: m for s, m in zip(syms, t.column("vol_10d_mean").to_pylist()) if m}


def output_path(d: date) -> Path:
    return config.out_root() / "premarket_live" / f"shortlist-{d.isoformat()}.csv"


def write_shortlist(rows: list[tuple[str, float, str]], d: date) -> Path:
    path = output_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "score", "reasons"])
            for s, sc, why in rows:
                w.writerow([s, f"{sc:.6f}", why])
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def run_precheck(d: date, *, night: Path | None = None, panel_next: Path | None = None,
                 snapshot: dict[str, dict] | None = None, rows: dict[str, dict] | None = None,
                 client: httpx.Client | None = None, redis_client=None, use_snapshot: bool = True) -> dict:
    """Score and write shortlist-D.csv. `snapshot` / `rows` are injected by tests; otherwise they
    are fetched (a failed snapshot call or an unreachable Redis just removes that input)."""
    pcfg = config.premarket_config()
    snap_error = None
    if snapshot is None and use_snapshot:
        try:
            snapshot = fetch_snapshot(client)
        except Exception as exc:  # no snapshot: the night quantiles (and the watchlist) rank
            snapshot, snap_error = {}, f"{type(exc).__name__}: {exc}"
    snapshot = snapshot or {}
    rows = rows if rows is not None else read_watchlist(client=redis_client)
    quant = read_night_quantiles(night)
    panel_syms, mean10 = read_panel_next(
        panel_next if panel_next is not None else config.out_root() / "tft" / "panel.next.parquet")
    symbols = sorted(set(panel_syms) or set(quant) or set(rows))
    feats = build_features(symbols, snapshot, rows, quant, mean10)
    scores = score(feats)
    short = shortlist(feats, scores, size=pcfg["shortlist_size"], move_pct=pcfg["move_pct"],
                      vol_ratio=pcfg["vol_ratio"])
    path = write_shortlist(short, d)
    return {
        "date": d.isoformat(), "symbols": len(symbols), "shortlist": len(short),
        "pulled_by_threshold": sum(1 for _, _, w in short if "top" not in w),
        "snapshot": len(snapshot), "watchlist": len(rows), "night": len(quant),
        "snapshot_error": snap_error, "path": str(path),
    }
