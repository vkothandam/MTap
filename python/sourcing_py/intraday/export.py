"""`sourcing-py intraday export-tft`: the intraday ML panel from the canonical feed.

One row per (symbol, date, decision second t). t steps every `step_s` seconds from the
open, with one random phase per date (shared by all symbols, so cross-symbol aggregates
line up). Every window is computed exactly at the feed's own resolution (2 s MAIN,
~20 s MidCap) from cumulative price·volume and volume sums, never from 1-minute bars:

    PV(x), V(x)       = sums over the feed's volume changes up to time x
    VWAP over (a, b]  = (PV(b) − PV(a)) / (V(b) − V(a))

Inputs at t use only events at or before t. Targets are the VWAPs of windows that start
at t, as log moves from the last price at t (`r_k = ln(fvwap_k / last_t)`), stored on the
decision row; ML shifts them for the TFT. The field contract is ml/FEATURES-intraday.md.

Reads `DATA_ROOT/intraday_feed/date=<D>/source=*/*.parquet` and the E*TRADE DuckDB
(read-only: daily_bars, sectors, results releases). Writes, atomically,
`OUT_ROOT/tft_intraday/panel/date=<D>/part.parquet` and `panel.meta.json`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..common import config
from ..utils import trading_calendar
from . import feed
from .upload import nas_dir

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TARGETS = ("r_1m", "r_5m", "r_15m", "r_60m", "r_rod", "r_close")  # bit i of `tmask`
HORIZON_S = {"r_1m": 60, "r_5m": 300, "r_15m": 900, "r_60m": 3600}
TRAIL_S = {"1": 60, "5": 300, "15": 900, "60": 3600}
CLOSE_GRACE_S = 30  # the closing cross prints in the first poll after the close
BASIS_TOL = 0.01  # feed previous close vs daily_bars close(D−1)
# 60-minute and rest-of-day windows tolerate gaps up to this (volume still carries across a
# gap in the cumulative sums; only its pricing is approximate)
LONG_GAP_S = 600
MIN_GAP_S = 30  # shorter windows: max(this, 2.5 × the symbol's poll interval)
TYPICAL_GRID = np.linspace(0.0, 1.0, 101)

# Columns by role; ML reads these names from panel.meta.json.
FEATURES = {
    "static_categoricals": ["sector_code", "watchlist", "day_of_week", "is_half_day"],
    "static_reals": [
        "log_dollar_vol_10d", "vwap_pct_prev_day", "intraday_spread_pct", "close_position_pct",
        "days_since_prev_session", "days_since_results",
    ],
    "known_categoricals": ["tod_bucket"],
    "known_reals": ["t_s", "session_frac", "minutes_to_close", "expected_vol_remaining"],
    "observed_reals": [
        # price vs VWAP (levels vs prev_vwap)
        "y", "L_trail_1", "L_trail_5", "L_trail_15", "L_trail_60", "L_session",
        "slope_1_5", "slope_5_15", "vwap_dev",
        # volume
        "vol_frac_10d", "rvol", "vol_step", "sv_1", "sv_5", "sv_15",
        # sub-minute
        "vslope_10s", "vslope_30s",
        # returns and volatility
        "ret_1", "ret_5", "ret_15", "ret_60", "rv_5", "rv_15", "rv_60",
        # day so far
        "y_open", "y_high_sofar", "y_low_sofar", "range_pos", "cum_sv", "rv_sofar",
        "frac_above_vwap", "z_trail_5",
        # results released today
        "released_today", "results_age_min",
        # other symbols at the same second (excluding self)
        "mkt_trail_5", "mkt_trail_15", "mkt_breadth", "sec_trail_5", "sec_rvol", "sec_breadth",
        "excess_5", "rank_trail_5", "rank_rvol",
        # quality
        "staleness_s", "since_trade_s",
    ],
}
REFERENCE_COLUMNS = [
    "last", "prev_vwap", "prev_close", "avg_vol_10d", "ref_trail_1m", "ref_trail_5m",
    "ref_trail_15m", "ref_trail_60m", "ref_session",
]


def panel_root() -> Path:
    return config.out_root() / "tft_intraday" / "panel"


def feed_days() -> list[date]:
    root = nas_dir()
    if not root.exists():
        return []
    out = []
    for p in root.glob("date=*"):
        try:
            out.append(date.fromisoformat(p.name.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(d for d in out if trading_calendar.is_trading_day(d))


# --------------------------------------------------------------------------- feed


def load_feed(day: date) -> pa.Table:
    """The day's canonical rows (all sources, deduped on symbol + time), sorted."""
    files = sorted((nas_dir() / f"date={day.isoformat()}").glob("source=*/*.parquet"))
    cols = ["symbol", "ts_utc", "price", "volume_cum", "day_open", "prev_close", "source_meta"]
    tables = [pq.read_table(f, columns=cols) for f in files]
    tables = [t for t in tables if t.num_rows]
    if not tables:
        return pa.table({c: [] for c in cols})
    t = pa.concat_tables(tables)
    t = t.append_column("ms", pc.cast(t.column("ts_utc"), pa.int64()))
    t = t.sort_by([("symbol", "ascending"), ("ms", "ascending")])
    sym = t.column("symbol").to_numpy(zero_copy_only=False)
    ms = t.column("ms").to_numpy()
    keep = np.ones(len(ms), bool)
    keep[1:] = (sym[1:] != sym[:-1]) | (ms[1:] != ms[:-1])
    return t.filter(pa.array(keep))


@dataclass
class Series:
    """One symbol's RTH event arrays, from its last premarket poll to the close."""

    symbol: str
    watchlist: str
    T: np.ndarray  # poll times, ms (T[0] = the last premarket poll)
    P: np.ndarray  # last price at each poll
    CPV: np.ndarray  # cumulative price·volume since the open, at each poll
    CV: np.ndarray  # cumulative volume since the open
    CSV: np.ndarray  # cumulative signed volume (tick rule)
    CRV: np.ndarray  # cumulative squared log returns of the last price
    traded_at: np.ndarray  # time of the latest poll with a volume change, at each poll
    day_open: float | None
    prev_close_feed: float | None  # E*TRADE's Previous Close, else the premarket price
    covered_close: bool
    gap_short_s: float


def build_series(symbol: str, watchlist: str, ms: np.ndarray, price: np.ndarray,
                 vol: np.ndarray, day_open: np.ndarray, prev_close: np.ndarray,
                 open_ms: int, close_ms: int) -> tuple[Series | None, str]:
    """(series, "") or (None, reason it was dropped)."""
    price = _ffill(price)
    vol = np.maximum.accumulate(np.nan_to_num(_ffill(vol), nan=0.0))
    pre = np.flatnonzero(ms <= open_ms)
    if len(pre) == 0:
        return None, "no_premarket"
    base = pre[-1]
    after = np.flatnonzero(ms >= close_ms + CLOSE_GRACE_S * 1000)
    end = after[0] if len(after) else len(ms) - 1
    covered_close = bool(len(after))
    if end <= base or np.isnan(price[base:end + 1]).all():
        return None, "no_rth"
    sl = slice(base, end + 1)
    T, P, V = ms[sl], price[sl], vol[sl]
    if np.isnan(P[0]):
        P = P.copy()
        P[0] = P[~np.isnan(P)][0]
    P = _ffill(P)
    rth_open = day_open[sl][~np.isnan(day_open[sl])]
    opened = float(rth_open[0]) if len(rth_open) else None
    pcs = prev_close[~np.isnan(prev_close)]
    pc_feed = float(pcs[0]) if len(pcs) else (float(price[base]) if not np.isnan(price[base]) else None)

    dv = np.diff(V)
    px = 0.5 * (P[:-1] + P[1:])
    first = np.flatnonzero(dv > 0)
    if len(first) and opened is not None:
        px[first[0]] = opened  # the opening cross, not the premarket print
    zero = np.zeros(1)
    CPV = np.concatenate([zero, np.cumsum(dv * px)])
    CV = np.concatenate([zero, np.cumsum(dv)])
    ret = np.diff(np.log(P))
    sign = np.sign(ret)
    for i in range(1, len(sign)):  # tick rule: no change keeps the last direction
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    CSV = np.concatenate([zero, np.cumsum(dv * sign)])
    CRV = np.concatenate([zero, np.cumsum(ret * ret)])
    traded = np.where(np.concatenate([[False], dv > 0]), T, -1)
    traded_at = np.maximum.accumulate(traded)
    traded_at = np.where(traded_at < 0, T[0], traded_at)
    step = np.median(np.diff(T)) / 1000 if len(T) > 1 else MIN_GAP_S
    return Series(symbol, watchlist, T, P, CPV, CV, CSV, CRV, traded_at, opened, pc_feed,
                  covered_close, max(MIN_GAP_S, 2.5 * step)), ""


def _ffill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    return out


def series_for_day(table: pa.Table, open_ms: int, close_ms: int) -> tuple[list[Series], dict]:
    sym = table.column("symbol").to_numpy(zero_copy_only=False)
    ms = table.column("ms").to_numpy()
    cols = {c: table.column(c).to_numpy(zero_copy_only=False).astype(float)
            for c in ("price", "volume_cum", "day_open", "prev_close")}
    metas = table.column("source_meta").to_pylist()
    starts = np.flatnonzero(np.concatenate([[True], sym[1:] != sym[:-1]]))
    ends = np.concatenate([starts[1:], [len(sym)]])
    out, dropped = [], {}
    for a, b in zip(starts, ends):
        meta = json.loads(metas[a]) if metas[a] else {}
        s, why = build_series(str(sym[a]), meta.get("watchlist") or "unknown", ms[a:b],
                              cols["price"][a:b], cols["volume_cum"][a:b], cols["day_open"][a:b],
                              cols["prev_close"][a:b], open_ms, close_ms)
        if s is None:
            dropped[why] = dropped.get(why, 0) + 1
        else:
            out.append(s)
    return out, dropped


# --------------------------------------------------------------------------- windows


def at(s: Series, x: np.ndarray) -> np.ndarray:
    """Index of the latest poll at or before each time x (−1 before the first)."""
    return np.searchsorted(s.T, x, side="right") - 1


def cum(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.where(idx >= 0, arr[np.clip(idx, 0, None)], 0.0)


def window_vwap(s: Series, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(VWAP, volume) over (a, b] for each pair; VWAP is NaN without volume."""
    ia, ib = at(s, a), at(s, b)
    dv = cum(s.CV, ib) - cum(s.CV, ia)
    dpv = cum(s.CPV, ib) - cum(s.CPV, ia)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dv > 0, dpv / dv, np.nan), dv


def gap_free(s: Series, a: np.ndarray, b: np.ndarray, max_gap_s: float) -> np.ndarray:
    """True where no feed gap longer than max_gap_s overlaps (a, b]."""
    g = np.diff(s.T)
    big = np.flatnonzero(g > max_gap_s * 1000)
    if not len(big):
        return np.ones(len(a), bool)
    gs, ge = s.T[big], s.T[big + 1]  # gap (gs, ge), sorted
    n_end_after_a = len(ge) - np.searchsorted(ge, a, side="right")
    n_start_from_b = len(gs) - np.searchsorted(gs, b, side="left")
    return (n_end_after_a - n_start_from_b) <= 0


def polls_in(s: Series, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.searchsorted(s.T, b, side="right") - np.searchsorted(s.T, a, side="right")


# --------------------------------------------------------------------------- daily context


def _duckdb():
    import duckdb

    # Read-only, and not etrade.db.connect(): that one runs the schema DDL.
    return duckdb.connect(str(config.etrade_config()["db_path"]), read_only=True)


def daily_context(day: date, symbols: list[str]) -> dict[str, dict]:
    """Per symbol: what was known before the open of D, plus D's own VWAP for r_close.

    Raw daily_bars values are on the basis of their own day; `f` = split_factor brings a
    row onto today's basis. Volume over D−10 … D−1 is only used when f is the same on
    every one of those days and D (no split inside the window)."""
    sess = trading_calendar.sessions(day - timedelta(days=40), day)
    if not sess or sess[-1] != day or len(sess) < 3:
        return {}
    prior = sess[:-1][-10:]
    d1, d2 = sess[-2], sess[-3]
    con = _duckdb()
    try:
        con.execute("CREATE TEMP TABLE _syms (symbol VARCHAR)")
        con.executemany("INSERT INTO _syms VALUES (?)", [(s,) for s in symbols])
        rows = con.execute(
            """
            SELECT b.symbol, b.date, b.close, b.vwap, b.high, b.low, b.volume,
                   COALESCE(b.split_factor, 1.0) AS f
            FROM daily_bars b JOIN _syms USING (symbol)
            WHERE b.date BETWEEN ? AND ?
            """,
            [min(prior[0], d2), day],
        ).fetchall()
        sectors = dict(con.execute(
            """
            SELECT s.symbol, min(i.sector_code) FROM symbol_industry s
            JOIN industries i USING (industry_code) JOIN _syms USING (symbol)
            GROUP BY s.symbol
            """
        ).fetchall())
        rel = con.execute(
            """
            SELECT r.symbol, r.accepted_at FROM results_announcements r JOIN _syms USING (symbol)
            WHERE r.accepted_at IS NOT NULL AND r.accepted_at < ?
            ORDER BY r.symbol, r.accepted_at
            """,
            [datetime.combine(day + timedelta(days=1), time(0))],
        ).fetchall()
    finally:
        con.close()

    bars: dict[str, dict[date, tuple]] = {}
    for sym, d, close, vwap, high, low, vol, f in rows:
        bars.setdefault(sym, {})[d] = (close, vwap, high, low, vol, f)
    releases: dict[str, list[datetime]] = {}
    for sym, acc in rel:
        releases.setdefault(sym, []).append(acc)

    open_ny = datetime.combine(day, time(9, 30))
    prev_close_ny = datetime.combine(d1, time(16, 0))
    out = {}
    for sym in symbols:
        b = bars.get(sym, {})
        if d1 not in b or b[d1][0] is None or b[d1][1] is None:
            continue
        c1, v1, h1, l1, _, f1 = b[d1]
        fs = [b[d][5] for d in prior if d in b]
        vols = [b[d][4] for d in prior if d in b and b[d][4] is not None]
        f_day = b[day][5] if day in b else f1
        avg_vol = (float(np.mean(vols)) if len(vols) >= 5 and len(set(fs)) == 1 and fs[0] == f_day
                   else None)
        vwap_d = b[day][1] if day in b else None
        v2 = b[d2][1] if d2 in b else None
        rel_s = releases.get(sym, [])
        before = [a for a in rel_s if a < open_ny]
        today = [a for a in rel_s if a >= prev_close_ny]
        if before:
            last_rel = before[-1]
            dsr = 0 if last_rel >= prev_close_ny else len([s for s in sess if s > last_rel.date()]) - 1
        else:
            dsr = 63
        out[sym] = {
            "close_1": c1, "vwap_1": v1, "high_1": h1, "low_1": l1, "f_1": f1, "f_day": f_day,
            # VWAP(D) / VWAP(D−1) on one basis: the daily model's target.
            "close_ratio": (vwap_d * f_day) / (v1 * f1) if vwap_d else None,
            "vwap_pct_prev_day": (v1 / (v2 * (b[d2][5] / f1)) - 1) if v2 else 0.0,
            "intraday_spread_pct": (h1 - l1) / v1 if h1 is not None and l1 is not None else 0.0,
            "close_position_pct": (c1 - v1) / v1,
            "avg_vol_10d": avg_vol,
            "sector_code": sectors.get(sym),
            "days_since_results": float(np.clip(dsr, 0, 63)),
            "released_at": today[0] if today else None,
            "vwap_d_raw": vwap_d,
        }
    out["__meta__"] = {"days_since_prev_session": (day - d1).days}
    return out


# --------------------------------------------------------------------------- typical volume curve


def default_typical() -> np.ndarray:
    """A U-shaped intraday volume curve (cumulative share of the day by session fraction),
    used until a full session has been collected."""
    x = TYPICAL_GRID
    return x + 0.6 * np.sin(2 * np.pi * x) / (2 * np.pi)


def day_profile(series: list[Series], open_ms: int, close_ms: int) -> np.ndarray | None:
    """The median symbol's cumulative volume share by session fraction, for a full day."""
    curves = []
    length = close_ms - open_ms
    xs = open_ms + TYPICAL_GRID * length
    xs[-1] = close_ms + CLOSE_GRACE_S * 1000
    for s in series:
        if not s.covered_close or s.CV[-1] <= 0:
            continue
        curves.append(cum(s.CV, at(s, xs)) / s.CV[-1])
    if len(curves) < 10:
        return None
    return np.median(np.vstack(curves), axis=0)


def profile_path(day: date) -> Path:
    return panel_root().parent / "profiles" / f"{day.isoformat()}.json"


def typical_for(day: date) -> tuple[np.ndarray, list[str]]:
    """Mean profile over full sessions before D (a default curve when there are none)."""
    used, curves = [], []
    for p in sorted(profile_path(day).parent.glob("*.json")):
        d = date.fromisoformat(p.stem)
        if d >= day:
            continue
        body = json.loads(p.read_text())
        if body.get("curve"):
            curves.append(np.array(body["curve"]))
            used.append(d.isoformat())
    if not curves:
        return default_typical(), []
    return np.mean(np.vstack(curves), axis=0), used


def _write_profile(day: date, curve: np.ndarray | None, full: bool) -> None:
    path = profile_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"date": day.isoformat(), "full_session": bool(full),
            "curve": [round(float(v), 6) for v in curve] if curve is not None and full else None}
    _atomic_text(path, json.dumps(body))


def ensure_profiles(days: list[date]) -> None:
    """Profiles for every feed day before the last exported day, computed if missing."""
    if not days:
        return
    for d in feed_days():
        if d >= max(days) or profile_path(d).exists():
            continue
        open_ms, close_ms = _session_ms(d)
        series, _ = series_for_day(load_feed(d), open_ms, close_ms)
        _write_profile(d, day_profile(series, open_ms, close_ms), _is_full_day(d, series))


def _is_full_day(day: date, series: list[Series]) -> bool:
    open_ms, close_ms = _session_ms(day)
    if close_ms - open_ms < 6.5 * 3600 * 1000 - 1:
        return False  # half days have their own shape
    return bool(series) and np.mean([s.covered_close for s in series]) > 0.9


# --------------------------------------------------------------------------- the panel


def _session_ms(day: date) -> tuple[int, int]:
    o = trading_calendar.session_open_utc(day)
    c = trading_calendar.session_close_utc(day)
    return int(o.timestamp() * 1000), int(c.timestamp() * 1000)


def phase_for(day: date, step_s: int) -> int:
    """One random phase per date (stable across reruns)."""
    return int(np.random.default_rng(int(day.strftime("%Y%m%d"))).integers(0, step_s))


def build_day(day: date, step_s: int = 15) -> tuple[pa.Table | None, dict]:
    open_ms, close_ms = _session_ms(day)
    table = load_feed(day)
    series, dropped = series_for_day(table, open_ms, close_ms)
    info: dict = {"date": day.isoformat(), "feed_rows": table.num_rows, "dropped": dropped}
    del table
    _write_profile(day, day_profile(series, open_ms, close_ms), _is_full_day(day, series))
    if not series:
        return None, info
    ctx = daily_context(day, [s.symbol for s in series])
    typical, typical_days = typical_for(day)
    info["typical_from"] = typical_days

    # Keep series with a matching basis and a 10-day volume.
    kept: list[Series] = []
    basis_bad, no_daily, no_vol = [], 0, 0
    for s in series:
        c = ctx.get(s.symbol)
        if c is None:
            no_daily += 1
            continue
        pc_feed = s.prev_close_feed
        if pc_feed is None or abs(pc_feed / c["close_1"] - 1) > BASIS_TOL:
            basis_bad.append(s.symbol)
            continue
        if not c["avg_vol_10d"]:
            no_vol += 1
            continue
        kept.append(s)
    dropped.update(no_daily_bars=no_daily, basis_mismatch=len(basis_bad), no_avg_vol=no_vol)
    info["basis_mismatch"] = basis_bad[:50]
    if not kept:
        return None, info

    phase = phase_for(day, step_s)
    cov_end = max(int(s.T[-1]) for s in kept)
    G = open_ms + phase * 1000 + step_s * 1000 * np.arange(0, (cov_end - open_ms) // (step_s * 1000) + 1)
    G = G[G <= min(cov_end, close_ms)]
    n, m = len(kept), len(G)
    t_s = ((G - open_ms) // 1000).astype(np.int32)
    length_s = (close_ms - open_ms) / 1000
    sess_frac = t_s / length_s
    typ = np.interp(sess_frac, TYPICAL_GRID, typical)
    is_half = int(length_s < 6.5 * 3600 - 1)
    released_cut = {}

    F: dict[str, np.ndarray] = {}

    def mat(name: str, dtype=np.float64) -> np.ndarray:
        F[name] = np.full((n, m), np.nan, dtype)
        return F[name]

    for name in (FEATURES["observed_reals"] + list(TARGETS) + REFERENCE_COLUMNS):
        mat(name)
    valid = np.zeros((n, m), bool)
    tmask = np.zeros((n, m), np.int8)
    sectors = np.array([ctx[s.symbol]["sector_code"] or "" for s in kept], object)

    for i, s in enumerate(kept):
        c = ctx[s.symbol]
        idx = at(s, G)
        ok = (idx >= 0) & (G <= s.T[-1]) & (cum(s.CV, idx) > 0)
        if not ok.any():
            continue
        valid[i] = ok
        last = s.P[np.clip(idx, 0, None)]
        # Levels vs prev_vwap without mixing bases: ratios to the feed's previous close,
        # then D−1's close/VWAP ratio from daily_bars (basis-free).
        pc_feed = s.prev_close_feed
        lvl0 = np.log(c["close_1"] / c["vwap_1"])
        prev_vwap = pc_feed * c["vwap_1"] / c["close_1"]

        def L(x, pc_feed=pc_feed, lvl0=lvl0):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(x / pc_feed) + lvl0

        y = L(last)
        sv_now = cum(s.CV, idx)
        spv = cum(s.CPV, idx)
        session_vwap = spv / np.where(sv_now > 0, sv_now, np.nan)
        F["y"][i] = y
        F["last"][i] = last
        F["prev_vwap"][i] = prev_vwap
        F["prev_close"][i] = pc_feed
        F["avg_vol_10d"][i] = c["avg_vol_10d"]
        F["L_session"][i] = L(session_vwap)
        F["vwap_dev"][i] = last / session_vwap - 1
        F["ref_session"][i] = np.log(session_vwap / last)
        trail = {}
        for k, sec in TRAIL_S.items():
            v, _ = window_vwap(s, G - sec * 1000, G)
            v = np.where(np.isnan(v), last, v)
            trail[k] = v
            F[f"L_trail_{k}"][i] = L(v)
            F[f"ref_trail_{k}m"][i] = np.log(v / last)
        F["slope_1_5"][i] = F["L_trail_1"][i] - F["L_trail_5"][i]
        F["slope_5_15"][i] = F["L_trail_5"][i] - F["L_trail_15"][i]
        for name, sec in (("vslope_10s", 10), ("vslope_30s", 30)):
            now, _ = window_vwap(s, G - sec * 1000, G)
            before, _ = window_vwap(s, G - 2 * sec * 1000, G - sec * 1000)
            F[name][i] = np.nan_to_num(np.log(now / before), nan=0.0, posinf=0.0, neginf=0.0)

        av = c["avg_vol_10d"]
        F["vol_frac_10d"][i] = sv_now / av
        F["rvol"][i] = (sv_now / av) / np.maximum(typ, 0.02)
        F["vol_step"][i] = (sv_now - cum(s.CV, at(s, G - step_s * 1000))) / av
        csv_now = cum(s.CSV, idx)
        for k in ("1", "5", "15"):
            F[f"sv_{k}"][i] = (csv_now - cum(s.CSV, at(s, G - TRAIL_S[k] * 1000))) / av
        F["cum_sv"][i] = csv_now / av
        logp = np.log(last)
        crv_now = cum(s.CRV, idx)
        for k in ("1", "5", "15", "60"):
            j = at(s, G - TRAIL_S[k] * 1000)
            past = s.P[np.clip(j, 0, None)]
            F[f"ret_{k}"][i] = logp - np.log(past)
            if k != "1":
                F[f"rv_{k}"][i] = np.sqrt(np.maximum(crv_now - cum(s.CRV, j), 0))
        F["rv_sofar"][i] = np.sqrt(np.maximum(crv_now, 0))
        F["z_trail_5"][i] = np.clip(np.nan_to_num((F["L_trail_5"][i] - y) / np.maximum(F["rv_sofar"][i], 1e-3)), -10, 10)

        opened = s.day_open if s.day_open is not None else s.P[1]
        F["y_open"][i] = L(np.array(opened))
        hi = np.maximum.accumulate(s.P[1:]) if len(s.P) > 1 else s.P
        lo = np.minimum.accumulate(s.P[1:]) if len(s.P) > 1 else s.P
        hidx = np.clip(idx - 1, 0, len(hi) - 1)
        high, low = np.maximum(hi[hidx], opened), np.minimum(lo[hidx], opened)
        F["y_high_sofar"][i] = L(high)
        F["y_low_sofar"][i] = L(low)
        F["range_pos"][i] = np.where(high > low, (last - low) / np.where(high > low, high - low, 1), 0.5)
        above = (last > session_vwap).astype(float)
        F["frac_above_vwap"][i] = np.cumsum(np.where(ok, above, 0)) / np.maximum(np.cumsum(ok), 1)

        rel = c["released_at"]
        if rel is not None:
            rel_ms = int(rel.replace(tzinfo=feed.NY).timestamp() * 1000)
            released_cut[s.symbol] = rel.isoformat()
            F["released_today"][i] = (G >= rel_ms).astype(float)
            F["results_age_min"][i] = np.where(G >= rel_ms, np.clip((G - rel_ms) / 60000, 0, 390), 0)
        else:
            F["released_today"][i] = 0.0
            F["results_age_min"][i] = 0.0
        F["staleness_s"][i] = np.clip((G - s.T[np.clip(idx, 0, None)]) / 1000, 0, 600)
        F["since_trade_s"][i] = np.clip((G - s.traded_at[np.clip(idx, 0, None)]) / 1000, 0, 600)

        # Targets: VWAP of windows starting at t, as log moves from the last price at t.
        end_time = s.T[-1] if s.covered_close else close_ms + CLOSE_GRACE_S * 1000
        bits = np.zeros(m, np.int8)
        for bit, name in enumerate(TARGETS[:4]):
            k = HORIZON_S[name]
            b = np.minimum(G + k * 1000, end_time)
            fv, fvol = window_vwap(s, G, b)
            gap = LONG_GAP_S if k >= 3600 else s.gap_short_s
            good = (b <= s.T[-1]) & (fvol > 0) & gap_free(s, G, b, gap) & (b > G)
            if name == "r_1m":
                good &= polls_in(s, G, b) >= 2
            F[name][i] = np.where(good, np.log(fv / last), np.nan)
            bits |= (good.astype(np.int8) << bit)
        fv, fvol = window_vwap(s, G, np.full(m, end_time))
        good = s.covered_close & (fvol > 0) & gap_free(s, G, np.full(m, end_time), LONG_GAP_S)
        F["r_rod"][i] = np.where(good, np.log(fv / last), np.nan)
        bits |= (good.astype(np.int8) << 4)
        if c["close_ratio"]:
            # ln(VWAP(D)/VWAP(D−1)) − y: the day's VWAP as a move from the last price.
            F["r_close"][i] = np.log(c["close_ratio"]) - y
            bits |= np.int8(1 << 5)
        tmask[i] = np.where(ok, bits, 0)

    # Other symbols at the same second, excluding self.
    _group_context(F, valid, sectors)

    # Flatten the valid cells.
    ii, jj = np.nonzero(valid)
    cols: dict[str, pa.Array] = {
        "symbol": pa.array([kept[i].symbol for i in ii], pa.string()),
        "date": pa.array([day.isoformat()] * len(ii), pa.string()),
        "step": pa.array(jj.astype(np.int32)),
        "t_s": pa.array(t_s[jj]),
        "phase_s": pa.array(np.full(len(ii), phase, np.int16)),
        "tod_bucket": pa.array((t_s[jj] // 900).astype(np.int16)),
        "session_frac": pa.array(sess_frac[jj].astype(np.float32)),
        "minutes_to_close": pa.array(((length_s - t_s[jj]) / 60).astype(np.float32)),
        "expected_vol_remaining": pa.array((1 - typ[jj]).astype(np.float32)),
        "sector_code": pa.array([sectors[i] or None for i in ii], pa.string()),
        "watchlist": pa.array([kept[i].watchlist for i in ii], pa.string()),
        "day_of_week": pa.array(np.full(len(ii), day.weekday(), np.int8)),
        "is_half_day": pa.array(np.full(len(ii), is_half, np.int8)),
    }
    for name in FEATURES["static_reals"]:
        if name == "days_since_prev_session":
            vals = np.full(len(ii), ctx["__meta__"]["days_since_prev_session"], float)
        elif name == "log_dollar_vol_10d":
            vals = np.array([np.log(ctx[kept[i].symbol]["avg_vol_10d"] * ctx[kept[i].symbol]["close_1"])
                             for i in ii])
        else:
            vals = np.array([ctx[kept[i].symbol][name] for i in ii], float)
        cols[name] = pa.array(vals.astype(np.float32))
    for name in FEATURES["observed_reals"]:
        cols[name] = pa.array(np.nan_to_num(F[name][ii, jj], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    for name in TARGETS:
        cols[name] = pa.array(F[name][ii, jj].astype(np.float32))
    cols["tmask"] = pa.array(tmask[ii, jj])
    for name in REFERENCE_COLUMNS:
        cols[name] = pa.array(F[name][ii, jj].astype(np.float64 if name in ("last", "prev_vwap", "prev_close") else np.float32))
    out = pa.table(cols)

    info.update(_coverage(kept, open_ms, close_ms), symbols=n, rows=out.num_rows, phase_s=phase,
                steps=m, results_released=len(released_cut),
                target_valid={t: float(np.mean((cols["tmask"].to_numpy() >> b) & 1)) for b, t in enumerate(TARGETS)},
                vwap_check=_vwap_check(kept, ctx) if _is_full_day(day, series) else None)
    return out, info


def _group_context(F: dict[str, np.ndarray], valid: np.ndarray, sectors: np.ndarray) -> None:
    """Universe and sector aggregates per column (decision second), each excluding the
    symbol itself. A sector with fewer than 5 other members falls back to the universe."""
    v = valid.astype(float)
    above = np.where(valid, (F["vwap_dev"] > 0).astype(float), 0.0)
    log_rvol = np.where(valid, np.log(np.maximum(np.nan_to_num(F["rvol"], nan=1.0), 1e-3)), 0.0)

    def loo(x: np.ndarray, members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.where(members, np.nan_to_num(x), 0.0)
        tot, cnt = xs.sum(0, keepdims=True), members.sum(0, keepdims=True)
        others = cnt - members
        with np.errstate(invalid="ignore", divide="ignore"):
            return (tot - xs) / np.where(others > 0, others, np.nan), others

    t5 = F["L_trail_5"] - F["y"]  # the trailing VWAP vs the last price: basis-free
    t15 = F["L_trail_15"] - F["y"]
    members = v > 0
    F["mkt_trail_5"][:], _ = loo(t5, members)
    F["mkt_trail_15"][:], _ = loo(t15, members)
    F["mkt_breadth"][:], _ = loo(above, members)
    mkt = {"t5": F["mkt_trail_5"].copy(), "rvol": loo(log_rvol, members)[0], "br": F["mkt_breadth"].copy()}
    F["sec_trail_5"][:] = mkt["t5"]
    F["sec_rvol"][:] = mkt["rvol"]
    F["sec_breadth"][:] = mkt["br"]
    for sec in set(sectors) - {""}:
        rows = sectors == sec
        mem = members & rows[:, None]
        for name, x in (("sec_trail_5", t5), ("sec_rvol", log_rvol), ("sec_breadth", above)):
            val, others = loo(x, mem)
            use = mem & (others >= 5)
            F[name][use] = val[use]
    F["excess_5"][:] = t5 - F["sec_trail_5"]
    for name, x in (("rank_trail_5", t5), ("rank_rvol", log_rvol)):
        filled = np.where(members, np.nan_to_num(x), np.inf)
        order = np.argsort(np.argsort(filled, axis=0), axis=0).astype(float)
        cnt = members.sum(0, keepdims=True)
        F[name][:] = np.where(members, order / np.maximum(cnt - 1, 1), np.nan)


def _coverage(series: list[Series], open_ms: int, close_ms: int) -> dict:
    by = {}
    for s in series:
        w = by.setdefault(s.watchlist, {"symbols": 0, "last_poll": 0, "covered_close": 0})
        w["symbols"] += 1
        w["last_poll"] = max(w["last_poll"], int(s.T[-1]))
        w["covered_close"] += int(s.covered_close)
    for w in by.values():
        w["last_poll"] = datetime.fromtimestamp(w["last_poll"] / 1000, feed.NY).strftime("%H:%M:%S")
    return {"coverage": by}


def _vwap_check(series: list[Series], ctx: dict) -> dict:
    """The feed's RTH VWAP vs daily_bars VWAP(D) on a full day, in basis points."""
    diffs = {}
    for s in series:
        c = ctx.get(s.symbol)
        if not c or not s.covered_close or s.CV[-1] <= 0 or not c["vwap_d_raw"]:
            continue
        diffs[s.symbol] = 1e4 * (s.CPV[-1] / s.CV[-1] / c["vwap_d_raw"] - 1)
    if not diffs:
        return {}
    a = np.abs(np.array(list(diffs.values())))
    worst = sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:20]
    return {"n": len(a), "median_abs_bp": round(float(np.median(a)), 2),
            "p90_abs_bp": round(float(np.percentile(a, 90)), 2),
            "share_within_10bp": round(float(np.mean(a <= 10)), 3),
            "worst": [[k, round(v, 1)] for k, v in worst]}


# --------------------------------------------------------------------------- driver


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_day(table: pa.Table, day: date) -> Path:
    path = panel_root() / f"date={day.isoformat()}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def export(days: list[date] | None = None, step_s: int = 15) -> dict:
    days = sorted(days) if days else feed_days()
    ensure_profiles(days)
    meta_path = panel_root() / "panel.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("step_s") not in (None, step_s):
        meta = {}  # a different grid: start the day list over
    meta.update({
        "schema_version": SCHEMA_VERSION, "step_s": step_s, "targets": list(TARGETS),
        "tmask_bits": {t: b for b, t in enumerate(TARGETS)},
        "horizon_s": {**HORIZON_S, "r_rod": None, "r_close": None},
        "features": FEATURES, "reference_columns": REFERENCE_COLUMNS,
        "target_unit": "log move from the last price at t",
        "panel_dir": str(panel_root()),
    })
    per_day = meta.setdefault("days", {})
    for day in days:
        table, info = build_day(day, step_s)
        if table is None:
            log.warning("%s: nothing to export (%s)", day, info.get("dropped"))
            per_day.pop(day.isoformat(), None)
            continue
        write_day(table, day)
        info["exported_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        per_day[day.isoformat()] = info
        log.info("%s: %d rows, %d symbols, valid %s", day, info["rows"], info["symbols"],
                 {k: round(v, 2) for k, v in info["target_valid"].items()})
    meta["dates"] = sorted(per_day)
    meta["rows"] = int(sum(d["rows"] for d in per_day.values()))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(meta_path, json.dumps(meta, indent=2, default=str))
    return meta
