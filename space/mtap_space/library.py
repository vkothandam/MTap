"""The moment library: every eligible stock-day of the last N sessions, one moment per 15 minutes.

A **moment** is one stock at one decision time of one day (09:45 … 15:45). For each moment it
keeps the parameters (what the day looked like so far, prices in the stock's σ) and the
readings of what followed (rest-of-day VWAP, bulge, sag). The minute volume and VWAP of every
eligible stock-day are kept too, so pockets can be drawn for any set of moments.

No look-ahead: σ, normal volume and dollar volume use the 20 sessions before the day;
parameters use the bars before the decision minute; only the readings use the bars from it on.

On disk (`<out>/lib-<last session>/`):
  meta.json          sessions, decision minutes, parameters, build settings
  stockdays.parquet  one row per eligible stock-day (row number = `e`)
  moments.parquet    one row per moment, sorted by (p, e)
  V.npy, VW.npy      (stock-days, 390) float32 minute volume and VWAP, opened with mmap
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mtap_space.bars import NMIN, dense, month_dirs, read_month

VERSION = 1
P = np.arange(15, 376, 15, dtype=np.int16)   # decision minutes after 09:30: 09:45 … 15:45
WARM = 25                                     # sessions read before the window (σ needs 20)
N_ROLL, MIN_ROLL = 20, 15
EPOCH = date(1970, 1, 1)


@dataclass(frozen=True)
class Param:
    label: str
    unit: str          # "σ" (a log price move over the stock's σ) or "log" (a log ratio)
    fn: Callable


def _rep(x):
    return np.repeat(np.asarray(x)[:, None], len(P), 1)


# A new parameter is one entry here. `c` holds per stock-day vectors (sig, prev, open, d1..d3,
# nv) and per-moment (stock-day x P) arrays (now, vws, low, high, back30, cv, curve).
PARAMS: dict[str, Param] = {
    "vwap_vs_prev": Param("VWAP so far vs the previous VWAP", "σ",
                          lambda c: np.log(c.vws / c.prev[:, None]) / c.sig[:, None]),
    "open_vs_prev": Param("The open vs the previous VWAP", "σ",
                          lambda c: _rep(np.log(c.open / c.prev) / c.sig)),
    "move_since_open": Param("The move since the open", "σ",
                             lambda c: np.log(c.now / c.open[:, None]) / c.sig[:, None]),
    "day_1": Param("Day −1: yesterday's VWAP move", "σ", lambda c: _rep(c.d1 / c.sig)),
    "day_2": Param("Day −2: the VWAP move the day before", "σ", lambda c: _rep(c.d2 / c.sig)),
    "path_30m": Param("The last 30 minutes", "σ",
                      lambda c: np.log(c.now / c.back30) / c.sig[:, None]),
    "day_3": Param("Day −3: the VWAP move three days back", "σ", lambda c: _rep(c.d3 / c.sig)),
    "rvol": Param("Volume so far vs normal for the time of day", "log",
                  lambda c: np.log(c.cv / (c.nv[:, None] * c.curve[None, :]))),
    "from_high": Param("The price now vs the day's high so far", "σ",
                       lambda c: np.log(c.now / c.high) / c.sig[:, None]),
    "from_low": Param("The price now vs the day's low so far", "σ",
                      lambda c: np.log(c.now / c.low) / c.sig[:, None]),
}
DEFAULT_PARAMS = ["vwap_vs_prev", "open_vs_prev", "move_since_open", "day_1", "day_2"]
READINGS = ["rod", "bul", "sag"]   # rest-of-day VWAP, highest high and lowest low after now (σ)


def to_date(day: int) -> date:
    return date.fromordinal(EPOCH.toordinal() + int(day))


def hhmm(p: int) -> str:
    m = 570 + int(p)
    return f"{m // 60:02d}:{m % 60:02d}"


def _prior(x: np.ndarray, start: np.ndarray, f: str) -> np.ndarray:
    """Rolling mean / std of the 20 rows before each row of the same ticker (rows sorted by
    ticker, day); NaN when fewer than 15 of them are known."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    c0, c1, c2 = (np.concatenate([[0], np.cumsum(a)]) for a in (ok, v, v * v))
    i = np.arange(len(x))
    lo = np.maximum(i - N_ROLL, start)
    n, s1, s2 = c0[i] - c0[lo], c1[i] - c1[lo], c2[i] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s1 / n if f == "mean" else np.sqrt(np.maximum(s2 - s1 * s1 / n, 0) / (n - 1))
    return np.where(n >= MIN_ROLL, out, np.nan)


def _lag(x: np.ndarray, start: np.ndarray, k: int) -> np.ndarray:
    i = np.arange(len(x))
    return np.where(i - k >= start, x[np.maximum(i - k, 0)], np.nan)


def stock_days(d: pd.DataFrame) -> pd.DataFrame:
    """Add the prior-only columns to the daily table: prev (yesterday's VWAP), r (today's log VWAP
    move, NaN beyond ±0.4 as a split guard), sig, nv (normal volume), ndol (normal dollar
    volume), d1..d3 (the last three days' moves)."""
    d = d.sort_values(["tk", "day"], kind="stable").reset_index(drop=True)
    tk = d.tk.to_numpy()
    first = np.r_[True, tk[1:] != tk[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(d)), 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(d.dv > 0, d.dpv / d.dv.where(d.dv > 0), np.nan)
        prev = _lag(vwap, start, 1)
        r = np.log(vwap / prev)
    r[np.abs(r) > 0.4] = np.nan
    d["vwap"], d["prev"], d["r"] = vwap, prev, r
    d["sig"] = _prior(r, start, "std")
    d["nv"] = _prior(d.dv.to_numpy(float), start, "mean")
    d["ndol"] = _prior(d.dpv.to_numpy(float), start, "mean")
    for k in (1, 2, 3):
        d[f"d{k}"] = _lag(r, start, k)
    return d


def _eligible(d: pd.DataFrame, min_dollar: float, min_price: float) -> pd.Series:
    return ((d.sig > 0.003) & (d.nv > 0) & (d.ndol >= min_dollar) & (d.prev >= min_price)
            & d.open.notna() & (d.open > 0) & d[["d1", "d2", "d3"]].notna().all(axis=1))


def _curve(dailies, bars_by_month, warm_days, min_dollar):
    """The typical share of the day's volume traded by the end of each minute, from the warm-up
    sessions (before the window, so no moment's parameter sees its own day). Uniform if none."""
    hist, n = np.zeros(NMIN), 0
    for daily, bars in zip(dailies, bars_by_month):
        use = daily.day.isin(warm_days).to_numpy() & (daily.dpv >= min_dollar).to_numpy() \
            & (daily.dv > 0).to_numpy()
        if not use.any():
            continue
        w = np.where(use, 1.0 / np.where(daily.dv > 0, daily.dv, 1), 0.0)
        hist += np.bincount(bars["mi"], weights=bars["v"] * w[bars["sd"]], minlength=NMIN)
        n += int(use.sum())
    if n == 0:
        return np.arange(1, NMIN + 1) / NMIN
    return np.cumsum(hist) / n


def build(bars_root: str | Path, sessions: int = 90, until: str | None = None,
          min_dollar: float = 5e6, min_price: float = 3.0, log=print) -> SimpleNamespace:
    """Read the last `sessions` + 25 sessions of minute bars and build the library in memory."""
    t0 = time.time()
    until_day = (date.fromisoformat(until) - EPOCH).days if until else None
    months = [m for m in month_dirs(bars_root)
              if until is None or m.name.removeprefix("month=") <= until[:7]]
    if not months:
        raise FileNotFoundError(f"no month=YYYY-MM folders in {bars_root}")
    dailies, bars_by_month, days = [], [], set()
    for folder in reversed(months):
        daily, bars = read_month(folder)
        if until_day is not None:
            daily = daily.assign(keep=daily.day <= until_day)
        else:
            daily = daily.assign(keep=True)
        dailies.insert(0, daily)
        bars_by_month.insert(0, bars)
        days |= set(daily.day[daily.keep].tolist())
        log(f"  {folder.name}: {daily.day.nunique()} sessions, {len(bars['v']):,} bars "
            f"({time.time() - t0:.0f}s)")
        if len(days) >= sessions + WARM:
            break
    allday = pd.concat([d.assign(month=i, row=np.arange(len(d)))
                        for i, d in enumerate(dailies)], ignore_index=True)
    allday = stock_days(allday[allday.keep].drop(columns="keep"))
    sess = np.sort(allday.day.unique())
    win = sess[max(0, len(sess) - sessions):]
    el = _eligible(allday, min_dollar, min_price) & allday.day.isin(win)
    de = allday[el].copy()
    de["s"] = np.searchsorted(win, de.day.to_numpy()).astype(np.int16)
    de = de.sort_values(["s", "tk"], kind="stable").reset_index(drop=True)
    if de.empty:
        raise ValueError("no eligible stock-days: too few sessions, or the filters drop all")
    curve = _curve(dailies, bars_by_month, sess[sess < win[0]], min_dollar)
    rows = [np.full(len(d), -1, np.int64) for d in dailies]
    for i, g in de.groupby("month"):
        rows[i][g.row.to_numpy()] = g.index.to_numpy()
    A = dense(list(zip(bars_by_month, rows)), len(de))
    del bars_by_month, dailies
    lib = moments(de, A, curve)
    lib.meta.update(
        version=VERSION, built=datetime.now(UTC).isoformat(timespec="seconds"),
        bars_root=str(bars_root), sessions=[str(to_date(x)) for x in win],
        sessions_wanted=sessions, until=until, min_dollar=min_dollar, min_price=min_price,
        build_s=round(time.time() - t0, 1))
    log(f"  library: {lib.meta['n_moments']:,} moments, {len(lib.sd):,} stock-days, "
        f"{len(win)} sessions ({to_date(win[0])} … {to_date(win[-1])}), {time.time() - t0:.0f}s")
    return lib


def moments(de: pd.DataFrame, A: dict[str, np.ndarray], curve: np.ndarray) -> SimpleNamespace:
    """Parameters and readings for every stock-day x decision minute. `A` holds the dense
    minute arrays (V, VW, C, H, L) of the stock-days in `de`, row for row."""
    V, VW, C, H, L = (A[k] for k in ("V", "VW", "C", "H", "L"))
    PV = np.where(V > 0, V * VW, 0).astype(np.float64)
    cV, cPV = np.cumsum(V, 1, dtype=np.float64), np.cumsum(PV, 1)
    Pm = P.astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = SimpleNamespace(
            sig=de.sig.to_numpy(), prev=de.prev.to_numpy(), open=de.open.to_numpy(),
            nv=de.nv.to_numpy(), d1=de.d1.to_numpy(), d2=de.d2.to_numpy(), d3=de.d3.to_numpy(),
            now=C[:, Pm - 1].astype(np.float64), back30=C[:, np.maximum(Pm - 31, 0)],
            vws=cPV[:, Pm - 1] / np.where(cV[:, Pm - 1] > 0, cV[:, Pm - 1], np.nan),
            low=np.fmin.accumulate(L, 1)[:, Pm - 1], high=np.fmax.accumulate(H, 1)[:, Pm - 1],
            cv=cV[:, Pm - 1], curve=curve[Pm - 1])
        rest_v = cV[:, -1:] - cV[:, Pm - 1]
        rod = (cPV[:, -1:] - cPV[:, Pm - 1]) / np.where(rest_v > 0, rest_v, np.nan)
        hs = np.flip(np.fmax.accumulate(np.flip(H, 1), 1), 1)[:, Pm]
        ls = np.flip(np.fmin.accumulate(np.flip(L, 1), 1), 1)[:, Pm]
        sg = c.sig[:, None]
        F = {k: p.fn(c) for k, p in PARAMS.items()}
        F["rod"] = np.log(rod / c.now) / sg
        F["bul"] = np.log(hs / c.now) / sg
        F["sag"] = np.log(ls / c.now) / sg
    # p-major order, so every decision minute is one contiguous block sorted by e (and by s)
    m = {k: np.asarray(v, np.float32).T.ravel() for k, v in F.items()}
    m["now"] = c.now.T.ravel().astype(np.float32)
    n = len(de)
    m["e"] = np.tile(np.arange(n, dtype=np.int32), len(P))
    m["p"] = np.repeat(P, n)
    ok = np.ones(len(m["e"]), bool)
    for k in list(PARAMS) + READINGS:
        ok &= np.isfinite(m[k])
    ok &= m["now"] > 0
    m = {k: v[ok] for k, v in m.items()}
    m["s"] = de.s.to_numpy()[m["e"]].astype(np.int16)
    sd = de[["tk", "day", "s", "sig", "nv", "ndol", "prev", "open", "vwap", "dv",
             "d1", "d2", "d3"]].copy()
    # each parameter's spread (IQR / 1.349, a robust std): the unit of the match tolerance
    spread = {k: float(np.subtract(*np.percentile(m[k], [75, 25])) / 1.349) or 1.0
              for k in PARAMS}
    meta = {"P": P.tolist(), "params": {k: {"label": p.label, "unit": p.unit}
                                        for k, p in PARAMS.items()}, "spread": spread,
            "default_params": DEFAULT_PARAMS, "n_moments": int(ok.sum()), "n_stockdays": n,
            "curve": np.round(curve, 6).tolist()}
    return SimpleNamespace(sd=sd, m=m, V=V, VW=VW, meta=meta)


def save(lib: SimpleNamespace, out: str | Path) -> Path:
    """Write `<out>/lib-<last session>/` (replacing one of the same name)."""
    out = Path(out)
    final = out / f"lib-{lib.meta['sessions'][-1]}"
    tmp = out / f".{final.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(lib.sd, preserve_index=False), tmp / "stockdays.parquet")
    pq.write_table(pa.table(lib.m), tmp / "moments.parquet")
    np.save(tmp / "V.npy", lib.V)
    np.save(tmp / "VW.npy", lib.VW)
    (tmp / "meta.json").write_text(json.dumps(lib.meta, indent=1))
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    return final


def find(root: str | Path) -> list[Path]:
    """Library directories under `root` (or `root` itself), newest session first."""
    root = Path(root)
    if (root / "meta.json").exists():
        return [root]
    return sorted((p for p in root.glob("lib-*") if (p / "meta.json").exists()), reverse=True)


class Library:
    """A saved library, opened read-only. Moments are column arrays in `m`, sorted by (p, e)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.sd = pq.read_table(self.path / "stockdays.parquet").to_pandas()
        t = pq.read_table(self.path / "moments.parquet")
        self.m = {k: t.column(k).to_numpy() for k in t.column_names}
        self.V = np.load(self.path / "V.npy", mmap_mode="r")
        self.VW = np.load(self.path / "VW.npy", mmap_mode="r")
        self._index()

    @classmethod
    def from_memory(cls, lib: SimpleNamespace) -> Library:
        """A Library over a `build()` result without writing it (tests)."""
        self = cls.__new__(cls)
        self.path, self.meta, self.sd, self.m, self.V, self.VW = (
            None, lib.meta, lib.sd, lib.m, lib.V, lib.VW)
        self._index()
        return self

    def _index(self):
        self.sessions = self.meta["sessions"]
        self.params = list(self.meta["params"])
        self.spread = self.meta["spread"]
        self.sig, self.nv = self.sd.sig.to_numpy(), self.sd.nv.to_numpy()
        self.tk, self.day = self.sd.tk.to_numpy(), self.sd.day.to_numpy()
        self.P = np.asarray(self.meta["P"])
        p = self.m["p"]
        self._block = {int(x): (int(np.searchsorted(p, x)), int(np.searchsorted(p, x, "right")))
                       for x in self.P}
        self._sd_index = {(t, int(d)): i for i, (t, d) in enumerate(zip(self.tk, self.day))}

    # ---- looking things up --------------------------------------------------------------------

    def stocks(self) -> pd.DataFrame:
        """Every stock in the library, most traded first: tk, days, ndol (median $ a day)."""
        g = self.sd.groupby("tk")
        return (pd.DataFrame({"days": g.size(), "ndol": g.ndol.median()})
                .sort_values("ndol", ascending=False).reset_index())

    def days_of(self, tk: str) -> pd.DataFrame:
        """The stock's library days, newest first: e, day, date."""
        e = np.flatnonzero(self.tk == tk)[::-1]
        return pd.DataFrame({"e": e, "day": self.day[e],
                             "date": [str(to_date(x)) for x in self.day[e]]})

    def stockday(self, tk: str, date_: str) -> int | None:
        return self._sd_index.get((tk, (date.fromisoformat(date_) - EPOCH).days))

    def moment(self, e: int, p: int) -> int | None:
        """The row in `m` of stock-day e at decision minute p (None if it was dropped)."""
        lo, hi = self._block.get(int(p), (0, 0))
        i = lo + int(np.searchsorted(self.m["e"][lo:hi], e))
        return i if i < hi and self.m["e"][i] == e else None

    def times_of(self, e: int) -> list[int]:
        return [int(p) for p in self.P if self.moment(e, p) is not None]

    def pool(self, p0: int, s0: int, window: int, tod: int = 0, min_dollar: float | None = None,
             min_price: float | None = None, tk: str | None = None) -> np.ndarray:
        """Moments at decision minutes p0 ± tod, on the `window` sessions before s0 (never s0 or
        later), optionally only stocks above a dollar volume / price, or only stock `tk`."""
        out = []
        for p in self.P[np.abs(self.P - p0) <= tod]:
            lo, hi = self._block[int(p)]
            s = self.m["s"][lo:hi]
            a = lo + int(np.searchsorted(s, s0 - window))
            b = lo + int(np.searchsorted(s, s0))
            out.append(np.arange(a, b))
        idx = np.concatenate(out) if out else np.zeros(0, np.int64)
        e = self.m["e"][idx]
        keep = np.ones(len(idx), bool)
        if min_dollar:
            keep &= self.sd.ndol.to_numpy()[e] >= min_dollar
        if min_price:
            keep &= self.sd.prev.to_numpy()[e] >= min_price
        if tk:
            keep &= self.tk[e] == tk
        return idx[keep]

    def X(self, rows: np.ndarray, params: list[str]) -> np.ndarray:
        return np.column_stack([self.m[k][rows] for k in params]).astype(np.float64)

    def label(self, row: int) -> str:
        e = int(self.m["e"][row])
        return f"{self.tk[e]} {to_date(self.day[e])} {hhmm(self.m['p'][row])}"
