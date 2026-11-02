"""Tables for Intraday → Space, from a moment library (the mtap_space package, built with
`space-build`). Pure functions: a library and a query in, tables out; the page shows them.

A **moment** is one stock at one decision time of one day. The query moment is matched against
the moments of the same time of day on earlier sessions of every stock, at two tolerances (the
close and the loose dial). A **pocket** is where the later volume of a set of moments traded,
on a map of time ahead × price vs now (in the stock's σ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mtap_space import pockets as K
from mtap_space.library import Library, hhmm, to_date
from mtap_space.match import Match, Query, neff_days, run

VIEWS = {"close": "close match", "loose": "loose match", "clim": "everything",
         "actual": "what it did"}
REGIONS = ["inside 50%", "inside 80%", "inside 95%", "outside"]
CURVE_DIALS = np.round(np.geomspace(0.25, 2.0, 10), 3)
CURVE_ROWS = 3_000        # pocket sample per dial on the dial curve (quantiles use every row)
CELL_ROWS = 5_000         # picked cell: the heaviest matches searched for their volume there


@dataclass
class Space:
    """Everything shown for one query moment."""
    lib: Library
    q: Query
    dials: tuple[float, float]
    pool: np.ndarray                 # moments the query is matched against
    M: dict[str, Match]              # close, loose
    R: dict[str, K.Reading]          # close, loose, clim (match readings blended to clim)
    A: np.ndarray                    # what the query moment's later volume did (unsmoothed)

    @property
    def e(self) -> int:
        return int(self.lib.m["e"][self.q.row])

    @property
    def p(self) -> int:
        return int(self.lib.m["p"][self.q.row])

    def maps(self) -> dict[str, np.ndarray]:
        """The four pocket maps; what it did is smoothed like the others, for display only."""
        return {**{k: self.R[k].Q for k in ("close", "loose", "clim")},
                "actual": K.density(self.A) if self.A.sum() > 0 else self.R["clim"].Q * 0}


def evaluate(lib: Library, q: Query, dials: tuple[float, float], pool: np.ndarray,
             clim: K.Reading) -> Space:
    """Match at both dials and read the pockets (the pool and its climatology are given, so the
    page can cache them across dial and parameter changes)."""
    M = {k: run(lib, q, d, pool) for k, d in zip(("close", "loose"), dials)}
    R = {k: K.matched(lib, m, clim) for k, m in M.items()}
    R["clim"] = clim
    return Space(lib, q, dials, pool, M, R, K.actual(lib, q.row))


def query_values(lib: Library, row: int) -> pd.Series:
    """The query moment's value of every parameter in the library."""
    return pd.Series(lib.X(np.array([row]), lib.params)[0], index=lib.params)


def moment_info(lib: Library, row: int) -> dict:
    e = int(lib.m["e"][row])
    sd = lib.sd.iloc[e]
    return {"symbol": lib.tk[e], "date": str(to_date(lib.day[e])), "time": hhmm(lib.m["p"][row]),
            "now": float(lib.m["now"][row]), "sig": float(sd.sig), "prev": float(sd.prev),
            "open": float(sd.open), "nv": float(sd.nv), "ndol": float(sd.ndol)}


def evidence(sp: Space) -> pd.DataFrame:
    """One row per view: how many matches and days stand behind it, its readings (σ vs the
    price now) and, against what the moment then did, its pocket coverage and log score."""
    lib, m = sp.lib, sp.lib.m
    clim_ls = K.log_score(sp.A, sp.R["clim"].Q)
    has_actual = sp.A.sum() > 0
    rows = []
    for k in ("close", "loose", "clim"):
        r = sp.R[k]
        if k == "clim":
            e = m["e"][sp.pool]
            n, stocks = len(sp.pool), len(np.unique(lib.tk[e]))
            days = neff_days(m["s"][sp.pool].astype(np.int64), np.ones(len(sp.pool)))
            trust = np.nan
        else:
            M = sp.M[k]
            n, stocks, days, trust = M.n, M.stocks, M.nd, M.shrink if M.n else 0.0
        cov = K.coverage(sp.A, r.Q) if has_actual else {}
        rows.append({
            "view": VIEWS[k], "matches": n, "stocks": stocks, "days": days, "trust": trust,
            "p_up": r.pu, "rod_p10": r.rq[0], "rod_p50": r.rq[1], "rod_p90": r.rq[2],
            "bulge_p50": r.bq[1], "sag_p50": r.sq[1], "above_now": r.above,
            "blobs_50": r.blobs()[0], "in_50": cov.get(0.5, np.nan),
            "in_80": cov.get(0.8, np.nan),
            "log_score": K.log_score(sp.A, r.Q) - clim_ls if has_actual else np.nan})
    row = sp.q.row
    rows.append({
        "view": VIEWS["actual"], "p_up": float(m["rod"][row] > 0), "rod_p50": m["rod"][row],
        "bulge_p50": m["bul"][row], "sag_p50": m["sag"][row],
        "above_now": float(sp.A[:, K.NY // 2:].sum()) if has_actual else np.nan})
    out = pd.DataFrame(rows, index=list(VIEWS))
    for c in ("rod_p10", "rod_p50", "rod_p90", "bulge_p50", "sag_p50"):
        out[c] = out[c].astype(float)
    return out


def matches(sp: Space, n: int = 50) -> tuple[pd.DataFrame, np.ndarray]:
    """The loose match's `n` heaviest moments, closest first, with what followed each. Returns
    the table and the moments' rows in the library."""
    lib, M = sp.lib, sp.M["loose"]
    if M.n == 0:
        return pd.DataFrame(), np.zeros(0, np.int64)
    top = np.argsort(-M.w, kind="stable")[:n]
    rows, w = M.rows[top], M.w[top]
    e = lib.m["e"][rows]
    up, tot = K.split(lib, rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        above = np.where(tot > 0, up / tot, np.nan)
    t = pd.DataFrame({
        "symbol": lib.tk[e], "date": [str(to_date(d)) for d in lib.day[e]],
        "time": [hhmm(p) for p in lib.m["p"][rows]], "weight": w,
        "close": np.isin(rows, sp.M["close"].rows)})
    for k in sp.q.params:
        t[k] = lib.m[k][rows].astype(float)
    t["rest_of_day"] = lib.m["rod"][rows].astype(float)
    t["bulge"] = lib.m["bul"][rows].astype(float)
    t["sag"] = lib.m["sag"][rows].astype(float)
    t["above_now"] = above
    return t, rows


def thresholds(Q: np.ndarray) -> list[float]:
    """Density thresholds of the 50 / 80 / 95% pockets."""
    return [K.level(Q, a) for a in K.LEVELS]


def cell_table(sp: Space, i: int, j: int) -> pd.DataFrame:
    """Per view, its share of the later volume in cell (i, j), vs everything, and which pocket
    the cell is in."""
    maps = {**{k: sp.R[k].Q for k in ("close", "loose", "clim")}, "actual": sp.A}
    clim = sp.R["clim"].Q[i, j]
    out = []
    for k, Q in maps.items():
        reg = K.region(Q)[i, j] if k != "actual" else None
        out.append({"view": VIEWS[k], "share_here": float(Q[i, j]),
                    "vs_everything": float(Q[i, j] / clim) if clim > 0 else np.nan,
                    "pocket": REGIONS[reg] if reg is not None else "n/a"})
    return pd.DataFrame(out)


def cell_matches(sp: Space, i: int, j: int, k: int = 10) -> pd.DataFrame:
    """The loose matches that put the most (weighted) later volume into cell (i, j)."""
    lib, M = sp.lib, sp.M["loose"]
    if M.n == 0:
        return pd.DataFrame()
    top = np.argsort(-M.w, kind="stable")[:CELL_ROWS]
    rows, w = M.rows[top], M.w[top]
    mass = K.cell_mass(lib, rows, w, i, j)
    if mass.sum() <= 0:
        return pd.DataFrame()
    best = np.argsort(-mass, kind="stable")[:k]
    best = best[mass[best] > 0]
    e = lib.m["e"][rows[best]]
    return pd.DataFrame({
        "symbol": lib.tk[e], "date": [str(to_date(d)) for d in lib.day[e]],
        "time": [hhmm(p) for p in lib.m["p"][rows[best]]], "weight": w[best],
        "of_cell": mass[best] / mass.sum()})


def dial_curve(lib: Library, q: Query, pool: np.ndarray, clim: K.Reading,
               dials=CURVE_DIALS) -> pd.DataFrame:
    """For a range of tolerances: matches, stocks and N_eff days, and the blended P(up) and
    share of later volume above now (pockets from a sample of CURVE_ROWS per dial)."""
    out = []
    for d in dials:
        M = run(lib, q, float(d), pool)
        if M.n:
            raw = K.read(lib, M.rows, M.w, rng=np.random.default_rng(0), max_rows=CURVE_ROWS)
            r = K.shrink(raw, clim, M.shrink)
        else:
            r = clim
        out.append({"dial": float(d), "matches": M.n, "stocks": M.stocks, "days": M.nd,
                    "p_up": r.pu, "above_now": r.above})
    return pd.DataFrame(out)


def day_path(lib: Library, row: int) -> pd.DataFrame:
    """The query stock's day, minute by minute: VWAP of the minute, volume, the session VWAP so
    far, and whether the minute is after the decision time."""
    e, p = int(lib.m["e"][row]), int(lib.m["p"][row])
    V = np.asarray(lib.V[e], float)
    VW = np.asarray(lib.VW[e], float)
    cv = np.cumsum(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        sess = np.where(cv > 0, np.cumsum(V * VW) / cv, np.nan)
    mi = np.arange(len(V))
    return pd.DataFrame({"minute": mi, "time": [hhmm(x) for x in mi], "vwap": VW, "volume": V,
                         "session_vwap": sess, "after": mi >= p})


def to_price(now: float, sig: float, x):
    """σ vs now → $."""
    return now * np.exp(np.asarray(x, float) * sig)
