"""Matching: weight past moments by how close their parameters are to the query's."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mtap_space.library import Library

N0 = 10   # shrinkage: a = N_eff days / (N_eff days + N0)


@dataclass
class Query:
    """What to match: the query moment, the parameters and how close a match must be."""
    row: int                          # the query moment's row in lib.m
    params: list[str]
    window: int = 60                  # sessions of history before the query's day
    tod: int = 0                      # also match decision minutes within ± this many minutes
    min_dollar: float | None = None   # pool: only stocks trading ≥ this a day (prior 20 days)
    min_price: float | None = None
    own_stock: bool = False           # pool: only the query's own stock
    tol: dict[str, float] = field(default_factory=dict)   # per-parameter tolerance multiplier


@dataclass
class Match:
    rows: np.ndarray       # matched moments (rows of lib.m)
    w: np.ndarray          # their weights, 0 < w ≤ 1
    pool: int              # moments in the pool before matching
    nd: float              # N_eff days
    stocks: int

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def shrink(self) -> float:
        return self.nd / (self.nd + N0)


def weights(X: np.ndarray, x0: np.ndarray, tol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Epanechnikov product kernel: z = |x - x0| / tol per parameter; a moment matches if every
    z < 1, with weight Π(1 - z²) (1 at an exact match, falling to 0 at the edge)."""
    z = np.abs(X - x0) / tol
    ok = (z < 1).all(1)
    return np.flatnonzero(ok), np.prod(1 - z[ok] ** 2, 1)


def neff_days(s: np.ndarray, w: np.ndarray) -> float:
    """How many independent days the weights stand for: (Σ W_d)² / Σ W_d², W_d = a day's weight."""
    if len(w) == 0:
        return 0.0
    Wd = np.bincount(s - s.min(), weights=w)
    return float(Wd.sum() ** 2 / (Wd ** 2).sum())


def pool(lib: Library, q: Query) -> np.ndarray:
    e = int(lib.m["e"][q.row])
    return lib.pool(int(lib.m["p"][q.row]), int(lib.m["s"][q.row]), q.window, q.tod,
                    q.min_dollar, q.min_price, lib.tk[e] if q.own_stock else None)


def tolerances(lib: Library, q: Query, dial: float) -> np.ndarray:
    return np.array([dial * lib.spread[k] * q.tol.get(k, 1.0) for k in q.params])


def run(lib: Library, q: Query, dial: float, rows: np.ndarray | None = None) -> Match:
    """Match the query against its pool (or `rows`, a pool computed once for several dials)."""
    c0 = pool(lib, q) if rows is None else rows
    if not q.params:
        hit, w = np.arange(len(c0)), np.ones(len(c0))
    else:
        hit, w = weights(lib.X(c0, q.params), lib.X(np.array([q.row]), q.params)[0],
                         tolerances(lib, q, dial))
    r = c0[hit]
    e = lib.m["e"][r]
    return Match(rows=r, w=w, pool=len(c0), nd=neff_days(lib.m["s"][r].astype(np.int64), w),
                 stocks=len(np.unique(lib.tk[e])))
