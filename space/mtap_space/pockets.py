"""Pockets: where the later volume of a set of moments traded, as a smoothed density.

The map is NX x NY cells: x = log2(minutes ahead + 1) from now to the close, y = that minute's
VWAP vs the price now in σ (±YL). Each later minute adds its volume ÷ the stock's normal daily
volume × the moment's weight. The map is smoothed and normalised to 1; the a-pocket is the
smallest set of cells holding share a of it (a highest-density region), and can be several
separate blobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtap_space.bars import NMIN
from mtap_space.library import Library
from mtap_space.match import Match

NX, NY, YL = 36, 32, 4.0
XMAX = np.log2(NMIN + 1)
LEVELS = (0.5, 0.8, 0.95)
BW = 0.7                 # smoothing, in cells (the check found 0.7 closer to nominal than 1.0)
MAX_ROWS = 20_000        # above this many matches, pockets come from a weighted sample
CLIM_ROWS = 6_000        # climatology pockets from a sample this size
MINS = np.arange(NMIN)


def x_edges() -> np.ndarray:
    """Cell edges in minutes ahead (NX + 1 values, 0 … 390)."""
    return 2 ** (np.arange(NX + 1) * XMAX / NX) - 1


def y_edges() -> np.ndarray:
    """Cell edges in σ vs now (NY + 1 values, −YL … +YL)."""
    return np.linspace(-YL, YL, NY + 1)


def _cells(lib: Library, rows: np.ndarray, w: np.ndarray):
    """For moments `rows` (sorted by stock-day for the memory-mapped reads): each later minute
    with volume as (position in `rows`, flat cell index, weighted mass)."""
    e = lib.m["e"][rows]
    V = np.asarray(lib.V[e], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log(np.asarray(lib.VW[e], np.float64) / lib.m["now"][rows][:, None]) \
            / lib.sig[e][:, None]
    ah = MINS[None, :] - lib.m["p"][rows].astype(np.int64)[:, None]
    xb = np.minimum((np.log2(np.maximum(ah, 0) + 1) / XMAX * NX).astype(np.int64), NX - 1)
    yb = np.clip(((np.nan_to_num(y) + YL) / (2 * YL) * NY).astype(np.int64), 0, NY - 1)
    ok = (ah >= 0) & (V > 0) & np.isfinite(y)
    mass = V / lib.nv[e][:, None] * np.asarray(w, float)[:, None]
    return np.nonzero(ok)[0], (xb * NY + yb)[ok], mass[ok]


def _chunks(lib: Library, rows: np.ndarray, w: np.ndarray, size: int = 4096):
    order = np.argsort(lib.m["e"][rows], kind="stable")
    for k in range(0, len(rows), size):
        o = order[k:k + size]
        pos, cell, mass = _cells(lib, rows[o], np.asarray(w, float)[o])
        yield o[pos], cell, mass


def grid(lib: Library, rows: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted later volume of moments `rows` on the NX x NY map (not smoothed or normalised)."""
    G = np.zeros(NX * NY)
    for _, cell, mass in _chunks(lib, np.asarray(rows), w):
        G += np.bincount(cell, weights=mass, minlength=NX * NY)
    return G.reshape(NX, NY)


def _kernel(bw: float) -> np.ndarray:
    r = int(4 * bw + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / bw) ** 2)
    return k / k.sum()


def smooth(G: np.ndarray, bw: float = BW) -> np.ndarray:
    """Separable Gaussian blur, edges mirrored (as scipy.ndimage.gaussian_filter's default)."""
    if bw <= 0:
        return G.astype(float)
    k = _kernel(bw)
    r = len(k) // 2
    out = G.astype(float)
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r, r)
        P_ = np.pad(out, pad, mode="symmetric")
        n = out.shape[ax]
        out = sum(k[i] * np.take(P_, np.arange(i, i + n), axis=ax) for i in range(len(k)))
    return out


def density(G: np.ndarray, bw: float = BW) -> np.ndarray:
    """Smoothed and normalised to 1, with a floor so every cell has a (tiny) density."""
    Q = smooth(G, bw) + 1e-9
    return Q / Q.sum()


def level(Q: np.ndarray, a: float) -> float:
    """The density threshold of the a-pocket: cells with Q ≥ this hold share a of the mass."""
    s = np.sort(Q.ravel())[::-1]
    return float(s[min(int(np.searchsorted(np.cumsum(s), a)), len(s) - 1)])


def region(Q: np.ndarray) -> np.ndarray:
    """Per cell, the smallest pocket it is in: 0 = 50%, 1 = 80%, 2 = 95%, 3 = outside."""
    r = np.full(Q.shape, len(LEVELS), np.int8)
    for i, a in reversed(list(enumerate(LEVELS))):
        r[Q >= level(Q, a)] = i
    return r


def blobs(mask: np.ndarray) -> int:
    """Number of separate blobs (4-connected components) in a boolean map."""
    seen = np.zeros_like(mask, bool)
    n = 0
    for i, j in zip(*np.nonzero(mask)):
        if seen[i, j]:
            continue
        n += 1
        stack = [(i, j)]
        seen[i, j] = True
        while stack:
            a, b = stack.pop()
            for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= c < mask.shape[0] and 0 <= d < mask.shape[1] and mask[c, d] \
                        and not seen[c, d]:
                    seen[c, d] = True
                    stack.append((c, d))
    return n


def wquant(x: np.ndarray, w: np.ndarray, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(qs, c / c[-1], x[o])


@dataclass
class Reading:
    """What followed a set of moments (prices in σ vs the price now)."""
    Q: np.ndarray          # pocket density, NX x NY, sums to 1
    pu: float              # P(rest-of-day VWAP > now)
    rq: np.ndarray         # rest-of-day VWAP p10 / p50 / p90
    bq: np.ndarray         # bulge (highest high after now) p10 / p50 / p90
    sq: np.ndarray         # sag (lowest low after now) p10 / p50 / p90
    n: int = 0

    @property
    def above(self) -> float:
        """Share of the later volume above the price now."""
        return float(self.Q[:, NY // 2:].sum())

    def blobs(self) -> list[int]:
        return [blobs(self.Q >= level(self.Q, a)) for a in LEVELS]


def _sample(rows, w, k, rng):
    if len(rows) <= k:
        return rows, w
    pick = rng.choice(len(rows), k, replace=False, p=w / w.sum())
    return rows[pick], np.ones(k)


def read(lib: Library, rows: np.ndarray, w: np.ndarray, bw: float = BW,
         rng: np.random.Generator | None = None, max_rows: int = MAX_ROWS) -> Reading:
    """Readings of a weighted set of moments. Quantiles use every row; the pocket map uses a
    weighted sample of `max_rows` when there are more (weights then become equal)."""
    rng = rng or np.random.default_rng(0)
    w = np.asarray(w, float)
    m = lib.m
    gr, gw = _sample(rows, w, max_rows, rng)
    return Reading(Q=density(grid(lib, gr, gw), bw), pu=float(np.average(m["rod"][rows] > 0,
                                                                            weights=w)),
                   rq=wquant(m["rod"][rows], w), bq=wquant(m["bul"][rows], w),
                   sq=wquant(m["sag"][rows], w), n=len(rows))


def climatology(lib: Library, rows: np.ndarray, bw: float = BW, seed: int = 0) -> Reading:
    """Everything in the pool, unmatched: the baseline every match must beat."""
    return read(lib, rows, np.ones(len(rows)), bw, np.random.default_rng(seed), CLIM_ROWS)


def shrink(S: Reading, C: Reading, a: float) -> Reading:
    """Blend a match reading with climatology: a x match + (1 - a) x climatology."""
    return Reading(Q=a * S.Q + (1 - a) * C.Q, pu=a * S.pu + (1 - a) * C.pu,
                   rq=a * S.rq + (1 - a) * C.rq, bq=a * S.bq + (1 - a) * C.bq,
                   sq=a * S.sq + (1 - a) * C.sq, n=S.n)


def matched(lib: Library, M: Match, C: Reading, bw: float = BW, seed: int = 0) -> Reading:
    """The match's reading, shrunk to climatology by its N_eff days (climatology if no match)."""
    if M.n == 0:
        return Reading(Q=C.Q, pu=C.pu, rq=C.rq, bq=C.bq, sq=C.sq, n=0)
    return shrink(read(lib, M.rows, M.w, bw, np.random.default_rng(seed)), C, M.shrink)


def actual(lib: Library, row: int) -> np.ndarray:
    """What the query moment actually did: its own later volume on the map, normalised (zeros
    if nothing traded after it). Not smoothed."""
    G = grid(lib, np.array([row]), np.ones(1))
    return G / G.sum() if G.sum() > 0 else G


def coverage(A: np.ndarray, Q: np.ndarray) -> dict[float, float]:
    """Share of the actual later volume inside each predicted pocket (nominal = the level)."""
    return {a: float(A[Q >= level(Q, a)].sum()) for a in LEVELS}


def log_score(A: np.ndarray, Q: np.ndarray) -> float:
    """Σ actual share x log predicted density over the map (higher is better)."""
    return float((A * np.log(Q)).sum())


def split(lib: Library, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per moment (in the order of `rows`), its later volume ÷ normal volume that traded above
    the price now, and in total (unsmoothed; the share above is up / total)."""
    rows = np.asarray(rows)
    tot, up = np.zeros(len(rows)), np.zeros(len(rows))
    for pos, cell, mass in _chunks(lib, rows, np.ones(len(rows))):
        tot += np.bincount(pos, weights=mass, minlength=len(rows))
        hi = cell % NY >= NY // 2
        up += np.bincount(pos[hi], weights=mass[hi], minlength=len(rows))
    return up, tot


def cell_of(minutes_ahead: float, sigma: float) -> tuple[int, int]:
    i = min(int(np.log2(max(minutes_ahead, 0) + 1) / XMAX * NX), NX - 1)
    j = int(np.clip((sigma + YL) / (2 * YL) * NY, 0, NY - 1))
    return i, j


def cell_mass(lib: Library, rows: np.ndarray, w: np.ndarray, i: int, j: int) -> np.ndarray:
    """Each moment's weighted later volume in cell (i, j), unsmoothed, in the order of `rows`."""
    out = np.zeros(len(rows))
    for pos, cell, mass in _chunks(lib, np.asarray(rows), w):
        hit = cell == i * NY + j
        out += np.bincount(pos[hit], weights=mass[hit], minlength=len(rows))
    return out
