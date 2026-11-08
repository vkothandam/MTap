"""Causal zig-zag (docs/intraday-features.md, "Zig-zag").

A leg ends only when the counter-move from the leg's extreme reaches the threshold k * sigma_1m (as a
log change). The pivot is confirmed at that step, never earlier, so a value at step t depends only on
steps <= t (no repainting). Vectorised over rows, one loop over the steps.
"""
from __future__ import annotations

import numpy as np

COLS = ("zz_dir", "zz_len", "zz_age", "zz_pull", "zz_prev", "zz_legs")


def zigzag(x, thr, sig, at=None):
    """The six zig-zag columns per step of the log-price series x (rows x T).

    thr: rows x T thresholds in log units (k * sigma_1m for that step's time of day); sig: the daily
    sigma per row (sizes are in sigma). NaN prices hold the state. Before the first move that reaches the
    threshold there is no leg: zz_dir 0, zz_legs 0, the rest NaN. Steps before the first price are NaN.
    at: the steps to return (rows x len(at)), e.g. the minute ends of a 1 s series; default every step."""
    x = np.asarray(x, np.float64)
    thr = np.broadcast_to(np.asarray(thr, np.float64), x.shape)
    sig = np.asarray(sig, np.float64)
    n, T = x.shape
    col = np.arange(T) if at is None else np.full(T, -1)
    if at is not None:
        col[np.asarray(at)] = np.arange(len(at))
    out = {c: np.full((n, T if at is None else len(at)), np.nan, np.float32) for c in COLS}
    d = np.zeros(n)                    # 0 = no leg yet, +1 up, -1 down
    p0 = np.full(n, np.nan)            # the leg's start (the last pivot)
    ext = np.full(n, np.nan)           # the leg's extreme so far
    t0 = np.zeros(n)                   # step of the leg's start
    text = np.zeros(n)                 # step of the extreme
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)  # the range before the first leg
    tlo, thi = np.zeros(n), np.zeros(n)
    prev, legs = np.full(n, np.nan), np.zeros(n)
    seen = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        for t in range(T):
            v, th = x[:, t], thr[:, t]
            fin = np.isfinite(v)
            first = fin & ~seen
            lo[first], hi[first], tlo[first], thi[first] = v[first], v[first], t, t
            seen |= fin
            # before the first leg: track the range, start a leg once a move from it reaches thr
            pre = fin & (d == 0)
            nl, nh = pre & (v < lo), pre & (v > hi)
            lo[nl], tlo[nl], hi[nh], thi[nh] = v[nl], t, v[nh], t
            up = pre & (v - lo >= th)
            dn = pre & ~up & (hi - v >= th)
            for m, s, base, tb in ((up, 1, lo, tlo), (dn, -1, hi, thi)):
                d[m], p0[m], ext[m], t0[m], text[m], legs[m] = s, base[m], v[m], tb[m], t, 1
            # in a leg: extend the extreme, or confirm the pivot when the counter-move reaches thr
            on = fin & (d != 0) & ~up & ~dn
            ext_new = on & ((v - ext) * d > 0)
            ext[ext_new], text[ext_new] = v[ext_new], t
            brk = on & ((ext - v) * d >= th)
            prev[brk] = (ext[brk] - p0[brk]) * d[brk] / sig[brk]
            p0[brk], t0[brk] = ext[brk], text[brk]
            d[brk] = -d[brk]
            ext[brk], text[brk] = v[brk], t
            legs[brk] += 1
            c = col[t]
            if c < 0:
                continue
            leg = seen & (d != 0)
            out["zz_dir"][seen, c] = d[seen]
            out["zz_legs"][seen, c] = legs[seen]
            hv = np.where(fin, v, ext)
            out["zz_len"][leg, c] = ((ext - p0) * d / sig)[leg]
            out["zz_age"][leg, c] = (t - t0)[leg]
            out["zz_pull"][leg, c] = np.clip((ext - hv) * d / np.where(th > 0, th, np.nan), 0, 1)[leg]
            out["zz_prev"][leg, c] = prev[leg]
    return out
