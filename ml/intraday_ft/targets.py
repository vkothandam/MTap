"""Pocket targets over duration bins (docs/intraday-features.md §5).

Decision at the end of minute i (0-based column, 09:30 + i); the entry P0 = minute i+1's open. The
bins count minutes after the entry: b1 = minutes i+1 .. i+5, b2 = i+6 .. i+15, b3 = i+16 .. i+30,
b4 = i+31 .. i+60, b5 = i+61 .. close. Every target reads bars i+1 .. only.
"""
from __future__ import annotations

import numpy as np

BINS = ((0, 5), (5, 15), (15, 30), (30, 60), (60, 390))
NB = len(BINS)
BIN_TARGETS = ("bulge", "sag", "mass_up", "cm_up", "cm_dn", "net", "rvol", "occ_up", "occ_dn",
               "pi_up", "pi_dn")
ROW_TARGETS = ("t50_up", "t50_dn", "t_peak_up", "t_peak_dn", "vol_to_up1", "vol_to_dn1", "race",
               "mass_up60", "bulge60", "sag60", "net60")
CM_MIN = 0.05
RACE = 0.5
VOL_TO_CAP = 1.0  # vol_to share of a typical day when the level is not touched within 60 min


def pocket(d, v):
    """The pocket numbers of one bin from per-minute positions d (sigma vs P0) and volumes v (same
    shape, last axis = minutes; NaN d = no price). Returns bulge, sag, mass_up, mass_dn, mass_at, cm_up,
    cm_dn, net and the bin volume."""
    v = np.where(np.isfinite(d), v, 0.0)
    tot = v.sum(-1)
    with np.errstate(all="ignore"):
        s = v / np.where(tot > 0, tot, np.nan)[..., None]
        dd = np.nan_to_num(d)
        bulge = (s * np.maximum(dd, 0)).sum(-1)
        sag = (s * np.maximum(-dd, 0)).sum(-1)
        up, dn = (s * (dd > 0)).sum(-1), (s * (dd < 0)).sum(-1)
        at = (s * (dd == 0)).sum(-1)
        cm_up = np.where(up >= CM_MIN, bulge / up, np.nan)
        cm_dn = np.where(dn >= CM_MIN, sag / dn, np.nan)
    empty = ~(tot > 0)
    res = {"bulge": bulge, "sag": sag, "mass_up": up, "mass_dn": dn, "mass_at": at, "cm_up": cm_up,
           "cm_dn": cm_dn, "net": bulge - sag}
    res = {k: np.where(empty, np.nan, v) for k, v in res.items()}
    res["vol"] = tot
    return res


def bin_columns(i, lo, hi):
    """Minute columns of a bin for decision minutes i (the bin's minutes after the entry), clipped at
    the close: (cols rows x (hi - lo), valid mask, cut-at-close flag)."""
    w = hi - lo if hi < 390 else 390
    cols = np.asarray(i)[:, None] + 1 + lo + np.arange(w)[None, :]
    valid = cols < 390
    cut = (np.asarray(i) + 1 + min(hi, 390) > 390) if hi < 390 else np.zeros(len(cols), bool)
    return np.minimum(cols, 389), valid, cut


def targets(M, r, i):
    """Realised targets for decisions (rows r, minutes i): a dict of arrays, `<target>_b<k>` for the bin
    targets and the row-level ones, plus p0 (the entry). M: V, VW, H, L, O, Cf (rows x 390), sig, nvm
    (normal volume per minute, rows x 390), typical (rows). occ_up / occ_dn need the seconds: occupancy()."""
    r, i = np.asarray(r), np.asarray(i)
    s = M.sig[r]
    nxt = np.minimum(i + 1, 389)
    p0 = np.where(np.isfinite(M.O[r, nxt]), M.O[r, nxt], M.Cf[r, i])  # no trade in i+1: the held price
    p0 = np.where(i + 1 < 390, p0, np.nan)
    lp0 = np.log(p0)
    Y = {"p0": p0}
    for k, (lo, hi) in enumerate(BINS, 1):
        cols, valid, cut = bin_columns(i, lo, hi)
        vv = np.where(valid, M.V[r[:, None], cols], 0.0)
        with np.errstate(all="ignore"):
            d = (np.log(M.VW[r[:, None], cols]) - lp0[:, None]) / s[:, None]
        d = np.where(valid, d, np.nan)
        pk = pocket(d, vv)
        nv = np.where(valid, M.nvm[r[:, None], cols], 0.0).sum(1)
        mins = valid.sum(1).astype(np.float64)
        with np.errstate(all="ignore"):
            ratio = np.where(nv > 0, pk["vol"] / nv, np.nan)   # no normal volume: pi undefined (frames
            # built before 2026-11-07 hold inf there; windows.batch masks both the same way)
            Y[f"rvol_b{k}"] = np.log((pk["vol"] + 1) / (nv + 1))
            Y[f"pi_up_b{k}"] = pk["bulge"] * ratio / mins
            Y[f"pi_dn_b{k}"] = pk["sag"] * ratio / mins
        for t in ("bulge", "sag", "mass_up", "cm_up", "cm_dn", "net"):
            Y[f"{t}_b{k}"] = pk[t]
        Y[f"cut_b{k}"] = cut.astype(np.int8)
        Y[f"mins_b{k}"] = mins
    Y.update(_row_level(M, r, i, lp0, s, Y))
    return Y


def _row_level(M, r, i, lp0, s, Y):
    """t50_up / t50_dn, t_peak_up / t_peak_dn, vol_to_up1 / vol_to_dn1, the +-0.5 sigma race and the
    pocket of the whole 60 minutes after the entry (mass_up60, bulge60, sag60, net60: the run-file
    readouts)."""
    out = {}
    cols = np.asarray(i)[:, None] + 1 + np.arange(60)[None, :]
    valid = cols < 390
    cols = np.minimum(cols, 389)
    vv = np.where(valid, M.V[r[:, None], cols], 0.0)
    with np.errstate(all="ignore"):
        d = (np.log(M.VW[r[:, None], cols]) - lp0[:, None]) / s[:, None]
        hi = np.log(M.H[r[:, None], cols]) - lp0[:, None]
        lo = np.log(M.L[r[:, None], cols]) - lp0[:, None]
    p60 = pocket(np.where(valid, d, np.nan), vv)
    for t in ("mass_up", "bulge", "sag", "net"):
        out[f"{t}60"] = p60[t]
    d = np.where(valid, np.nan_to_num(d), 0.0)
    for side, sgn in (("up", 1), ("dn", -1)):
        w = vv * np.maximum(sgn * d, 0)
        c = np.cumsum(w, 1)
        tot = c[:, -1]
        half = (c >= 0.5 * tot[:, None]) & (tot[:, None] > 0)
        out[f"t50_{side}"] = np.where(tot > 0, half.argmax(1) + 1, np.nan).astype(np.float64)
        pis = np.stack([Y[f"pi_{side}_b{k}"] for k in range(1, len(BINS) + 1)], 1)
        okp = np.isfinite(pis).any(1)
        out[f"t_peak_{side}"] = np.where(okp, np.nanargmax(np.where(np.isfinite(pis), pis, -np.inf), 1), np.nan)
        lvl = np.log(1.01)
        touch = valid & ((hi >= lvl) if sgn > 0 else (lo <= -lvl))
        first = np.where(touch.any(1), touch.argmax(1), 60)
        cv = np.cumsum(vv, 1)
        upto = np.where(first < 60, cv[np.arange(len(r)), np.minimum(first, 59)], np.nan)
        out[f"vol_to_{side}1"] = np.where(first < 60, upto / M.typical[r], VOL_TO_CAP)
    with np.errstate(invalid="ignore"):
        L = RACE * s[:, None]
        up, dn = valid & (hi >= L), valid & (lo <= -L)
    fu = np.where(up.any(1), up.argmax(1), 999)
    fd = np.where(dn.any(1), dn.argmax(1), 999)
    out["race"] = np.where(fu < fd, 0, np.where(fd < fu, 1, 2)).astype(np.int8)  # 0 up first, 1 down, 2 neither/tie
    return out


def persistence(M, r, i):
    """The persistence baseline: the realised pocket of the 15 minutes up to the decision (minutes
    i-14 .. i), measured from the open of minute i-14. Known at i."""
    r, i = np.asarray(r), np.asarray(i)
    a = np.maximum(i - 14, 0)
    cols = a[:, None] + np.arange(15)[None, :]
    valid = cols <= i[:, None]
    cols = np.minimum(cols, 389)
    p = np.where(np.isfinite(M.O[r, a]), M.O[r, a], M.Cf[r, a])
    with np.errstate(all="ignore"):
        d = (np.log(M.VW[r[:, None], cols]) - np.log(p)[:, None]) / M.sig[r][:, None]
    d = np.where(valid, d, np.nan)
    return pocket(d, np.where(valid, M.V[r[:, None], cols], 0.0))


def occupancy(P, i, p0):
    """occ_up_b<k> / occ_dn_b<k> for one stock-day: the share of each bin's seconds whose held price P
    (23400 seconds, NaN before the first trade) is above / below the entry p0, for decisions at minutes i.
    Seconds with no price yet are left out."""
    i, p0 = np.asarray(i), np.asarray(p0, np.float64)
    fin = np.isfinite(P)
    z = np.concatenate([[0], np.cumsum(fin)])
    with np.errstate(invalid="ignore"):
        up = np.concatenate([np.zeros((len(i), 1), np.int32),
                             np.cumsum(fin & (P[None, :] > p0[:, None]), 1, dtype=np.int32)], 1)
        dn = np.concatenate([np.zeros((len(i), 1), np.int32),
                             np.cumsum(fin & (P[None, :] < p0[:, None]), 1, dtype=np.int32)], 1)
    out = {}
    rr = np.arange(len(i))
    for k, (lo, hi) in enumerate(BINS, 1):
        a = np.minimum((i + 1 + lo) * 60, 23400)
        b = np.minimum((i + 1 + min(hi, 390)) * 60, 23400)
        n = (z[b] - z[a]).astype(np.float64)
        with np.errstate(all="ignore"):
            out[f"occ_up_b{k}"] = np.where(n > 0, (up[rr, b] - up[rr, a]) / n, np.nan)
            out[f"occ_dn_b{k}"] = np.where(n > 0, (dn[rr, b] - dn[rr, a]) / n, np.nan)
    return out
