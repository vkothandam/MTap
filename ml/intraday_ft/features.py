"""The intraday v2 inputs: the FEATURES table and one function per chunk (docs/intraday-features.md §3).

FEATURES is the source of the dictionary in docs/intraday-features.md; test_intraday_ft checks that
every name here appears there. Every function returns arrays shaped (stock-days x 390): column i is the
value known at the end of minute i (09:30 + i), built from minutes 0 .. i only. Price units =
ln(x / prev_vwap) / sigma clipped to +-10; volumes = ln((x + 1) / (normal + 1)).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from .zigzag import zigzag

F = namedtuple("F", "name chunk role tier")
CLIP = 10.0
MKT_CAP_W = 0.25   # a member's weight in its sector is capped at 25%
IND_MIN = 5        # ind_mv needs >= 5 other members, else the sector
ZZ_K = (2, 5)
RES_W = ("1", "5", "15", "30", "open")
MINS = np.arange(390)


def _rows(chunk, role, tier, *names):
    return [F(n, chunk, role, tier) for n in names]


FEATURES = [
    *_rows(0, "known", "M", "fc_med", "fc_iqr", "fc_w90", "fc_skew", "fc_pup", "fc_stage"),
    *_rows(0, "observed", "M", "band_px", "band_vwap"),
    *_rows(1, "observed", "M", "m_o", "m_h", "m_l", "m_c", "m_vw", "m_rvol", "m_n", "vol_used", "vol_pace",
           "from_high", "from_low", "min_since_high", "min_since_low"),
    *_rows(1, "observed", "S", *[f"res_{s}_{w}" for w in RES_W for s in ("up", "dn", "bal")],
           "vol_to_up1", "vol_to_dn1"),
    *_rows(1, "observed", "S", "twap", "linger_pos", "twap_gap", "tpo_poc", "tpo_lo70", "tpo_hi70",
           "tpo_lo50", "tpo_hi50", "time_above", "dwell_now", "tpo_poc_hl30", "tpo_lo70_hl30",
           "tpo_hi70_hl30", "time_above_hl30", "dwell_now_hl30", "linger_hi", "linger_lo", "visits_hi",
           "visits_lo", "dwell_hi", "dwell_lo", "accept"),
    *_rows(1, "observed", "M", *[f"{c}_k{k}" for k in ZZ_K for c in
                                 ("zz_dir", "zz_len", "zz_age", "zz_pull", "zz_prev", "zz_legs")]),
    *_rows(1, "observed", "S", *[f"{c}_k{k}_s" for k in ZZ_K for c in
                                 ("zz_dir", "zz_len", "zz_age", "zz_pull", "zz_prev", "zz_legs")]),
    *_rows(2, "observed", "M", "sec_mv", "sec_mv_ew", "ind_mv", "mkt_mv", "resid", "breadth",
           "spy_mv", "spy_vwap_mv", "spy_rvol"),
    *_rows(3, "static", "M", "y_close", "y_post", "gap", "pm_open", "pm_vwap", "pm_rvol", "spy_gap", "spx_y",
           "spx_gap", "sigma", "log_dvol", "log_mcap", "beta60", "results"),
    *_rows(3, "static_cat", "-", "cap_band"),
    *_rows("overlay", "static_cat", "-", "sector", "industry", "symbol"),
    *_rows("time", "known", "-", "minute", "to_close", "dow", "normal_rvol", "pm_cut"),
    *_rows("flags", "static", "-", "poll_tier"),
]
NAMES = [f.name for f in FEATURES]
OBSERVED = [f.name for f in FEATURES if f.role == "observed"]
KNOWN = [f.name for f in FEATURES if f.role == "known" and f.chunk != "time"]
STATIC = [f.name for f in FEATURES if f.role == "static"]
CATEGORICAL = [f.name for f in FEATURES if f.role == "static_cat"]
S_TIER = [f.name for f in FEATURES if f.tier == "S"]


def unit(x, ref, sig):
    """Price unit: ln(x / ref) / sigma, clipped to +-10 (ref and sig per row)."""
    with np.errstate(all="ignore"):
        return np.clip(np.log(x / np.asarray(ref)[:, None]) / np.asarray(sig)[:, None], -CLIP, CLIP)


def rlog(x, normal):
    with np.errstate(all="ignore"):
        return np.log((x + 1) / (normal + 1))


# ---------------------------------------------------------------- chunk 0

def fc_shape(Q, prev, sig):
    """fc_med, fc_iqr, fc_w90, fc_skew, fc_pup from 19 quantile price levels (rows x 19), in price units
    vs yesterday's VWAP (the run's prev_vwap)."""
    with np.errstate(all="ignore"):
        q = np.log(Q / np.asarray(prev)[:, None]) / np.asarray(sig)[:, None]
        q05, q25, q50, q75, q95 = q[:, 0], q[:, 4], q[:, 9], q[:, 14], q[:, 18]
        w = q95 - q05
        skew = np.where(w > 0, ((q95 - q50) - (q50 - q05)) / w, 0.0)
        pup = 1 - pct(np.zeros((len(q), 1)), q)[:, 0]
    return {"fc_med": q50, "fc_iqr": q75 - q25, "fc_w90": w, "fc_skew": skew, "fc_pup": pup}


def pct(X, Q):
    """Percentile of X (rows x m) inside each row's 19 quantiles: linear between them, 0.025 / 0.975
    outside, NaN when either is missing."""
    tau = np.arange(1, 20) * 0.05
    out = np.where(X < Q[:, :1], 0.025, np.where(X >= Q[:, -1:], 0.975, np.nan))
    for k in range(18):
        a, b = Q[:, k:k + 1], Q[:, k + 1:k + 2]
        out = np.where((X >= a) & (X < b), tau[k] + 0.05 * (X - a) / np.maximum(b - a, 1e-12), out)
    return out


def chunk0(M):
    """The daily forecast: shape numbers from the 09:30 run for minutes 0-14 (09:30-09:44 decisions; 09:30
    itself has the 09:30 run too), the 09:45 run from minute 15. band_* = percentile of the price now and
    of today's VWAP so far inside the curve in force."""
    n = len(M.sig)
    out = {}
    stage = np.where(MINS[None, :] < 15, 930, 945) * np.ones((n, 1))
    a, b = fc_shape(M.Q930, M.fc_prev930, M.sig), fc_shape(M.Q945, M.fc_prev945, M.sig)
    for k in a:
        out[k] = np.where(MINS[None, :] < 15, a[k][:, None], b[k][:, None]).astype(np.float32)
    out["fc_stage"] = np.where(np.isfinite(out["fc_med"]), stage, np.nan).astype(np.float32)
    P, vws = M.Cf, _vwap_sofar(M)
    e = MINS[None, :] < 15
    for name, X in (("band_px", P), ("band_vwap", vws)):
        out[name] = np.where(e, pct(X, M.Q930), pct(X, M.Q945)).astype(np.float32)
    return out


def _vwap_sofar(M):
    pv = np.where(M.V > 0, M.V * M.VW, 0)
    cv = np.cumsum(M.V, 1, dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.cumsum(pv, 1, dtype=np.float64) / np.where(cv > 0, cv, np.nan)


# ---------------------------------------------------------------- chunk 1 (M tier)

def chunk1_minute(M):
    """m_o .. m_vw, m_rvol, m_n, vol_used, vol_pace, from_high / from_low, min_since_high / low, and the
    zig-zag on the minute VWAPs."""
    out = {}
    for k, X in (("m_o", np.where(np.isfinite(M.O), M.O, M.Cf)), ("m_h", M.H), ("m_l", M.L), ("m_c", M.Cf),
                 ("m_vw", M.VW)):
        out[k] = unit(X, M.prev, M.sig)
    out["m_rvol"] = rlog(M.V, M.nvm)
    out["m_n"] = rlog(M.N, M.nnm)
    cv = np.cumsum(M.V, 1, dtype=np.float64)
    out["vol_used"] = cv / M.typical[:, None]
    out["vol_pace"] = cv / np.maximum(np.cumsum(M.nvm, 1), 1)
    Hc, Lc = np.fmax.accumulate(M.H, 1), np.fmin.accumulate(M.L, 1)
    with np.errstate(all="ignore"):
        out["from_high"] = np.clip(np.log(M.Cf / Hc) / M.sig[:, None], -CLIP, CLIP)
        out["from_low"] = np.clip(np.log(M.Cf / Lc) / M.sig[:, None], -CLIP, CLIP)
    for name, X, Xc in (("min_since_high", M.H, Hc), ("min_since_low", M.L, Lc)):
        pos = np.maximum.accumulate(np.where(X == Xc, MINS, -1), 1)
        out[name] = np.where(pos >= 0, MINS - pos, np.nan)
    with np.errstate(all="ignore"):
        lx = np.log(M.VW)
    for k in ZZ_K:
        z = zigzag(lx, k * M.sig1m, M.sig)
        for c, v in z.items():
            out[f"{c}_k{k}"] = v
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


# ---------------------------------------------------------------- chunk 2

def capped_weights(w, cap=MKT_CAP_W):
    """Normalised weights with no member above `cap` (the excess is spread over the others, repeated)."""
    w = np.asarray(w, np.float64)
    w = w / w.sum() if w.sum() > 0 else w
    if cap * len(w) < 1:
        return w
    for _ in range(len(w)):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        w[~over] += excess * w[~over] / w[~over].sum()
    return w


def group_moves(mv, cap, group, cap_w=MKT_CAP_W):
    """Leave-one-out group moves per member, per minute: the cap-weighted (each capped at cap_w within the
    group) and the equal-weight mean of the other members' mv. mv: rows x T; cap, group: rows. Returns
    (cw, ew, n_others)."""
    cw, ew = np.full(mv.shape, np.nan, np.float32), np.full(mv.shape, np.nan, np.float32)
    n_oth = np.zeros(len(mv))
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        n_oth[idx] = len(idx) - 1
        if len(idx) < 2:
            continue
        X = mv[idx]
        ok = np.isfinite(X)
        Xz = np.where(ok, X, 0)
        c = np.nan_to_num(np.asarray(cap, np.float64)[idx])
        for j, r in enumerate(idx):
            oth = np.delete(np.arange(len(idx)), j)
            w = capped_weights(c[oth], cap_w)[:, None] * ok[oth]
            ws = w.sum(0)
            with np.errstate(all="ignore"):
                cw[r] = np.where(ws > 0, (w * Xz[oth]).sum(0) / ws, np.nan)
                ne = ok[oth].sum(0)
                ew[r] = np.where(ne > 0, Xz[oth].sum(0) / ne, np.nan)
    return cw, ew, n_oth


def chunk2(M, spy_cf, spy_v, spy_nv, spy_prevclose, spy_prevvwap):
    """Sector, industry and market moves (leave-one-out, members = the day's live set), breadth, resid and
    SPY. Moves are % log changes vs each stock's own prev VWAP. M covers one day's stock-days."""
    with np.errstate(all="ignore"):
        mv = np.log(M.Cf / M.prev[:, None]) * 100
    live = M.live.astype(bool)
    out = {k: np.full(mv.shape, np.nan, np.float32) for k in ("sec_mv", "sec_mv_ew", "ind_mv", "mkt_mv")}
    L = np.flatnonzero(live)
    for key, grp in (("sec", M.sector), ("ind", M.industry), ("mkt", np.zeros(len(mv), int))):
        # every stock gets the move of the live members of its group, leaving itself out
        cw, ew, n_oth = _loo_with_outsiders(mv, M.mcap, grp, L)
        if key == "sec":
            out["sec_mv"], out["sec_mv_ew"] = cw, ew
            sec_cw = cw
        elif key == "ind":
            out["ind_mv"] = np.where((n_oth >= IND_MIN)[:, None], cw, sec_cw)
        else:
            out["mkt_mv"] = cw
    with np.errstate(all="ignore"):
        spy_mv = np.log(spy_cf / spy_prevclose) * 100
        out["spy_mv"] = np.broadcast_to(spy_mv, mv.shape).astype(np.float32)
        out["spy_vwap_mv"] = np.broadcast_to(np.log(spy_cf / spy_prevvwap) * 100, mv.shape).astype(np.float32)
        out["spy_rvol"] = np.broadcast_to(rlog(spy_v, spy_nv), mv.shape).astype(np.float32)
        out["resid"] = (mv - M.beta60[:, None] * spy_mv[None, :]).astype(np.float32)
        up = (mv[L] > 0).sum(0) / np.maximum(np.isfinite(mv[L]).sum(0), 1)
    out["breadth"] = np.broadcast_to(up, mv.shape).astype(np.float32)
    return out


def _loo_with_outsiders(mv, cap, grp, live_idx):
    """group_moves over the live members; a stock outside the live set gets its group's full live
    mean (it is not in it, so nothing to leave out)."""
    n = len(mv)
    cw, ew = np.full(mv.shape, np.nan, np.float32), np.full(mv.shape, np.nan, np.float32)
    n_oth = np.zeros(n)
    inside = np.zeros(n, bool)
    inside[live_idx] = True
    gl = np.asarray(grp)
    for g in np.unique(gl):
        mem = np.flatnonzero((gl == g) & inside)
        out_ = np.flatnonzero((gl == g) & ~inside)
        if len(mem):
            a, b, k = group_moves(mv[mem], np.asarray(cap)[mem], np.zeros(len(mem), int))
            cw[mem], ew[mem], n_oth[mem] = a, b, k
        if len(out_) and len(mem):
            X = mv[mem]
            ok = np.isfinite(X)
            w = capped_weights(np.nan_to_num(np.asarray(cap, np.float64)[mem]))[:, None] * ok
            with np.errstate(all="ignore"):
                cw[out_] = (w * np.where(ok, X, 0)).sum(0) / w.sum(0)
                ew[out_] = np.where(ok, X, 0).sum(0) / ok.sum(0)
            n_oth[out_] = len(mem)
    return cw, ew, n_oth


# ---------------------------------------------------------------- chunk 3

def spy_gap(open_, prevclose, dividend):
    """ln(SPY open / previous close) + the dividend / previous close on ex-dates (%)."""
    return (np.log(open_ / prevclose) + np.nan_to_num(dividend) / prevclose) * 100


def chunk3(M):
    """Static per stock-day: the previous session, the pre-market, reference data. pm_vwap / pm_rvol come
    as two cuts (555 for minutes 0-14, 570 from 15); the window builder picks per decision minute."""
    s = M.sig
    with np.errstate(all="ignore"):
        out = {"y_close": np.log(M.prevclose / M.prev) / s, "y_post": np.log(M.prevpost / M.prev) / s,
               "gap": np.log(M.open / M.prevclose) / s, "pm_open": np.log(M.open / M.pm570_last) / s,
               "sigma": np.log(s), "log_dvol": np.log(M.ndol), "log_mcap": np.log(M.mcap),
               "beta60": M.beta60, "results": M.results,
               "spy_gap": M.spy_gap, "spx_y": M.spx_y, "spx_gap": M.spx_gap}
        for cut in (555, 570):
            out[f"pm_vwap{cut}"] = np.log(M.pm[f"pm{cut}_pv"] / M.pm[f"pm{cut}_v"] / M.prev) / s
            out[f"pm_rvol{cut}"] = rlog(np.nan_to_num(M.pm[f"pm{cut}_v"]), M.npm[cut])
    return {k: np.clip(np.asarray(v, np.float64), -50, 50).astype(np.float32) for k, v in out.items()}
