"""The second tier (S): Massive 1 s bars sampled on the live poll grid (docs/intraday-features.md §1, Q-E).

Per stock-day the session's 23,400 seconds hold the held price (the last trade carried forward) and
the volume of each poll. Every column is computed at the end of each minute m from seconds
< 60 (m + 1) only:
- time at price on a 0.1-sigma grid anchored at yesterday's VWAP (bins -10 .. +10 sigma; the price-now
  bin = the grid bin holding the price now), the value area, the extremes, the volume in the way;
- resistance by the tick rule over 1 / 5 / 15 / 30 min and since the open;
- per minute: twap, linger_pos, twap_gap.
visits_* and dwell_* need the order of the seconds, so they are computed every VISIT_EVERY minutes and
carried forward in between (the value is then up to 4 minutes old, never newer than the minute).
"""
from __future__ import annotations

import numpy as np

SESSION = 23400
OPEN_S = 9 * 3600 + 30 * 60
BIN = 0.1           # time-at-price bin, sigma
NB = 201            # bins -10 .. +10 sigma
MID = NB // 2
NEAR_EXT = 0.25     # within 0.25 sigma of the day's high / low
VISIT_GAP = 30      # a visit ends after >= 30 s away
VISIT_EVERY = 5
HALF_LIFE = 30 * 60
RES_WIN = {"1": 60, "5": 300, "15": 900, "30": 1800, "open": SESSION}
POLL_STEP = {"1s": 1, "2s": 2, "20s": 20}
ENDS = 60 * (np.arange(390) + 1)
S_COLS = ("twap", "linger_pos", "twap_gap", "tpo_poc", "tpo_lo70", "tpo_hi70", "tpo_lo50", "tpo_hi50",
          "time_above", "dwell_now", "tpo_poc_hl30", "tpo_lo70_hl30", "tpo_hi70_hl30", "time_above_hl30",
          "dwell_now_hl30", "linger_hi", "linger_lo", "visits_hi", "visits_lo", "dwell_hi", "dwell_lo",
          "accept", "vol_to_up1", "vol_to_dn1",
          *[f"res_{s}_{w}" for w in RES_WIN for s in ("up", "dn", "bal")])


def value_area(counts, center, share):
    """The value-area rule on one histogram: start at the bin with the most seconds; add the neighbour
    above or below the band with more seconds (ties: the one nearer `center`, the price-now bin; then
    the upper one); stop once the band holds >= share of the seconds. Returns (lo_bin, hi_bin)."""
    c = np.asarray(counts, np.float64)
    tot = c.sum()
    if tot <= 0:
        return -1, -1
    lo = hi = int(np.argmax(c))
    held = c[lo]
    while held < share * tot - 1e-9:
        a = c[hi + 1] if hi + 1 < len(c) else -1.0
        b = c[lo - 1] if lo - 1 >= 0 else -1.0
        if a < 0 and b < 0:
            break
        if a > b or (a == b and abs(hi + 1 - center) <= abs(lo - 1 - center)):
            hi += 1
            held += a
        else:
            lo -= 1
            held += b
    return lo, hi


def value_area_rows(C, center, share):
    """value_area for every row of C (rows x bins) at once; center per row. (lo, hi) per row, -1 when the
    row is empty."""
    C = np.asarray(C, np.float64)
    n, nb = C.shape
    r = np.arange(n)
    tot = C.sum(1)
    lo = C.argmax(1)
    hi = lo.copy()
    held = C[r, lo]
    for _ in range(nb):
        go = held < share * tot - 1e-9
        if not go.any():
            break
        a = np.where(hi + 1 < nb, C[r, np.minimum(hi + 1, nb - 1)], -1.0)
        b = np.where(lo - 1 >= 0, C[r, np.maximum(lo - 1, 0)], -1.0)
        go &= (a >= 0) | (b >= 0)
        up = (a > b) | ((a == b) & (np.abs(hi + 1 - center) <= np.abs(lo - 1 - center)))
        hi = np.where(go & up, hi + 1, hi)
        lo = np.where(go & ~up, lo - 1, lo)
        held = held + np.where(go, np.where(up, a, b), 0)
    empty = ~(tot > 0)
    return np.where(empty, -1, lo), np.where(empty, -1, hi)


def grid_day(sec, price, volume, step=1):
    """Held price and volume per session second (09:30:00 ... 15:59:59) as a poll every `step` seconds
    would see them: the held price at the poll second, and the volume since the previous poll, put on the
    poll second (step 1 = Massive's own seconds). Seconds before the first poll that saw a trade are NaN."""
    s = np.asarray(sec) - OPEN_S
    ok = (s >= 0) & (s < SESSION)
    s, p, v = s[ok].astype(np.int64), np.asarray(price, np.float64)[ok], np.asarray(volume, np.float64)[ok]
    P = np.full(SESSION, np.nan)
    V = np.zeros(SESSION)
    P[s] = p
    np.add.at(V, s, v)
    has = np.isfinite(P)
    idx = np.maximum.accumulate(np.where(has, np.arange(SESSION), -1))
    P = np.where(idx >= 0, P[np.maximum(idx, 0)], np.nan)
    if step > 1:
        polls = np.arange(step - 1, SESSION, step)
        Vp = np.add.reduceat(V, np.arange(0, SESSION, step))[:len(polls)]
        P2, V2 = np.full(SESSION, np.nan), np.zeros(SESSION)
        P2[polls], V2[polls] = P[polls], Vp
        idx = np.maximum.accumulate(np.where(np.isfinite(P2), np.arange(SESSION), -1))
        P, V = np.where(idx >= 0, P2[np.maximum(idx, 0)], np.nan), V2
    return P, V


def tick_sides(P, V):
    """Tick rule on the grid: per second, the log change since the previous second with volume, and the
    side (+1 / -1) of the last change (0 before the first). A second with volume and no change keeps the
    last side: volume absorbed without moving the price. P: log prices."""
    n = len(P)
    trade = V > 0
    idx = np.maximum.accumulate(np.where(trade, np.arange(n), -1))
    prev_idx = np.concatenate([[-1], idx[:-1]])
    prev_p = np.where(prev_idx >= 0, P[np.maximum(prev_idx, 0)], np.nan)
    dlt = np.nan_to_num(np.where(trade, P - prev_p, 0.0))
    sgn = np.sign(dlt)
    last = np.maximum.accumulate(np.where(sgn != 0, np.arange(n), -1))
    side = np.where(last >= 0, sgn[np.maximum(last, 0)], 0)
    return dlt, side


def resistance(dlt, side, V, typical, ends=ENDS):
    """res_up / res_dn per window, at each end second (exclusive): the up-side volume (% of the typical
    day) per 1% of summed up-moves; res_bal = ln(res_dn / res_up). NaN where a side has no move."""
    cu = np.concatenate([[0], np.cumsum(np.where(side > 0, V, 0))])
    cd = np.concatenate([[0], np.cumsum(np.where(side < 0, V, 0))])
    mu = np.concatenate([[0], np.cumsum(np.maximum(dlt, 0))])
    md = np.concatenate([[0], np.cumsum(np.maximum(-dlt, 0))])
    out = {}
    with np.errstate(all="ignore"):
        for name, w in RES_WIN.items():
            a = np.maximum(ends - w, 0)
            vu, vd = (cu[ends] - cu[a]) / typical * 100, (cd[ends] - cd[a]) / typical * 100
            pu, pd_ = (mu[ends] - mu[a]) * 100, (md[ends] - md[a]) * 100
            ru, rd = np.where(pu > 0, vu / pu, np.nan), np.where(pd_ > 0, vd / pd_, np.nan)
            out[f"res_up_{name}"], out[f"res_dn_{name}"], out[f"res_bal_{name}"] = ru, rd, np.log(rd / ru)
    return out


def visits(near):
    """Number of visits and mean seconds per visit in a boolean series; a visit ends after >= 30 s away."""
    idx = np.flatnonzero(near)
    if not len(idx):
        return 0, np.nan
    n = 1 + int((np.diff(idx) > VISIT_GAP).sum())
    return n, len(idx) / n


def day_columns(P, V, lprev, sig, typical):
    """Every S column per minute end for one stock-day. P: held price per second (NaN before the first
    trade), V: poll volume per second, lprev: ln(yesterday's VWAP), sig: daily sigma (log), typical: the
    typical day's volume. Returns {name: float32 array (390)}."""
    out = {k: np.full(390, np.nan, np.float32) for k in S_COLS}
    with np.errstate(all="ignore"):
        lp = np.log(P)
    ok = np.isfinite(lp)
    if ok.sum() < 2 or not (sig > 0) or not (typical > 0):
        return out
    y = (lp - lprev) / sig
    b = np.clip(np.rint(y / BIN), -MID, MID).astype(np.int64) + MID
    sec = np.arange(SESSION)
    m = sec // 60
    w_hl = np.exp2(sec / HALF_LIFE)
    cnt = np.zeros((390, NB))
    vol = np.zeros((390, NB))
    wt = np.zeros((390, NB))
    k = m[ok] * NB + b[ok]
    cnt.ravel()[:] = np.bincount(k, minlength=390 * NB)
    vol.ravel()[:] = np.bincount(k, weights=V[ok], minlength=390 * NB)
    wt.ravel()[:] = np.bincount(k, weights=w_hl[ok], minlength=390 * NB)
    C, CV = np.cumsum(cnt, 0), np.cumsum(vol, 0)
    CW = np.cumsum(wt, 0) * np.exp2(-ENDS / HALF_LIFE)[:, None]
    last = ENDS - 1
    live = ok[last]
    c = np.where(live, b[last], MID)
    r = np.arange(390)
    tot, vtot = C.sum(1), CV.sum(1)
    CB, CVB, CWB = np.cumsum(C, 1), np.cumsum(CV, 1), np.cumsum(CW, 1)

    def band(H, share):
        lo, hi = value_area_rows(H, c, share)
        return (lo - c - 0.5) * BIN, (hi - c + 0.5) * BIN
    with np.errstate(all="ignore"):
        for suf, H, HB in (("", C, CB), ("_hl30", CW, CWB)):
            t = H.sum(1)
            out[f"tpo_poc{suf}"] = (H.argmax(1) - c) * BIN
            out[f"tpo_lo70{suf}"], out[f"tpo_hi70{suf}"] = band(H, 0.7)
            out[f"time_above{suf}"] = (t - HB[r, c]) / t
            out[f"dwell_now{suf}"] = H[r, c] / t
        out["tpo_lo50"], out["tpo_hi50"] = band(C, 0.5)
        out["accept"] = np.where(CV[r, c] > 0, (C[r, c] / tot) / (CV[r, c] / vtot), np.nan)
        hib = np.maximum.accumulate(np.where(cnt.sum(1) > 0, NB - 1 - np.argmax(cnt[:, ::-1] > 0, 1), -1))
        lob = np.minimum.accumulate(np.where(cnt.sum(1) > 0, np.argmax(cnt > 0, 1), NB))
        near = int(np.floor(NEAR_EXT / BIN + 1e-9))
        a_hi = np.clip(hib - near - 1, -1, NB - 1)
        out["linger_hi"] = (CB[r, np.clip(hib, 0, NB - 1)] - np.where(a_hi >= 0, CB[r, np.maximum(a_hi, 0)], 0)) / tot
        out["linger_lo"] = (CB[r, np.clip(lob + near, 0, NB - 1)]
                            - np.where(lob >= 1, CB[r, np.clip(lob - 1, 0, NB - 1)], 0)) / tot
        kb = np.maximum(np.rint(np.log(1.01) / sig / BIN).astype(np.int64), 1)
        out["vol_to_up1"] = (CVB[r, np.minimum(c + kb, NB - 1)] - CVB[r, c]) / typical
        out["vol_to_dn1"] = (np.where(c >= 1, CVB[r, np.maximum(c - 1, 0)], 0)
                             - np.where(c - kb - 1 >= 0, CVB[r, np.maximum(c - kb - 1, 0)], 0)) / typical
        # per minute: twap, where in the minute's range the time was spent, VWAP - TWAP
        Pm, Vm = lp.reshape(390, 60), V.reshape(390, 60)
        twap = np.nanmean(Pm, 1)
        lo_, hi_ = np.nanmin(Pm, 1), np.nanmax(Pm, 1)
        vw = np.nansum(np.where(np.isfinite(Pm), Pm, 0) * Vm, 1) / np.where(Vm.sum(1) > 0, Vm.sum(1), np.nan)
        out["twap"] = np.clip((twap - lprev) / sig, -10, 10)
        out["linger_pos"] = np.where(np.isfinite(twap), np.where(hi_ > lo_, (twap - lo_) / (hi_ - lo_), 0.5), np.nan)
        out["twap_gap"] = (vw - twap) / sig
    dlt, side = tick_sides(np.where(ok, lp, 0.0), np.where(ok, V, 0.0))
    out.update(resistance(dlt, side, np.where(ok, V, 0.0), typical))
    thr = NEAR_EXT * sig
    for j in range(VISIT_EVERY - 1, 390, VISIT_EVERY):
        x = lp[:ENDS[j]]
        x = x[np.isfinite(x)]
        if len(x) < 2:
            continue
        for s_, ext in (("hi", x.max()), ("lo", x.min())):
            n, d = visits(np.abs(x - ext) <= thr)
            out[f"visits_{s_}"][j:j + VISIT_EVERY], out[f"dwell_{s_}"][j:j + VISIT_EVERY] = n, d
    for k_ in ("visits_hi", "visits_lo", "dwell_hi", "dwell_lo"):
        out[k_][:VISIT_EVERY - 1] = np.nan
    dead = ~live
    return {k_: np.where(dead, np.nan, v).astype(np.float32) for k_, v in out.items()}
