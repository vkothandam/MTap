"""Rolling recalibration of the stored quantiles (go rule Q41: calibrated, or fixed by a recalibration).

The calibration level drifts between years while the ranking holds (rvol P50 hits 0.43 in 2026, 0.55 in
2025: LEARNINGS "Intraday v2 IA1"). So each day's quantiles get, per target and quantile, the offset that
would have made the previous WINDOW sessions' outcomes hit it at its level:

    off[d, t, k] = quantile_TAUS[k]( y - q[:, t, k] )  over the rows of sessions d-WINDOW .. d-1
    q_cal = clip(sort(q + off), LO, HI)

Only realised targets of past sessions are used (all known live by the next morning), the same rule as
the daily model's cal20s. The first WINDOW test sessions have no offsets (raw quantiles, `n_days` < WINDOW).
One offset for every stock at a minute, so the rank IC of any quantile is unchanged. Offsets go to
fold_<k>/recal.parquet (date, target, o10 .. o90, lo, hi, n_days): the dashboard adds them itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
WINDOW = 20
OCOLS = [f"o{round(t * 100)}" for t in TAUS]
UNIT = ("mass_up", "occ_up", "occ_dn", "mass_up60")   # shares: 0 .. 1
SIGNED = ("net", "rvol", "net60")                     # net in sigma, rvol in log: either sign


def bounds(target):
    """(lo, hi) a target's quantiles are clipped to after the offset."""
    base = target.rsplit("_b", 1)[0]
    if base in SIGNED:
        return -np.inf, np.inf
    return 0.0, 1.0 if base in UNIT else np.inf


def fit(q, y, day, window=WINDOW):
    """Offsets per session from the previous `window` sessions. q (rows, 5), y (rows,), day (rows,) ->
    (sessions sorted, offsets (sessions, 5), n_days used)."""
    days = np.sort(pd.unique(day))
    di = pd.Index(days).get_indexer(day)
    order = np.argsort(di, kind="stable")
    cut = np.searchsorted(di[order], np.arange(len(days) + 1))
    err = y[:, None] - q
    off = np.zeros((len(days), len(TAUS)))
    n = np.minimum(np.arange(len(days)), window)
    for i in range(window, len(days)):
        e = err[order[cut[i - window]:cut[i]]]
        e = e[np.isfinite(e).all(1)]
        if len(e):
            off[i] = [np.quantile(e[:, k], t) for k, t in enumerate(TAUS)]
        else:
            n[i] = 0
    off[n < window] = 0.0
    return days, off, n


def apply(q, off, lo=-np.inf, hi=np.inf):
    """q (rows, 5) + per-row offsets (rows, 5), kept in order and inside the target's range."""
    return np.clip(np.sort(q + off, axis=-1), lo, hi)


def fit_fold(Q, names, f, window=WINDOW, zero_mass=(), dust=0.0):
    """Offsets for every stored target of a fold. Q (rows, targets, 5) aligned to f (the run file rows,
    via dec_row); f has date and real_<target>. Returns the recal.parquet frame."""
    day = f.date.astype(str).to_numpy()
    parts = []
    for j, t in enumerate(names):
        if f"real_{t}" not in f:
            continue
        q = np.asarray(Q[:, j], np.float64)
        if t.rsplit("_b", 1)[0].removesuffix("60") in zero_mass:
            q = np.where(q < dust, 0.0, q)
        days, off, n = fit(q, f[f"real_{t}"].to_numpy(np.float64), day, window)
        lo, hi = bounds(t)
        parts.append(pd.DataFrame({"date": days, "target": t, **{c: off[:, k] for k, c in enumerate(OCOLS)},
                                   "lo": lo, "hi": hi, "n_days": n}))
    return pd.concat(parts, ignore_index=True)


def offsets_for(rc, target, day):
    """Per-row offsets (rows, 5) of `target` from a recal frame, 0 where a day has none."""
    r = rc[rc.target == target].set_index("date")[OCOLS]
    return r.reindex(pd.Index(day).astype(str)).fillna(0.0).to_numpy(np.float64)
