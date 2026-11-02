"""Analog matching across stocks: the first check behind docs/analog-pockets.md.

For a stock at a moment of the day, find past moments of any stock whose parameters were close,
and compare what followed: the "pockets", i.e. where the later volume traded in price and time.
Read-only: bars come from the NAS, nothing is written.

  example   one query (MU 2026-10-02 11:00), its matches narrow -> broad, the pockets as text maps
  windows   October queries, 30/60/90-session windows x narrow/medium/broad, skill vs climatology
  hgb       HistGradientBoosting on the same inputs (trained before October, tested on October)
  coverage  share of the actual later volume inside the 50/80/95% pockets

uv run python scripts_space_analog_check.py [--parts example,windows,hgb,coverage] [--queries 500] [--seed 1]
"""
import argparse
import glob
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy import ndimage

BARS = "/Volumes/Backups/MTap-Data/bars_1m"
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]  # June = warm-up for the 20-session sigma
TEST_FROM = pd.Timestamp("2026-10-01")
P = np.arange(15, 376, 15)           # decision minutes after 09:30: 09:45 ... 15:45
QP = [60, 90, 120, 180, 240, 300]    # query times: 10:30, 11:00, 11:30, 12:30, 13:30, 14:30
NX, NY, YL = 36, 32, 4.0             # pocket grid: log2 time ahead x price vs now (+-4 sigma)
XMAX = np.log2(391)
MINS = np.arange(390)
DIALS = {"narrow": 0.25, "medium": 0.5, "broad": 1.0}
YOURS = ["vwap_vs_prev", "open_vs_prev", "move_since_open", "day_1", "day_2"]
EXAMPLE = ("MU", "2026-10-02", 90, ["vwap_vs_prev", "open_vs_prev", "move_since_open", "path_30m",
                                    "day_1", "day_2", "rvol"])
N0 = 10                              # shrinkage to climatology: a = N_eff days / (N_eff days + N0)
EPOCH = pd.Timestamp("1970-01-01")


def et_minutes(t_ms):
    """Epoch minutes on the New York clock, DST included (one offset per UTC date)."""
    m = t_ms // 60000
    day = m // 1440
    u = np.unique(day)
    noon = pd.DatetimeIndex(pd.to_datetime(u * 1440 + 720, unit="m", utc=True))
    off = (noon.tz_convert("America/New_York").tz_localize(None) - noon.tz_localize(None)) // pd.Timedelta(minutes=1)
    return m + np.asarray(off)[np.searchsorted(u, day)]


def load_month(month):
    f = glob.glob(f"{BARS}/month={month}/*.parquet")[0]  # the folder holds this month only, whatever the file name says
    tb = ds.dataset(f).to_table(columns=["ticker", "t", "o", "h", "l", "c", "v", "vw"])
    tk = pc.dictionary_encode(tb["ticker"].combine_chunks())
    et = et_minutes(tb["t"].to_numpy())
    mod = et % 1440
    reg = (mod >= 570) & (mod < 960)
    a = {k: tb[k].combine_chunks().fill_null(np.nan).to_numpy(zero_copy_only=False)[reg].astype(np.float32)
         for k in ("o", "h", "l", "c", "v", "vw")}
    code = tk.indices.to_numpy()[reg].astype(np.int64)
    u, inv = np.unique(code * 100000 + et[reg] // 1440, return_inverse=True)
    mi = mod[reg] - 570

    def dense(x, fill):
        D = np.full((len(u), 390), fill, np.float32)
        D[inv, mi] = x
        return D
    V = dense(a["v"], 0)
    C, VW, H, Lo, Op = (dense(a[k], np.nan) for k in ("c", "vw", "h", "l", "o"))
    idx = np.where(~np.isnan(C), MINS, 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    Cf = np.take_along_axis(C, idx, 1)  # last price carried forward (NaN before the first trade)
    has = ~np.isnan(Op)
    names = np.array(tk.dictionary.to_pylist(), dtype=object)
    return dict(tk=names[u // 100000], day=u % 100000, V=V, VW=np.where(np.isnan(VW), Cf, VW), Cf=Cf,
                H=np.where(np.isnan(H), Cf, H), L=np.where(np.isnan(Lo), Cf, Lo), open=Op[np.arange(len(u)), has.argmax(1)])


def build(months):
    """Stock-days and the moment library (one row per eligible stock-day per 15 minutes)."""
    t0 = time.time()
    parts = [load_month(m) for m in months]
    A = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    V, VW, Cf, H, Lo = (A.pop(k) for k in ("V", "VW", "Cf", "H", "L"))
    d = pd.DataFrame(A)
    PV = np.where(V > 0, V * VW, 0).astype(np.float32)
    d["row"] = np.arange(len(d))
    d["dv"] = V.sum(1)
    d["dpv"] = PV.sum(1)
    d["vwap"] = d.dpv / d.dv.where(d.dv > 0)
    d = d.sort_values(["tk", "day"])
    g = d.groupby("tk", sort=False)
    d["prev"] = g.vwap.shift(1)
    d["r"] = np.log(d.vwap / d.prev)
    d.loc[d.r.abs() > 0.4, "r"] = np.nan  # split guard
    g = d.groupby("tk", sort=False)
    d["sig"] = g.r.transform(lambda s: s.rolling(20, min_periods=15).std().shift(1))
    d["nv"] = g.dv.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    d["ndol"] = g.dpv.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    for k in (1, 2, 3):
        d[f"d{k}"] = g.r.shift(k)
    el = ((d.sig > 0.003) & (d.nv > 0) & (d.ndol >= 5e6) & (d.prev >= 3) & d.open.notna()
          & d[["d1", "d2", "d3"]].notna().all(axis=1))
    de = d[el].reset_index(drop=True)
    sess = np.sort(d.day.unique())
    de["s"] = np.searchsorted(sess, de.day.values)
    R = de.row.values
    Ve, VWe, Ce = V[R], VW[R], Cf[R]
    cV, cPV = np.cumsum(Ve, 1), np.cumsum(PV[R], 1)
    He, Le = H[R], Lo[R]
    del V, VW, Cf, H, Lo, PV
    test = de.day.values >= (TEST_FROM - EPOCH).days
    curve = np.mean((cV / np.maximum(cV[:, -1:], 1))[~test], 0)  # typical share of the day's volume by each minute
    sg = de.sig.values[:, None]
    now = Ce[:, P - 1]
    vws = cPV[:, P - 1] / np.where(cV[:, P - 1] > 0, cV[:, P - 1], np.nan)
    low = np.fmin.accumulate(Le, 1)[:, P - 1]
    high = np.fmax.accumulate(He, 1)[:, P - 1]
    rod = (cPV[:, -1:] - cPV[:, P - 1]) / np.maximum(cV[:, -1:] - cV[:, P - 1], 1e-9)
    hs = np.flip(np.fmax.accumulate(np.flip(He, 1), 1), 1)[:, P]  # max high after now
    ls = np.flip(np.fmin.accumulate(np.flip(Le, 1), 1), 1)[:, P]  # min low after now
    rep = lambda x: np.repeat(x[:, None], len(P), 1)
    F = {"vwap_vs_prev": np.log(vws / de.prev.values[:, None]) / sg,
         "open_vs_prev": rep(np.log(de.open.values / de.prev.values) / de.sig.values),
         "move_since_open": np.log(now / de.open.values[:, None]) / sg,
         "path_30m": np.log(now / Ce[:, np.maximum(P - 31, 0)]) / sg,
         "from_low": np.log(now / low) / sg,
         "from_high": np.log(now / high) / sg,
         "rvol": np.log(cV[:, P - 1] / (de.nv.values[:, None] * curve[P - 1])),
         **{f"day_{k}": rep(de[f"d{k}"].values / de.sig.values) for k in (1, 2, 3)}}
    M = pd.DataFrame({k: np.asarray(v, np.float32).ravel() for k, v in F.items()})
    M["e"] = np.repeat(np.arange(len(de)), len(P))
    M["p"] = np.tile(P, len(de))
    M["now"] = now.ravel()
    M["rod"] = (np.log(rod / now) / sg).ravel()
    M["bul"] = (np.log(hs / now) / sg).ravel()
    M["sag"] = (np.log(ls / now) / sg).ravel()
    ok = np.isfinite(M[list(F) + ["rod", "bul", "sag"]].values).all(1) & (M.now.values > 0)
    M = M[ok].reset_index(drop=True)
    e = M.e.values
    M["s"] = de.s.values[e]
    M["day"] = de.day.values[e]
    M["test"] = test[e]
    first = pd.Timestamp(EPOCH + pd.Timedelta(days=int(de.day.min()))).date()
    print(f"library: {len(M):,} moments from {len(de):,} stock-days, {de.s.nunique()} sessions "
          f"({first} … {(EPOCH + pd.Timedelta(days=int(de.day.max()))).date()}), "
          f"{de.day[~test].nunique()} before {TEST_FROM.date()}; built {time.time() - t0:.0f}s")
    return SimpleNamespace(de=de, M=M, V=Ve, VW=VWe, sig=de.sig.values, nv=de.nv.values, params=list(F))


def grid(L, e, p, now, w):
    """Weighted later volume of moments (e, p) on the grid: log time ahead x price vs now (sigma)."""
    Vk = L.V[e]
    ah = MINS[None, :] - p[:, None]
    fut = ah >= 0
    xb = np.minimum((np.log2(np.maximum(ah, 0) + 1) / XMAX * NX).astype(np.int64), NX - 1)
    y = np.log(L.VW[e] / now[:, None]) / L.sig[e][:, None]
    yb = np.clip(((np.nan_to_num(y) + YL) / (2 * YL) * NY).astype(np.int64), 0, NY - 1)
    m = fut & (Vk > 0) & np.isfinite(y)
    mass = Vk / L.nv[e][:, None] * w[:, None]
    return np.bincount((xb * NY + yb)[m], weights=mass[m], minlength=NX * NY).reshape(NX, NY)


def dens(G, bw=1.0):
    Gs = ndimage.gaussian_filter(G, bw) + 1e-9
    return Gs / Gs.sum()


def hdr(Q, a):
    """Cells of the smallest region holding share a of the mass (the a-pocket)."""
    s = np.sort(Q.ravel())[::-1]
    return Q >= s[min(np.searchsorted(np.cumsum(s), a), len(s) - 1)]


def wq(x, w, qs):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(qs, c / c[-1], x[o])


def readings(L, c, w, bw=1.0):
    M = L.M
    return dict(Q=dens(grid(L, M.e.values[c], M.p.values[c], M.now.values[c], w), bw),
                pu=np.average(M.rod.values[c] > 0, weights=w), rq=wq(M.rod.values[c], w, [.1, .5, .9]),
                bq=wq(M.bul.values[c], w, [.1, .5, .9]), sq=wq(M.sag.values[c], w, [.1, .5, .9]))


def shrink(S, C, nd):
    a = nd / (nd + N0)
    return {k: a * S[k] + (1 - a) * C[k] for k in ("Q", "pu", "rq", "bq", "sq")}


def neff_days(s, w):
    Wd = np.bincount(s - s.min(), weights=w)
    return Wd.sum() ** 2 / (Wd ** 2).sum()


def match(X, scale, c0, qr, b):
    """Epanechnikov product weights: every parameter within +-b x its spread, weight falls to 0 at the edge."""
    z = np.abs(X[c0] - X[qr]) / (b * scale)
    ok = (z < 1).all(1)
    return c0[ok], np.prod(1 - z[ok] ** 2, 1)


class Climatology:
    """Everything at the same time of day in the window (pockets from a 6k sample)."""

    def __init__(self, L, rng, bw=1.0):
        self.L, self.rng, self.bw, self.cache = L, rng, bw, {}

    def pool(self, p0, s0, W):
        M = self.L.M
        c = np.flatnonzero((M.p.values == p0) & (M.s.values < s0) & (M.s.values >= s0 - W))
        return c

    def __call__(self, p0, s0, W):
        k = (p0, s0, W)
        if k not in self.cache:
            self.cache[k] = self.of(self.pool(p0, s0, W))
        return self.cache[k]

    def of(self, c):
        M = self.L.M
        cs = self.rng.choice(c, min(6000, len(c)), replace=False)
        return dict(Q=readings(self.L, cs, np.ones(len(cs)), self.bw)["Q"], pu=np.mean(M.rod.values[c] > 0),
                    rq=np.quantile(M.rod.values[c], [.1, .5, .9]), bq=np.quantile(M.bul.values[c], [.1, .5, .9]),
                    sq=np.quantile(M.sag.values[c], [.1, .5, .9]), n=len(c))


def actual(L, qr, bw=None):
    M = L.M
    G = grid(L, M.e.values[[qr]], M.p.values[[qr]], M.now.values[[qr]], np.ones(1))
    return G / max(G.sum(), 1e-12)


pin = lambda q, y: np.mean([max(t * (y - v), (t - 1) * (y - v)) for t, v in zip((.1, .5, .9), q)])


def ci(x, days, rng):
    m = pd.Series(x).groupby(days).mean().values
    bs = [np.mean(rng.choice(m, len(m))) for _ in range(2000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def scale_of(L, params):
    X = L.M[params].values.astype(np.float32)
    return X, np.subtract(*np.nanpercentile(X[~L.M.test.values], [75, 25], axis=0)) / 1.349


def queries(L, n, rng):
    M = L.M
    qm = np.flatnonzero(M.test.values & np.isin(M.p.values, QP) & (L.de.ndol.values[M.e.values] >= 2e7))
    return rng.choice(qm, min(n, len(qm)), replace=False)


def ascii_map(Q, title):
    nb = [ndimage.label(hdr(Q, a))[1] for a in (.5, .8, .95)]
    Q2 = Q[:, 0::2] + Q[:, 1::2]
    lv = [hdr(Q2, a) for a in (.5, .8, .95)]
    print(f"\n  {title}\n  later volume above now {Q[:, NY // 2:].sum():.0%} | separate blobs in the 50/80/95% pockets: "
          f"{nb[0]}/{nb[1]}/{nb[2]}")
    for j in range(11, 3, -1):
        lo = -4 + j * 0.5
        row = "".join("@" if lv[0][i, j] else "o" if lv[1][i, j] else "." if lv[2][i, j] else " " for i in range(NX))
        print(f"   {lo:+.1f}..{lo + 0.5:+.1f}σ |{row}|")
        if j == 8:
            print("   ---- now -----  |" + "-" * NX + "|")
    lab = [" "] * NX
    for m_, t in [(1, "1m"), (4, "4m"), (16, "16m"), (64, "64m"), (256, "4h")]:
        x = int(np.log2(m_ + 1) / XMAX * NX)
        for i, ch in enumerate(t):
            if x + i < NX:
                lab[x + i] = ch
    print("                    " + "".join(lab) + "  (log time ahead; @ 50% pocket, o 80%, . 95%)")


def part_example(L, rng):
    tk, date, p0, params = EXAMPLE
    M, de = L.M, L.de
    X, scale = scale_of(L, params)
    day = (pd.Timestamp(date) - EPOCH).days
    hit = np.flatnonzero((de.tk.values[M.e.values] == tk) & (M.day.values == day) & (M.p.values == p0))
    if not len(hit):
        print(f"\n=== example: {tk} {date} not in the library")
        return
    qr = hit[0]
    hhmm = f"{9 + (30 + p0) // 60:02d}:{(30 + p0) % 60:02d}"
    print(f"\n=== EXAMPLE {tk} {date} {hhmm} ET (σ = {L.sig[M.e.values[qr]]:.2%}): "
          + ", ".join(f"{k} {M[k].values[qr]:+.2f}" for k in params))
    print(f"    library: every moment at {hhmm} before {TEST_FROM.date()}; matching on {', '.join(params)}")
    c0 = np.flatnonzero((M.p.values == p0) & ~M.test.values)
    C = Climatology(L, rng).of(c0)
    for lab, b in DIALS.items():
        t1 = time.time()
        c, w = match(X, scale, c0, qr, b)
        ms = 1000 * (time.time() - t1)
        if not len(c):
            print(f"\n  [{lab}: each parameter within ±{b} × its spread] no matches")
            continue
        S = readings(L, c, w)
        nd = neff_days(M.s.values[c], w)
        ascii_map(S["Q"], f"[{lab}: each parameter within ±{b} × its spread] {len(c)} matches, "
                  f"{len(np.unique(de.tk.values[M.e.values[c]]))} stocks, N_eff {nd:.1f} days, "
                  f"match + pockets {1000 * (time.time() - t1):.0f} ms (match {ms:.0f} ms)")
        print(f"  P(rest-of-day VWAP > now) {S['pu']:.0%} | rest-of-day VWAP p10/50/90 "
              f"{S['rq'][0]:+.2f}/{S['rq'][1]:+.2f}/{S['rq'][2]:+.2f}σ | bulge p50 {S['bq'][1]:+.2f}σ, sag p50 {S['sq'][1]:+.2f}σ")
        Sh = shrink(S, C, nd)
        print(f"  shrunk to climatology (a = {nd / (nd + N0):.2f}): P(up) {Sh['pu']:.0%}, later volume above now "
              f"{Sh['Q'][:, NY // 2:].sum():.0%}")
        if lab == "medium":
            top = c[np.argsort(-w)[:6]]
            print("  closest matches: " + "; ".join(
                f"{de.tk.values[M.e.values[i]]} {(EPOCH + pd.Timedelta(days=int(M.day.values[i]))).date()}" for i in top))
    ascii_map(C["Q"], f"[everything at {hhmm}, no matching = climatology] {C['n']:,} moments")
    print(f"  P(up) {C['pu']:.0%} | rest-of-day VWAP p10/50/90 {C['rq'][0]:+.2f}/{C['rq'][1]:+.2f}/{C['rq'][2]:+.2f}σ | "
          f"bulge p50 {C['bq'][1]:+.2f}σ, sag p50 {C['sq'][1]:+.2f}σ")
    ascii_map(actual(L, qr), f"[what {tk} actually did]")
    print(f"  rest-of-day VWAP {M.rod.values[qr]:+.2f}σ, bulge {M.bul.values[qr]:+.2f}σ, sag {M.sag.values[qr]:+.2f}σ")


def part_windows(L, nq, rng):
    M = L.M
    qs = queries(L, nq, rng)
    clim = Climatology(L, rng)
    sets = {"yours (5): " + ", ".join(YOURS): YOURS, "3: vwap_vs_prev, open_vs_prev, move_since_open": YOURS[:3]}
    for sname, params in sets.items():
        X, scale = scale_of(L, params)
        print(f"\n##### parameters {sname}")
        for W in (30, 60, 90):
            rows, t0 = [], time.time()
            for qr in qs:
                p0, s0 = M.p.values[qr], M.s.values[qr]
                C = clim(p0, s0, W)
                A = actual(L, qr)
                if A.sum() == 0:
                    continue
                y, b_, sa = M.rod.values[qr], M.bul.values[qr], M.sag.values[qr]
                score = lambda S: ((S["pu"] - float(y > 0)) ** 2, pin(S["rq"], y), pin(S["bq"], b_), pin(S["sq"], sa),
                                   (A * np.log(S["Q"])).sum())
                r = {"day": M.day.values[qr], **{f"c{i}": v for i, v in enumerate(score(C))}}
                c0 = clim.pool(p0, s0, W)
                for lab, b in DIALS.items():
                    t1 = time.time()
                    c, w = match(X, scale, c0, qr, b)
                    r[f"{lab}_ms"] = 1000 * (time.time() - t1)
                    r[f"{lab}_n"] = len(c)
                    nd = neff_days(M.s.values[c], w) if len(c) else 0.0
                    S = shrink(readings(L, c, w), C, nd) if len(c) else C
                    r[f"{lab}_ms2"] = 1000 * (time.time() - t1)
                    r[f"{lab}_nd"] = nd
                    r.update({f"{lab}{i}": v for i, v in enumerate(score(S))})
                rows.append(r)
            B = pd.DataFrame(rows)
            print(f"\n  window = last {W} sessions before the query day ({len(B)} October queries, {time.time() - t0:.0f}s; "
                  f"medium: match {B.medium_ms.median():.1f} ms, match + pockets {B.medium_ms2.median():.1f} ms; "
                  f"climatology pool ~{np.median([v['n'] for k, v in clim.cache.items() if k[2] == W]):,.0f} moments)")
            print("   dial     matches(med) N_eff days(med) <5 matches |  skill vs climatology %, shrunk to it by N_eff days [95% CI over days]")
            print("                                                     |  P(up) Brier     rest-of-day VWAP   bulge         sag          pocket density (nats)")
            for lab in DIALS:
                o = []
                for i in range(4):
                    x = 1 - B[f"{lab}{i}"] / B[f"c{i}"].mean()
                    lo, hi = ci(x.values, B.day.values, rng)
                    o.append(f"{100 * (1 - B[f'{lab}{i}'].mean() / B[f'c{i}'].mean()):+5.1f} [{100 * lo:+.0f},{100 * hi:+.0f}]")
                dd = B[f"{lab}4"] - B["c4"]
                lo, hi = ci(dd.values, B.day.values, rng)
                print(f"   {lab:7s}  {B[lab + '_n'].median():9.0f}  {B[lab + '_nd'].median():12.1f}  {np.mean(B[lab + '_n'] < 5):9.0%}  |  "
                      + "   ".join(o) + f"   {dd.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")


def part_hgb(L, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    M = L.M
    print(f"\n##### trained model (HGB) on the same inputs: train = moments before {TEST_FROM.date()}, "
          f"test = all October moments of stocks with ≥ $20M a day")
    tr_all = np.flatnonzero(~M.test.values)
    te = np.flatnonzero(M.test.values & (L.de.ndol.values[M.e.values] >= 2e7))
    for sname, params in [("yours (5)", YOURS), ("all 10 + time of day", L.params)]:
        feats = params + ["p"]
        tr = rng.choice(tr_all, min(1_500_000, len(tr_all)), replace=False)
        Xtr, Xte, t0 = M[feats].values[tr], M[feats].values[te], time.time()
        kw = dict(max_iter=300, learning_rate=0.05, min_samples_leaf=500, early_stopping=True, random_state=0)
        clf = HistGradientBoostingClassifier(max_leaf_nodes=31, **kw).fit(Xtr, M.rod.values[tr] > 0)
        pu = clf.predict_proba(Xte)[:, 1]
        yv = (M.rod.values[te] > 0).astype(float)
        pb = pd.Series(M.rod.values[tr] > 0).groupby(M.p.values[tr]).mean().reindex(M.p.values[te]).values
        bs, bb = (pu - yv) ** 2, (pb - yv) ** 2
        lo, hi = ci(1 - bs / bb.mean(), M.day.values[te], rng)
        yy = M.rod.values[te]
        qh = [HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw).fit(Xtr, M.rod.values[tr]).predict(Xte)
              for q in (.1, .5, .9)]
        cq = pd.DataFrame({"p": M.p.values[tr], "y": M.rod.values[tr]}).groupby("p").y.quantile([.1, .5, .9]).unstack()
        pl = lambda qv: np.mean([np.maximum(t * (yy - v), (t - 1) * (yy - v)) for t, v in zip((.1, .5, .9), qv)], 0)
        pm, pc_ = pl(qh), pl([cq[t].reindex(M.p.values[te]).values for t in (.1, .5, .9)])
        lo2, hi2 = ci(1 - pm / pc_.mean(), M.day.values[te], rng)
        print(f"  {sname:22s} P(up) Brier skill {100 * (1 - bs.mean() / bb.mean()):+.1f}% [{100 * lo:+.1f},{100 * hi:+.1f}] | "
              f"rest-of-day VWAP pinball skill {100 * (1 - pm.mean() / pc_.mean()):+.1f}% [{100 * lo2:+.1f},{100 * hi2:+.1f}] | "
              f"{len(te):,} test moments, train {time.time() - t0:.0f}s")


def part_coverage(L, nq, rng, W=60):
    M = L.M
    X, scale = scale_of(L, YOURS)
    qs = queries(L, nq, rng)
    clims = {bw: Climatology(L, rng, bw) for bw in (0.7, 1.5)}
    out, t0 = [], time.time()
    for qr in qs:
        p0, s0 = M.p.values[qr], M.s.values[qr]
        A = actual(L, qr)
        if A.sum() == 0:
            continue
        r = {"day": M.day.values[qr]}
        c0 = clims[0.7].pool(p0, s0, W)
        for bw, clim in clims.items():
            C = clim(p0, s0, W)
            for lab, b in (("clim", None), ("medium", .5), ("broad", 1.)):
                Q = C["Q"]
                if b is not None:
                    c, w = match(X, scale, c0, qr, b)
                    if len(c):
                        Q = shrink(readings(L, c, w, bw), C, neff_days(M.s.values[c], w))["Q"]
                for a in (.5, .8, .95):
                    r[f"{lab}_{bw}_{a}"] = A[hdr(Q, a)].sum()
                r[f"{lab}_{bw}_ls"] = (A * np.log(Q)).sum()
                r[f"{lab}_{bw}_area"] = hdr(Q, .8).mean()
                r[f"{lab}_{bw}_blobs"] = ndimage.label(hdr(Q, .5))[1]
        out.append(r)
    B = pd.DataFrame(out)
    print(f"\n##### pocket coverage: {len(B)} October queries, {W}-session window, your 5 parameters, {time.time() - t0:.0f}s")
    print("share of the ACTUAL later volume that landed inside each predicted pocket (on target = the pocket's nominal mass)")
    print("  smoothing  pockets from            inside 50%      inside 80%      inside 95%      80% pocket covers   50% blobs(med)  log score")
    for bw in clims:
        for lab in ("clim", "medium", "broad"):
            cells = []
            for a in (.5, .8, .95):
                lo, hi = ci(B[f"{lab}_{bw}_{a}"].values, B.day.values, rng)
                cells.append(f"{100 * B[f'{lab}_{bw}_{a}'].mean():5.1f}% [{100 * lo:.0f},{100 * hi:.0f}]")
            name = "climatology" if lab == "clim" else lab + " match + shrink"
            print(f"  {bw:4.1f} cell  {name:22s} " + "  ".join(cells)
                  + f"   {100 * B[f'{lab}_{bw}_area'].mean():5.1f}% of the map   {B[f'{lab}_{bw}_blobs'].median():5.0f}"
                  + f"           {B[f'{lab}_{bw}_ls'].mean():+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="example,windows,hgb,coverage")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    pd.set_option("display.width", 250)
    t0, rng, parts = time.time(), np.random.default_rng(a.seed), a.parts.split(",")
    L = build(MONTHS)
    if "example" in parts:
        part_example(L, rng)
    if "windows" in parts:
        part_windows(L, a.queries, rng)
    if "hgb" in parts:
        part_hgb(L, rng)
    if "coverage" in parts:
        part_coverage(L, a.queries, rng)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
