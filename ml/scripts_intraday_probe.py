"""Intraday probe: Step 0 (skill per input group), Step 0b (opportunity and predictability maps)
and the recommendation count, from docs/intraday-design.md.

Read-only on the NAS (bars_1m, share snapshots, agg_1s); caches and results go to
/tmp/mtap/intraday_probe/ (status.txt, run.log, report.txt, results.json).

  features  per month: point-in-time inputs at delay d = 0 and 15 min, window targets, slices
  ladder    HGB per input group (yearly walk-forward), Brier skill, rank IC, top-20 lists
  scan      opportunity map (how often big moves follow) and predictability map (skill), per slice
  recs      2026, every minute: how many P(touch) >= 60% recommendations a day, and how many came true
  retime    1 s bars: how much of the move after a new day high / low is gone by the minute end

uv run python scripts_intraday_probe.py [--parts features,ladder,scan,recs,retime] [--quick]
"""
import argparse
import glob
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy import sparse
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

NAS = "/Volumes/Backups/MTap-Data"
BARS = f"{NAS}/bars_1m"
HERE = Path(__file__).resolve().parent
FORECAST = HERE / "runs/20261029-211059Z-gbm-4438sym-cal20s"  # R5-cal20s walk-forward folds
PANEL = HERE.parent / "out/tft/exp5y-r1/panel.parquet"
OUT = Path("/tmp/mtap/intraday_probe")
EPOCH = pd.Timestamp("1970-01-01")
MINS = np.arange(390)
WIN = (5, 15, 60)
HS = (5, 15, 30, 60, 120, 390)        # window lengths for up/dn (390 = to the close)
REC_H = (15, 60, 390)
LEVELS = (("0.5s", 0.5, "s"), ("1s", 1.0, "s"), ("0.5%", np.log(1.005), "p"), ("1%", np.log(1.01), "p"))
TOUCH = [f"t{u}_{n}_{h}" for u in "ud" for n, _, _ in LEVELS for h in REC_H]
TAU = np.arange(1, 20) * 0.05
RES = ["Normal_Trading", "Pre_Earnings_Runup", "Earnings_Eve", "Earnings_Day", "Post_Earnings_Reaction",
       "Post_Earnings_Drift"]
CAPS = [("mega", 200e9), ("large", 10e9), ("mid", 2e9), ("small", 250e6), ("micro", 0)]
HGB = {"max_iter": 300, "learning_rate": 0.05, "min_samples_leaf": 500, "early_stopping": True,
       "validation_fraction": 0.1, "n_iter_no_change": 20}
LEVEL = {n: (v, k) for n, v, k in LEVELS}

G1 = ["minute", "to_close", "px_prev", "px_open", "px_prevclose", "vwap_sofar", "px_vwap", "from_high",
      "from_low", "day_range", "min_since_high", "min_since_low", "rvol_sofar", "rvol_accel", "gap_prev_vwap",
      "d1", "d2", "d3", *[f"{k}_{w}" for w in WIN for k in ("ret", "rvol", "range", "clv", "vwapdev")]]
G2 = ["d_med", "d_iqr", "d_w90", "d_pup", "band_px", "band_vwap", "gap_prev_close", "pm_last", "pm_vwap",
      "pm_vol", "pm_range", "prev_clv", "sigma", "dollar_vol", "log_mcap"]
G3 = ["mkt_open", "sec_open", *[f"{k}_{w}" for w in WIN for k in ("mkt", "sec", "resid")], "breadth",
      "rank_open", "rank_rvol"]
G4 = ["above_share", "poc", "va_lo", "va_hi", "node_up", "node_dn", "thin_up", "thin_dn", "above_dec",
      "bulge30", "sag30"]
G5 = ["imb_open", *[f"imb_{w}" for w in WIN], "rank_imb"]
G6 = ["res_code", "days_since", "days_to"]
FULL = G1 + G2 + G3 + G4 + G5 + G6
DAILY_ONLY = ["minute", "to_close", "d_med", "d_iqr", "d_w90", "d_pup", "band_px", "band_vwap", "sigma",
              "dollar_vol"]
ARMS = {"daily_only": (DAILY_ONLY, 15), "s1_path": (G1, 15), "s2_daily": (G1 + G2, 15),
        "s3_market": (G1 + G2 + G3, 15), "s4_vap": (G1 + G2 + G3 + G4, 15),
        "s5_flow": (G1 + G2 + G3 + G4 + G5, 15), "s6_results": (FULL, 15), "full_d0": (FULL, 0)}
LADDER_Y = {"tu": "tu_0.5s_60", "td": "td_0.5s_60", "uf": "upfirst", "vb": "volbal_up"}
SLICES = ["burst", "new_hl", "near_hi", "near_lo", "vwap_x", "band_x", "first05", "gap", "results",
          "mkt_move", "turn_hi", "turn_lo"]
EXT = ["near_hi", "near_lo", "turn_hi", "turn_lo"]  # densely sampled evaluation points (user, 2026-11-01)


# ---------------------------------------------------------------- status

class Status:
    """status.txt after every unit of work: part, done / total, elapsed, ETA from the measured rate."""

    def __init__(self, parts):
        self.t0, self.parts = time.time(), parts

    def start(self, part, total):
        self.part, self.total, self.done, self.tp = part, max(total, 1), 0, time.time()
        self.write("started")

    def step(self, msg=""):
        self.done += 1
        self.write(msg)

    def write(self, msg):
        el = time.time() - self.tp
        eta = el / self.done * (self.total - self.done) if self.done else float("nan")
        end = time.strftime("%H:%M", time.localtime(time.time() + eta)) if np.isfinite(eta) else "?"
        left = self.parts[self.parts.index(self.part) + 1:]
        line = (f"[{self.part} {self.parts.index(self.part) + 1}/{len(self.parts)}] {self.done}/{self.total}"
                f" · part {el / 60:.0f} min · run {(time.time() - self.t0) / 60:.0f} min · part ETA {end}"
                f" · next: {','.join(left) or '-'} · {msg}")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "status.txt").write_text(line + "\n")
        print(line, flush=True)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- loading

def et_minutes(t_ms):
    """Epoch minutes on the New York clock, DST included (one offset per UTC date)."""
    m = t_ms // 60000
    day = m // 1440
    u = np.unique(day)
    noon = pd.DatetimeIndex(pd.to_datetime(u * 1440 + 720, unit="m", utc=True))
    off = (noon.tz_convert("America/New_York").tz_localize(None) - noon.tz_localize(None)) // pd.Timedelta(minutes=1)
    return m + np.asarray(off)[np.searchsorted(u, day)]


def read_month(month):
    f = glob.glob(f"{BARS}/month={month}/*.parquet")[0]  # the folder holds this month only, whatever the file name says
    tb = ds.dataset(f).to_table(columns=["ticker", "t", "o", "h", "l", "c", "v", "vw"])
    tk = pc.dictionary_encode(tb["ticker"].combine_chunks())
    et = et_minutes(tb["t"].to_numpy())
    a = {k: tb[k].combine_chunks().fill_null(np.nan).to_numpy(zero_copy_only=False).astype(np.float32)
         for k in ("o", "h", "l", "c", "v", "vw")}
    names = np.array(tk.dictionary.to_pylist(), dtype=object)
    return names, tk.indices.to_numpy().astype(np.int64), et, a


def summarise(month):
    """One row per stock-day (regular session): vwap, volume, dollar volume, open, close, high, low."""
    names, code, et, a = read_month(month)
    mod = et % 1440
    reg = (mod >= 570) & (mod < 960)
    d = pd.DataFrame({"code": code[reg], "day": et[reg] // 1440, "et": et[reg],
                      **{k: a[k][reg] for k in a}}).sort_values(["code", "day", "et"])
    d["pv"] = np.where(d.v > 0, d.v * d.vw, 0)
    g = d.groupby(["code", "day"], sort=False)
    s = g.agg(dv=("v", "sum"), dpv=("pv", "sum"), open=("o", "first"), close=("c", "last"), hi=("h", "max"),
              lo=("l", "min")).reset_index()
    s["tk"] = names[s.code.values]
    s["vwap"] = s.dpv / s.dv.where(s.dv > 0)
    return s.drop(columns="code")


def load_month(month, keep):
    """Dense stock-day x 390 arrays for the stock-days in `keep` (a set of (ticker, day)), plus pre-market
    aggregates up to 09:15 and 09:30."""
    names, code, et, a = read_month(month)
    mod = et % 1440
    key = code * 100000 + et // 1440
    kk = pd.Index(names[np.unique(key) // 100000] + "|" + (np.unique(key) % 100000).astype(str))
    want = np.unique(key)[kk.isin(keep)]
    reg = (mod >= 570) & (mod < 960) & np.isin(key, want)
    u, inv = np.unique(key[reg], return_inverse=True)
    mi = mod[reg] - 570

    def dense(x, fill):
        D = np.full((len(u), 390), fill, np.float32)
        D[inv, mi] = x[reg]
        return D
    V = dense(a["v"], 0)
    C, VW, H, Lo, Op = (dense(a[k], np.nan) for k in ("c", "vw", "h", "l", "o"))
    idx = np.where(~np.isnan(C), MINS, 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    Cf = np.take_along_axis(C, idx, 1)  # last price carried forward (NaN before the first trade)
    has = ~np.isnan(Op)
    pm = (mod >= 240) & (mod < 570) & np.isin(key, u)
    pmd = pd.DataFrame({"pos": np.searchsorted(u, key[pm]), "mod": mod[pm], "c": a["c"][pm], "h": a["h"][pm],
                        "l": a["l"][pm], "v": a["v"][pm], "vw": a["vw"][pm]}).sort_values(["pos", "mod"])
    pmd["pv"] = pmd.v * pmd.vw
    PM = {}
    for cut in (555, 570):
        g = pmd[pmd["mod"] < cut].groupby("pos")
        agg = g.agg(v=("v", "sum"), pv=("pv", "sum"), last=("c", "last"), hi=("h", "max"), lo=("l", "min"))
        for k in agg:
            arr = np.full(len(u), np.nan)
            arr[agg.index.values] = agg[k].values
            PM[f"pm{cut}_{k}"] = arr
    return dict(tk=names[u // 100000], day=u % 100000, V=V, VW=np.where(np.isnan(VW), Cf, VW), Cf=Cf,
                H=np.where(np.isnan(H), Cf, H), L=np.where(np.isnan(Lo), Cf, Lo),
                open=Op[np.arange(len(u)), has.argmax(1)], **PM)


def months(first, last):
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def sector_map():
    p = pd.read_parquet(PANEL, columns=["symbol", "date", "industry_code"])
    p = p[p.industry_code.str[:2].str.isdigit().fillna(False)]  # drops "unknown"
    s = p.sort_values("date").groupby("symbol").industry_code.last()
    return s.str[:2].astype(int)


def day_stats(S, secmap):
    """Point-in-time per stock-day: sigma, normal volume, prior days, eligibility; market and sector sigma."""
    d = S.sort_values(["tk", "day"]).reset_index(drop=True)
    g = d.groupby("tk", sort=False)
    d["prev"] = g.vwap.shift(1)
    d["prevclose"], d["prevhi"], d["prevlo"] = g.close.shift(1), g.hi.shift(1), g.lo.shift(1)
    d["r"] = np.log(d.vwap / d.prev)
    d.loc[d.r.abs() > 0.4, "r"] = np.nan  # split guard
    g = d.groupby("tk", sort=False)
    d["sig"] = g.r.transform(lambda s: s.rolling(20, min_periods=15).std().shift(1))
    d["nv"] = g.dv.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    d["ndol"] = g.dpv.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    for k in (1, 2, 3):
        d[f"r{k}"] = g.r.shift(k)
    d["sec"] = d.tk.map(secmap).fillna(-1).astype(int)
    d["elig"] = ((d.sig > 0.003) & (d.nv > 0) & (d.ndol >= 2e7) & (d.prev >= 3) & d.open.notna()
                 & d[["r1", "r2", "r3"]].notna().all(axis=1))
    e = d[d.elig].copy()
    e["oc"] = np.log(e.close / e.open)
    e["w"] = e.ndol
    for key, name in ((["day"], "msig"), (["day", "sec"], "ssig")):
        k = e.groupby(key).apply(lambda x: np.average(x.oc, weights=x.w), include_groups=False).rename("x")
        k = k.reset_index()
        gk = k.groupby(key[1:], sort=True) if len(key) > 1 else None
        k = k.sort_values(key[::-1])
        k[name] = (k.groupby(key[1:]).x.transform(lambda s: s.rolling(20, min_periods=15).std().shift(1))
                   if gk is not None else k.x.rolling(20, min_periods=15).std().shift(1))
        d = d.merge(k[key + [name]], on=key, how="left")
    d["ssig"] = d.ssig.fillna(d.msig)
    return d


def market_caps(d):
    """Daily market cap as of the previous close = shares outstanding from the latest quarter-end snapshot
    before the day x the previous close. Massive's own market_cap is weighted shares x close (all share
    classes); share_class_shares_outstanding (that class only: GOOGL 5.9B of 12.2B) is the fallback."""
    rows = []
    for f in sorted(glob.glob(f"{NAS}/out/shares/shares-*.jsonl")):
        x = pd.read_json(f, lines=True)
        rows.append(x[["symbol", "snap_date", "weighted_shares_outstanding", "share_class_shares_outstanding"]])
    sh = pd.concat(rows)
    sh["key"] = (pd.to_datetime(sh.snap_date) - EPOCH).dt.days
    sh["shares"] = sh.weighted_shares_outstanding.where(sh.weighted_shares_outstanding > 0,
                                                        sh.share_class_shares_outstanding)
    sh = sh[sh.shares > 0].rename(columns={"symbol": "tk"}).sort_values("key")
    q = d[["tk", "day", "prevclose"]].copy()
    q["key"] = q.day - 1
    q = pd.merge_asof(q.sort_values("key"), sh[["tk", "key", "shares"]], on="key", by="tk", direction="backward")
    q["mcap"] = q.shares * q.prevclose
    return d.merge(q[["tk", "day", "mcap"]], on=["tk", "day"], how="left")


def cap_band(mcap):
    out = np.full(len(mcap), "", object)
    for name, lo in CAPS[::-1]:
        out[np.asarray(mcap) >= lo] = name
    return out


def forecasts():
    fs = sorted(glob.glob(f"{FORECAST}/fold_*/test_predictions.parquet"))
    f = pd.concat([pd.read_parquet(x) for x in fs])
    f["day"] = (pd.to_datetime(f.date) - EPOCH).dt.days
    return f.rename(columns={"symbol": "tk", "prev_vwap": "fc_prev"}).drop_duplicates(["tk", "day"], keep="last")


def results_days():
    p = pd.read_parquet(PANEL, columns=["symbol", "date", "results_window_pit", "days_since_results",
                                        "days_to_results_est"])
    p["day"] = (pd.to_datetime(p.date) - EPOCH).dt.days
    p["res_code"] = p.results_window_pit.map({k: i for i, k in enumerate(RES)}).astype(float)
    return p.rename(columns={"symbol": "tk", "days_since_results": "days_since", "days_to_results_est": "days_to"})[
        ["tk", "day", "res_code", "days_since", "days_to"]]


def build_month(raw, info):
    """The month's eligible stock-days as one namespace: dense bars plus per-row point-in-time values."""
    k = pd.DataFrame({"tk": raw["tk"], "day": raw["day"]}).merge(info, on=["tk", "day"], how="left")
    Q = k[[f"q{round(t * 100):02d}" for t in TAU]].to_numpy(np.float64)
    M = SimpleNamespace(**{n: raw[n] for n in ("V", "VW", "Cf", "H", "L")}, Q=Q)
    for c in ("tk", "day", "open", "prev", "prevclose", "prevhi", "prevlo", "sig", "nv", "ndol", "r1", "r2", "r3",
              "sec", "msig", "ssig", "mcap", "res_code", "days_since", "days_to", "fc_prev"):
        setattr(M, c, k[c].to_numpy() if c == "tk" else k[c].to_numpy(np.float64))
    M.open = raw["open"].astype(np.float64)
    M.sec = np.nan_to_num(M.sec, nan=-1).astype(int)
    M.day = M.day.astype(int)
    M.pm = {c: raw[c] for c in raw if c.startswith("pm")}
    return M


# ---------------------------------------------------------------- features

def shr(X, s):
    Y = np.full_like(X, np.nan)
    Y[:, s:] = X[:, :-s]
    return Y


def roll(X, w, f):
    """f (np.fmax / np.fmin) over the last w columns ending at each column, NaN-padded at the start."""
    out, k = X.copy(), 1
    while 2 * k <= w:
        out = f(out, shr(out, k))
        k *= 2
    return f(out, shr(out, w - k)) if w > k else out


def fwd(X, h, f):
    """f over columns i .. i+h-1 (to the close when h >= 390), per column i."""
    R = X[:, ::-1]
    out = f.accumulate(R, axis=1) if h >= 390 else roll(R, h, f)
    return np.ascontiguousarray(out[:, ::-1])


def lead(X, fill):
    """Column c of the result = column c-1 of X (column 0 = fill): the value known at the start of minute c."""
    return np.concatenate([np.full((len(X), 1), fill, X.dtype), X], 1)


def lagc(X, w):
    """Column c of the result = column max(c - w, 0) of X."""
    return np.concatenate([np.repeat(X[:, :1], w, 1), X[:, :-w]], 1)


def pct(X, Q):
    """Percentile of X (rows x m) inside each row's 19 forecast quantiles: linear between them, 0.025 / 0.975
    outside, NaN when either is missing."""
    out = np.where(X < Q[:, :1], 0.025, np.where(X >= Q[:, -1:], 0.975, np.nan))
    for k in range(18):
        a, b = Q[:, k:k + 1], Q[:, k + 1:k + 2]
        out = np.where((X >= a) & (X < b), TAU[k] + 0.05 * (X - a) / np.maximum(b - a, 1e-12), out)
    return out


def indicator(key):
    u, kix = np.unique(key, return_inverse=True)
    return sparse.csr_matrix((np.ones(len(key)), (kix, np.arange(len(key)))), shape=(len(u), len(key))), kix


def prank(X, dix):
    """Percentile rank across the stocks of the same day, per column (NaN stays NaN)."""
    out = np.full(X.shape, np.nan, np.float32)
    for d in np.unique(dix):
        r = np.flatnonzero(dix == d)
        x = X[r]
        fin = np.isfinite(x)
        o = np.argsort(np.where(fin, x, np.inf), axis=0)
        rk = np.empty_like(o)
        np.put_along_axis(rk, o, np.arange(len(r))[:, None], axis=0)
        out[r] = np.where(fin, rk / np.maximum(fin.sum(0) - 1, 1), np.nan)
    return out


def dense_features(M, curve):
    """Every input as a (rows x 391) array: column c uses only bars before minute c (09:30 + c)."""
    lg, sg = np.log, M.sig[:, None]
    F = {}
    with np.errstate(all="ignore"):
        P = lead(M.Cf, np.nan)
        P[:, 0] = M.open  # only an anchor for "since the open" windows; every feature is blanked at c = 0
        PV = np.where(M.V > 0, M.V * M.VW, 0)
        cV, cPV = lead(np.cumsum(M.V, 1, dtype=np.float64), 0), lead(np.cumsum(PV, 1, dtype=np.float64), 0)
        nvc = M.nv[:, None] * curve[None, :]
        vws = cPV / np.where(cV > 0, cV, np.nan)
        Hc, Lc = lead(np.fmax.accumulate(M.H, 1), np.nan), lead(np.fmin.accumulate(M.L, 1), np.nan)
        F["_hc"], F["_lc"] = Hc, Lc
        F["px_prev"] = lg(P / M.prev[:, None]) / sg
        F["px_open"] = lg(P / M.open[:, None]) / sg
        F["px_prevclose"] = lg(P / M.prevclose[:, None]) / sg
        F["vwap_sofar"] = lg(vws / M.prev[:, None]) / sg
        F["px_vwap"] = lg(P / vws) / sg
        F["from_high"], F["from_low"] = lg(P / Hc) / sg, lg(P / Lc) / sg
        F["day_range"] = lg(Hc / Lc) / sg
        for name, X, f in (("min_since_high", M.H, np.fmax), ("min_since_low", M.L, np.fmin)):
            new = X == f.accumulate(X, 1)
            pos = np.maximum.accumulate(np.where(new, MINS, -1), 1)
            F[name] = lead(np.where(pos >= 0, MINS - pos, np.nan).astype(np.float32), np.nan)
        F["rvol_sofar"] = lg((cV + 1) / (nvc + 1))
        lr = np.concatenate([lg(M.Cf[:, :1] / M.open[:, None]), lg(M.Cf[:, 1:] / M.Cf[:, :-1])], 1)
        lr = np.nan_to_num(lr)
        sv = M.V * (2 * ndtr(lr / (M.sig[:, None] / np.sqrt(390))) - 1)  # bulk classification: signed volume
        cSV = lead(np.cumsum(sv, 1), 0)
        for w in WIN:
            dv = cV - lagc(cV, w)
            F[f"ret_{w}"] = lg(P / lagc(P, w)) / sg
            F[f"rvol_{w}"] = lg((dv + 1) / (nvc - lagc(nvc, w) + 1))
            hi, lo = lead(roll(M.H, w, np.fmax), np.nan), lead(roll(M.L, w, np.fmin), np.nan)
            F[f"range_{w}"] = lg(hi / lo) / sg
            F[f"clv_{w}"] = np.where(hi > lo, (P - lo) / (hi - lo), np.where(np.isfinite(hi), 0.5, np.nan))
            F[f"vwapdev_{w}"] = lg(P * dv / (cPV - lagc(cPV, w))) / sg
            F[f"imb_{w}"] = (cSV - lagc(cSV, w)) / np.where(dv > 0, dv, np.nan)
        F["rvol_accel"] = F["rvol_5"] - F["rvol_15"]
        F["imb_open"] = cSV / np.where(cV > 0, cV, np.nan)
        # market and sector: prior-dollar-volume-weighted mean of the eligible stocks' 1-min log returns
        w8 = np.nan_to_num(M.ndol)
        for name, key, s in (("mkt", M.day, M.msig), ("sec", M.day * 1000 + M.sec + 1, M.ssig)):
            A, kix = indicator(key)
            idx = (A @ (lr * w8[:, None])) / np.maximum(A @ w8, 1)[:, None]
            C = lead(np.cumsum(idx, 1), 0)[kix]
            F[f"{name}_open"] = C / s[:, None]
            for w in WIN:
                F[f"{name}_{w}"] = (C - lagc(C, w)) / s[:, None]
                if name == "mkt":
                    F[f"resid_{w}"] = F[f"ret_{w}"] - (C - lagc(C, w)) / sg
            if name == "mkt":
                F["_mkt15"] = (C - lagc(C, 15)) / (s[:, None] * np.sqrt(15 / 390))
        A, kix = indicator(M.day)
        F["breadth"] = ((A @ (P > M.open[:, None]).astype(np.float64)) / np.asarray(A.sum(1)))[kix]
        F["rank_open"] = prank(F["px_open"], kix)
        F["rank_rvol"] = prank(F["rvol_sofar"], kix)
        F["rank_imb"] = prank(F["imb_15"], kix)
        F["band_px"], F["band_vwap"] = pct(P, M.Q), pct(vws, M.Q)
    for k in list(F):
        F[k] = np.clip(F[k], -50, 50).astype(np.float32)
        F[k][:, 0] = np.nan
    for k in ("band_px", "band_vwap"):
        F[k][:, :15] = np.nan  # the daily forecast is known from 09:45
    return F


def fixed_features(M):
    lg, s = np.log, M.sig
    with np.errstate(all="ignore"):
        S = {"sigma": lg(s), "dollar_vol": lg(M.ndol), "log_mcap": lg(M.mcap),
             "gap_prev_vwap": lg(M.open / M.prev) / s, "gap_prev_close": lg(M.open / M.prevclose) / s,
             "d1": M.r1 / s, "d2": M.r2 / s, "d3": M.r3 / s,
             "prev_clv": (M.prevclose - M.prevlo) / (M.prevhi - M.prevlo),
             "d_med": lg(M.Q[:, 9] / M.fc_prev) / s, "d_iqr": lg(M.Q[:, 14] / M.Q[:, 4]) / s,
             "d_w90": lg(M.Q[:, 18] / M.Q[:, 0]) / s, "d_pup": 1 - pct(M.fc_prev[:, None], M.Q)[:, 0],
             "res_code": M.res_code, "days_since": M.days_since, "days_to": M.days_to}
        for cut in (555, 570):
            p = M.pm
            S[f"pm_last{cut}"] = lg(p[f"pm{cut}_last"] / M.prev) / s
            S[f"pm_vwap{cut}"] = lg(p[f"pm{cut}_pv"] / p[f"pm{cut}_v"] / M.prev) / s
            S[f"pm_vol{cut}"] = np.log1p(np.nan_to_num(p[f"pm{cut}_v"]) / M.nv)
            S[f"pm_range{cut}"] = lg(p[f"pm{cut}_hi"] / p[f"pm{cut}_lo"]) / s
    return {k: np.clip(np.asarray(v, np.float64), -50, 50) for k, v in S.items()}


VAP_CHUNK = 20000


def vap(M, r, c):
    """Volume at price for rows r at known column c: today's bars before c, in 0.25-sigma bins around p."""
    out = {k: np.full(len(r), np.nan, np.float32) for k in G4}
    sel = np.flatnonzero(c >= 1)
    for ch in np.array_split(sel, max(1, len(sel) // VAP_CHUNK)):
        if not len(ch):
            continue
        rr, cc = r[ch], c[ch]
        p = M.Cf[rr, cc - 1].astype(np.float64)
        known = MINS[None, :] < cc[:, None]
        with np.errstate(all="ignore"):
            y = np.log(M.VW[rr] / p[:, None]) / M.sig[rr][:, None]
        ok = known & np.isfinite(y) & (M.V[rr] > 0)
        v = np.where(ok, M.V[rr], 0).astype(np.float64)
        y = np.where(ok, y, 0)
        tot = v.sum(1)
        wv = v * 0.5 ** ((cc[:, None] - 1 - MINS[None, :]) / 30)
        wt = wv.sum(1)
        b = np.clip(np.rint(y / 0.25), -12, 12).astype(np.int64) + 12
        Hb = np.bincount((np.arange(len(rr))[:, None] * 25 + b).ravel(), weights=v.ravel(),
                         minlength=len(rr) * 25).reshape(-1, 25)
        cs = np.cumsum(Hb, 1) / np.maximum(tot, 1e-12)[:, None]
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(Hb > 0, Hb, np.nan), 1)
            heavy = Hb >= 2 * med[:, None]
            thin = Hb < 0.2 * Hb.max(1, keepdims=True)
            up, dn = heavy[:, 13:], heavy[:, :12][:, ::-1]
            res = {"above_share": (v * (y > 0)).sum(1) / tot, "poc": (Hb.argmax(1) - 12) * 0.25,
                   "va_lo": ((cs < 0.15).sum(1) - 12) * 0.25, "va_hi": ((cs < 0.85).sum(1) - 12) * 0.25,
                   "node_up": np.where(up.any(1), (up.argmax(1) + 1) * 0.25, 3.25),
                   "node_dn": -np.where(dn.any(1), (dn.argmax(1) + 1) * 0.25, 3.25),
                   "thin_up": thin[:, 13:17].mean(1), "thin_dn": thin[:, 8:12].mean(1),
                   "above_dec": (wv * (y > 0)).sum(1) / wt, "bulge30": (wv * np.maximum(y, 0)).sum(1) / wt,
                   "sag30": (wv * np.minimum(y, 0)).sum(1) / wt}
        empty = tot <= 0
        for k, x in res.items():
            out[k][ch] = np.where(empty, np.nan, x)
    return out


def gather(M, F, S, r, i, d):
    """Inputs for decisions at minute i (rows r) with data delay d: everything comes from bars before i - d."""
    c = np.maximum(i - d, 0)
    X = {k: F[k][r, c] for k in FULL if k in F}
    X["minute"], X["to_close"] = i.astype(np.float32), (390 - i).astype(np.float32)
    for k in ("sigma", "dollar_vol", "log_mcap", "d1", "d2", "d3", "prev_clv", "res_code", "days_since", "days_to"):
        X[k] = S[k][r]
    for k in ("gap_prev_vwap", "gap_prev_close"):
        X[k] = np.where(c >= 1, S[k][r], np.nan)
    for k in ("d_med", "d_iqr", "d_w90", "d_pup"):
        X[k] = np.where(c >= 15, S[k][r], np.nan)
    early = (d > 0) & (i <= d)  # decision before 09:30 + d: the pre-market up to 09:15 only
    for k in ("pm_last", "pm_vwap", "pm_vol", "pm_range"):
        X[k] = np.where(early, S[f"{k}555"][r], S[f"{k}570"][r])
    X.update(vap(M, r, c))
    return pd.DataFrame({k: np.asarray(X[k], np.float32) for k in FULL})


def future(M):
    """Highest high / lowest low over bars i .. i+h-1, per decision column i."""
    return {h: (fwd(M.H, h, np.fmax), fwd(M.L, h, np.fmin)) for h in HS}


WCHUNK = 100000


def targets(M, T, r, i, full=True):
    """Outcomes after decision minute i (bars i ..), measured from p = the last price before i, in sigma."""
    p = M.Cf[r, i - 1].astype(np.float64)
    s = M.sig[r]
    lg = np.log
    Y = {}
    with np.errstate(all="ignore"):
        raw = {h: (lg(T[h][0][r, i] / p), lg(T[h][1][r, i] / p)) for h in HS}
        for n, lev, kind in LEVELS:
            L = lev * s if kind == "s" else lev
            for h in REC_H:
                Y[f"tu_{n}_{h}"] = (raw[h][0] >= L).astype(np.int8)
                Y[f"td_{n}_{h}"] = (raw[h][1] <= -L).astype(np.int8)
        if not full:
            return Y
        for h in HS:
            Y[f"up_{h}"], Y[f"dn_{h}"] = raw[h][0] / s, raw[h][1] / s
        Y["lean_raw"] = sum(np.clip(Y[f"up_{h}"], 0, 5) + np.clip(Y[f"dn_{h}"], -5, 0) for h in HS)
        Y["ret_close"] = lg(M.Cf[r, 389] / p) / s
    Y["race"] = np.full(len(r), 2, np.int8)
    Y["volbal"] = np.full(len(r), np.nan, np.float32)
    for ch in np.array_split(np.arange(len(r)), max(1, len(r) // WCHUNK)):
        rr, ii, pp, ss = r[ch], i[ch], p[ch], s[ch]
        idx = ii[:, None] + MINS[None, :]
        valid = idx < 390
        idx = np.minimum(idx, 389)
        up = valid & (M.H[rr[:, None], idx] >= (pp * np.exp(0.5 * ss))[:, None])
        dn = valid & (M.L[rr[:, None], idx] <= (pp * np.exp(-0.5 * ss))[:, None])
        fu = np.where(up.any(1), up.argmax(1), 999)
        fd = np.where(dn.any(1), dn.argmax(1), 999)
        Y["race"][ch] = np.where(fu < fd, 0, np.where(fd < fu, 1, 2))
        w60 = valid[:, :60]
        vv = np.where(w60, M.V[rr[:, None], idx[:, :60]], 0)
        vw = M.VW[rr[:, None], idx[:, :60]]
        above = (vv * ((vw > pp[:, None]) + 0.5 * (vw == pp[:, None]))).sum(1)
        tot = vv.sum(1)
        Y["volbal"][ch] = np.where(tot > 0, above / np.maximum(tot, 1e-12), np.nan)
    Y["upfirst"] = (Y["race"] == 0).astype(np.int8)
    Y["dnfirst"] = (Y["race"] == 1).astype(np.int8)
    Y["volbal_up"] = (Y["volbal"] > 0.5).astype(np.int8)
    return Y


def slice_flags(M, F, S, r, i, hix, lox, first05):
    """Pre-registered slices at decision minute i, from what is known at i (except turn_*: hindsight)."""
    c = i
    with np.errstate(invalid="ignore"):
        rng = F["day_range"][r, c]
        s = {"burst": F["rvol_5"][r, c] >= np.log(3),
             "new_hl": (i >= 6) & ((F["_hc"][r, c] > F["_hc"][r, c - 1]) | (F["_lc"][r, c] < F["_lc"][r, c - 1])),
             "near_hi": (F["from_high"][r, c] >= -0.1) & (rng >= 0.5),
             "near_lo": (F["from_low"][r, c] <= 0.1) & (rng >= 0.5),
             "vwap_x": (i >= 2) & (np.sign(F["px_vwap"][r, c]) * np.sign(F["px_vwap"][r, c - 1]) < 0),
             "band_x": (i >= 16) & np.any([(np.minimum(F["band_px"][r, c], F["band_px"][r, c - 1]) < q)
                                           & (np.maximum(F["band_px"][r, c], F["band_px"][r, c - 1]) >= q)
                                           for q in (0.25, 0.75)], axis=0),
             "first05": i == first05[r],
             "gap": np.abs(S["gap_prev_close"][r]) >= 1,
             "results": np.isin(S["res_code"][r], [RES.index("Earnings_Day"), RES.index("Post_Earnings_Reaction")]),
             "mkt_move": np.abs(F["_mkt15"][r, c]) >= 1,
             "turn_hi": (np.abs(i - hix[r]) <= 5) & (i >= 15) & (i <= 375),
             "turn_lo": (np.abs(i - lox[r]) <= 5) & (i >= 15) & (i <= 375)}
    return {k: np.asarray(v, bool) for k, v in s.items()}


def turn_points(M):
    H = np.where(np.isfinite(M.H), M.H, -np.inf)
    L = np.where(np.isfinite(M.L), M.L, np.inf)
    return H.argmax(1), L.argmin(1)


def first_move(F):
    hit = np.abs(F["px_open"][:, 1:390]) >= 0.5
    return np.where(hit.any(1), hit.argmax(1) + 1, -1)


def ext_masks(F):
    """Dense (rows x 389) masks for the PIT near-extreme evaluation points, decisions i = 1 .. 389."""
    with np.errstate(invalid="ignore"):
        rng = F["day_range"][:, 1:390] >= 0.5
        return {"near_hi": rng & (F["from_high"][:, 1:390] >= -0.1), "near_lo": rng & (F["from_low"][:, 1:390] <= 0.1)}


# ---------------------------------------------------------------- features part

def curve_from(raw):
    V = raw["V"].astype(np.float64)
    tot = V.sum(1)
    c = np.cumsum(V, 1)[tot > 0] / tot[tot > 0, None]
    return np.concatenate([[0.0], c.mean(0)])


def part_features(a, st):
    mlist = months(a.first, a.last)
    (OUT / "summ").mkdir(parents=True, exist_ok=True)
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    st.start("features", 2 * len(mlist))
    for m in mlist:
        f = OUT / "summ" / f"{m}.parquet"
        if not f.exists():
            summarise(m).to_parquet(f)
        st.step(f"summary {m}")
    S = pd.concat([pd.read_parquet(OUT / "summ" / f"{m}.parquet") for m in mlist])
    info = info_table(S)
    rng = np.random.default_rng(a.seed)
    curve_f = OUT / "curve.npy"
    for m in mlist:
        f = OUT / "rows" / f"{m}.parquet"
        if f.exists():
            st.step(f"{m} cached")
            continue
        keep = set(info.tk[info.month == m] + "|" + info.day[info.month == m].astype(str))
        if not keep:
            st.step(f"{m} no eligible stock-days")
            continue
        raw = load_month(m, keep)
        if not curve_f.exists():
            np.save(curve_f, curve_from(raw))  # the typical intraday volume curve, from the first month only
        M = build_month(raw, info)
        rows = month_rows(M, np.load(curve_f), a, rng, m)
        rows.to_parquet(f)
        st.step(f"{m}: {len(rows):,} rows, {len(M.sig):,} stock-days")


def info_table(S):
    secmap = sector_map()
    d = day_stats(S, secmap)
    d = market_caps(d)
    d = d[d.elig].merge(forecasts().drop(columns=["date", "day_idx", "actual", "industry_code"], errors="ignore"),
                        on=["tk", "day"], how="left").merge(results_days(), on=["tk", "day"], how="left")
    d["month"] = pd.to_datetime(d.day, unit="D").dt.strftime("%Y-%m")
    return d


def month_rows(M, curve, a, rng, m):
    """The month's sampled decisions: kind 0 = training sample, 1 = test grid (every stock, every
    `grid` minutes from a per-day random offset), 2 = densely sampled near-extreme / turning points."""
    F = dense_features(M, curve)
    S = fixed_features(M)
    T = future(M)
    hix, lox = turn_points(M)
    f05 = first_move(F)
    n = len(M.sig)
    day0 = (pd.Timestamp(m + "-01") - EPOCH).days
    picks = []
    if day0 <= a.train_last:
        k = min(a.train_per_month, n * 389)
        flat = rng.choice(n * 389, k, replace=False)
        picks.append((flat // 389, flat % 389 + 1, 0))
    if day0 >= a.test_first:
        rr, ii = [], []
        for d in np.unique(M.day):
            rows = np.flatnonzero(M.day == d)
            mins = np.arange(rng.integers(1, a.grid + 1), 390, a.grid)
            rr.append(np.repeat(rows, len(mins)))
            ii.append(np.tile(mins, len(rows)))
        picks.append((np.concatenate(rr), np.concatenate(ii), 1))
        E = ext_masks(F)
        I = np.repeat(np.arange(1, 390)[None, :], n, 0)
        R = np.repeat(np.arange(n)[:, None], 389, 1)
        E["turn_hi"] = (np.abs(I - hix[:, None]) <= 5) & (I >= 15) & (I <= 375)
        E["turn_lo"] = (np.abs(I - lox[:, None]) <= 5) & (I >= 15) & (I <= 375)
        for k in EXT:
            cand = np.flatnonzero(E[k].ravel())
            cand = rng.choice(cand, min(a.ext_per_month, len(cand)), replace=False)
            picks.append((R.ravel()[cand], I.ravel()[cand], 2))
    r = np.concatenate([p[0] for p in picks]).astype(np.int64)
    i = np.concatenate([p[1] for p in picks]).astype(np.int64)
    kind = np.concatenate([np.full(len(p[0]), p[2], np.int8) for p in picks])
    ok = np.isfinite(M.Cf[r, i - 1]) & (M.Cf[r, i - 1] > 0)
    r, i, kind = r[ok], i[ok], kind[ok]
    out = {"tk": M.tk[r], "day": M.day[r].astype(np.int32), "i": i.astype(np.int16), "kind": kind,
           "sig": M.sig[r].astype(np.float32), "ndol": M.ndol[r].astype(np.float32),
           "mcap": M.mcap[r].astype(np.float32), "sec": M.sec[r].astype(np.int16),
           "p": M.Cf[r, i - 1].astype(np.float32), "prior_hi": F["_hc"][r, np.maximum(i - 1, 0)],
           "prior_lo": F["_lc"][r, np.maximum(i - 1, 0)]}
    for d, pre in ((15, "b_"), (0, "a_")):
        X = gather(M, F, S, r, i, d)
        out.update({pre + k: X[k].values for k in X})
    out.update(targets(M, T, r, i))
    out.update({"s_" + k: v for k, v in slice_flags(M, F, S, r, i, hix, lox, f05).items()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- models and scoring

def folds(a):
    out = []
    for y in a.test_years:
        te0 = (pd.Timestamp(f"{y}-01-01") - EPOCH).days
        te1 = (pd.Timestamp(f"{y}-12-31") - EPOCH).days
        cal0 = (pd.Timestamp(f"{y}-01-01") - pd.DateOffset(months=a.cal_months) - EPOCH).days
        out.append(SimpleNamespace(year=y, test=(te0, te1), cal0=cal0, train_last=te0 - 1))
    return out


def read_rows(cols, kinds, day_lo, day_hi):
    fs = sorted((OUT / "rows").glob("*.parquet"))
    parts = []
    for f in fs:
        m0 = (pd.Timestamp(f.stem + "-01") - EPOCH).days
        if m0 > day_hi or m0 + 31 < day_lo:
            continue
        x = pd.read_parquet(f, columns=cols, filters=[("kind", "in", list(kinds))])
        parts.append(x[(x.day >= day_lo) & (x.day <= day_hi)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def groups(df, cuts):
    b = ((df.i.values - 1) // 5).astype(np.int64)
    s = np.searchsorted(cuts[0], df.sig.values)
    lq = np.searchsorted(cuts[1], df.ndol.values)
    return b * 9 + s * 3 + lq


def tercile_cuts(df):
    return (np.quantile(df.sig, [1 / 3, 2 / 3]), np.quantile(df.ndol, [1 / 3, 2 / 3]))


def clim(train, cuts, ycols):
    """Climatology: the training mean of each target per group (5-min bucket x sigma x liquidity tercile)."""
    g = groups(train, cuts)
    tab = pd.DataFrame({c: train[c].values.astype(np.float64) for c in ycols}).groupby(g).mean()
    overall = train[ycols].astype(np.float64).mean()
    return tab, overall


def clim_of(tab, overall, df, cuts, col):
    g = groups(df, cuts)
    return pd.Series(g).map(tab[col]).fillna(overall[col]).values


def fit_binary(X, y, Xc, yc, seed):
    m = HistGradientBoostingClassifier(**HGB, random_state=seed).fit(X, y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(m.predict_proba(Xc)[:, 1], yc)
    return m, iso


def predict_binary(model, X):
    m, iso = model
    return iso.predict(m.predict_proba(X)[:, 1])


def ci_skill(err_m, err_c, days, rng, n=2000):
    """Skill = 1 - sum(model error) / sum(base error), with a day-bootstrap 95% CI."""
    g = pd.DataFrame({"m": err_m, "c": err_c, "d": days}).groupby("d").sum()
    m, c = g.m.values, g.c.values
    pt = 1 - m.sum() / c.sum()
    idx = rng.integers(0, len(m), (n, len(m)))
    bs = 1 - m[idx].sum(1) / c[idx].sum(1)
    return pt, *np.percentile(bs, [2.5, 97.5]), len(m)


def ci_mean(x, days, rng, n=2000):
    m = pd.Series(x).groupby(days).mean().values
    if not len(m):
        return np.nan, np.nan, np.nan, 0
    idx = rng.integers(0, len(m), (n, len(m)))
    return m.mean(), *np.percentile(m[idx].mean(1), [2.5, 97.5]), len(m)


def rank_ic(df, pred, real):
    """Spearman per (day, minute) across stocks, then averaged per day."""
    d = pd.DataFrame({"day": df.day.values, "i": df.i.values, "p": pred, "y": real}).dropna()
    d["pr"] = d.groupby(["day", "i"]).p.rank()
    d["yr"] = d.groupby(["day", "i"]).y.rank()
    ic = d.groupby(["day", "i"]).apply(lambda x: np.corrcoef(x.pr, x.yr)[0, 1] if len(x) > 30 else np.nan,
                                       include_groups=False)
    ic = ic.dropna()
    return ic.groupby(level=0).mean()


def top_lists(df, score, k=20):
    """Per (day, minute): the top-k by score (up list) and bottom-k (down list); their realised outcomes."""
    d = pd.DataFrame({"day": df.day.values, "i": df.i.values, "s": score, "lean": df.lean_x.values,
                      "rc": df.ret_close.values, "up": df.up_60.values, "dn": df.dn_60.values}).dropna(subset=["s"])
    d["r_up"] = d.groupby(["day", "i"]).s.rank(ascending=False, method="first")
    d["r_dn"] = d.groupby(["day", "i"]).s.rank(ascending=True, method="first")
    u = d[d.r_up <= k].groupby(["day", "i"])[["lean", "rc", "up", "dn"]].mean()
    w = d[d.r_dn <= k].groupby(["day", "i"])[["lean", "rc", "up", "dn"]].mean()
    j = u.join(w, lsuffix="_u", rsuffix="_d").dropna()
    j["spread_lean"] = j.lean_u - j.lean_d
    j["spread_rc"] = j.rc_u - j.rc_d
    return j.groupby(level=0).mean()


def part_ladder(a, st, R):
    rng = np.random.default_rng(a.seed)
    fl = folds(a)
    st.start("ladder", len(fl) * len(ARMS))
    feats = sorted({p + f for f in FULL for p in ("a_", "b_")})
    ycols = list(LADDER_Y.values()) + ["lean_raw", "ret_close", "up_60", "dn_60"]
    meta = ["tk", "day", "i", "kind", "sig", "ndol", "mcap"] + [f"s_{s}" for s in SLICES]
    R["ladder"] = {}
    for fo in fl:
        tr = read_rows(meta + feats + ycols, [0], 0, fo.train_last)
        cuts = tercile_cuts(tr)
        tab, ov = clim(tr, cuts, ycols)
        tr["lean_x"] = tr.lean_raw - clim_of(tab, ov, tr, cuts, "lean_raw")
        fit = tr[tr.day < fo.cal0]
        fit = fit.iloc[rng.choice(len(fit), min(a.train_n, len(fit)), replace=False)]
        cal = tr[tr.day >= fo.cal0]
        te = read_rows(meta + feats + ycols, [1, 2], *fo.test)
        te["lean_x"] = te.lean_raw - clim_of(tab, ov, te, cuts, "lean_raw")
        P = {"tk": te.tk.values, "day": te.day.values, "i": te.i.values, "kind": te.kind.values}
        for k, col in LADDER_Y.items():
            P[f"clim_{k}"] = clim_of(tab, ov, te, cuts, col)
        log(f"fold {fo.year}: train {len(fit):,} (+{len(cal):,} recal), test {len(te):,}")
        for arm, (cols, d) in ARMS.items():
            pre = "a_" if d == 0 else "b_"
            X, Xc, Xt = (x[[pre + c for c in cols]].values for x in (fit, cal, te))
            for k, col in LADDER_Y.items():
                mdl = fit_binary(X, fit[col].values, Xc, cal[col].values, a.seed)
                P[f"{arm}_{k}"] = predict_binary(mdl, Xt).astype(np.float32)
            reg = HistGradientBoostingRegressor(**HGB, random_state=a.seed).fit(X, fit.lean_x.values)
            P[f"{arm}_lean"] = reg.predict(Xt).astype(np.float32)
            st.step(f"fold {fo.year} {arm}")
        pdf = pd.DataFrame(P)
        for c in ycols + ["lean_x", "sig", "ndol", "mcap", "b_px_open"] + [f"s_{s}" for s in SLICES]:
            pdf[c] = te[c].values
        pdf.to_parquet(OUT / f"pred_{fo.year}.parquet")
    R["ladder"] = ladder_report(a, rng)


def ladder_report(a, rng):
    out = {}
    for y in a.test_years:
        te = pd.read_parquet(OUT / f"pred_{y}.parquet")
        g = te[te.kind == 1]
        res = {}
        for arm in ARMS:
            r = {}
            for k, col in LADDER_Y.items():
                yv = g[col].values.astype(np.float64)
                r[f"brier_{k}"] = ci_skill((g[f"{arm}_{k}"] - yv) ** 2, (g[f"clim_{k}"] - yv) ** 2, g.day.values, rng)
            ic = rank_ic(g, g[f"{arm}_lean"].values, g.lean_x.values)
            r["rank_ic"] = ci_mean(ic.values, ic.index.values, rng)
            tl = top_lists(g, g[f"{arm}_lean"].values)
            r["top20_lean"] = ci_mean(tl.spread_lean.values, tl.index.values, rng)
            r["top20_retclose"] = ci_mean(tl.spread_rc.values, tl.index.values, rng)
            res[arm] = r
        tl = top_lists(g, g.b_px_open.values)
        ic = rank_ic(g, g.b_px_open.values, g.lean_x.values)
        res["open_rule"] = {"rank_ic": ci_mean(ic.values, ic.index.values, rng),
                            "top20_lean": ci_mean(tl.spread_lean.values, tl.index.values, rng),
                            "top20_retclose": ci_mean(tl.spread_rc.values, tl.index.values, rng)}
        out[str(y)] = res
    return out


def part_scan(a, st, R):
    """Opportunity map (no model) and predictability map (skill of full_d0 and s6_results) per slice."""
    rng = np.random.default_rng(a.seed + 1)
    st.start("scan", 3)
    te = pd.concat([pd.read_parquet(OUT / f"pred_{y}.parquet").assign(year=y) for y in a.test_years])
    grid = te[te.kind == 1]
    te["big_up"], te["big_dn"] = te.up_60 >= 1, te.dn_60 <= -1
    grid = te[te.kind == 1]
    b = (grid.i - 1) // 5
    opp = grid.groupby(b).agg(n=("big_up", "size"), p_up1=("big_up", "mean"), p_dn1=("big_dn", "mean"),
                              med_up=("up_60", "median"), med_dn=("dn_60", "median"))
    opp.index = [f"{9 + (570 + 5 * x) // 60 - 9:02d}:{(570 + 5 * x) % 60:02d}" for x in opp.index]
    st.step("opportunity by minute")
    rows = []
    disc = [y for y in a.test_years if y != a.test_years[-1]] or a.test_years
    conf = a.test_years[-1]
    for s in ["all"] + SLICES + ["cap_" + c for c, _ in CAPS]:
        if s == "all":
            sub = grid
        elif s.startswith("cap_"):
            sub = grid[cap_band(grid.mcap.values) == s[4:]]
        else:
            sub = te[te[f"s_{s}"] & ((te.kind == 1) | (s in EXT))]
        r = {"slice": s, "n": len(sub), "p_up1": sub.big_up.mean(), "p_dn1": sub.big_dn.mean(),
             "med_up": sub.up_60.median(), "med_dn": sub.dn_60.median()}
        for arm in ("full_d0", "s6_results"):
            for k, col in (("tu", "tu_0.5s_60"), ("td", "td_0.5s_60"), ("uf", "upfirst")):
                for tag, ys in (("disc", disc), ("conf", [conf])):
                    x = sub[sub.year.isin(ys)]
                    if len(x) < 500:
                        r[f"{arm}_{k}_{tag}"] = (np.nan, np.nan, np.nan, 0)
                        continue
                    yv = x[col].values.astype(np.float64)
                    r[f"{arm}_{k}_{tag}"] = ci_skill((x[f"{arm}_{k}"] - yv) ** 2, (x[f"clim_{k}"] - yv) ** 2,
                                                     x.day.values, rng)
        rows.append(r)
    st.step("predictability by slice")
    R["scan"] = {"by_minute": opp.reset_index().rename(columns={"index": "time"}).to_dict("records"),
                 "slices": rows}
    st.step("done")


# ---------------------------------------------------------------- recommendation count

def fire(P, ok, h, thr=0.6):
    """First minute with P >= thr per row; after a fire at column j the same key can fire again from j + h
    (h >= 390: once a day). Returns the boolean fire mask."""
    hit = (P >= thr) & ok
    out = np.zeros_like(hit)
    nxt = np.zeros(len(P), np.int64)
    for j in range(P.shape[1]):
        f = hit[:, j] & (j >= nxt)
        out[:, j] = f
        nxt = np.where(f, j + h, nxt)
    return out


def on_list(rank, join=20, leave=35):
    """Up / down list membership per minute with hysteresis: join at rank <= join, stay while <= leave."""
    out = np.zeros(rank.shape, bool)
    prev = np.zeros(rank.shape[0], bool)
    for j in range(rank.shape[1]):
        prev = (rank[:, j] <= join) | (prev & (rank[:, j] <= leave))
        out[:, j] = prev
    return out


def part_recs(a, st, R):
    rng = np.random.default_rng(a.seed + 2)
    fo = folds(a)[-1]
    feats = sorted({p + f for f in FULL for p in ("a_", "b_")})
    tr = read_rows(["day", "i", "sig", "ndol"] + feats + TOUCH + ["lean_raw"], [0], 0, fo.train_last)
    cuts = tercile_cuts(tr)
    tab, ov = clim(tr, cuts, TOUCH + ["lean_raw"])
    tr["lean_x"] = tr.lean_raw - clim_of(tab, ov, tr, cuts, "lean_raw")
    fit = tr[tr.day < fo.cal0]
    fit = fit.iloc[rng.choice(len(fit), min(a.train_n, len(fit)), replace=False)]
    cal = tr[tr.day >= fo.cal0]
    models = {}
    st.start("recs-train", 2 * len(TOUCH) + 1)
    for d, pre in ((0, "a_"), (15, "b_")):
        X, Xc = fit[[pre + c for c in FULL]].values, cal[[pre + c for c in FULL]].values
        for t in TOUCH:
            models[(d, t)] = fit_binary(X, fit[t].values, Xc, cal[t].values, a.seed)
            st.step(f"d={d} {t}")
    lean = HistGradientBoostingRegressor(**HGB, random_state=a.seed).fit(fit[["b_" + c for c in FULL]].values,
                                                                         fit.lean_x.values)
    st.step("lean d=15")
    del tr, fit, cal
    mlist = [m for m in months(a.first, a.last) if pd.Timestamp(m + "-01").year == fo.year]
    S = pd.concat([pd.read_parquet(OUT / "summ" / f"{m}.parquet") for m in months(a.first, a.last)])
    info = info_table(S)
    curve = np.load(OUT / "curve.npy")
    sess = np.sort(info.day[info.month.isin(mlist)].unique())[::a.recs_stride]
    st.start("recs", len(sess))
    rows, fired_log = [], []
    for m in mlist:
        days = sess[pd.to_datetime(sess, unit="D").strftime("%Y-%m") == m]
        if not len(days):
            continue
        sel = info[(info.month == m)]
        keep = set(sel.tk + "|" + sel.day.astype(str))
        M = build_month(load_month(m, keep), info)
        F, Sx, T = dense_features(M, curve), fixed_features(M), future(M)
        for dday in days:
            rr = np.flatnonzero(M.day == dday)
            n = len(rr)
            r = np.repeat(rr, 389)
            i = np.tile(np.arange(1, 390), n)
            ok = (np.isfinite(M.Cf[r, i - 1]) & (M.Cf[r, i - 1] > 0)).reshape(n, 389)
            Y = targets(M, T, r, i, full=False)
            meta = pd.DataFrame({"i": i, "sig": M.sig[r], "ndol": M.ndol[r]})
            Pr = {}
            for d, pre in ((0, "a_"), (15, "b_")):
                X = gather(M, F, Sx, r, i, d)[FULL].values
                for t in TOUCH:
                    Pr[(d, t)] = predict_binary(models[(d, t)], X).reshape(n, 389)
                if d == 15:
                    L = lean.predict(X).reshape(n, 389)
            L = np.where(ok, L, np.nan)
            o = np.argsort(np.where(np.isfinite(L), -L, np.inf), axis=0)
            rk_up = np.empty_like(o)
            np.put_along_axis(rk_up, o, np.arange(n)[:, None] + 1, axis=0)
            o = np.argsort(np.where(np.isfinite(L), L, np.inf), axis=0)
            rk_dn = np.empty_like(o)
            np.put_along_axis(rk_dn, o, np.arange(n)[:, None] + 1, axis=0)
            lists = {"u": on_list(rk_up), "d": on_list(rk_dn)}
            for t in TOUCH:
                u, lev, h = t[1], t.split("_")[1], int(t.split("_")[2])
                y = Y[t].reshape(n, 389).astype(bool)
                base = clim_of(tab, ov, meta, cuts, t).reshape(n, 389)
                lv, kind_l = LEVEL[lev]
                size = np.broadcast_to(lv * (M.sig[rr][:, None] if kind_l == "s" else 1.0), (n, 389))
                for setting, d, mask in (("all_d0", 0, ok), ("pipeline", 0, ok & lists[u]), ("all_d15", 15, ok)):
                    P = Pr[(d, t)]
                    for kind, m2 in (("all", mask), ("edge", mask & (base < 0.6)), ("lift10", mask & (P - base >= 0.1))):
                        fm = fire(P, m2, h)
                        nf = int(fm.sum())
                        rows.append({"day": dday, "setting": setting, "kind": kind, "target": t, "fired": nf,
                                     "true": int((fm & y).sum()), "stocks": int(fm.any(1).sum()),
                                     "mean_p": float(P[fm].mean()) if nf else np.nan,
                                     "mean_base": float(base[fm].mean()) if nf else np.nan,
                                     "size_pct": float(np.median(size[fm]) * 100) if nf else np.nan,
                                     **{f"h{x}": int(fm[:, x * 60:(x + 1) * 60].sum()) for x in range(7)}})
                        if kind == "all" and setting == "all_d0" and nf:
                            jj = np.argwhere(fm)
                            take = jj[rng.choice(len(jj), min(20, len(jj)), replace=False)]
                            fired_log.append(pd.DataFrame({"tk": M.tk[rr[take[:, 0]]], "day": dday, "i": take[:, 1] + 1,
                                                           "target": t, "p": P[take[:, 0], take[:, 1]],
                                                           "true": y[take[:, 0], take[:, 1]]}))
            st.step(f"{pd.Timestamp(EPOCH + pd.Timedelta(days=int(dday))).date()}: {n} stocks")
        del M, F, T
    rec = pd.DataFrame(rows)
    rec.to_parquet(OUT / "recs_days.parquet")
    if fired_log:
        pd.concat(fired_log).to_parquet(OUT / "recs_sample.parquet")
    R["recs"] = recs_report(rec)


def recs_report(rec):
    out = {}
    for (setting, kind), g in rec.groupby(["setting", "kind"]):
        per_day = g.groupby("day")[["fired", "true"]].sum()
        tgt = g.groupby("target").agg(fired_day=("fired", "mean"), true_day=("true", "mean"),
                                      stocks_day=("stocks", "mean"), size_pct=("size_pct", "median"))
        tot = g.groupby("target")[["fired", "true"]].sum()
        tgt["precision"] = tot.true / tot.fired.where(tot.fired > 0)
        for c in ("mean_p", "mean_base"):
            mp = g.assign(w=g[c] * g.fired).groupby("target")[["w", "fired"]].sum()
            tgt[c] = mp.w / mp.fired.where(mp.fired > 0)
        hours = g[[f"h{x}" for x in range(7)]].sum() / g.day.nunique()
        out[f"{setting}/{kind}"] = {
            "days": int(g.day.nunique()),
            "fired_per_day": {q: float(per_day.fired.quantile(v)) for q, v in (("p10", .1), ("median", .5), ("p90", .9))},
            "true_per_day": {q: float(per_day.true.quantile(v)) for q, v in (("p10", .1), ("median", .5), ("p90", .9))},
            "precision": float(per_day.true.sum() / max(per_day.fired.sum(), 1)),
            "mean_base": float((g.mean_base * g.fired).sum() / max(g.fired.sum(), 1)),
            "by_target": tgt.reset_index().to_dict("records"),
            "by_hour": hours.to_dict()}
    return out


# ---------------------------------------------------------------- 1 s re-timing

def part_retime(a, st, R):
    """For new-day-high / low minutes: the second the level broke, and how much of the next 15 min's move
    was already done by the minute end."""
    rng = np.random.default_rng(a.seed + 3)
    rows = read_rows(["tk", "day", "i", "kind", "p", "prior_hi", "prior_lo", "s_new_hl"], [1, 2],
                     *folds(a)[-1].test)
    ev = rows[rows.s_new_hl].copy()
    have = {d: Path(f"{NAS}/agg_1s/date={d}/source=massive_rest/day-{d}.parquet")
            for d in pd.to_datetime(ev.day.unique(), unit="D").strftime("%Y-%m-%d")}
    have = {d: f for d, f in have.items() if f.exists()}
    ev["date"] = pd.to_datetime(ev.day, unit="D").dt.strftime("%Y-%m-%d")
    days = sorted(have)
    days = list(rng.choice(days, min(a.retime_days, len(days)), replace=False)) if days else []
    ev = ev[ev.date.isin(days)]  # each 1 s day file is ~100 MB over SMB, so a sample of days
    ev = ev.iloc[rng.choice(len(ev), min(a.retime_n, len(ev)), replace=False)] if len(ev) else ev
    st.start("retime", max(ev.date.nunique(), 1))
    out = []
    for d, g in ev.groupby("date"):
        tb = ds.dataset(str(have[d])).to_table(columns=["symbol", "ts_utc", "price"],
                                               filter=pc.field("symbol").isin(list(g.tk.unique())))
        t = tb["ts_utc"].combine_chunks().cast(pa.int64()).to_numpy()  # epoch ms
        x = pd.DataFrame({"symbol": tb["symbol"].to_numpy(), "price": tb["price"].to_numpy(),
                          "et_s": (et_minutes(t) % 1440) * 60 + (t // 1000) % 60})
        by = dict(tuple(x.groupby("symbol")))
        for e in g.itertuples():
            s = by.get(e.tk)
            if s is None:
                continue
            m0 = (570 + e.i - 1) * 60
            minute = s[(s.et_s >= m0) & (s.et_s < m0 + 60)]
            after = s[(s.et_s >= m0) & (s.et_s < m0 + 60 * 16)]
            up = e.prior_hi > 0 and (minute.price > e.prior_hi).any()
            dn = e.prior_lo > 0 and (minute.price < e.prior_lo).any()
            if not (up or dn) or after.empty:
                continue
            br = minute[minute.price > e.prior_hi] if up else minute[minute.price < e.prior_lo]
            s0, p0 = br.et_s.iloc[0], br.price.iloc[0]
            ext = after[after.et_s >= s0].price.max() if up else after[after.et_s >= s0].price.min()
            move = (ext - p0) if up else (p0 - ext)
            done = ((e.p - p0) if up else (p0 - e.p))
            out.append({"secs_left": m0 + 60 - s0, "move_pct": move / p0 * 100,
                        "gone": done / move if move > 0 else np.nan})
        st.step(d)
    o = pd.DataFrame(out)
    R["retime"] = {"events": len(o), "median_secs_left": float(o.secs_left.median()) if len(o) else None,
                   "median_gone": float(o.gone.median()) if len(o) else None,
                   "median_move_pct": float(o.move_pct.median()) if len(o) else None}


# ---------------------------------------------------------------- report

def fmt_ci(t):
    pt, lo, hi, _ = t
    return f"{pt * 100:+.1f}% [{lo * 100:+.1f}, {hi * 100:+.1f}]" if np.isfinite(pt) else "n/a"


def fmt_num(t, k=3):
    pt, lo, hi, _ = t
    return f"{pt:+.{k}f} [{lo:+.{k}f}, {hi:+.{k}f}]" if np.isfinite(pt) else "n/a"


def write_report(R):
    L = []
    if "ladder" in R:
        L.append("== Step 0: skill per input group (test grid rows; Brier skill vs climatology, 95% day CI)")
        for y, res in R["ladder"].items():
            L.append(f"-- test {y}")
            L.append(f"{'arm':12} {'touch+0.5s/60':>24} {'touch-0.5s/60':>24} {'up first':>24} {'vol above':>24}"
                     f" {'rank IC lean':>26} {'top20 lean spread (s)':>26} {'top20 ret-close (s)':>26}")
            for arm, r in res.items():
                if arm == "open_rule":
                    L.append(f"{arm:12} {'':>24} {'':>24} {'':>24} {'':>24} {fmt_num(r['rank_ic']):>26}"
                             f" {fmt_num(r['top20_lean'], 2):>26} {fmt_num(r['top20_retclose'], 2):>26}")
                    continue
                L.append(f"{arm:12} " + " ".join(f"{fmt_ci(r[f'brier_{k}']):>24}" for k in LADDER_Y)
                         + f" {fmt_num(r['rank_ic']):>26} {fmt_num(r['top20_lean'], 2):>26}"
                           f" {fmt_num(r['top20_retclose'], 2):>26}")
    if "scan" in R:
        L.append("\n== Step 0b: opportunity map by time of day (grid rows, all test years)")
        L.append(f"{'time':>6} {'n':>9} {'P(up60>=1s)':>12} {'P(dn60<=-1s)':>13} {'med up60':>9} {'med dn60':>9}")
        for r in R["scan"]["by_minute"]:
            L.append(f"{r['time']:>6} {r['n']:>9,} {r['p_up1']:>12.3f} {r['p_dn1']:>13.3f} {r['med_up']:>9.2f}"
                     f" {r['med_dn']:>9.2f}")
        L.append("\n== Step 0b: slices (opportunity; predictability = Brier skill of full_d0, discover vs confirm)")
        for r in R["scan"]["slices"]:
            L.append(f"{r['slice']:10} n={r['n']:>9,} P(up>=1s)={r['p_up1']:.3f} P(dn<=-1s)={r['p_dn1']:.3f}"
                     f" | tu disc {fmt_ci(r['full_d0_tu_disc'])} conf {fmt_ci(r['full_d0_tu_conf'])}"
                     f" | td disc {fmt_ci(r['full_d0_td_disc'])} conf {fmt_ci(r['full_d0_td_conf'])}"
                     f" | upfirst disc {fmt_ci(r['full_d0_uf_disc'])} conf {fmt_ci(r['full_d0_uf_conf'])}"
                     f" | d15 tu conf {fmt_ci(r['s6_results_tu_conf'])}")
    if "recs" in R:
        L.append("\n== Recommendations per day (P(touch) >= 60%, once per window; came true = touched in the window)")
        for k, r in R["recs"].items():
            fd, td = r["fired_per_day"], r["true_per_day"]
            L.append(f"-- {k}: {r['days']} days; per day fired median {fd['median']:.0f} (p10 {fd['p10']:.0f},"
                     f" p90 {fd['p90']:.0f}); came true median {td['median']:.0f} (p10 {td['p10']:.0f},"
                     f" p90 {td['p90']:.0f}); precision {r['precision']:.1%} (typical rate {r['mean_base']:.1%})")
            for t in r["by_target"]:
                pr, mp, mb = (t[k] if t[k] == t[k] else 0 for k in ("precision", "mean_p", "mean_base"))
                L.append(f"   {t['target']:14} fired/day {t['fired_day']:8.1f} true/day {t['true_day']:8.1f}"
                         f" stocks/day {t['stocks_day']:7.1f} precision {pr:6.1%} mean P {mp:6.1%}"
                         f" typical {mb:6.1%} size {t['size_pct']:.2f}%")
    if "retime" in R:
        L.append(f"\n== 1 s re-timing of new day high / low minutes: {R['retime']}")
    txt = "\n".join(L)
    (OUT / "report.txt").write_text(txt + "\n")
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item")
                                                 else list(o) if isinstance(o, tuple) else str(o)))
    log(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="features,ladder,scan,recs,retime")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-n", type=int, default=2_000_000)
    ap.add_argument("--quick", action="store_true", help="a small end-to-end run (2023-09 .. 2024-02)")
    ap.add_argument("--cache", default="/tmp/mtap/intraday_probe")
    a = ap.parse_args()
    global OUT
    OUT = Path(a.cache)
    if a.quick:
        a.first, a.last, a.test_years, a.cal_months = "2023-09", "2024-02", [2024], 1
        a.train_per_month, a.grid, a.ext_per_month, a.train_n, a.recs_stride, a.retime_n = 20000, 60, 3000, 40000, 20, 50
        a.retime_days = 2
    else:
        a.first, a.last, a.test_years, a.cal_months = "2021-10", "2026-10", [2024, 2025, 2026], 3
        a.train_per_month, a.grid, a.ext_per_month, a.recs_stride, a.retime_n = 80000, 30, 30000, 2, 2000
        a.retime_days = 40
    a.train_last = (pd.Timestamp(f"{a.test_years[-1] - 1}-12-31") - EPOCH).days
    a.test_first = (pd.Timestamp(f"{a.test_years[0]}-01-01") - EPOCH).days
    parts = a.parts.split(",")
    names = [p for p in ("features", "ladder", "scan", "recs-train", "recs", "retime")
             if p in parts or (p == "recs-train" and "recs" in parts)]
    st = Status(names)
    R = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    for p, fn in (("features", part_features), ("ladder", part_ladder), ("scan", part_scan), ("recs", part_recs),
                  ("retime", part_retime)):
        if p in parts:
            fn(a, st, R) if p != "features" else fn(a, st)
            write_report(R)
    (OUT / "status.txt").write_text(f"[done] run {(time.time() - st.t0) / 60:.0f} min · report: {OUT}/report.txt\n")


if __name__ == "__main__":
    main()
