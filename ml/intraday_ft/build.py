"""intraday-ft-build: the month frames for the intraday v2 model (docs/intraday-features.md §2-§5).

    intraday-ft-build --from 2023-01 --to 2026-10 [--check] [--no-seconds] [--every 5]

Per month, from the NAS (read only) into CACHE/frames/month=YYYY-MM/:
- rows.parquet   one row per stock-day of the day's top --stocks eligible stocks (the live stand-in):
                 keys, slices, the static inputs (chunk 3) and the
                 per-stock-day scalars the windows need (sig, prev);
- seq.npy        float16 (stock-days x 390 x n_seq): every observed / known input per minute, in the
                 column order of seq_cols in meta.json;
- dec.parquet    one row per sampled decision (stock-day, minute i): the realised targets of §5 and the
                 persistence baseline. Decisions are every `--every` minutes (staggered per stock-day)
                 from minute 4 (the encoder needs >= 5 steps) to 384.
- meta.json      columns, counts, timings, NaN shares, S-tier coverage.
--check builds one month and prints the S2 check (timings, NaN shares, S-tier coverage) without
writing more. Reference values (sigma, the normals, beta) use only sessions before the day. The months
are streamed in date order and each month's minute bars are read from the NAS once: its summary joins
a window of the WARM months before it (enough for 20 sessions and beta's 60), which gives the month's
reference values, and the same read builds its frames. The WARM months before --from are read for
the window only.
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data as D
from . import features as Fe
from . import profile as Pf
from . import targets as Tg
from .zigzag import zigzag

WARM = 4          # months of summaries before each month: >= 61 sessions for beta's 60 changes, any month
N_NORM = 20
N_BETA = 60
MIN_SESS = 15     # sigma and the normals need >= 15 of the 20 sessions
FIRST_DEC, LAST_DEC = 4, 384
PM_CUT_SWITCH = 15  # pm555 for minutes 0-14, pm570 from 15
FRAME_VERSION = 1   # bump when a change alters the frames; months built by another version are rebuilt


# ---------------------------------------------------------------- reference data (prior sessions only)

def reference(S, A, t):
    """Per stock-day, from prior sessions only: the stock id, the previous session (prev = VWAP,
    prevclose, prevpost), sig, ndol, typical, prior and the pre-market normals npm555 / npm570. S, A = the
    summarise() outputs of consecutive months, concatenated. Returns S and A sorted by (stock id, day);
    the per-minute normals come from normals() for the rows that need them."""
    S = S.reset_index(drop=True).copy()
    S["sid"] = D.stock_ids(S.tk.to_numpy(), S.day.to_numpy(), t)
    o = np.lexsort((S.day.to_numpy(), S.sid.to_numpy()))
    S, A = S.iloc[o].reset_index(drop=True), {k: v[o] for k, v in A.items()}
    g = S.groupby("sid", sort=False)
    S["prev"], S["prevclose"], S["prevpost"] = g.vwap.shift(1), g.close.shift(1), g.post.shift(1)
    S["r"] = np.log(S.vwap / S.prev)
    S.loc[S.r.abs() > 0.4, "r"] = np.nan  # split guard
    g = S.groupby("sid", sort=False)
    def roll(col, f, n=N_NORM):
        return g[col].transform(lambda s: getattr(s.rolling(n, min_periods=MIN_SESS), f)().shift(1))
    S["sig"] = roll("r", "std")
    S["ndol"] = roll("dpv", "mean")
    S["typical"] = roll("dv", "median")
    S["prior"] = g.cumcount()
    sid, rows = S.sid.to_numpy(), np.arange(len(S))
    for c in (555, 570):
        S[f"npm{c}"] = _prior_median(A[f"pmv{c}"][:, None], sid, rows)[:, 0]
    return S, A


def normals(S, A, rows):
    """The per-minute normals of the rows `rows` (positions in reference()'s sorted S): normal volume /
    trades per minute (the median over the previous 20 sessions of the same stock) and sigma_1m per
    15-min bucket. Only the frame rows need them: the whole window would be 20 x stock-days x 390."""
    sid = S.sid.to_numpy()
    out = {"nvm": _prior_median(A["V"], sid, rows), "nnm": _prior_median(A["N"], sid, rows)}
    ss, cnt = _prior_sum(A["ss"], sid, rows), _prior_sum(A["cnt"], sid, rows)
    with np.errstate(all="ignore"):
        out["sig1m"] = np.sqrt(ss / np.where(cnt >= 20, cnt, np.nan))
    return out


def _prior_window(X, sid, rows, n=N_NORM):
    """(n, len(rows), ...) stack: each row's values from 1 .. n sessions before (same stock, X sorted by
    stock and day), NaN past the stock's first row."""
    rows = np.asarray(rows)
    src = rows[None, :] - np.arange(1, n + 1)[:, None]
    ok = (src >= 0) & (sid[np.maximum(src, 0)] == sid[rows][None, :])
    W = np.asarray(X, np.float32)[np.maximum(src, 0)]
    W[~ok] = np.nan
    return W


def _prior_median(X, sid, rows, n=N_NORM):
    W = _prior_window(X, sid, rows, n)
    ok = np.isfinite(W).sum(0) >= MIN_SESS
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(W, 0)
    return np.where(ok, m, np.nan).astype(np.float32)


def _prior_sum(X, sid, rows, n=N_NORM):
    return np.nansum(_prior_window(X, sid, rows, n), 0)


def beta60(S, spy_daily):
    """beta vs SPY of the daily VWAP-to-VWAP change over the previous 60 sessions (>= 40 needed)."""
    sp = spy_daily[["day", "vw"]].sort_values("day").copy()
    sp["sr"] = np.log(sp.vw / sp.vw.shift(1))
    d = S[["sid", "day", "r"]].merge(sp[["day", "sr"]], on="day", how="left")
    d["xy"], d["xx"] = d.r * d.sr, d.sr ** 2
    g = d.groupby("sid", sort=False)
    m = {}
    for c in ("r", "sr", "xy", "xx"):
        m[c] = g[c].transform(lambda s: s.rolling(N_BETA, min_periods=40).mean().shift(1))
    with np.errstate(all="ignore"):
        return ((m["xy"] - m["r"] * m["sr"]) / (m["xx"] - m["sr"] ** 2)).to_numpy()


def eligible(S, live_n=D.LIVE_N):
    """§2 universe: prior-20 mean dollar volume >= $20M, sigma > 0.3%, >= 3 prior sessions, an open
    today, a common stock. live_n: the day's top eligible stocks by prior dollar volume stand in for the
    live set in training (the E*TRADE lists exist only from 2026-10-12). The frames hold only those: the
    whole universe would be ~2.5 GB a month of float16 sequences, and the model is used on the ~1,000
    live symbols."""
    e = ((S.ndol >= D.MIN_DOLLAR_VOL) & (S.sig > D.MIN_SIGMA) & (S.prior >= D.MIN_PRIOR) & S.open.notna()
         & S.prev.notna() & S.typical.gt(0) & S.common)
    rank = S.ndol.where(e).groupby(S.day).rank(ascending=False, method="first")
    return e.to_numpy(), (rank <= live_n).to_numpy()


# ---------------------------------------------------------------- one month

def build_month(month, ctx, R, norms, raw=None, seconds=True, every=5, seed=0, workers=1):
    """Every frame of one month. ctx: static_tables(); R, norms: month_reference() (the month's frame
    stock-days and their per-minute normals); raw: D.read_month(month) when the caller has it; workers:
    processes for the S tier (one day per job). Returns (rows, seq, dec, meta) without writing."""
    tm = {}
    t0 = time.time()
    raw = raw if raw is not None else D.read_month(month)
    tm["read"] = time.time() - t0
    keep = list(zip(R.tk, R.day))
    t0 = time.time()
    L = D.load_month(month, keep, raw)
    del raw
    tm["load"] = time.time() - t0
    k = pd.DataFrame({"tk": L["tk"], "day": L["day"]}).merge(R, on=["tk", "day"], how="left")
    sel = k.nrow.to_numpy(np.int64)
    M = SimpleNamespace(V=L["V"].astype(np.float64), N=L["N"].astype(np.float64), VW=L["VW"].astype(np.float64),
                        Cf=L["Cf"].astype(np.float64), H=L["H"].astype(np.float64), L=L["L"].astype(np.float64),
                        O=L["O"].astype(np.float64))
    for c in ("sig", "prev", "prevclose", "prevpost", "ndol", "typical", "mcap", "beta60", "results",
              "npm555", "npm570"):
        setattr(M, c, k[c].to_numpy(np.float64))
    M.open = L["open"].astype(np.float64)
    M.nvm, M.nnm, M.sig1m = (norms[n][sel].astype(np.float64) for n in ("nvm", "nnm", "sig1m"))
    M.sig1m = np.repeat(M.sig1m, 15, 1)[:, :390]
    M.pm = {c: L[c] for c in L if c.startswith("pm")}
    M.npm = {555: M.npm555, 570: M.npm570}
    M.pm570_last = M.pm["pm570_last"]
    M.sector, M.industry = k.sector.to_numpy(), k.industry.to_numpy()
    M.live = k.live.to_numpy()
    n = len(k)
    # forecasts: both stages, NaN where the run has no prediction for the day
    for st in ("930", "945"):
        f = k[["tk", "day"]].merge(ctx["fc"]["0" + st], on=["tk", "day"], how="left")
        setattr(M, f"Q{st}", f[D.QCOLS].to_numpy(np.float64))
        setattr(M, f"fc_prev{st}", f.fc_prev.to_numpy(np.float64))
    # SPY of each day
    spy = ctx["spy"]
    pos = np.searchsorted(spy["days"], k.day.to_numpy())
    pos_ok = (pos < len(spy["days"])) & (spy["days"][np.minimum(pos, len(spy["days"]) - 1)] == k.day.to_numpy())
    t0 = time.time()
    cols = {}
    cols.update(Fe.chunk0(M))
    cols.update(Fe.chunk1_minute(M))
    st = chunk3_inputs(M, k, ctx)
    cols.update(_chunk2_by_day(M, k, spy, pos, pos_ok))
    tm["minute"] = time.time() - t0
    # decisions and targets
    t0 = time.time()
    rng = np.random.default_rng(seed + int(month.replace("-", "")))
    off = rng.integers(0, every, n)
    r_, i_ = [], []
    for j in range(n):
        ii = np.arange(FIRST_DEC + off[j], LAST_DEC + 1, every)
        r_.append(np.full(len(ii), j))
        i_.append(ii)
    r_, i_ = np.concatenate(r_), np.concatenate(i_)
    Y = Tg.targets(M, r_, i_)
    per = Tg.persistence(M, r_, i_)
    dec = pd.DataFrame({"row": r_.astype(np.int32), "i": i_.astype(np.int16)})
    for c, v in Y.items():
        dec[c] = np.asarray(v, np.float32)
    first = np.searchsorted(r_, np.arange(n + 1))  # r_ is sorted: row j's decisions are first[j]:first[j+1]
    occ = {f"occ_{s}_b{b}": np.full(len(r_), np.nan, np.float32) for s in ("up", "dn") for b in range(1, Tg.NB + 1)}
    for c in ("bulge", "sag", "mass_up", "cm_up", "cm_dn", "net"):
        dec[f"pers_{c}"] = per[c].astype(np.float32)
    tm["targets"] = time.time() - t0
    # the S tier: one job per day (its 1 s file read once), the days spread over `workers` processes
    t0 = time.time()
    S_cols = {c: np.full((n, 390), np.nan, np.float32) for c in Fe.S_TIER}
    poll = np.zeros(n, np.int8)
    if seconds:
        days, tks = k.day.to_numpy(), k.tk.to_numpy()
        groups = [np.flatnonzero(days == day) for day in np.unique(days)]
        jobs = [(D.date_of(days[idx[0]]), tks[idx], M.prev[idx], M.sig[idx], M.typical[idx], M.sig1m[idx],
                 [i_[first[j]:first[j + 1]] for j in idx], [Y["p0"][first[j]:first[j + 1]] for j in idx])
                for idx in groups]
        for idx, (out, occ_d, ok) in zip(groups, _map(seconds_day, jobs, workers)):
            for c, v in out.items():
                S_cols[c][idx[ok]] = v
            poll[idx[ok]] = 1
            for j, o in zip(idx, occ_d):
                if o is not None:
                    for c, v in o.items():
                        occ[c][first[j]:first[j + 1]] = v
    for c, v in occ.items():
        dec[c] = v
    cols.update(S_cols)
    tm["seconds"] = time.time() - t0
    cols.update(time_columns(M, k))
    seq_cols = [c for c in Fe.NAMES if c in cols]
    seq = np.stack([np.asarray(cols[c], np.float32) for c in seq_cols], -1).astype(np.float16)
    rows = k[["tk", "day", "sid", "sector", "industry", "live", "cap_band", "delisted_later", "prior"]].copy()
    rows["poll_tier"] = poll
    for c, v in st.items():
        rows[c] = v
    rows["sig"], rows["prev"] = M.sig, M.prev
    meta = {"month": month, "rows": n, "decisions": len(dec), "seq_cols": seq_cols, "timings_s": tm,
            "s_tier_share": float(poll.mean()) if n else 0.0,
            "nan_share": {c: float(np.isnan(seq[..., j].astype(np.float32)).mean()) for j, c in enumerate(seq_cols)},
            "target_nan_share": {c: float(dec[c].isna().mean()) for c in dec.columns if c not in ("row", "i")}}
    return rows, seq, dec, meta


def second_columns(P, V, prev, sig, typical, sig1m):
    """The S-tier columns of one stock-day (seconds_rows on one row)."""
    out = seconds_rows(P[None], V[None], np.array([prev]), np.array([sig]), np.array([typical]), sig1m[None])
    return {c: v[0] for c, v in out.items()}


def seconds_rows(P, V, prev, sig, typical, sig1m):
    """The S-tier columns of several stock-days (rows x 23400 held prices and poll volumes): the profile
    and resistance (profile.day_columns, per row) and the zig-zag on the 1 s grid, all rows at once (its
    loop over the 23,400 steps is the costly part), read at each minute end. sig1m: rows x 390."""
    n = len(P)
    out = {c: np.full((n, 390), np.nan, np.float32) for c in Fe.S_TIER}
    for j in range(n):
        for c, v in Pf.day_columns(P[j], V[j], np.log(prev[j]), sig[j], typical[j]).items():
            out[c][j] = v
    if not n:
        return out
    with np.errstate(all="ignore"):
        lp = np.log(P)
    th = np.repeat(np.asarray(sig1m, np.float64), 60, 1)
    for kk in Fe.ZZ_K:
        for c, v in zigzag(lp, kk * th, np.asarray(sig, np.float64), at=Pf.ENDS - 1).items():
            out[f"{c}_k{kk}_s"] = v
    return out


def seconds_day(job):
    """One day's S tier: reads the day's 1 s bars for its stock-days once, grids them, and returns
    (S columns for the rows with seconds, per-row occupancy targets or None, the rows-with-seconds mask).
    A module-level function so a process pool can run it."""
    date, tks, prev, sig, typical, sig1m, dec_i, dec_p0 = job
    n = len(tks)
    ok = np.zeros(n, bool)
    P, V = np.full((n, Pf.SESSION), np.nan), np.zeros((n, Pf.SESSION))
    sec = D.read_seconds(date, set(tks))
    if not sec.empty:
        by = {s: g for s, g in sec.groupby("symbol", sort=False)}
        for j, tk in enumerate(tks):
            g = by.get(tk)
            if g is not None:
                P[j], V[j] = Pf.grid_day(g.sec.to_numpy(), g.price.to_numpy(), g.volume.to_numpy())
                ok[j] = True
    out = seconds_rows(P[ok], V[ok], prev[ok], sig[ok], typical[ok], sig1m[ok])
    occ = [Tg.occupancy(P[j], dec_i[j], dec_p0[j]) if ok[j] else None for j in range(n)]
    return out, occ, ok


def _map(f, jobs, workers):
    """f over jobs in order; in this process when workers <= 1, else in a spawn pool of `workers`."""
    if workers <= 1 or len(jobs) <= 1:
        yield from map(f, jobs)
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        yield from ex.map(f, jobs)


def _chunk2_by_day(M, k, spy, pos, pos_ok):
    """chunk2 per day (the cross-section is one day's stock-days)."""
    n = len(k)
    out = {c: np.full((n, 390), np.nan, np.float32) for c in
           ("sec_mv", "sec_mv_ew", "ind_mv", "mkt_mv", "resid", "breadth", "spy_mv", "spy_vwap_mv", "spy_rvol")}
    days = k.day.to_numpy()
    for day in np.unique(days):
        idx = np.flatnonzero(days == day)
        if not pos_ok[idx[0]]:
            continue
        p = pos[idx[0]]
        sub = SimpleNamespace(Cf=M.Cf[idx], prev=M.prev[idx], live=M.live[idx], sector=M.sector[idx],
                              industry=M.industry[idx], mcap=np.nan_to_num(M.mcap[idx]),
                              beta60=np.nan_to_num(M.beta60[idx], nan=1.0))
        prevc = spy["daily_close"][p - 1] if p else np.nan
        prevv = spy["vwap"][p - 1] if p else np.nan
        o = Fe.chunk2(sub, spy["Cf"][p], spy["V"][p], spy["nv"][p], prevc, prevv)
        for c, v in o.items():
            out[c][idx] = v
    return out


def chunk3_inputs(M, k, ctx):
    """chunk3 plus the market statics of the day (spy_gap with the dividend added back, spx_y,
    spx_gap)."""
    sd, sx = ctx["spy"]["daily"], ctx["spx"]
    day = k.day.to_numpy()
    sd = sd.assign(prevc=sd.c.shift(1))
    a = pd.DataFrame({"day": day}).merge(sd[["day", "o", "prevc"]], on="day", how="left")
    div = np.array([ctx["spy"]["div"].get(d, 0.0) for d in day])
    M.spy_gap = Fe.spy_gap(a.o.to_numpy(), a.prevc.to_numpy(), div)
    sx = sx.assign(prevc=sx.c.shift(1), prevc2=sx.c.shift(2))
    b = pd.DataFrame({"day": day}).merge(sx[["day", "o", "prevc", "prevc2"]], on="day", how="left")
    with np.errstate(all="ignore"):
        M.spx_y = (np.log(b.prevc / b.prevc2) * 100).to_numpy()
        M.spx_gap = (np.log(b.o / b.prevc) * 100).to_numpy()
    return Fe.chunk3(M)


def time_columns(M, k):
    """The known clock columns per minute: minute, to_close, dow, normal_rvol (the stock's normal share of
    the day's volume at each minute), pm_cut."""
    n = len(k)
    dow = (k.day.to_numpy() + 3) % 7  # epoch day 0 = Thursday; Monday = 0
    tot = np.nansum(M.nvm, 1, keepdims=True)
    with np.errstate(all="ignore"):
        share = M.nvm / np.where(tot > 0, tot, np.nan)
    return {"minute": np.broadcast_to(Fe.MINS, (n, 390)).astype(np.float32),
            "to_close": np.broadcast_to(389 - Fe.MINS, (n, 390)).astype(np.float32),
            "dow": np.repeat(dow[:, None], 390, 1).astype(np.float32),
            "normal_rvol": share.astype(np.float32),
            "pm_cut": np.broadcast_to(np.where(Fe.MINS < PM_CUT_SWITCH, 555, 570), (n, 390)).astype(np.float32)}


# ---------------------------------------------------------------- reference tables

def static_tables():
    """The tables read once for the whole build: tickers, SPY (with its previous close and the normal
    minute volume over 20 prior days), SPX, both daily forecasts, results days, shares, industries."""
    t = D.ticker_table()
    spy = D.spy()
    dd = spy["daily"].set_index("day")
    spy["daily_close"] = dd.c.reindex(spy["days"]).to_numpy()
    nv = np.full_like(spy["V"], np.nan)
    for j in range(N_NORM, len(spy["days"])):
        nv[j] = np.median(spy["V"][j - N_NORM:j], 0)
    spy["nv"] = nv
    return {"t": t, "common": D.common_tickers(t), "spy": spy, "spx": D.spx(),
            "fc": {st: D.forecasts(st) for st in ("0930", "0945")}, "res": D.results_days(),
            "shares": D.shares_table(), "ind": D.industry_codes(t), "gone": D.delist_days(t)}


def month_reference(win, month, ctx, live_n=D.LIVE_N):
    """The month's frame stock-days and their per-minute normals. win = [(summary, arrays), ...] of the
    WARM months before `month` and the month itself, from D.summarise. Returns R (the month's eligible
    top-live_n stock-days with every reference column; R.nrow indexes the normals) and normals()."""
    S = pd.concat([w[0] for w in win], ignore_index=True)
    A = {key: np.concatenate([w[1][key] for w in win]) for key in win[0][1]}
    S, A = reference(S, A, ctx["t"])
    S["beta60"] = beta60(S, ctx["spy"]["daily"])
    R = S[S.month.to_numpy() == month].copy()
    R["pos"] = np.flatnonzero(S.month.to_numpy() == month)
    R["common"] = R.tk.isin(ctx["common"])
    R = D.market_caps(R, ctx["shares"])
    R["cap_band"] = D.cap_band(R.mcap)
    R["industry"] = R.sid.map(ctx["ind"]).fillna("")
    R["sector"] = R.industry.str[:2]
    R = R.merge(ctx["res"], on=["tk", "day"], how="left")
    R["results"] = R.results.fillna(0.0)
    R["delisted_later"] = R.sid.map(ctx["gone"]).notna()
    R["elig"], R["live"] = eligible(R, live_n)
    R = R[R.elig & R.live].reset_index(drop=True)
    R["nrow"] = np.arange(len(R))
    return R, normals(S, A, R.pos.to_numpy())


# ---------------------------------------------------------------- main

def write_month(month, rows, seq, dec, meta, root):
    """meta.json is written last, so a month cut short has none and is built again."""
    out = root / "frames" / f"month={month}"
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "rows.parquet", index=False, compression="zstd")
    dec.to_parquet(out / "dec.parquet", index=False, compression="zstd")
    np.save(out / "seq.npy", seq)
    tmp = out / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(out / "meta.json")


def built(month, root, settings):
    """True when the month's frames exist from this FRAME_VERSION and the same settings."""
    f = root / "frames" / f"month={month}" / "meta.json"
    if not f.exists():
        return False
    m = json.loads(f.read_text())
    return m.get("frame_version") == FRAME_VERSION and m.get("settings") == settings


def check_report(meta):
    """The S2 one-month check: timings, NaN shares, S-tier coverage."""
    lines = [(f"month {meta['month']}: {meta['rows']} stock-days, {meta['decisions']} decisions, "
              f"S tier on {meta['s_tier_share'] * 100:.0f}% of stock-days"),
             "timings (s): " + ", ".join(f"{k} {v:.0f}" for k, v in meta["timings_s"].items())]
    worst = sorted(meta["nan_share"].items(), key=lambda x: -x[1])[:15]
    lines.append("highest input NaN shares: " + ", ".join(f"{k} {v:.2f}" for k, v in worst))
    tw = sorted(meta["target_nan_share"].items(), key=lambda x: -x[1])[:10]
    lines.append("highest target NaN shares: " + ", ".join(f"{k} {v:.2f}" for k, v in tw))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="intraday-ft-build", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="first", default="2023-01")
    p.add_argument("--to", dest="last", default="2026-10")
    p.add_argument("--check", action="store_true", help="build --from only and print the S2 check")
    p.add_argument("--no-seconds", action="store_true", help="skip the S tier (M tier only)")
    p.add_argument("--every", type=int, default=5, help="decision every N minutes per stock-day")
    p.add_argument("--stocks", type=int, default=D.LIVE_N, help="top N eligible stocks per day by prior "
                   "dollar volume (the live stand-in)")
    p.add_argument("--workers", type=int, default=8, help="processes for the S tier (one day per job)")
    p.add_argument("--force", action="store_true", help="rebuild months that are already built")
    p.add_argument("--status-dir", help="where status.txt goes (default --out); one per build when two run at once")
    p.add_argument("--out", default=str(D.CACHE))
    a = p.parse_args(argv)
    from pathlib import Path
    root = Path(a.out)
    settings = {"stocks": a.stocks, "every": a.every, "seconds": not a.no_seconds}
    ms = [a.first] if a.check else D.months(a.first, a.last)
    # a month built before (a check, or a run cut short later) counts: it is read for the window only
    skip = set() if a.force else {m for m in ms if built(m, root, settings)}
    status = D.Status(["tables", "months"], out=Path(a.status_dir) if a.status_dir else root)
    status.start("tables", 1)
    ctx = static_tables()
    status.step("tickers, SPY, SPX, forecasts, results, shares, industries")
    todo = D.months(pd.Period(ms[0]) - WARM, ms[-1])
    status.start("months", len(todo))
    win = []
    for m in todo:
        t0 = time.time()
        raw = D.read_month(m)
        s, arr = D.summarise(m, raw)
        s["month"] = m
        win = [*win[-WARM:], (s, arr)]
        if m not in ms or m in skip:
            what = "already built, window only" if m in skip else "warm-up only"
            status.step(f"{m}: {what} ({time.time() - t0:.0f} s)")
            continue
        R, norms = month_reference(win, m, ctx, a.stocks)
        tref = time.time() - t0
        rows, seq, dec, meta = build_month(m, ctx, R, norms, raw, seconds=not a.no_seconds, every=a.every,
                                          workers=a.workers)
        del raw
        meta["timings_s"] = {"read+reference": tref, **meta["timings_s"]}
        meta["frame_version"], meta["settings"] = FRAME_VERSION, settings
        write_month(m, rows, seq, dec, meta, root)
        status.step(f"{m}: {meta['rows']} stock-days, {sum(meta['timings_s'].values()) / 60:.1f} min")
        if a.check:
            print(check_report(meta))


if __name__ == "__main__":
    main()
