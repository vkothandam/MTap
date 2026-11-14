"""Read-only loaders for the intraday model (docs/intraday-features.md §2).

The NAS is only read; caches go under CACHE (<repo>/temp/intraday_ft, git-ignored; never /tmp, which
macOS clears on a restart: the 86 GB frame cache was lost that way on 2026-11-14). Lifted from
scripts_intraday_probe.py (et_minutes, read_month, summarise, load_month, day_stats, market_caps,
cap_band, forecasts, results_days, Status) with these changes:
- a stock is its stock id (composite_figi, else CIK), so sigma and the normals carry across a
  ticker change (FI -> FISV);
- an agg_1s day is every *.parquet in its folder (day-D plus day-D.extra);
- both daily forecast runs (09:30 and 09:45) are read;
- SPY and SPX come from out/market/.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

NAS = Path(os.environ.get("MTAP_NAS", "/Volumes/Backups/MTap-Data"))
BARS = NAS / "bars_1m"
AGG1S = NAS / "agg_1s"
MARKET = NAS / "out" / "market"
TICKERS = NAS / "out" / "tickers_cs"
SHARES = NAS / "out" / "shares"
FEED = NAS / "intraday_feed"
HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / "runs"
FORECAST = {"0930": RUNS / "20261029-211059Z-gbm-4438sym~px-cal20s",  # R5~px-cal20s, prices only
            "0945": RUNS / "20261029-211059Z-gbm-4438sym-cal20s"}     # R5-cal20s, full pre-market
PANEL = HERE.parent / "out" / "tft" / "exp5y-r1" / "panel.parquet"
TRANSIENT = HERE.parent / "temp"   # repo-local, git-ignored: expensive rebuildable data lives here
CACHE = Path(os.environ.get("MTAP_INTRADAY_FT_CACHE", str(TRANSIENT / "intraday_ft")))
EPOCH = pd.Timestamp("1970-01-01")
MINS = np.arange(390)
TAU = np.arange(1, 20) * 0.05
QCOLS = [f"q{round(t * 100):02d}" for t in TAU]
BUCKETS = 26  # 15-minute buckets of the session, for sigma_1m
CAPS = [("mega", 200e9), ("large", 10e9), ("mid", 2e9), ("small", 250e6), ("micro", 0)]
RESULT_DAYS = ("Earnings_Day", "Post_Earnings_Reaction")
MIN_DOLLAR_VOL, MIN_SIGMA, MIN_PRIOR = 2e7, 0.003, 3
LIVE_N = 1000  # training stand-in for the live set: the day's top eligible stocks by prior dollar volume

# The massive_1s universe rule (python/sourcing_py/massive_1s/prepare.py exclusion()), repeated here
# because ml/ does not import sourcing_py. Keep the two in step.
_NOT_EQUITY = (
    ("note", re.compile(r"\bnotes?\b|\bdue \d{4}\b|debenture|subordinated", re.IGNORECASE)),
    ("preferred", re.compile(r"preferred|depositary|\bpfd\b", re.IGNORECASE)),
    ("fund", re.compile(r"\bfund\b", re.IGNORECASE)),
    ("rights", re.compile(r"\brights?\b", re.IGNORECASE)),
)


def exclusion(name: str | None) -> str | None:
    for reason, pat in _NOT_EQUITY:
        if pat.search(name or ""):
            return reason
    return None


# ---------------------------------------------------------------- status

class Status:
    """status.txt after every unit of work: `[step k/n · name] done x/y · elapsed · ETA hh:mm`."""

    def __init__(self, steps, out=CACHE):
        self.t0, self.steps, self.out = time.time(), list(steps), Path(out)

    def start(self, step, total):
        self.step_name, self.total, self.done, self.tp = step, max(total, 1), 0, time.time()
        self.write("started")

    def step(self, msg=""):
        self.done += 1
        self.write(msg)

    def write(self, msg):
        el = time.time() - self.tp
        eta = el / self.done * (self.total - self.done) if self.done else float("nan")
        end = time.strftime("%H:%M", time.localtime(time.time() + eta)) if np.isfinite(eta) else "?"
        k = self.steps.index(self.step_name) + 1
        line = (f"[step {k}/{len(self.steps)} · {self.step_name}] done {self.done}/{self.total} · "
                f"elapsed {el / 60:.0f} min (run {(time.time() - self.t0) / 60:.0f}) · ETA {end} · {msg}")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "status.txt").write_text(line + "\n")
        print(line, flush=True)


# ---------------------------------------------------------------- clock

def et_minutes(t_ms):
    """Epoch minutes on the New York clock, DST included (one offset per UTC date)."""
    m = np.asarray(t_ms) // 60000
    day = m // 1440
    u = np.unique(day)
    noon = pd.DatetimeIndex(pd.to_datetime(u * 1440 + 720, unit="m", utc=True))
    off = (noon.tz_convert("America/New_York").tz_localize(None) - noon.tz_localize(None)) // pd.Timedelta(minutes=1)
    return m + np.asarray(off)[np.searchsorted(u, day)]


def et_seconds(t_ms):
    """Seconds since midnight on the New York clock."""
    t_ms = np.asarray(t_ms)
    return (et_minutes(t_ms) % 1440) * 60 + (t_ms // 1000) % 60


def months(first, last):
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def day_of(date) -> int:
    return int((pd.Timestamp(date) - EPOCH).days)


def date_of(day: int) -> str:
    return (EPOCH + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- stock identity and universe

def ticker_table() -> pd.DataFrame:
    """Every tickers_cs row ever written, one per (ticker, composite_figi, cik)."""
    rows = []
    for f in sorted(glob.glob(str(TICKERS / "tickers_cs-*.jsonl"))):
        with open(f) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    t = pd.DataFrame(rows)
    for c in ("composite_figi", "cik", "delisted_utc", "name", "type"):
        if c not in t:
            t[c] = None
    t["delisted"] = pd.to_datetime(t.delisted_utc, utc=True, errors="coerce").dt.tz_localize(None)
    t["active"] = t.active.fillna(False).astype(bool)
    return t.drop_duplicates(["ticker", "composite_figi", "cik", "active", "delisted_utc"])


def common_tickers(t: pd.DataFrame) -> set[str]:
    """Tickers typed CS in some tickers_cs row, with a name that passes exclusion(). ADRs are typed
    ADRC, so they are not in it."""
    cs = t[t.type.isna() | (t.type == "CS")]
    return set(cs.ticker[[exclusion(n) is None for n in cs.name]])


def stock_ids(tk, day, t: pd.DataFrame) -> np.ndarray:
    """The stock id of ticker tk on epoch day `day`: composite_figi, else CIK, else the ticker.

    A recycled ticker has several rows; the one in force on the day is the row delisted soonest on
    or after it, else the active row, else the last delisted one."""
    tk, day = np.asarray(tk, object), np.asarray(day)
    t = t.assign(sid=_sid(t), dday=np.where(t.delisted.notna(), (t.delisted - EPOCH).dt.days,
                                            np.iinfo(np.int32).max))
    t = t.drop_duplicates(["ticker", "sid", "dday"]).sort_values(["ticker", "dday"])
    one = t.groupby("ticker").sid.nunique() == 1
    out = pd.Series(tk).map(t[t.ticker.isin(one.index[one])].drop_duplicates("ticker").set_index("ticker").sid)
    out = out.to_numpy(object)
    by = {k: g[["dday", "sid"]].to_numpy() for k, g in t[t.ticker.isin(one.index[~one])].groupby("ticker")}
    for i in np.flatnonzero(pd.isna(out)):  # recycled tickers only
        g = by.get(tk[i])
        if g is None:
            out[i] = "tk:" + tk[i]
            continue
        later = g[g[:, 0] >= day[i]]
        out[i] = later[0, 1] if len(later) else g[-1, 1]
    return out


def _sid(t: pd.DataFrame) -> np.ndarray:
    return np.where(t.composite_figi.notna() & (t.composite_figi != ""), t.composite_figi,
                    np.where(t.cik.notna() & (t.cik != ""), "cik:" + t.cik.astype(str), "tk:" + t.ticker))


def delist_days(t: pd.DataFrame) -> dict:
    """stock id -> the epoch day it was delisted (only for ids with no active row): a report slice, never
    an input."""
    t = t.assign(sid=_sid(t))
    act = set(t.sid[t.active])
    gone = t[~t.sid.isin(act) & t.delisted.notna()]
    return ((gone.groupby("sid").delisted.max() - EPOCH).dt.days).to_dict()


def live_symbols() -> set[str]:
    """The symbols of the newest E*TRADE watchlist day (report slice `live_set`)."""
    days = sorted(glob.glob(str(FEED / "date=*" / "source=etrade_watchlist")))
    if not days:
        return set()
    t = ds.dataset(days[-1]).to_table(columns=["symbol"])
    return set(pc.unique(t["symbol"]).to_pylist())


# ---------------------------------------------------------------- minute bars

def read_month(month):
    """The month's minute bars. The file in a month=YYYY-MM folder is named after its quarter
    (bars-2026-01-01_2026-03-31) but holds that month only (checked 2026-11-07); the rows are still cut
    to the month's ET days as a guard, so a stock-day can never come from two folders."""
    files = sorted(glob.glob(str(BARS / f"month={month}" / "*.parquet")))
    d0, d1 = day_of(f"{month}-01"), day_of(pd.Period(month, "M").end_time.normalize()) + 1
    near = (pc.field("t") >= (d0 - 1) * 86_400_000) & (pc.field("t") < (d1 + 1) * 86_400_000)
    tb = ds.dataset(files).to_table(columns=["ticker", "t", "o", "h", "l", "c", "v", "vw", "n"], filter=near)
    et = et_minutes(tb["t"].to_numpy())
    inside = (et // 1440 >= d0) & (et // 1440 < d1)
    tb, et = tb.filter(pa.array(inside)), et[inside]
    tk = pc.dictionary_encode(tb["ticker"].combine_chunks())
    a = {k: tb[k].combine_chunks().cast(pa.float64()).fill_null(np.nan).to_numpy().astype(np.float32)
         for k in ("o", "h", "l", "c", "v", "vw", "n")}
    names = np.array(tk.dictionary.to_pylist(), dtype=object)
    return names, tk.indices.to_numpy().astype(np.int64), et, a


def summarise(month, raw=None):
    """Per stock-day: the regular-session summary (dv, dpv, n, open, close, hi, lo, vwap, post = the last
    trade 16:00-20:00), and per-minute arrays for the normals: V and N (390), the sum of squared 1-minute
    VWAP log changes and their count per 15-minute bucket (26), and the pre-market volume to 09:15 / 09:30.
    raw = read_month(month) when the caller already has it (one NAS read per month)."""
    names, code, et, a = raw or read_month(month)
    mod = et % 1440
    key = code * 100000 + et // 1440
    reg = (mod >= 570) & (mod < 960)
    u, inv = np.unique(key[reg], return_inverse=True)
    mi = mod[reg] - 570

    def dense(x, fill):
        D = np.full((len(u), 390), fill, np.float32)
        D[inv, mi] = x[reg]
        return D
    V, N, VW, C = dense(a["v"], 0), dense(a["n"], 0), dense(a["vw"], np.nan), dense(a["c"], np.nan)
    O, H, L = dense(a["o"], np.nan), dense(a["h"], np.nan), dense(a["l"], np.nan)
    with np.errstate(all="ignore"):
        dv, dpv = V.sum(1, dtype=np.float64), np.nansum(np.where(V > 0, V * VW, 0), 1, dtype=np.float64)
        has = ~np.isnan(C)
        first, last = has.argmax(1), 389 - has[:, ::-1].argmax(1)
        r = np.arange(len(u))
        s = pd.DataFrame({"tk": names[u // 100000], "day": u % 100000, "dv": dv, "dpv": dpv,
                          "n": N.sum(1, dtype=np.float64), "open": O[r, first], "close": C[r, last],
                          "hi": np.nanmax(H, 1), "lo": np.nanmin(L, 1)})
        s["vwap"] = s.dpv / s.dv.where(s.dv > 0)
        idx = np.maximum.accumulate(np.where(~np.isnan(VW), MINS, 0), 1)
        VWf = np.take_along_axis(VW, idx, 1)
        lr = np.diff(np.log(VWf), axis=1)
        ok = np.isfinite(lr) & (V[:, 1:] > 0)
        b = (MINS[1:] // 15)
        ss = np.zeros((len(u), BUCKETS), np.float32)
        cnt = np.zeros((len(u), BUCKETS), np.float32)
        for k in range(BUCKETS):
            m = b == k
            ss[:, k] = np.where(ok[:, m], lr[:, m] ** 2, 0).sum(1)
            cnt[:, k] = ok[:, m].sum(1)
    post = (mod >= 960) & (mod < 1200) & np.isin(key, u)
    pq_ = pd.DataFrame({"pos": np.searchsorted(u, key[post]), "mod": mod[post], "c": a["c"][post]})
    pl = pq_.sort_values(["pos", "mod"]).groupby("pos").c.last()
    s["post"] = np.nan
    s.loc[pl.index.values, "post"] = pl.values
    pm = (mod >= 240) & (mod < 570) & np.isin(key, u)
    pmv = {}
    for cut in (555, 570):
        m = pm & (mod < cut)
        pmv[cut] = np.bincount(np.searchsorted(u, key[m]), weights=a["v"][m], minlength=len(u))
    arrays = {"V": V, "N": N, "ss": ss, "cnt": cnt, "pmv555": pmv[555].astype(np.float32),
              "pmv570": pmv[570].astype(np.float32)}
    return s, arrays


def load_month(month, keep, raw=None):
    """Dense stock-day x 390 arrays for the (ticker, day) pairs in `keep`, plus pre-market aggregates up
    to 09:15 and 09:30 (v, pv, last, hi, lo). Missing minutes: volume 0, prices = the last close carried
    forward (NaN before the first trade); O stays NaN where the minute had no trade."""
    names, code, et, a = raw or read_month(month)
    mod = et % 1440
    key = code * 100000 + et // 1440
    ukey = np.unique(key)
    kk = pd.MultiIndex.from_arrays([names[ukey // 100000], ukey % 100000])
    want = ukey[kk.isin(pd.MultiIndex.from_tuples(list(keep)))] if keep else ukey[:0]
    reg = (mod >= 570) & (mod < 960) & np.isin(key, want)
    u, inv = np.unique(key[reg], return_inverse=True)
    mi = mod[reg] - 570

    def dense(x, fill):
        D = np.full((len(u), 390), fill, np.float32)
        D[inv, mi] = x[reg]
        return D
    V, N = dense(a["v"], 0), dense(a["n"], 0)
    C, VW, H, Lo, Op = (dense(a[k], np.nan) for k in ("c", "vw", "h", "l", "o"))
    idx = np.where(~np.isnan(C), MINS, 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    Cf = np.take_along_axis(C, idx, 1)
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
    return dict(tk=names[u // 100000], day=u % 100000, V=V, N=N, VW=np.where(np.isnan(VW), Cf, VW), Cf=Cf,
                H=np.where(np.isnan(H), Cf, H), L=np.where(np.isnan(Lo), Cf, Lo), O=Op,
                open=Op[np.arange(len(u)), has.argmax(1)], **PM)


# ---------------------------------------------------------------- one second tier

def second_files(date: str) -> list[str]:
    """Every file of an agg_1s massive_rest day: day-D.parquet and day-D.extra.parquet."""
    return sorted(glob.glob(str(AGG1S / f"date={date}" / "source=massive_rest" / "*.parquet")))


def read_seconds(date: str, symbols) -> pd.DataFrame:
    """The day's 1 s bars for `symbols`: symbol, sec (New York seconds since midnight), price, volume."""
    files = second_files(date)
    if not files:
        return pd.DataFrame({"symbol": [], "sec": [], "price": [], "volume": []})
    tb = ds.dataset(files).to_table(columns=["symbol", "ts_utc", "price", "volume_i"],
                                    filter=pc.field("symbol").isin(list(symbols)))
    t = tb["ts_utc"].combine_chunks().cast(pa.int64()).to_numpy()
    return pd.DataFrame({"symbol": tb["symbol"].to_numpy(zero_copy_only=False), "sec": et_seconds(t),
                         "price": tb["price"].to_numpy(), "volume": tb["volume_i"].to_numpy().astype(np.float64)})


# ---------------------------------------------------------------- reference data

def industry_codes(t: pd.DataFrame) -> dict:
    """stock id -> the TRBC industry code (6 digits) last seen in the panel. Codes that aren't digits
    ("unknown") are dropped."""
    p = pd.read_parquet(PANEL, columns=["symbol", "date", "industry_code"])
    p = p[p.industry_code.astype(str).str.fullmatch(r"\d{6}").fillna(False)]
    last = p.sort_values("date").groupby("symbol").agg(date=("date", "last"), code=("industry_code", "last"))
    sid = stock_ids(last.index.values, [day_of(d) for d in last.date], t)
    return dict(zip(sid, last.code.astype(str)))


def shares_table() -> pd.DataFrame:
    """tk, key (epoch day of the quarter-end snapshot), shares (weighted, else that share class's)."""
    rows = []
    for f in sorted(glob.glob(str(SHARES / "shares-*.jsonl"))):
        x = pd.read_json(f, lines=True)
        rows.append(x[["symbol", "snap_date", "weighted_shares_outstanding", "share_class_shares_outstanding"]])
    sh = pd.concat(rows)
    sh["key"] = (pd.to_datetime(sh.snap_date) - EPOCH).dt.days
    sh["shares"] = sh.weighted_shares_outstanding.where(sh.weighted_shares_outstanding > 0,
                                                        sh.share_class_shares_outstanding)
    return sh[sh.shares > 0].rename(columns={"symbol": "tk"}).sort_values("key")[["tk", "key", "shares"]]


def market_caps(d, sh=None):
    """Market cap as of the previous close = shares outstanding from the latest quarter-end snapshot
    before the day x the previous close. sh = shares_table() when the caller already has it."""
    sh = shares_table() if sh is None else sh
    q = d[["tk", "day", "prevclose"]].copy()
    q["key"] = q.day - 1
    q = pd.merge_asof(q.sort_values("key"), sh[["tk", "key", "shares"]], on="key", by="tk", direction="backward")
    q["mcap"] = q.shares * q.prevclose
    return d.merge(q[["tk", "day", "mcap"]], on=["tk", "day"], how="left")


def cap_band(mcap):
    out = np.full(len(mcap), "", object)
    m = np.nan_to_num(np.asarray(mcap, np.float64), nan=-1)
    for name, lo in CAPS[::-1]:
        out[m >= lo] = name
    out[m < 0] = "unknown"
    return out


def forecasts(stage: str) -> pd.DataFrame:
    """The walk-forward test predictions of one daily run: tk, day, q05..q95 (price levels), fc_prev."""
    fs = sorted(glob.glob(str(FORECAST[stage] / "fold_*" / "test_predictions.parquet")))
    if not fs:
        return pd.DataFrame(columns=["tk", "day", *QCOLS, "fc_prev"])
    f = pd.concat([pd.read_parquet(x, columns=["symbol", "date", "prev_vwap", *QCOLS]) for x in fs])
    f["day"] = (pd.to_datetime(f.date) - EPOCH).dt.days
    f = f.rename(columns={"symbol": "tk", "prev_vwap": "fc_prev"}).drop(columns="date")
    return f.drop_duplicates(["tk", "day"], keep="last")


def results_days() -> pd.DataFrame:
    p = pd.read_parquet(PANEL, columns=["symbol", "date", "results_window_pit"])
    p["day"] = (pd.to_datetime(p.date) - EPOCH).dt.days
    p["results"] = p.results_window_pit.isin(RESULT_DAYS).astype(np.float32)
    return p.rename(columns={"symbol": "tk"})[["tk", "day", "results"]]


def spy() -> dict:
    """SPY daily (day, o, c, vw), the minute bars as dense day x 390 arrays (Cf, V), the regular-session
    VWAP per minute-bar day (vwap, the same rule as the stocks' prev_vwap), and the dividends."""
    d = pd.read_parquet(MARKET / "spy_daily.parquet")
    d["day"] = (pd.to_datetime(d.date) - EPOCH).dt.days
    m = pd.read_parquet(MARKET / "spy_1m.parquet")
    et = et_minutes(m.t.to_numpy())
    mod, day = et % 1440, et // 1440
    reg = (mod >= 570) & (mod < 960)
    days = np.unique(day[reg])
    r, c = np.searchsorted(days, day[reg]), mod[reg] - 570
    C = np.full((len(days), 390), np.nan)
    V = np.zeros((len(days), 390))
    C[r, c], V[r, c] = m.c.to_numpy()[reg], m.v.to_numpy()[reg]
    pv = np.bincount(r, weights=(m.v * m.vw).to_numpy()[reg], minlength=len(days))
    idx = np.maximum.accumulate(np.where(~np.isnan(C), MINS, 0), 1)
    div = pd.read_parquet(MARKET / "spy_dividends.parquet")
    div["day"] = (pd.to_datetime(div.ex_date) - EPOCH).dt.days
    return {"daily": d.sort_values("day").reset_index(drop=True), "days": days,
            "Cf": np.take_along_axis(C, idx, 1), "V": V, "vwap": pv / np.maximum(V.sum(1), 1), "div": div.groupby("day").cash.sum().to_dict()}


def spx() -> pd.DataFrame:
    s = pd.read_parquet(MARKET / "spx_daily.parquet")
    s["day"] = (pd.to_datetime(s.date) - EPOCH).dt.days
    return s.sort_values("day").reset_index(drop=True)
