"""intraday-ft-extremes: the realised prices after each decision of a scored run (no training).

For every run-file row (stock, day, minute i) and duration bin bK (targets.BINS: the bin's minutes after
the entry, entry = minute i+1), from the cached frames' minutes:

    hi_bK = ln(max high / P0)     lo_bK = ln(min low / P0)       (P0 = real_p0, the entry price)
    t_hi_bK / t_lo_bK = the minute of that high / low, counted from the decision (the entry minute = 1)
    vw_bK / tw_bK = ln(VWAP / P0), ln(TWAP / P0) over the bin's minutes (VWAP: the minute VWAPs
                    weighted by volume; TWAP: the mean of the minute closes, a quiet minute holding
                    the last price)
    cvw_bK / ctw_bK = the same from the entry to the bin's end (cvw_b4 = the first 60 minutes)
    cl_bK = ln(the last price at the bin's end / P0)

The frames hold the prices as ln(x / prev) / sig in float16, clipped to +-10 sig from the previous
close: a price is good to ~0.01%, and a minute at the clip (the stock more than 10 sig from yesterday's
close, e.g. after a results gap) is not a price, so a window holding one has NaN. The volume comes back
from m_rvol x normal_rvol (a share of the normal day, so only its ratios count): the VWAPs match the
run file's real_net_bK to ~0.01 bp (median), 3 bp at the 99th percentile. A window without a trade has
NaN too; t = -1 for both. Writes fold_<k>/extremes.parquet (dec_row + the 45 columns); the dashboard's
Intraday -> Pockets -> Moves tab reads it.

    intraday-ft-extremes runs/intraday_ft/<run> [--cache DIR]   (default: the run's config.json cache)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import features as Fe
from . import targets as Tg
from . import windows as W

PRED = "intraday_ft_pred.parquet"
OUT = "extremes.parquet"


def excursions(H, L, r, i, lp0, bad=None):
    """Per bin: the max high / min low (log vs the entry) of rows r after decision minutes i, and their
    minute from the decision. H, L: (rows, 390) prices (NaN = no trade); lp0 = ln P0 per decision;
    bad: (rows, 390) minutes whose price is unknown (the bin is NaN if it holds one)."""
    r, i = np.asarray(r), np.asarray(i)
    out = {}
    for k, (lo, hi) in enumerate(Tg.BINS, 1):
        cols, valid, _ = Tg.bin_columns(i, lo, hi)
        unknown = (bad[r[:, None], cols] & valid).any(1) if bad is not None else np.zeros(len(r), bool)
        for name, X, pick in (("hi", H, np.nanargmax), ("lo", L, np.nanargmin)):
            x = np.where(valid, X[r[:, None], cols], np.nan)
            ok = np.isfinite(x).any(1) & ~unknown
            j = np.zeros(len(x), np.int64)
            if ok.any():
                j[ok] = pick(x[ok], 1)
            v = x[np.arange(len(x)), j]
            out[f"{name}_b{k}"] = np.where(ok, np.log(v) - lp0, np.nan).astype(np.float32)
            out[f"t_{name}_b{k}"] = np.where(ok, cols[np.arange(len(x)), j] - i, -1).astype(np.int16)
    return out


def averages(VW, Wt, C, r, i, lp0, bad=None):
    """Per bin, and from the entry to each bin's end: the VWAP and TWAP after decision minutes i (log vs
    the entry) and the last price at the bin's end. VW: (rows, 390) minute VWAPs, Wt: their volumes (any
    scale; 0 = no trade), C: held minute closes (NaN before the first trade); bad as in excursions."""
    r, i = np.asarray(r), np.asarray(i)
    out = {}
    for k, (lo, hi) in enumerate(Tg.BINS, 1):
        for pre, a in (("", lo), ("c", 0)):
            cols, valid, _ = Tg.bin_columns(i, a, hi)
            unknown = (bad[r[:, None], cols] & valid).any(1) if bad is not None else np.zeros(len(r), bool)
            x = np.where(valid, VW[r[:, None], cols], np.nan)
            w = np.where(valid & np.isfinite(x), Wt[r[:, None], cols], 0.0)
            c = np.where(valid, C[r[:, None], cols], np.nan)
            with np.errstate(all="ignore"):
                vw = np.log(np.nansum(w * x, 1) / w.sum(1)) - lp0
                tw = np.log(np.nanmean(c, 1)) - lp0
            out[f"{pre}vw_b{k}"] = np.where(unknown, np.nan, vw).astype(np.float32)
            out[f"{pre}tw_b{k}"] = np.where(unknown, np.nan, tw).astype(np.float32)
            if not pre:
                last = np.where(valid.any(1), cols[np.arange(len(r)), valid.sum(1) - 1], 0)
                with np.errstate(all="ignore"):
                    cl = np.log(C[r, last]) - lp0
                out[f"cl_b{k}"] = np.where(unknown | ~valid.any(1), np.nan, cl).astype(np.float32)
    return out


def month_prices(mo: W.Month, rows):
    """Minute highs, lows, VWAPs and held closes in price units for frame rows `rows` of an open month,
    the minute volumes (as a share of the normal day) and the minutes at the frames' clip (price
    unknown)."""
    rr = mo.rows.iloc[rows]
    ref, sig = np.log(rr.prev.to_numpy(np.float64)), rr.sig.to_numpy(np.float64)
    seq = np.asarray(mo.seq[np.asarray(rows)], np.float32)

    def col(c):
        return seq[..., mo.seq_cols.index(c)].astype(np.float64)

    def price(c):
        return np.exp(col(c) * sig[:, None] + ref[:, None])

    h, lw = col("m_h"), col("m_l")
    bad = (np.abs(h) >= Fe.CLIP - 1e-3) | (np.abs(lw) >= Fe.CLIP - 1e-3)
    vol = np.nan_to_num(np.exp(col("m_rvol")) * col("normal_rvol"))
    return {"H": price("m_h"), "L": price("m_l"), "VW": price("m_vw"), "C": price("m_c"), "V": vol, "bad": bad}


def fold_extremes(f: pd.DataFrame, root) -> pd.DataFrame:
    """extremes.parquet for a run file `f` (date, minute, stock_id, symbol, dec_row, real_p0)."""
    day = (pd.to_datetime(f["date"].astype(str)) - D.EPOCH).dt.days.to_numpy()
    month = f["date"].astype(str).str[:7].to_numpy()
    parts = []
    for m in sorted(set(month)):
        g = np.flatnonzero(month == m)
        mo = W.open_month(m, root)
        # (stock, symbol, day): a when-issued line (APTVw) shares its company's stock_id
        key = pd.MultiIndex.from_arrays([mo.rows.sid.to_numpy(), mo.rows.tk.to_numpy(), mo.rows.day.to_numpy()])
        fr = key.get_indexer(pd.MultiIndex.from_arrays([f["stock_id"].to_numpy()[g], f["symbol"].to_numpy()[g],
                                                        day[g]]))
        if (fr < 0).any():
            raise ValueError(f"{m}: {int((fr < 0).sum())} run-file rows not in the frames under {root}")
        uniq, r = np.unique(fr, return_inverse=True)
        m_ = month_prices(mo, uniq)
        lp0 = np.log(f["real_p0"].to_numpy(np.float64)[g])
        i = f["minute"].to_numpy()[g]
        ex = excursions(m_["H"], m_["L"], r, i, lp0, m_["bad"])
        av = averages(m_["VW"], m_["V"], m_["C"], r, i, lp0, m_["bad"])
        parts.append(pd.DataFrame({"dec_row": f["dec_row"].to_numpy()[g], **ex, **av}))
        print(f"  {m}: {len(g):,} decisions", flush=True)
    return pd.concat(parts, ignore_index=True).sort_values("dec_row", ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--cache", help="the frame cache the run was trained on (default: config.json cache)")
    a = ap.parse_args(argv)
    cfg = Path(a.run_dir) / "config.json"
    root = a.cache or (json.loads(cfg.read_text()).get("cache") if cfg.is_file() else None) or D.CACHE
    for p in sorted(Path(a.run_dir).glob(f"fold_*/{PRED}")):
        print(f"{p.parent.name}", flush=True)
        f = pd.read_parquet(p, columns=["date", "minute", "stock_id", "symbol", "dec_row", "real_p0"])
        fold_extremes(f, root).to_parquet(p.parent / OUT, index=False, compression="zstd")
        print(f"  wrote {p.parent / OUT}", flush=True)


if __name__ == "__main__":
    main()
