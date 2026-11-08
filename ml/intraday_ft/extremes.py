"""intraday-ft-extremes: the realised high and low after each decision of a scored run (no training).

For every run-file row (stock, day, minute i) and duration bin bK (targets.BINS: the bin's minutes after
the entry, entry = minute i+1), from the cached frames' minute highs / lows:

    hi_bK = ln(max high / P0)     lo_bK = ln(min low / P0)       (P0 = real_p0, the entry price)
    t_hi_bK / t_lo_bK = the minute of that high / low, counted from the decision (the entry minute = 1)

The frames hold m_h / m_l = ln(x / prev) / sig in float16, clipped to +-10 sig from the previous close:
a price is good to ~0.01%, and a minute at the clip (the stock more than 10 sig from yesterday's close,
e.g. after a results gap) is not a price, so a bin holding one has NaN. A bin without a trade has NaN
too; t = -1 for both. Writes fold_<k>/extremes.parquet (dec_row + the 20 columns); the dashboard's
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


def month_prices(mo: W.Month, rows):
    """Minute highs and lows in price units for frame rows `rows` of an open month, and the minutes
    at the frames' clip (price unknown)."""
    rr = mo.rows.iloc[rows]
    ref, sig = np.log(rr.prev.to_numpy(np.float64)), rr.sig.to_numpy(np.float64)
    seq = np.asarray(mo.seq[np.asarray(rows)], np.float32)
    h = seq[..., mo.seq_cols.index("m_h")].astype(np.float64)
    lw = seq[..., mo.seq_cols.index("m_l")].astype(np.float64)
    bad = (np.abs(h) >= Fe.CLIP - 1e-3) | (np.abs(lw) >= Fe.CLIP - 1e-3)
    return np.exp(h * sig[:, None] + ref[:, None]), np.exp(lw * sig[:, None] + ref[:, None]), bad


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
        H, L, bad = month_prices(mo, uniq)
        lp0 = np.log(f["real_p0"].to_numpy(np.float64)[g])
        ex = excursions(H, L, r, f["minute"].to_numpy()[g], lp0, bad)
        parts.append(pd.DataFrame({"dec_row": f["dec_row"].to_numpy()[g], **ex}))
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
