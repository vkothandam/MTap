"""The morning daily forecast for each (symbol, date), from a finished tft_vwap run.

The daily model forecasts VWAP(D) from data up to D-1, so its forecast for D is known before D
opens. Only one fold is read (IntradayConfig.daily_fold): its test window has to contain D,
otherwise the forecast was made by a model trained on D itself.

    dq_<q>    ln(q / VWAP(D-1)): the forecast quantile as a log move from yesterday's VWAP
    d_med, d_width, d_skew, d_p_up
              median, P10-P90 width, skew of the band, share of the band above yesterday's VWAP
    hist_mae, hist_cov80
              the daily model's own record on this symbol over the 20 sessions before D
    sec_d_med the mean d_med of the other symbols in the sector that day (attached per panel)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import QUANTILES

QCOLS = [f"q{round(q * 100):02d}" for q in QUANTILES]
DQCOLS = [f"dq{c[1:]}" for c in QCOLS]
HIST_SESSIONS = 20


def load_daily(run: Path, fold: int) -> pd.DataFrame:
    """One row per (symbol, date): dq_* and the daily features (no sector column yet)."""
    f = Path(run) / f"fold_{fold}" / "test_predictions.parquet"
    d = pd.read_parquet(f, columns=["symbol", "date", *QCOLS, "prev_vwap", "actual"])
    d["date"] = d["date"].astype(str)
    # series-break suffixes (A~1 is an earlier security under ticker A): the latest part wins
    d["base"] = d["symbol"].str.split("~").str[0]
    d["older"] = d["symbol"].str.contains("~", regex=False)
    d = (d.sort_values(["base", "date", "older"]).drop_duplicates(["base", "date"])
         .drop(columns=["symbol", "older"]).rename(columns={"base": "symbol"}))
    d = d[(d["prev_vwap"] > 0) & (d[QCOLS] > 0).all(axis=1)]
    dq = np.log(d[QCOLS].to_numpy(float) / d["prev_vwap"].to_numpy(float)[:, None])
    dq = np.sort(dq, axis=1)
    out = pd.DataFrame(dq, columns=DQCOLS, index=d.index)
    out.insert(0, "symbol", d["symbol"].to_numpy())
    out.insert(1, "date", d["date"].to_numpy())
    p10, p50, p90 = dq[:, 1], dq[:, 9], dq[:, 17]
    width = p90 - p10
    out["has_daily"] = 1.0
    out["d_med"] = p50
    out["d_width"] = width
    out["d_skew"] = np.where(width > 0, (p90 + p10 - 2 * p50) / np.maximum(width, 1e-12), 0.0)
    out["d_p_up"] = (dq > 0).mean(axis=1)
    # the model's record before D: rows dated before D only (shift 1), the actual of D-1 being
    # VWAP(D-1), known at D's open
    act = np.log(d["actual"].to_numpy(float) / d["prev_vwap"].to_numpy(float))
    out["_err"] = np.abs(act - p50)
    out["_in"] = ((act >= p10) & (act <= p90)).astype(float)
    out.loc[~np.isfinite(act), ["_err", "_in"]] = np.nan
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = out.groupby("symbol", sort=False)
    for src, dst in (("_err", "hist_mae"), ("_in", "hist_cov80")):
        out[dst] = g[src].transform(
            lambda s: s.shift(1).rolling(HIST_SESSIONS, min_periods=1).mean())
    return out.drop(columns=["_err", "_in"])


def attach(df: pd.DataFrame, daily: pd.DataFrame | None) -> pd.DataFrame:
    """Add the daily columns (DAILY_REALS and dq_*) to panel rows; 0 where there is no
    forecast. `sec_d_med` is the mean d_med of the other symbols in the row's sector that day."""
    keys = df[["symbol", "date", "sector_code"]].drop_duplicates(["symbol", "date"])
    if daily is None or daily.empty:
        k = keys.assign(**{c: 0.0 for c in ["has_daily", "d_med", "d_width", "d_skew", "d_p_up",
                                            "hist_mae", "hist_cov80", "sec_d_med", *DQCOLS]})
    else:
        k = keys.merge(daily, on=["symbol", "date"], how="left")
        k["has_daily"] = k["has_daily"].fillna(0.0)
        k = k.fillna({c: 0.0 for c in k.columns if c not in ("symbol", "date", "sector_code")})
        g = k.groupby(["date", "sector_code"], dropna=False)
        s = g["d_med"].transform("sum") - k["d_med"]
        n = g["has_daily"].transform("sum") - k["has_daily"]
        k["sec_d_med"] = np.where(n > 0, s / n.clip(lower=1), 0.0)
    k = k.drop(columns="sector_code")
    cols = [c for c in k.columns if c not in ("symbol", "date")]
    k[cols] = k[cols].astype("float32")
    return df.merge(k, on=["symbol", "date"], how="left", validate="many_to_one")
