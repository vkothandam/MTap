"""Panel -> screener frames. The one rule: day d's high and low appear only in the target.

`history_features` holds what is known at the close of session t; a target day d takes the
features of its previous session (d-1), plus its own open, pre-market and TFT forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ("px", "full")
PX_FEATURES = ["pm_last_pct_prev_vwap"]
FULL_FEATURES = [*PX_FEATURES, "pm_vwap_pct_prev_vwap", "pm_vol_pct_10d", "pm_range_pct"]
COMMON_FEATURES = [
    "gap_prev_close", "gap_prev_vwap", "tft_width90", "tft_width50", "tft_median",
    "prev_range", "prev_range_open", "atr14_pct", "log_dollar_volume", "mkt_ret", "sector_ret",
    "days_to_results", "days_since_results", "earn_near", "day_of_week",
]
EARN_FAR = 63
PM_CLIP = 0.3
QUANTILE_COLS = ("q05", "q25", "q50", "q75", "q95")


def feature_names(state: str) -> list[str]:
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    return [*COMMON_FEATURES, *(FULL_FEATURES if state == "full" else PX_FEATURES)]


def bars(df: pd.DataFrame) -> pd.DataFrame:
    """open/high/low/close/vwap/volume per (symbol, day_idx). Uses the columns when the panel
    has them, else rebuilds the prices from today_vwap and the *_position_pct columns. Rows
    whose bar is not coherent (NULL-filled zeros) get NaN prices."""
    out = df[["symbol", "day_idx"]].copy()
    vwap = df["today_vwap"]
    out["vwap"] = vwap
    for name in ("open", "high", "low", "close"):
        if name in df.columns:
            out[name] = df[name]
            continue
        pos = f"{name}_position_pct"
        if pos not in df.columns:
            raise ValueError(f"panel has neither `{name}` nor `{pos}`; re-export with schema 4")
        out[name] = vwap * (1 + df[pos])
    out["volume"] = df["volume"] if "volume" in df.columns else np.nan
    ok = ((out["high"] >= out[["open", "close"]].max(axis=1))
          & (out["low"] <= out[["open", "close"]].min(axis=1)) & (out["low"] > 0))
    out.loc[~ok, ["open", "high", "low", "close"]] = np.nan
    return out


def targets(b: pd.DataFrame) -> pd.DataFrame:
    """range_from_open = max(high - open, open - low) / open; hl_range_open = (high - low) / open."""
    t = b[["symbol", "day_idx"]].copy()
    t["range_from_open"] = np.maximum(b["high"] - b["open"], b["open"] - b["low"]) / b["open"]
    t["hl_range_open"] = (b["high"] - b["low"]) / b["open"]
    return t


def history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, day_idx t): what is known at t's close. Join it to target day d on
    day_idx + 1 == d."""
    b = bars(panel).sort_values(["symbol", "day_idx"]).reset_index(drop=True)
    g = b.groupby("symbol", sort=False)
    prev_close = g["close"].shift(1)
    tr = (np.maximum(b["high"], prev_close) - np.minimum(b["low"], prev_close)) / b["close"]
    h = pd.DataFrame({"symbol": b["symbol"], "day_idx": b["day_idx"],
                      "prev_vwap": b["vwap"], "prev_close": b["close"]})
    h["prev_range"] = (b["high"] - b["low"]) / b["vwap"]
    h["prev_range_open"] = targets(b)["range_from_open"]
    h["atr14_pct"] = tr.groupby(b["symbol"], sort=False).transform(
        lambda s: s.rolling(14, min_periods=5).mean())
    dv = (b["volume"] * b["vwap"]).groupby(b["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    h["dollar_volume_20d"] = dv
    h["log_dollar_volume"] = np.log1p(dv)
    src = panel.set_index(["symbol", "day_idx"]).reindex(
        pd.MultiIndex.from_frame(b[["symbol", "day_idx"]]))
    for out_col, col in (("mkt_ret", "mkt_cap_ret"), ("sector_ret", "own_sector_cap_ret")):
        h[out_col] = src[col].to_numpy() if col in src.columns else np.nan
    to = next((c for c in ("days_to_results", "days_to_results_est") if c in src.columns), None)
    h["days_to_results"] = src[to].to_numpy() if to else np.nan
    h["days_since_results"] = (src["days_since_results"].to_numpy()
                               if "days_since_results" in src.columns else np.nan)
    if b["volume"].notna().any():
        h["vol_10d_mean"] = b["volume"].groupby(b["symbol"], sort=False).transform(
            lambda s: s.rolling(10, min_periods=5).mean())
    elif "vol_10d_mean" in src.columns:
        h["vol_10d_mean"] = src["vol_10d_mean"].to_numpy()
    return h


def advance_earnings(h: pd.DataFrame, sessions: int = 1) -> pd.DataFrame:
    """Move the t-row earnings counters forward to the target session and flag a release
    within a session of it (earn_near). NaN counters (no release known) become EARN_FAR."""
    h = h.copy()
    to = (h["days_to_results"] - sessions).fillna(EARN_FAR).clip(upper=EARN_FAR)
    since = (h["days_since_results"] + sessions).fillna(EARN_FAR).clip(upper=EARN_FAR)
    h["days_to_results"], h["days_since_results"] = to, since
    h["earn_near"] = ((to.abs() <= 1) | (since <= 1)).astype(float)
    return h


def tft_features(fc: pd.DataFrame, *, prices: bool) -> pd.DataFrame:
    """Width P5-P95, width P25-P75 and the median move vs the previous VWAP. `prices`: the
    qNN columns are prices with `prev_vwap` beside them (test_predictions); otherwise they are
    fractional moves (forecast_D)."""
    raw = fc[list(QUANTILE_COLS)].to_numpy(float)
    q = dict(zip(QUANTILE_COLS, np.sort(raw, axis=1).T))  # rearranged: never crossing
    scale = fc["prev_vwap"].to_numpy(float) if prices else 1.0
    out = fc[[c for c in ("symbol", "day_idx") if c in fc.columns]].copy()
    out["tft_width90"] = (q["q95"] - q["q05"]) / scale
    out["tft_width50"] = (q["q75"] - q["q25"]) / scale
    out["tft_median"] = q["q50"] / scale - 1 if prices else q["q50"]
    return out


def finish(df: pd.DataFrame) -> pd.DataFrame:
    """Gap features from open, previous close and previous VWAP (all known by 09:31)."""
    df = df.copy()
    df["gap_prev_close"] = df["open"] / df["prev_close"] - 1
    df["gap_prev_vwap"] = df["open"] / df["prev_vwap"] - 1
    df["lean"] = np.where(df["tft_median"] > 0, "up", "down")
    return df


def mask_premarket(df: pd.DataFrame) -> pd.DataFrame:
    """Panel pm columns are 0 when unknown; with the known flags they become NaN."""
    df = df.copy()
    px_known = (df["pm_px_known"] if "pm_px_known" in df
                else df.get("pm_known", pd.Series(np.nan, index=df.index)))
    full_known = (df["pm_full_known"] if "pm_full_known" in df
                  else df.get("pm_known", pd.Series(np.nan, index=df.index)))
    for cols, known in ((PX_FEATURES, px_known), (FULL_FEATURES[1:], full_known)):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
            df.loc[~(known == 1), c] = np.nan
    return df


def training_frame(panel: pd.DataFrame, tft: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, target day d) that has an out-of-fold TFT forecast: features,
    targets, date and open. `tft` is tft_features(test_predictions, prices=True)."""
    b = bars(panel)
    t = targets(b)
    keep = ["symbol", "day_idx", *[c for c in ("date", "day_of_week", "split_day")
                                   if c in panel.columns]]
    pm = [c for c in (*FULL_FEATURES, "pm_px_known", "pm_full_known", "pm_known")
          if c in panel.columns]
    d = panel[[*keep, *pm]].merge(b[["symbol", "day_idx", "open"]], on=["symbol", "day_idx"])
    d = d.merge(t, on=["symbol", "day_idx"]).merge(tft, on=["symbol", "day_idx"])
    h = advance_earnings(history_features(panel))
    h = h.drop(columns=["vol_10d_mean"], errors="ignore")
    h["day_idx"] = h["day_idx"] + 1
    d = d.merge(h, on=["symbol", "day_idx"])
    if "day_of_week" not in d.columns:
        d["day_of_week"] = pd.to_datetime(d["date"]).dt.dayofweek + 1
    if "split_day" not in d.columns:
        d["split_day"] = False
    d = finish(mask_premarket(d))
    return d.dropna(subset=["range_from_open", "open", "prev_close"]).query(
        "range_from_open > 0").reset_index(drop=True)
