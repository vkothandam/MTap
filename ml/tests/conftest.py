import json

import numpy as np
import pandas as pd
import pytest

from tft_vwap.config import (
    AFTER_HOURS_REALS,
    MACRO_REALS,
    MARKET_REALS,
    PREMARKET_REALS,
    SESSION_SHAPE_REALS,
)

SCHEMA4_COLUMNS = [*MARKET_REALS, *MACRO_REALS, *SESSION_SHAPE_REALS, *AFTER_HOURS_REALS,
                   *PREMARKET_REALS, "vol_10d_mean"]


def synthetic_panel(n_symbols=6, n_days=150, start=1000, seed=0, gap_symbol="S1", gap_day=1050):
    """Panel shaped like export-tft output: target = next session's vwap, rows whose next bar is
    missing are dropped, and the last session keeps a NULL target (inference row)."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_symbols):
        sym = f"S{i}"
        days = np.arange(start, start + n_days)
        if sym == gap_symbol:
            days = days[days != gap_day]
        vwap = (10 + 5 * i) * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
        f = pd.DataFrame({"symbol": sym, "day_idx": days, "today_vwap": vwap})
        nxt = f["today_vwap"].shift(-1).where(f["day_idx"].shift(-1) == f["day_idx"] + 1)
        f["target_tomorrow_vwap"] = nxt
        f = f[f["target_tomorrow_vwap"].notna() | (f["day_idx"] == start + n_days - 1)]
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["target_return"] = df["target_tomorrow_vwap"] / df["today_vwap"] - 1
    df["industry_code"] = np.where(df["symbol"].isin(["S0", "S1"]), "unknown", "531010")
    df["exchange"], df["issue_type"] = "NASDAQ", "CS"
    df["date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day_idx"] - start, unit="D")
    df["day_of_week"] = df["day_idx"] % 5 + 1
    df["month"] = (df["day_idx"] // 21) % 12 + 1
    df["is_month_end"] = (df["day_idx"] % 21 == 20).astype(int)
    df["is_quarter_end"] = (df["day_idx"] % 63 == 62).astype(int)
    df["sessions_gap_next"] = np.where(df["day_of_week"] == 5, 3, 1)
    prev = df.groupby("symbol")["today_vwap"].shift(1)
    df["vwap_pct_prev_day"] = df["today_vwap"] / prev - 1  # NaN on first bar, like the export
    df["intraday_spread_pct"] = rng.uniform(0.005, 0.05, len(df))
    df["close_position_pct"] = rng.normal(0, 0.01, len(df))
    df["open_position_pct"] = rng.normal(0, 0.01, len(df))
    df["volume_velocity"] = rng.lognormal(0, 0.5, len(df))
    df["transactions_per_volume"] = rng.uniform(0.001, 0.02, len(df))
    df["daily_sentiment"] = df["sentiment_volume"] = df["has_news"] = 0.0
    df["is_10k"] = df["is_10q"] = 0
    # a results release every 63 sessions, staggered by symbol; none before the first one.
    # The estimate runs a session early, so the exact and point-in-time windows differ.
    since = (df["day_idx"] + 7 * df["symbol"].str[1:].astype(int)) % 63
    first = df["day_idx"] - since >= start
    df["days_since_results"] = since.where(first)
    df["days_to_results"] = 63 - since
    df["days_to_results_is_est"] = 0
    df["days_to_results_est"] = (62 - since).where(first)

    def window(till):
        return np.select(
            [first & (since == 0), first & (since <= 3), first & (since <= 15),
             till.between(2, 10), till.between(-10, 1)],
            ["Earnings_Day", "Post_Earnings_Reaction", "Post_Earnings_Drift",
             "Pre_Earnings_Runup", "Earnings_Eve"], "Normal_Trading")

    df["results_window"] = window(df["days_to_results"])
    df["results_window_pit"] = window(df["days_to_results_est"])
    df["is_10q"] = (first & (since == 0)).astype(int)
    add_schema4(df, rng)
    return df


def add_schema4(df, rng):
    """The schema-4 columns, with NULL gaps like the export (flags are never NULL)."""
    n = len(df)
    for c in (*MARKET_REALS, *MACRO_REALS, *SESSION_SHAPE_REALS):
        df[c] = rng.normal(0, 0.01, n)
        df.loc[rng.random(n) < 0.05, c] = np.nan
    df["ah_known"] = (rng.random(n) < 0.8).astype(float)
    df["ah_vwap_pct_rth_vwap"] = np.where(df["ah_known"] == 1, rng.normal(0, 0.005, n), np.nan)
    df["ah_vol_pct_10d"] = np.where(df["ah_known"] == 1, rng.uniform(0, 3, n), np.nan)
    df["close_pct_rth_vwap"] = np.where(df["ah_known"] == 1, rng.normal(0, 0.004, n), np.nan)
    df["vol_10d_mean"] = rng.uniform(1e5, 1e6, n)  # not a model input (only the export uses it)
    df["pm_px_known"] = (rng.random(n) < 0.8).astype(float)
    df["pm_full_known"] = df["pm_px_known"] * (rng.random(n) < 0.8)  # full implies prices
    for c in ("pm_last_pct_prev_vwap", "open_pct_prev_vwap"):
        df[c] = np.where(df["pm_px_known"] == 1, rng.normal(0, 0.01, n), np.nan)
    df["pm_vwap_pct_prev_vwap"] = np.where(df["pm_full_known"] == 1, rng.normal(0, 0.01, n), np.nan)
    df["pm_vol_pct_10d"] = np.where(df["pm_full_known"] == 1, rng.uniform(0, 4, n), np.nan)
    df["pm_range_pct"] = np.where(df["pm_full_known"] == 1, rng.uniform(0, 0.05, n), np.nan)


def make_next(panel_df, export_id="exp-1"):
    """panel.next.parquet: the per-symbol known inputs of the session after the panel."""
    last = int(panel_df["day_idx"].max())
    syms = sorted(panel_df.loc[panel_df["day_idx"] == last, "symbol"].unique())
    rng = np.random.default_rng(5)
    nxt = pd.DataFrame({"symbol": syms, "day_idx": last + 1,
                        "pm_last_pct_prev_vwap": rng.normal(0, 0.01, len(syms)),
                        "open_pct_prev_vwap": 0.0, "pm_px_known": 1.0,
                        "pm_vwap_pct_prev_vwap": rng.normal(0, 0.01, len(syms)),
                        "pm_vol_pct_10d": rng.uniform(0, 4, len(syms)),
                        "pm_range_pct": rng.uniform(0, 0.05, len(syms)),
                        "pm_full_known": 1.0, "vol_10d_mean": np.nan, "export_id": export_id})
    return nxt.iloc[:-1]  # the last symbol has no pre-market row: it gets a blank one


def write_panel(tmp_path, panel_df, schema, export_id="exp-1"):
    df = panel_df if schema == 4 else panel_df.drop(columns=SCHEMA4_COLUMNS)
    path = tmp_path / "panel.parquet"
    df.to_parquet(path, index=False)
    last = int(panel_df["day_idx"].max())
    nxt = {"date": (panel_df["date"].max() + pd.Timedelta(days=1)).date().isoformat(),
           "day_idx": last + 1, "day_of_week": (last + 1) % 5 + 1,
           "month": ((last + 1) // 21) % 12 + 1, "is_month_end": 0, "is_quarter_end": 0,
           "sessions_gap_next": 1}
    meta = {"schema_version": schema, "next_session": nxt}
    if schema == 4:
        meta.update(export_id=export_id, premarket_cutoff="09:25")
        make_next(panel_df, export_id).to_parquet(tmp_path / "panel.next.parquet", index=False)
    (tmp_path / "panel.meta.json").write_text(json.dumps(meta))
    return path


@pytest.fixture
def panel_df():
    return synthetic_panel()


@pytest.fixture
def panel_path(tmp_path, panel_df):
    """A schema-3 panel (none of the schema-4 columns), as production runs read."""
    return write_panel(tmp_path, panel_df, 3)


@pytest.fixture
def panel4_path(tmp_path, panel_df):
    """A schema-4 panel with panel.next.parquet and an export_id in the meta."""
    d = tmp_path / "p4"
    d.mkdir()
    return write_panel(d, panel_df, 4)
