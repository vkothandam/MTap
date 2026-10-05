import json

import numpy as np
import pandas as pd
import pytest


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
    return df


@pytest.fixture
def panel_df():
    return synthetic_panel()


@pytest.fixture
def panel_path(tmp_path, panel_df):
    path = tmp_path / "panel.parquet"
    panel_df.to_parquet(path, index=False)
    last = int(panel_df["day_idx"].max())
    nxt = {"date": (panel_df["date"].max() + pd.Timedelta(days=1)).date().isoformat(),
           "day_idx": last + 1, "day_of_week": (last + 1) % 5 + 1,
           "month": ((last + 1) // 21) % 12 + 1, "is_month_end": 0, "is_quarter_end": 0,
           "sessions_gap_next": 1}
    (tmp_path / "panel.meta.json").write_text(
        json.dumps({"schema_version": 1, "next_session": nxt}))
    return path
