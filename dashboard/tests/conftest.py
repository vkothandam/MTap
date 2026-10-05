import json
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

LEVELS = [round(k * 0.05, 2) for k in range(1, 20)]


def synthetic_predictions(n_symbols=8, n_days=60, start=1000, seed=0, shift=None, folds=2):
    """Test predictions in the contract's format. Forecasts are N(mu, sigma) quantiles and the
    actual is drawn from that same distribution (+ `shift[symbol]`), so with no shift the
    forecasts are calibrated by construction."""
    rng = np.random.default_rng(seed)
    z = np.array([NormalDist().inv_cdf(lv) for lv in LEVELS])
    shift = shift or {}
    rows = []
    for i in range(n_symbols):
        sym = f"S{i}"
        days = np.arange(start, start + n_days)
        mu = rng.normal(0, 0.003, n_days)
        sigma = rng.uniform(0.01, 0.03, n_days)
        r_act = mu + sigma * rng.standard_normal(n_days) + shift.get(sym, 0.0)
        prev = 10 + i * 5 * np.exp(rng.normal(0, 0.01, n_days))
        f = pd.DataFrame({"symbol": sym, "day_idx": days, "prev_vwap": prev,
                          "actual": prev * (1 + r_act)})
        for lv, zz in zip(LEVELS, z, strict=True):
            f[f"q{round(lv * 100):02d}"] = prev * (1 + mu + sigma * zz)
        f["fold"] = np.minimum((days - start) * folds // n_days, folds - 1)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def write_run(root, name="run-a", panel=None, with_keys=True, forecast=True, **kw):
    run = root / name
    df = synthetic_predictions(**kw)
    if with_keys:
        df["date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_idx"] - 1000, unit="D")
        df["industry_code"] = np.where(df["symbol"].isin(["S0", "S1"]), "A", "B")
    for k, d in df.groupby("fold"):
        (run / f"fold_{k}").mkdir(parents=True, exist_ok=True)
        d.drop(columns="fold").to_parquet(run / f"fold_{k}" / "test_predictions.parquet",
                                          index=False)
    cfg = {"config": {"target": "return", "panel": str(panel) if panel else ""}}
    (run / "config.json").write_text(json.dumps(cfg))
    if forecast:
        z = np.array([NormalDist().inv_cdf(lv) for lv in LEVELS])
        syms = sorted(df["symbol"].unique())
        fc = pd.DataFrame({"date": "2025-03-03", "symbol": syms, "today_vwap": 10.0})
        for lv, zz in zip(LEVELS, z, strict=True):
            fc[f"q{round(lv * 100):02d}"] = 0.004 + 0.015 * zz  # fractional moves
        fc["up_70"] = 0.0
        fc.to_parquet(run / "forecast_2025-03-03.parquet", index=False)
    return run


@pytest.fixture
def run_dir(tmp_path):
    return write_run(tmp_path / "runs")
