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
        # one earlier forecast day that has closed: actuals drawn from the forecast itself
        past = fc.assign(date="2025-02-28", asof="2025-02-27")
        past.to_parquet(run / "forecast_2025-02-28.parquet", index=False)
        move = 0.004 + 0.015 * np.random.default_rng(7).standard_normal(len(syms))
        pd.DataFrame({"date": "2025-02-28", "symbol": syms, "actual_move": move,
                      "actual_vwap": 10.0 * (1 + move)}).to_parquet(run / "actuals.parquet",
                                                                    index=False)
    return run


@pytest.fixture
def run_dir(tmp_path):
    return write_run(tmp_path / "runs")


ID_TARGETS = ["r_1m", "r_5m", "r_15m", "r_60m", "r_rod", "r_close"]
ID_HORIZON = {"r_1m": 60, "r_5m": 300, "r_15m": 900, "r_60m": 3600, "r_rod": None,
              "r_close": None}
ID_DATES = [["2025-03-03", "2025-03-04"], ["2025-03-05", "2025-03-06"]]  # test dates per fold
ID_SYMBOLS = ["S0", "S1", "S2", "S3", "S4", "S5"]
ID_QCOLS = [f"q{round(lv * 100):02d}" for lv in LEVELS]


def intraday_frame(target, dates, seed=0, bias=0.0, with_daily=False, step=900, with_clim=True):
    """Decision points (every `step` s) in the intraday contract's format. The actual is drawn
    from the forecast itself (+ `bias`), so with bias=0 the forecasts are calibrated. ~5% of
    rows are mask=0. Everything is a log move from the last price. with_clim=False writes a run
    from before the climatology baseline."""
    rng = np.random.default_rng(seed + ID_TARGETS.index(target))
    z = np.array([NormalDist().inv_cdf(lv) for lv in LEVELS])
    ts = np.arange(0, 23401, step)
    keys = pd.DataFrame([(s, d, t) for s in ID_SYMBOLS for d in dates for t in ts],
                        columns=["symbol", "date", "t_s"])
    n = len(keys)
    scale = {"r_1m": 0.001, "r_5m": 0.002, "r_15m": 0.003, "r_60m": 0.005, "r_rod": 0.006,
             "r_close": 0.008}[target]
    sigma = scale * rng.uniform(0.8, 1.2, n)
    mu = rng.normal(0, scale / 4, n)
    f = keys.assign(
        t_s=keys["t_s"].astype("int32"), tod_bucket=(keys["t_s"] // 900).astype(int),
        sector_code=np.where(keys["symbol"].isin(["S0", "S1", "S2"]), "tech", "energy"),
        rvol=rng.uniform(0.5, 2, n), mask=(rng.random(n) > 0.05).astype("int8"))
    f["actual"] = mu + sigma * rng.standard_normal(n) + bias
    for c, zz in zip(ID_QCOLS, z, strict=True):
        f[c] = mu + sigma * zz
        f[f"mom_{c}"] = 1.3 * sigma * zz  # wider and uncentred: a weaker baseline
        if with_clim:  # the right width on average, but blind to mu and sigma
            f[f"clim_{c}"] = scale * 1.05 * zz
    f["ref_trail"] = rng.normal(0, scale / 3, n)
    f["ref_session"] = rng.normal(0, scale / 2, n)
    f["y"] = rng.normal(0, 0.01, n)
    f["last"] = 20 + keys["symbol"].map({s: i * 7 for i, s in enumerate(ID_SYMBOLS)}) \
        + rng.normal(0, 0.05, n)
    f["prev_vwap"] = f["last"] * 0.998
    if with_daily:
        f["has_daily"] = (rng.random(n) > 0.1).astype("int8")
        for c, zz in zip(ID_QCOLS, z, strict=True):
            f[f"daily_{c}"] = mu + 1.4 * sigma * zz
    return f


def write_intraday_run(root, name="id-a", seed=0, report=False, **kw):
    """A synthetic intraday run: 2 folds x 2 test dates, 6 symbols in 2 sectors, every target."""
    run = root / name
    folds = []
    for k, dates in enumerate(ID_DATES):
        (run / f"fold_{k}").mkdir(parents=True, exist_ok=True)
        folds.append({"fold": k, "train_dates": ["2025-02-03"], "es_dates": ["2025-02-04"],
                      "test_dates": dates})
        for t in ID_TARGETS:
            intraday_frame(t, dates, seed + k * 10, with_daily=t == "r_close", **kw).to_parquet(
                run / f"fold_{k}" / f"intraday_{t}.parquet", index=False)
    cfg = {"model": "tft_intraday", "step_s": 15, "eval_stride_s": 60, "targets": ID_TARGETS,
           "horizon_s": ID_HORIZON, "daily_run": "run-a", "folds": folds}
    (run / "config.json").write_text(json.dumps(cfg))
    if report:
        (run / "report").mkdir()
        rows = [{"target": t, "model": m, "n": 1000, "coverage80": 0.81,
                 "pinball": 0.001, "skill_clim": 0.045, "skill_rw": 0.05, "skill_mom": 0.123,
                 "skill_daily": None,
                 "dir_last": 0.5, "dir_trail": 0.577, "dir_session": 0.52, "pit_bias": 0.0,
                 "width80_med": 0.01}
                for t in ID_TARGETS for m in ("tft", "climatology", "momentum")]
        pd.DataFrame(rows).to_parquet(run / "report" / "by_target.parquet", index=False)
    return run
