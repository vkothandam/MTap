"""tft_intraday: masked loss, target shift, folds, daily join, baselines and a tiny full run."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import model_card
from tft_intraday import baselines, daily_join, report, train
from tft_intraday import dataset as ds
from tft_intraday.config import QUANTILES, TARGETS, IntradayConfig
from tft_intraday.losses import MaskedMultiLoss, bit_weights

STATIC_CATS = ["sector_code", "watchlist", "day_of_week", "is_half_day"]
STATIC_REALS = ["log_dollar_vol_10d", "vwap_pct_prev_day", "intraday_spread_pct",
                "close_position_pct", "days_since_prev_session", "days_since_results"]
KNOWN_REALS = ["t_s", "session_frac", "minutes_to_close", "expected_vol_remaining"]
OBSERVED = ["y", "L_trail_1", "L_trail_5", "rvol", "mkt_trail_5", "sec_trail_5"]
REFS = ["ref_trail_1m", "ref_trail_5m", "ref_trail_15m", "ref_trail_60m", "ref_session"]
DATES = ["2025-12-19", "2025-12-22", "2025-12-23", "2025-12-24"]


def synth_day(date: str, symbols: list[str], steps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for j, sym in enumerate(symbols):
        n = steps
        y = np.cumsum(rng.normal(0, 1e-4, n))
        d = pd.DataFrame({"symbol": sym, "date": date, "step": np.arange(n, dtype=np.int32)})
        d["t_s"] = (7 + 15 * d["step"]).astype(np.int32)
        d["tod_bucket"] = (d["t_s"] // 900).astype(np.int16)
        d["session_frac"] = d["t_s"] / 23400
        d["minutes_to_close"] = (23400 - d["t_s"]) / 60
        d["expected_vol_remaining"] = 1 - d["session_frac"]
        d["sector_code"] = None if j == 0 else f"S{j % 2}"
        d["watchlist"] = "Experimental"
        d["day_of_week"] = pd.Timestamp(date).dayofweek
        d["is_half_day"] = 0
        for c in STATIC_REALS:
            d[c] = 0.1 * j
        d["y"] = y
        for c in OBSERVED[1:]:
            d[c] = y + rng.normal(0, 1e-5, n)
        drift = np.r_[np.diff(y), 0]
        for k, t in enumerate(TARGETS):
            d[t] = drift * (k + 1) + rng.normal(0, 1e-4, n)
        mask = np.full(n, 0b111111, dtype=np.int8)
        mask[-3:] &= ~np.int8(0b000010)  # the 5-minute target runs past the end
        mask[::7] &= ~np.int8(0b010000)  # rest of day invalid on some rows
        d["tmask"] = mask
        for k, t in enumerate(TARGETS):
            d.loc[(d["tmask"].to_numpy() >> k) & 1 == 0, t] = np.nan
        d["last"] = 50.0 * np.exp(y)
        d["prev_vwap"] = 50.0
        d["prev_close"] = 50.0
        d["avg_vol_10d"] = 1e6
        for c in REFS:
            d[c] = rng.normal(0, 1e-4, n)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def panel(tmp_path):
    root = tmp_path / "panel"
    syms = [f"S{i}" for i in range(6)]
    days = {}
    for i, date in enumerate(DATES):
        d = synth_day(date, syms, 120, i)
        (root / f"date={date}").mkdir(parents=True)
        d.to_parquet(root / f"date={date}" / "part.parquet", index=False)
        days[date] = {"rows": len(d), "symbols": len(syms)}
    meta = {"schema_version": 1, "step_s": 15, "targets": list(TARGETS),
            "tmask_bits": {t: k for k, t in enumerate(TARGETS)},
            "horizon_s": {"r_1m": 60, "r_5m": 300, "r_15m": 900, "r_60m": 3600,
                          "r_rod": None, "r_close": None},
            "target_unit": "log move from the last price at t", "dates": DATES, "rows": 0,
            "days": days,
            "features": {"static_categoricals": STATIC_CATS, "static_reals": STATIC_REALS,
                         "known_categoricals": ["tod_bucket"], "known_reals": KNOWN_REALS,
                         "observed_reals": OBSERVED}}
    (root / "panel.meta.json").write_text(json.dumps(meta))
    return root


def daily_run(tmp_path, symbols=("S1", "S1~1", "S2"), dates=DATES):
    run = tmp_path / "daily"
    (run / "fold_1").mkdir(parents=True)
    rows = []
    for sym in symbols:
        for k, date in enumerate(dates):
            prev = 100.0 if "~" not in sym else 7.0
            qs = {f"q{round(q * 100):02d}": prev * (1 + (q - 0.5) * 0.02 + 0.001 * k)
                  for q in QUANTILES}
            rows.append({"symbol": sym, "day_idx": k, **qs, "prev_vwap": prev,
                         "actual": prev * (1.001 if k % 2 else 1.03), "date": date,
                         "industry_code": "1"})
    pd.DataFrame(rows).to_parquet(run / "fold_1" / "test_predictions.parquet", index=False)
    return run


# --------------------------------------------------------------------------- the loss


def _batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = [torch.randn(n, 1, len(QUANTILES), generator=g, requires_grad=True)
            for _ in TARGETS]
    target = [torch.randn(n, 1, generator=g) for _ in TARGETS]
    mask = torch.randint(0, 64, (n, 1), generator=g).float()
    return pred, target, mask


def test_bitmask_round_trips():
    w = torch.tensor([[float(0b101001)], [0.0], [63.0]])
    got = [x.squeeze(-1) > 0 for x in bit_weights(w, 6)]
    for i in range(6):
        assert got[i].tolist() == [bool(int(v) >> i & 1) for v in (0b101001, 0, 63)]


def test_loss_ignores_masked_targets():
    loss = MaskedMultiLoss(QUANTILES, len(TARGETS))
    pred, target, mask = _batch()
    a = loss(pred, (target, mask))
    a.backward()
    moved = [t.clone() for t in target]
    for i, t in enumerate(moved):
        off = ((mask.long() >> i) & 1) == 0
        t[off] += 1000.0
    assert torch.allclose(loss(pred, (moved, mask)), a)
    for i, p in enumerate(pred):  # no gradient from a masked target
        off = (((mask.long() >> i) & 1) == 0).squeeze(-1)
        assert torch.all(p.grad[off] == 0)
        assert torch.any(p.grad[~off] != 0)


def test_all_masked_batch_is_finite_and_zero():
    loss = MaskedMultiLoss(QUANTILES, len(TARGETS))
    pred, target, _ = _batch()
    v = loss(pred, (target, torch.zeros(64, 1)))
    assert torch.isfinite(v) and v.item() == 0.0


def test_loss_instances_are_separate():
    loss = MaskedMultiLoss(QUANTILES, 3)
    assert len({id(m) for m in loss.metrics}) == 3


# --------------------------------------------------------------------------- the frames


def test_model_frame_shifts_targets_one_row_down(panel):
    meta = ds.read_meta(panel)
    feats = ds.features(meta, IntradayConfig(panel=panel, daily_run=None))
    raw = ds.load_days(panel, DATES[:2])
    fr = ds.model_frame(raw, feats, 15)
    per = fr.groupby(["symbol", "date"]).size()
    assert (per == 121).all()  # one padding row per series
    s = fr[(fr["symbol"] == "S2") & (fr["date"] == DATES[0])].reset_index(drop=True)
    r = raw[(raw["symbol"] == "S2") & (raw["date"] == DATES[0])].reset_index(drop=True)
    assert s.loc[0, ds.WEIGHT] == 0 and s.loc[0, ds.MODEL_TARGETS].eq(0).all()
    np.testing.assert_allclose(s.loc[1:, ds.WEIGHT].to_numpy(), r["tmask"].to_numpy())
    np.testing.assert_allclose(s.loc[1:, "tgt_r_1m"].to_numpy(), r["r_1m"].fillna(0).to_numpy(),
                               rtol=1e-6)
    np.testing.assert_allclose(s.loc[:119, "y"].to_numpy(), r["y"].to_numpy(), rtol=1e-6)
    assert s.loc[120, "t_s"] == r.loc[119, "t_s"] + 15
    assert not set(TARGETS) & set(fr.columns)  # targets are never inputs
    assert set(fr["sector_code"]) == {"unknown", "S0", "S1"}


def test_make_folds_takes_early_stop_dates_just_before_the_test(panel):
    meta = ds.read_meta(panel)
    n = meta["days"][DATES[0]]["rows"]
    cfg = IntradayConfig(panel=panel, test_blocks=[[DATES[3]]], es_min_rows=n + 1)
    (f,) = ds.make_folds(meta, cfg)
    assert f.es_dates == DATES[1:3] and f.train_dates == DATES[:1] and f.test_dates == DATES[3:]
    with pytest.raises(ValueError):
        ds.make_folds(meta, IntradayConfig(panel=panel, test_blocks=[[DATES[1]]], es_min_rows=1e9))


def test_evaluation_set_holds_only_its_dates_on_the_stride(panel):
    meta = ds.read_meta(panel)
    cfg = IntradayConfig(panel=panel, daily_run=None, encoder_steps=20)
    feats = ds.features(meta, cfg)
    train_ds = ds.build_train(ds.model_frame(ds.load_days(panel, DATES[:2]), feats, 15), feats, cfg)
    ev = ds.evaluation_set(train_ds, ds.model_frame(ds.load_days(panel, DATES[3:]), feats, 15), 4)
    idx = ev.decoded_index
    assert set(idx["date"].astype(str)) == {DATES[3]}  # an unseen date encodes
    assert ((idx["time_idx_first_prediction"] - 1) % 4 == 0).all()
    assert not idx.duplicated(["symbol", "date", "time_idx_first_prediction"]).any()
    # the longest encoder: 20 steps once the series is long enough
    late = idx["time_idx_first_prediction"] > 40
    assert (idx.loc[late, "time_idx_last"] - idx.loc[late, "time_idx_first"] == 20).all()


# --------------------------------------------------------------------------- daily join


def test_daily_join_maps_suffixes_and_uses_only_earlier_history(tmp_path):
    d = daily_join.load_daily(daily_run(tmp_path), 1)
    assert set(d["symbol"]) == {"S1", "S2"} and len(d) == 2 * len(DATES)
    s1 = d[d["symbol"] == "S1"].reset_index(drop=True)
    assert np.allclose(s1["d_med"], np.log(1 + 0.001 * np.arange(len(DATES))))  # not S1~1's
    assert np.isnan(s1.loc[0, "hist_mae"])  # nothing before the first date
    first_err = abs(np.log(1.03) - s1.loc[0, "d_med"])
    assert s1.loc[1, "hist_mae"] == pytest.approx(first_err)
    assert s1.loc[0, "d_p_up"] > 0.4 and s1.loc[0, "d_width"] == pytest.approx(
        np.log(1.008) - np.log(0.992), rel=1e-6)


def test_attach_fills_missing_and_sector_mean_excludes_self(tmp_path, panel):
    daily = daily_join.load_daily(daily_run(tmp_path, symbols=("S1", "S3", "S5")), 1)
    raw = daily_join.attach(ds.load_days(panel, DATES[:1]), daily)
    one = raw.drop_duplicates("symbol").set_index("symbol")
    assert one.loc["S0", "has_daily"] == 0 and one.loc["S0", "d_med"] == 0
    assert one.loc["S1", "has_daily"] == 1
    # S1, S3, S5 share sector S1; each sees the mean of the other two
    assert one.loc["S1", "sec_d_med"] == pytest.approx(one.loc[["S3", "S5"], "d_med"].mean())


def test_daily_run_never_reads_another_fold(tmp_path):
    run = daily_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        daily_join.load_daily(run, 3)


# --------------------------------------------------------------------------- baselines


def test_momentum_is_fitted_on_training_rows_only(panel):
    train_raw = ds.load_days(panel, DATES[:2])
    table = baselines.fit_momentum(train_raw)
    ok = baselines.valid(train_raw, "r_5m")
    res = (train_raw["r_5m"] - train_raw["ref_trail_5m"])[ok]
    slot = train_raw["tod_bucket"][ok]
    assert np.allclose(table["r_5m"][1], np.quantile(res[slot == 1], QUANTILES))
    assert np.allclose(table["r_5m"][5], np.quantile(res, QUANTILES))  # empty slot: all day
    test = ds.load_days(panel, DATES[3:])
    q = baselines.momentum_quantiles(test, "r_5m", table)
    first = (test["tod_bucket"] == 1).to_numpy()
    assert np.allclose(q[first] - test["ref_trail_5m"].to_numpy()[first, None], table["r_5m"][1])


def test_climatology_is_the_training_spread_from_the_last_price(panel):
    train_raw = ds.load_days(panel, DATES[:2])
    table = baselines.fit_climatology(train_raw)
    ok = baselines.valid(train_raw, "r_15m")
    r, slot = train_raw["r_15m"][ok], train_raw["tod_bucket"][ok]
    assert np.allclose(table["r_15m"][1], np.quantile(r[slot == 1], QUANTILES))
    test = ds.load_days(panel, DATES[3:])
    q = baselines.climatology_quantiles(test, "r_15m", table)
    first = (test["tod_bucket"] == 1).to_numpy()
    assert np.allclose(q[first], table["r_15m"][1])  # no reference: the same for every row


def test_report_without_climatology_columns_scores_nan():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"date": "2025-12-29", "t_s": np.arange(n) * 60, "tod_bucket": 1,
                       "actual": rng.normal(0, 1e-3, n), "ref_trail": 0.0, "ref_session": 0.0})
    for cols in (report.QCOLS, report.MODELS["momentum"]):
        df[cols] = np.quantile(rng.normal(0, 1e-3, 1000), QUANTILES)
    bt = report.score_target(df, "r_5m")["by_target"]
    assert set(bt["model"]) == {"tft", "momentum"} and bt["skill_clim"].isna().all()
    assert bt["skill_mom"].notna().all()


# --------------------------------------------------------------------------- a whole run


def test_tiny_run_writes_the_dashboard_contract(tmp_path, panel):
    cfg = IntradayConfig(panel=panel, out_dir=tmp_path / "runs", daily_run=daily_run(tmp_path),
                         test_blocks=[[DATES[3]]], es_min_rows=1, encoder_steps=12,
                         hidden_size=8, attention_head_size=1, hidden_continuous_size=4,
                         batch_size=64, max_epochs=1, limit_train_batches=2,
                         accelerator="cpu", test_stride_s=60)
    run = train.run(cfg, card={"label": "I1"})
    card = model_card.read(run)
    assert (card["label"], card["forecast"], card["kind"], card["status"]) == (
        "I1", "intraday", "Intraday", "done")
    assert card["result"].startswith("Skill vs climatology") and "r_close" in card["result"]
    conf = json.loads((run / "config.json").read_text())
    assert conf["model"] == "tft_intraday" and conf["folds"][0]["test_dates"] == [DATES[3]]
    fold = run / "fold_0"
    for f in ("best.ckpt", "metrics.json", "baselines.json"):
        assert (fold / f).exists()
    for t in TARGETS:
        p = pd.read_parquet(fold / f"intraday_{t}.parquet")
        need = ["symbol", "date", "t_s", "tod_bucket", "sector_code", "rvol", "mask", "actual",
                *report.QCOLS, *report.MODELS["momentum"], *report.MODELS["climatology"],
                "ref_trail", "ref_session", "y",
                "last", "prev_vwap"]
        assert not set(need) - set(p.columns), t
        assert set(p["date"]) == {DATES[3]} and (p["t_s"] % 60 == 7).all()
        assert (np.diff(p[report.QCOLS].to_numpy(), axis=1) >= 0).all()
        assert p.loc[p["mask"] == 0, "actual"].isna().all()
        assert p.loc[p["mask"] == 1, "actual"].notna().all()
    close = pd.read_parquet(fold / "intraday_r_close.parquet")
    assert set(close["has_daily"]) == {0, 1}
    assert not set(report.MODELS["daily"]) - set(close.columns)
    # actuals line up with the panel's decision rows
    raw = ds.load_days(panel, DATES[3:])
    m = close.merge(raw, on=["symbol", "date", "t_s"])
    np.testing.assert_allclose(m["actual"], m["r_close"], rtol=1e-6)
    bt = pd.read_parquet(run / "report" / "by_target.parquet")
    assert set(bt["model"]) == {"tft", "climatology", "momentum", "daily"}
    assert set(bt["target"]) == set(TARGETS)
    assert {"skill_clim", "skill_clim_lo", "skill_clim_hi", "skill_mom", "skill_mom_lo",
            "skill_mom_hi", "coverage80", "dir_trail"} <= set(bt)
    clim = bt[bt["model"] == "climatology"]
    assert np.allclose(clim["skill_clim"], 0.0)  # scored against itself


def test_baselines_only_run(tmp_path, panel):
    cfg = IntradayConfig(panel=panel, out_dir=tmp_path / "runs", daily_run=None,
                         test_blocks=[[DATES[3]]], es_min_rows=1, baselines_only=True)
    run = train.run(cfg)
    assert run.name.endswith("-baselines")
    assert model_card.read(run)["result"].startswith("Baselines only")
    p = pd.read_parquet(run / "fold_0" / "intraday_r_5m.parquet")
    np.testing.assert_allclose(p[report.QCOLS].to_numpy(), p[report.MODELS["momentum"]].to_numpy())
    assert (p["t_s"] >= 7 + 15 * (cfg.min_encoder_steps - 1)).all()
