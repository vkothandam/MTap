"""Schema-4 input groups, pre-market masking, resumable runs, folds, compare and overlay."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import SCHEMA4_COLUMNS, make_next, write_panel
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer

from tft_vwap import compare, dataset, grade, overlay, predict, train
from tft_vwap.config import (
    AFTER_HOURS_REALS,
    MACRO_REALS,
    MARKET_REALS,
    PREMARKET_REALS,
    SESSION_SHAPE_REALS,
    TrainConfig,
    from_run,
)


def _cfg(panel, tmp_path, **kw):
    base = {"panel": panel, "out_dir": tmp_path / "runs", "n_folds": 2, "fold_len": 10,
            "early_stop_len": 5, "encoder_length": 20, "min_encoder_length": 5,
            "batch_size": 32, "hidden_size": 8, "attention_head_size": 1,
            "hidden_continuous_size": 4, "max_epochs": 1, "limit_train_batches": 2,
            "accelerator": "cpu"}
    return TrainConfig(**{**base, **kw})


GROUPS = {"market": MARKET_REALS, "macro": MACRO_REALS, "session_shape": SESSION_SHAPE_REALS,
          "after_hours": AFTER_HOURS_REALS}


@pytest.mark.parametrize("flag", GROUPS)
def test_each_flag_adds_exactly_its_columns(flag):
    off = TrainConfig()
    on = TrainConfig(**{flag: True})
    assert set(on.observed_reals) - set(off.observed_reals) == set(GROUPS[flag])
    assert on.known_reals == off.known_reals
    assert off.extra_reals == [] and on.extra_reals == list(GROUPS[flag])


def test_premarket_adds_known_reals_only():
    on = TrainConfig(premarket="optional")
    assert set(on.known_reals) - set(TrainConfig().known_reals) == set(PREMARKET_REALS)
    assert on.observed_reals == TrainConfig().observed_reals
    with pytest.raises(ValueError, match="premarket"):
        TrainConfig(premarket="maybe")


def test_old_config_loads_with_every_new_input_off(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    saved = TrainConfig().to_dict()
    for k in ("market", "macro", "session_shape", "after_hours", "premarket", "pm_mask",
              "premarket_cutoff", "dedupe_train", "min_train_sessions"):
        saved.pop(k)
    (run / "config.json").write_text(json.dumps({"config": saved}))
    cfg = from_run(run)
    assert not (cfg.market or cfg.macro or cfg.session_shape or cfg.after_hours)
    assert cfg.premarket == "off" and cfg.dedupe_train is False and cfg.extra_reals == []


def test_prepare_reads_only_enabled_columns(panel_df):
    # a schema-3 frame works with every group off, and says what is missing otherwise
    df = dataset.prepare(panel_df.drop(columns=SCHEMA4_COLUMNS))
    assert not set(SCHEMA4_COLUMNS) & set(df.columns)
    with pytest.raises(ValueError, match="schema 4"):
        dataset.prepare(panel_df.drop(columns=SCHEMA4_COLUMNS), extra=MARKET_REALS)
    full = dataset.prepare(panel_df, extra=[*MARKET_REALS, *PREMARKET_REALS])
    assert full[[*MARKET_REALS, *PREMARKET_REALS]].notna().all().all()  # NULL gaps filled with 0
    assert (full.loc[panel_df.loc[full.index, "pm_known"] == 0, "pm_vwap_pct_prev_vwap"] == 0).all()


def test_train_dedupe_keeps_one_sample_per_symbol_day(panel_df, tmp_path):
    df = dataset.prepare(panel_df)
    fold = dataset.make_folds(df, _cfg(tmp_path, tmp_path))[0]
    full = dataset.build_datasets(df, _cfg(tmp_path, tmp_path, dedupe_train=False), fold)[0]
    one = dataset.build_datasets(df, _cfg(tmp_path, tmp_path, dedupe_train=True), fold)[0]
    idx = one.decoded_index
    assert len(one) < len(full)
    assert not idx.duplicated(["symbol", "time_idx_first_prediction"]).any()
    # every series end appears once, with its longest encoder
    assert set(zip(idx["symbol"], idx["time_idx_first_prediction"])) == set(
        zip(full.decoded_index["symbol"], full.decoded_index["time_idx_first_prediction"]))


def _batch(panel_df, tmp_path, **kw):
    cfg = _cfg(tmp_path, tmp_path, premarket="optional", **kw)
    df = dataset.prepare(panel_df, extra=cfg.extra_reals)
    ds = dataset.build_datasets(df, cfg, dataset.make_folds(df, cfg)[0])[0]
    return cfg, ds, [ds[i] for i in range(200)]


def test_pm_mask_collate_blanks_the_decoder_only(panel_df, tmp_path):
    _, ds, items = _batch(panel_df, tmp_path)
    idx = [ds.reals.index(c) for c in PREMARKET_REALS]
    others = [i for i in range(len(ds.reals)) if i not in idx]
    x0, _ = ds._collate_fn(items)
    assert (x0["decoder_cont"][..., idx[2]] == 1).any()  # raw pm_known, unscaled
    for rate in (0.0, 1.0):
        x, _ = dataset.PmMaskCollate.for_dataset(ds, rate)(items)
        blank = (x["decoder_cont"][..., idx] == 0).all(dim=-1).all(dim=-1)
        assert blank.all() if rate else (blank == (x0["decoder_cont"][..., idx] == 0).all(-1).all(-1)).all()
        assert torch.equal(x["encoder_cont"], x0["encoder_cont"])  # history keeps real values
        assert torch.equal(x["decoder_cont"][..., others], x0["decoder_cont"][..., others])
    torch.manual_seed(0)
    x, _ = dataset.PmMaskCollate.for_dataset(ds, 0.5)(items)
    hit = ((x["decoder_cont"][..., idx[2]] == 0) & (x0["decoder_cont"][..., idx[2]] == 1)).flatten()
    was_known = (x0["decoder_cont"][..., idx[2]] == 1).flatten()
    assert 0.3 < hit.sum() / was_known.sum() < 0.7

    det = dataset.PmMaskCollate.for_dataset(ds, 0.5, deterministic=True)
    a, b = det(items)[0]["decoder_cont"], det(items)[0]["decoder_cont"]
    assert torch.equal(a, b)
    blanked = (a != x0["decoder_cont"]).flatten(1).any(dim=1)
    assert 0.2 < blanked.float().mean() < 0.8
    assert pickle.loads(pickle.dumps(det)).cols_idx == det.cols_idx


def test_optional_run_pm_in_decoder_scores_with_and_without(panel4_path, tmp_path):
    cfg = _cfg(panel4_path, tmp_path, premarket="optional", only_folds=[1], max_epochs=2)
    run_dir = train.run(cfg)
    cj = json.loads((run_dir / "config.json").read_text())
    assert cj["config"]["premarket_cutoff"] == "09:25" and cj["config"]["test_end"] == 1149
    model = TemporalFusionTransformer.load_from_checkpoint(run_dir / "fold_1" / "best.ckpt")
    assert set(PREMARKET_REALS) <= set(model.hparams.time_varying_reals_decoder)
    assert set(PREMARKET_REALS) <= set(model.dataset_parameters["time_varying_known_reals"])
    assert set(PREMARKET_REALS) & set(model.dataset_parameters["scalers"]) == set(PREMARKET_REALS)
    assert all(model.dataset_parameters["scalers"][c] is None for c in PREMARKET_REALS)
    nopm = run_dir.with_name(run_dir.name + "~nopm")
    assert json.loads((nopm / "config.json").read_text())["view_of"] == run_dir.name
    a = pd.read_parquet(run_dir / "fold_1" / "test_predictions.parquet")
    b = pd.read_parquet(nopm / "fold_1" / "test_predictions.parquet")
    assert len(a) == len(b) and (a[["symbol", "day_idx"]].values == b[["symbol", "day_idx"]].values).all()
    assert not np.allclose(a["q50"], b["q50"])  # the model does read the pre-market
    assert (nopm / "fold_1" / "metrics.json").exists()

    # the live path: pre-market from panel.next.parquet, both forecasts written and graded
    assert predict.main([str(run_dir)]) == 0
    day = json.loads(panel4_path.with_name("panel.meta.json").read_text())["next_session"]["date"]
    with_pm = pd.read_parquet(run_dir / f"forecast_{day}.parquet")
    without = pd.read_parquet(nopm / f"forecast_{day}.parquet")
    assert len(with_pm) == len(without) == 6 and not np.allclose(with_pm["q50"], without["q50"])
    assert (nopm / "actuals.parquet").exists()
    # a next file from another export is refused
    make_next(pd.read_parquet(panel4_path), "other").to_parquet(
        panel4_path.with_name("panel.next.parquet"), index=False)
    with pytest.raises(ValueError, match="export_id"):
        predict.forecast(run_dir)


def test_resume_continues_the_paused_fold(panel4_path, tmp_path, monkeypatch):
    cfg = _cfg(panel4_path, tmp_path, only_folds=[1], max_epochs=4, patience=3, premarket="on")
    run_dir = train.run(cfg, max_hours=1e-9)  # one epoch, then the budget is gone
    assert json.loads((run_dir / "state.json").read_text())["status"] == "paused"
    fold = run_dir / "fold_1"
    assert (fold / "resume.ckpt").exists() and not (fold / "test_predictions.parquet").exists()
    assert not (run_dir / "summary.json").exists()  # a half-trained fold is never scored
    ck = torch.load(fold / "resume.ckpt", map_location="cpu", weights_only=False)
    es_state = next(v for k, v in ck["callbacks"].items() if k.startswith("EarlyStopping"))
    assert "wait_count" in es_state

    seen = []
    orig = EarlyStopping.load_state_dict
    monkeypatch.setattr(EarlyStopping, "load_state_dict",
                        lambda self, sd: (seen.append(dict(sd)), orig(self, sd))[1])
    done = train.run(from_run(run_dir), resume=run_dir)
    assert done == run_dir and json.loads((run_dir / "state.json").read_text())["status"] == "done"
    assert seen and seen[0]["wait_count"] == es_state["wait_count"]
    assert seen[0]["best_score"] == es_state["best_score"]
    assert not list(fold.glob("best-v*.ckpt")) and (fold / "best.ckpt").exists()
    assert not (fold / "resume.ckpt").exists()
    metrics = json.loads((fold / "metrics.json").read_text())
    assert metrics["test"]["rows"] > 0 and 2 <= metrics["epochs"] <= 4  # continued, max_epochs is total
    log = pd.read_csv(fold / "logs" / "metrics.csv")
    assert log["epoch"].max() == metrics["epochs"] - 1
    assert (run_dir / "summary.json").exists()

    with pytest.raises(ValueError, match="panel changed"):
        meta = panel4_path.with_name("panel.meta.json")
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "export_id": "new"}))
        train.run(from_run(run_dir), resume=run_dir)


def test_resume_cli_refuses_model_overrides(tmp_path, capsys):
    with pytest.raises(SystemExit):
        train.main(["--resume", str(tmp_path), "--hidden-size", "64"])
    assert "only --max-hours" in capsys.readouterr().err


def test_stop_file_pauses(panel_path, tmp_path):
    cfg = _cfg(panel_path, tmp_path, only_folds=[1], max_epochs=3)
    cfg.out_dir.mkdir()
    # a STOP file in place before the start still lets the segment train one epoch
    run_dir = train.run(cfg, max_hours=1e-9)
    assert json.loads((run_dir / "state.json").read_text())["status"] == "paused"
    (run_dir / "STOP").write_text("")
    assert train.run(from_run(run_dir), resume=run_dir) == run_dir
    assert json.loads((run_dir / "state.json").read_text())["status"] == "paused"


def test_min_train_sessions_fits_folds(panel_df, tmp_path):
    df = dataset.prepare(panel_df)  # day_idx 1000..1149
    cfg = _cfg(tmp_path, tmp_path, n_folds=0, fold_len=20, early_stop_len=10,
               min_train_sessions=60)
    folds = dataset.make_folds(df, cfg)
    # 150 sessions: 60 train + 10 early-stop leave 80 -> 4 blocks of 20, ending at the panel's end
    assert [(f.es_start, f.test_start, f.test_end) for f in folds] == [
        (1060, 1070, 1089), (1080, 1090, 1109), (1100, 1110, 1129), (1120, 1130, 1149)]
    assert folds[0].es_start - 1000 >= 60
    assert len(dataset.make_folds(df, _cfg(tmp_path, tmp_path, n_folds=0, fold_len=20,
                                           early_stop_len=10, min_train_sessions=60,
                                           test_end=1129))) == 3
    default = dataset.make_folds(df, _cfg(tmp_path, tmp_path))  # unchanged without the flags
    assert len(default) == 2 and default[-1].test_end == 1149


def test_future_rows_raises_on_unfilled_known_real(panel_df):
    df = dataset.prepare(panel_df)
    last = int(df["day_idx"].max())
    nxt = {"day_idx": last + 1, "day_of_week": 2, "month": 3, "is_month_end": 0,
           "is_quarter_end": 0, "sessions_gap_next": 1}
    known = ["day_of_week", "month", "is_month_end", "is_quarter_end", "sessions_gap_next",
             *PREMARKET_REALS]
    with pytest.raises(ValueError, match="no next-session rows"):
        predict.future_rows(df, nxt, known, None)
    with pytest.raises(ValueError, match="lack known inputs"):
        predict.future_rows(df, nxt, known, pd.DataFrame({"symbol": ["S0"], "pm_known": [1.0]}))
    with pytest.raises(ValueError, match="unfilled"):
        predict.future_rows(df, {k: v for k, v in nxt.items()} | {"month": np.nan}, known,
                            make_next(panel_df))
    with pytest.raises(KeyError):  # a calendar field missing from the meta
        predict.future_rows(df, {"day_idx": last + 1}, known, make_next(panel_df))
    fut = predict.future_rows(df, nxt, known, make_next(panel_df))
    assert len(fut) == 6 and (fut["day_idx"] == last + 1).all()
    s5 = fut.set_index("symbol").loc["S5"]  # last symbol: not in the next file -> blank
    assert s5["pm_known"] == 0 and s5["pm_vwap_pct_prev_vwap"] == 0
    assert fut.set_index("symbol").loc["S0", "pm_known"] == 1
    assert predict.future_rows(df, nxt)["month"].eq(3).all()  # calendar-only model


def test_asof_copies_only_known_premarket_columns(panel4_path, tmp_path):
    cfg = _cfg(panel4_path, tmp_path, premarket="on", only_folds=[1])
    run_dir = train.run(cfg)
    panel = pd.read_parquet(panel4_path)
    day = panel.loc[panel["day_idx"] == 1147, "date"].iloc[0].date().isoformat()
    seen = {}
    orig = predict.future_rows

    def spy(df, nxt, known, next_df):
        seen["cols"], seen["next"] = list(next_df.columns), next_df
        return orig(df, nxt, known, next_df)

    predict.future_rows = spy
    try:
        fc, nxt = predict.forecast(run_dir, asof=day)
    finally:
        predict.future_rows = orig
    assert seen["cols"] == ["symbol", *PREMARKET_REALS]  # nothing observed from day d+1
    real = panel[panel["day_idx"] == 1148].set_index("symbol")
    got = seen["next"].set_index("symbol")
    assert np.allclose(got["pm_vol_pct_10d"].fillna(0), real.loc[got.index, "pm_vol_pct_10d"].fillna(0))
    assert nxt["day_idx"] == 1148 and (fc["asof"] == day).all()


def test_grade_ignores_nopm_forecast_files(tmp_path):
    pd.DataFrame({"symbol": ["A", "A"], "date": pd.to_datetime(["2026-10-02", "2026-10-05"]),
                  "today_vwap": [10.0, 11.0]}).to_parquet(tmp_path / "panel.parquet")
    run = tmp_path / "run"
    run.mkdir()
    fc = pd.DataFrame({"date": "2026-10-05", "asof": "2026-10-02", "symbol": ["A"],
                       "today_vwap": [10.0]})
    fc.to_parquet(run / "forecast_2026-10-05.parquet")
    fc.to_parquet(run / "forecast_2026-10-05_moves.parquet")
    fc.to_parquet(run / "forecast_2026-10-05~nopm.parquet")
    out = grade.grade(run, tmp_path / "panel.parquet")
    assert len(out) == 1


def _fake_run(root, name, folds, shift=0.0, seed=0, symbols=("S0", "S1", "S2", "S3"), panel=None):
    """A run dir with fold_k/test_predictions.parquet; q10..q90 around prev_vwap."""
    rng = np.random.default_rng(seed)
    run = root / name
    for k, days in folds.items():
        rows = []
        for s in symbols:
            for d in days:
                prev = 100.0
                act = prev * (1 + rng.normal(0, 0.02))
                rows.append({"symbol": s, "day_idx": d, "prev_vwap": prev, "actual": act})
        f = pd.DataFrame(rows)
        for q, z in ((0.1, -1.28), (0.5, 0.0), (0.9, 1.28)):
            col = f"q{round(q * 100):02d}"
            f[col] = f["prev_vwap"] * (1 + shift + z * 0.02)
        (run / f"fold_{k}").mkdir(parents=True, exist_ok=True)
        f.to_parquet(run / f"fold_{k}" / "test_predictions.parquet", index=False)
    (run / "config.json").write_text(json.dumps(
        {"config": {"panel": str(panel) if panel else "none"}}))
    return run


def test_compare_two_runs(tmp_path):
    folds = {0: range(1100, 1110), 1: range(1110, 1120)}
    a = _fake_run(tmp_path, "a", folds, seed=1)
    # same actuals (seed), but B's quantiles are shifted off: A must be better
    b = _fake_run(tmp_path, "b", folds, seed=1, shift=0.02)
    r = compare.compare(a, b)
    assert r["rows"] == 80 and r["paired"]["days"] == 20 and len(r["per_fold"]) == 2
    assert r["paired"]["diff"] < 0 and r["paired"]["ci_high"] < 0
    assert "A better" in compare.format_report(r)
    assert set(r["coverage"]["a"]) == {"p5_p95", "p10_p90"}
    assert compare.compare(a, b, folds=[1])["rows"] == 40
    assert compare.main([str(a), str(b), "--folds", "0"]) == 0
    assert (a / "compare.json").exists()
    # --drop-split removes series-break segments
    c = _fake_run(tmp_path, "c", folds, seed=1, symbols=("S0", "S0~1", "S1"))
    d = _fake_run(tmp_path, "d", folds, seed=1, symbols=("S0", "S0~1", "S1"), shift=0.02)
    assert compare.compare(c, d)["rows"] == 60
    assert compare.compare(c, d, drop_split=True)["rows"] == 40


def test_compare_splits_earnings_days(tmp_path, panel_df):
    panel = tmp_path / "panel.parquet"
    panel_df.to_parquet(panel, index=False)
    folds = {0: range(1100, 1140)}
    a = _fake_run(tmp_path, "a", folds, panel=panel, symbols=("S0", "S1", "S2"))
    b = _fake_run(tmp_path, "b", folds, panel=panel, symbols=("S0", "S1", "S2"), shift=0.01)
    r = compare.compare(a, b)
    assert {"earnings", "other"} == set(r["by_earnings"])
    assert r["by_earnings"]["earnings"]["rows"] + r["by_earnings"]["other"]["rows"] == 120


def test_overlay_never_crosses_and_never_uses_its_own_fold(tmp_path, panel4_path):
    panel = pd.read_parquet(panel4_path)
    syms = ("S0", "S1", "S2", "S3")
    folds = {0: range(1100, 1115), 1: range(1115, 1130), 2: range(1130, 1145)}
    base = _fake_run(tmp_path, "base", folds, symbols=syms)
    out = overlay.overlay(base, panel4_path)
    assert json.loads((out / "config.json").read_text())["model"] == "overlay"
    assert not (out / "fold_0").exists()  # nothing earlier to learn from
    for k in (1, 2):
        o = pd.read_parquet(out / f"fold_{k}" / "test_predictions.parquet")
        b = pd.read_parquet(base / f"fold_{k}" / "test_predictions.parquet")
        assert (o["q10"] <= o["q50"]).all() and (o["q50"] <= o["q90"]).all()
        assert (o[["symbol", "day_idx", "prev_vwap", "actual"]].values == b[
            ["symbol", "day_idx", "prev_vwap", "actual"]].values).all()

    # fold 1 is scored by a model fitted on fold 0 alone: corrupting fold 2 changes nothing
    # for fold 1, while corrupting fold 0 does
    first = pd.read_parquet(out / "fold_1" / "test_predictions.parquet")
    p2 = base / "fold_2" / "test_predictions.parquet"
    f2 = pd.read_parquet(p2)
    f2["actual"] *= 3
    f2.to_parquet(p2, index=False)
    again = overlay.overlay(base, panel4_path, folds=[1], out=tmp_path / "again")
    assert np.allclose(first["q50"], pd.read_parquet(again / "fold_1" / "test_predictions.parquet")["q50"])
    p0 = base / "fold_0" / "test_predictions.parquet"
    f0 = pd.read_parquet(p0)
    f0["actual"] *= 1.5
    f0.to_parquet(p0, index=False)
    moved = overlay.overlay(base, panel4_path, folds=[1], out=tmp_path / "moved")
    assert not np.allclose(first["q50"], pd.read_parquet(moved / "fold_1" / "test_predictions.parquet")["q50"])
    assert panel is not None


def test_write_panel_helper_schema4_meta(tmp_path, panel_df):
    d = tmp_path / "x"
    d.mkdir()
    path = write_panel(d, panel_df, 4)
    meta = json.loads(path.with_name("panel.meta.json").read_text())
    assert meta["schema_version"] == 4 and meta["export_id"] == "exp-1"
    assert dataset.read_meta(path)["premarket_cutoff"] == "09:25"
