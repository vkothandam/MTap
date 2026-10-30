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

import model_card
from tft_vwap import compare, dataset, grade, overlay, predict, train
from tft_vwap.config import (
    AFTER_HOURS_REALS,
    MACRO_REALS,
    MARKET_REALS,
    PM_FULL_REALS,
    PM_PX_REALS,
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
    for k in ("market", "macro", "session_shape", "after_hours", "premarket", "pm_groups",
              "pm_mask_full", "pm_mask_px", "premarket_cutoff", "dedupe_train", "min_train_sessions"):
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
    assert (full.loc[panel_df.loc[full.index, "pm_full_known"] == 0, "pm_vwap_pct_prev_vwap"] == 0).all()


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


def test_longest_encoder_drops_repeated_samples():
    # pytorch-forecasting emitted WALD's (691..750) sample twice next to missing days on the
    # 5-year panel, which broke the one-to-one merge in score_frame
    class Stub:
        decoded_index = pd.DataFrame({
            "symbol": ["A", "A", "A", "B"], "time_idx_first": [691, 700, 691, 5],
            "time_idx_last": [750] * 3 + [30], "time_idx_first_prediction": [750] * 3 + [30]})

        def filter(self, f, copy=True):
            return self.decoded_index[f(self.decoded_index)]

    kept = dataset.longest_encoder(Stub())
    assert list(kept.index) == [0, 3]


def _batch(panel_df, tmp_path, **kw):
    cfg = _cfg(tmp_path, tmp_path, premarket="optional", **kw)
    df = dataset.prepare(panel_df, extra=cfg.extra_reals)
    ds = dataset.build_datasets(df, cfg, dataset.make_folds(df, cfg)[0])[0]
    return cfg, ds, [ds[i] for i in range(200)]


def test_pm_mask_collate_blanks_groups_decoder_only(panel_df, tmp_path):
    _, ds, items = _batch(panel_df, tmp_path)
    px = [ds.reals.index(c) for c in PM_PX_REALS]
    full = [ds.reals.index(c) for c in PM_FULL_REALS]
    others = [i for i in range(len(ds.reals)) if i not in px + full]
    x0, _ = ds._collate_fn(items)
    assert (x0["decoder_cont"][..., px[-1]] == 1).any()  # raw flags, unscaled
    assert (x0["decoder_cont"][..., full[-1]] == 1).any()

    def blank(x, idx):  # per sample: every value and the flag are 0
        return (x["decoder_cont"][..., idx] == 0).all(dim=-1).all(dim=-1)

    for state, (px_off, full_off) in {"night": (True, True), "px": (False, True),
                                      "full": (False, False)}.items():
        x, _ = dataset.PmMaskCollate.for_state(ds, state)(items)
        assert torch.equal(x["encoder_cont"], x0["encoder_cont"])  # history keeps real values
        assert torch.equal(x["decoder_cont"][..., others], x0["decoder_cont"][..., others])
        assert blank(x, full).all() == full_off
        if px_off:
            assert blank(x, px).all()
        else:
            assert torch.equal(x["decoder_cont"][..., px], x0["decoder_cont"][..., px])
        if not full_off:
            assert torch.equal(x["decoder_cont"][..., full], x0["decoder_cont"][..., full])
    # random masking at the plan's rates: full off for about 50%, prices for 25%, and full is
    # never on where prices are off
    torch.manual_seed(0)
    rates = {"px": 0.25, "full": 0.5}
    x, _ = dataset.PmMaskCollate.for_dataset(ds, rates)(items)
    was_px = x0["decoder_cont"][..., px[-1]].flatten() == 1
    was_full = x0["decoder_cont"][..., full[-1]].flatten() == 1
    px_on = x["decoder_cont"][..., px[-1]].flatten() == 1
    full_on = x["decoder_cont"][..., full[-1]].flatten() == 1
    assert 0.55 < (px_on & was_px).sum() / was_px.sum() < 0.95
    assert 0.3 < (full_on & was_full).sum() / was_full.sum() < 0.7
    assert not (full_on & ~px_on).any()
    # prices-only models have only the px columns
    ds_px = _batch(panel_df, tmp_path, pm_groups="px")[1]
    assert dataset.PmMaskCollate.for_dataset(ds_px, rates).cols_idx.keys() == {"px"}

    det = dataset.PmMaskCollate.for_dataset(ds, rates, deterministic=True)
    a, b = det(items)[0]["decoder_cont"], det(items)[0]["decoder_cont"]
    assert torch.equal(a, b)
    assert 0.2 < (a != x0["decoder_cont"]).flatten(1).any(dim=1).float().mean() < 0.8
    assert pickle.loads(pickle.dumps(det)).cols_idx == det.cols_idx


def test_optional_run_pm_in_decoder_scores_per_state(panel4_path, tmp_path):
    cfg = _cfg(panel4_path, tmp_path, premarket="optional", only_folds=[1], max_epochs=2)
    run_dir = train.run(cfg, card={"label": "R4x", "purpose": "masking test"})
    cj = json.loads((run_dir / "config.json").read_text())
    assert cj["config"]["premarket_cutoff"] == "09:25" and cj["config"]["test_end"] == 1149
    # one card per state: the main run is the 09:45 forecast, siblings carry the suffix
    for suffix, label, forecast in (("", "R4x", "09:45"), ("~px", "R4x~px", "09:30"),
                                    ("~night", "R4x~night", "night")):
        card = model_card.read(run_dir.with_name(run_dir.name + suffix))
        assert (card["label"], card["forecast"], card["status"]) == (label, forecast, "done")
        assert card["purpose"] == "masking test" and card["result"].startswith("Skill ")
    model = TemporalFusionTransformer.load_from_checkpoint(run_dir / "fold_1" / "best.ckpt")
    assert set(PREMARKET_REALS) <= set(model.hparams.time_varying_reals_decoder)
    assert set(PREMARKET_REALS) <= set(model.dataset_parameters["time_varying_known_reals"])
    assert all(model.dataset_parameters["scalers"][c] is None for c in PREMARKET_REALS)
    sib = {s: run_dir.with_name(run_dir.name + f"~{s}") for s in ("night", "px")}
    a = pd.read_parquet(run_dir / "fold_1" / "test_predictions.parquet")
    frames = {}
    for s, d in sib.items():
        assert json.loads((d / "config.json").read_text())["view_of"] == run_dir.name
        frames[s] = pd.read_parquet(d / "fold_1" / "test_predictions.parquet")
        assert (frames[s][["symbol", "day_idx"]].values == a[["symbol", "day_idx"]].values).all()
        assert (d / "fold_1" / "metrics.json").exists()
    assert not np.allclose(a["q50"], frames["night"]["q50"])  # the model does read the pre-market
    assert not np.allclose(frames["px"]["q50"], frames["night"]["q50"])
    assert not np.allclose(a["q50"], frames["px"]["q50"])

    # variable importance: per fold file and a per-group table in the summary
    imp = json.loads((run_dir / "fold_1" / "importance.json").read_text())
    assert set(imp["variables"]) == {"encoder", "decoder", "static"}
    assert {"premarket_px", "premarket_full", "base"} <= set(imp["groups"]["decoder"])
    assert abs(sum(imp["groups"]["decoder"].values()) - 1) < 1e-3
    summ = json.loads((run_dir / "summary.json").read_text())
    assert "premarket_px" in summ["importance_by_group"]["mean"]["decoder"]

    # the live path: pre-market from panel.next.parquet; every state written and graded
    assert predict.main([str(run_dir)]) == 0
    day = json.loads(panel4_path.with_name("panel.meta.json").read_text())["next_session"]["date"]
    live = {"full": pd.read_parquet(run_dir / f"forecast_{day}.parquet"),
            **{s: pd.read_parquet(d / f"forecast_{day}.parquet") for s, d in sib.items()}}
    assert all(len(f) == 6 for f in live.values())
    assert not np.allclose(live["full"]["q50"], live["night"]["q50"])
    assert not np.allclose(live["px"]["q50"], live["night"]["q50"])
    assert not np.allclose(live["full"]["q50"], live["px"]["q50"])
    for d in (run_dir, *sib.values()):
        assert (d / "actuals.parquet").exists()
    # --state night reproduces the same forecast; --symbols-file keeps the same rows
    assert predict.main([str(run_dir), "--state", "night"]) == 0
    again = pd.read_parquet(sib["night"] / f"forecast_{day}.parquet")
    assert np.allclose(again["q50"], live["night"]["q50"])
    sf = tmp_path / "syms.txt"
    sf.write_text("S1\nS3\n")
    assert predict.main([str(run_dir), "--state", "px", "--symbols-file", str(sf)]) == 0
    sub = pd.read_parquet(sib["px"] / f"forecast_{day}_subset.parquet")
    assert list(sub["symbol"]) == ["S1", "S3"]
    ref = live["px"].set_index("symbol").loc[["S1", "S3"]]
    assert np.allclose(sub.set_index("symbol")["q50"], ref["q50"], atol=1e-6)
    assert not (sib["px"] / f"forecast_{day}_subset_moves.parquet").exists()
    # the morning's shortlist CSV (symbol, score, reasons) works as a symbols file too
    sl = tmp_path / "shortlist.csv"
    sl.write_text("symbol,score,reasons\nS1,2.0,top\nS3,1.0,move\n")
    assert predict.read_symbols(sl) == ["S1", "S3"]
    # a next file from another export is refused
    make_next(pd.read_parquet(panel4_path), "other").to_parquet(
        panel4_path.with_name("panel.next.parquet"), index=False)
    with pytest.raises(ValueError, match="export_id"):
        predict.forecast(run_dir)


def test_force_state_blanks_by_group():
    nxt = pd.DataFrame({"symbol": ["A", "B"], "pm_last_pct_prev_vwap": [0.01, 0.02],
                        "open_pct_prev_vwap": [0.0, 0.0], "pm_px_known": [1.0, 0.0],
                        "pm_vwap_pct_prev_vwap": [0.03, 0.04], "pm_vol_pct_10d": [1.0, 2.0],
                        "pm_range_pct": [0.1, 0.2], "pm_full_known": [1.0, 1.0]})
    cols = list(PREMARKET_REALS)
    assert (predict.force_state(nxt, "night", cols)[cols] == 0).all().all()
    px = predict.force_state(nxt, "px", cols).set_index("symbol")
    assert px.loc["A", "pm_px_known"] == 1 and px.loc["A", "pm_last_pct_prev_vwap"] == 0.01
    assert (px[list(PM_FULL_REALS)] == 0).all().all()
    full = predict.force_state(nxt, "full", cols).set_index("symbol")
    assert full.loc["A", "pm_full_known"] == 1 and full.loc["A", "pm_vol_pct_10d"] == 1.0
    assert full.loc["B", "pm_full_known"] == 0 and full.loc["B", "pm_vwap_pct_prev_vwap"] == 0


def test_prices_only_model_has_no_full_group(panel4_path, tmp_path):
    cfg = _cfg(panel4_path, tmp_path, premarket="optional", pm_groups="px", only_folds=[1])
    assert cfg.known_reals[-3:] == list(PM_PX_REALS) and cfg.scoring_states == ["night", "px"]
    run_dir = train.run(cfg)
    assert (run_dir.with_name(run_dir.name + "~night") / "fold_1" / "test_predictions.parquet").exists()
    assert not run_dir.with_name(run_dir.name + "~px").exists()
    assert predict.main([str(run_dir)]) == 0


def test_resume_continues_the_paused_fold(panel4_path, tmp_path, monkeypatch):
    cfg = _cfg(panel4_path, tmp_path, only_folds=[1], max_epochs=4, patience=3, premarket="on")
    run_dir = train.run(cfg, max_hours=1e-9, card={"label": "R9"})  # one epoch, then paused
    assert json.loads((run_dir / "state.json").read_text())["status"] == "paused"
    assert model_card.read(run_dir)["status"] == "paused"
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
    card = model_card.read(run_dir)
    assert card["label"] == "R9" and card["status"] == "done" and card["forecast"] == "09:45"
    assert seen and seen[0]["wait_count"] == es_state["wait_count"]
    assert seen[0]["best_score"] == es_state["best_score"]
    assert not list(fold.glob("best-v*.ckpt")) and (fold / "best.ckpt").exists()
    assert not (fold / "resume.ckpt").exists()
    metrics = json.loads((fold / "metrics.json").read_text())
    assert metrics["test"]["rows"] > 0 and 2 <= metrics["epochs"] <= 4  # continued, max_epochs is total
    log = pd.read_csv(fold / "logs" / "metrics.csv")
    assert log["epoch"].max() == metrics["epochs"] - 1
    assert (run_dir / "summary.json").exists()

    with pytest.raises(ValueError, match="changed since the run started"):
        meta = panel4_path.with_name("panel.meta.json")
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "export_id": "new"}))
        train.run(from_run(run_dir), resume=run_dir)


def test_resume_cli_refuses_model_overrides(tmp_path, capsys):
    with pytest.raises(SystemExit):
        train.main(["--resume", str(tmp_path), "--hidden-size", "64"])
    assert "only --max-hours" in capsys.readouterr().err


def test_resume_cli_takes_num_workers(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(train, "from_run", lambda d, **kw: seen.setdefault("kw", kw))
    monkeypatch.setattr(train, "run", lambda cfg, **kw: None)
    assert train.main(["--resume", str(tmp_path), "--num-workers", "4"]) == 0
    assert seen.pop("kw") == {"num_workers": 4}
    assert train.main(["--resume", str(tmp_path)]) == 0
    assert seen["kw"] == {"num_workers": None}  # the run's saved count
    assert train.main(["--resume", str(tmp_path), "--label", "R9"]) == 0  # card flags are fine


def test_paused_run_resumes_with_workers(panel_path, tmp_path):
    cfg = _cfg(panel_path, tmp_path, only_folds=[1], max_epochs=2, patience=3)
    run_dir = train.run(cfg, max_hours=1e-9)
    assert json.loads((run_dir / "state.json").read_text())["status"] == "paused"
    resumed = from_run(run_dir, num_workers=2)
    assert resumed.num_workers == 2
    assert train.run(resumed, resume=run_dir) == run_dir
    assert json.loads((run_dir / "state.json").read_text())["status"] == "done"
    assert json.loads((run_dir / "fold_1" / "metrics.json").read_text())["test"]["rows"] > 0


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
        predict.future_rows(df, nxt, known, pd.DataFrame({"symbol": ["S0"], "pm_px_known": [1.0]}))
    with pytest.raises(ValueError, match="unfilled"):
        predict.future_rows(df, {k: v for k, v in nxt.items()} | {"month": np.nan}, known,
                            make_next(panel_df))
    with pytest.raises(KeyError):  # a calendar field missing from the meta
        predict.future_rows(df, {"day_idx": last + 1}, known, make_next(panel_df))
    fut = predict.future_rows(df, nxt, known, make_next(panel_df))
    assert len(fut) == 6 and (fut["day_idx"] == last + 1).all()
    s5 = fut.set_index("symbol").loc["S5"]  # last symbol: not in the next file -> blank
    assert s5["pm_px_known"] == 0 and s5["pm_full_known"] == 0 and s5["pm_vwap_pct_prev_vwap"] == 0
    assert fut.set_index("symbol").loc["S0", "pm_px_known"] == 1
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


def test_compare_matches_on_date_across_panels_and_states(tmp_path, panel_df):
    # run B numbers its days differently (another panel) but has the same dates
    folds = {0: range(1100, 1110)}
    a = _fake_run(tmp_path, "a", folds, seed=1)
    b = _fake_run(tmp_path, "b", folds, seed=1, shift=0.02)
    night = _fake_run(tmp_path, "a~night", folds, seed=1, shift=0.02)
    for run, off in ((a, 0), (b, 500), (night, 0)):
        p = run / "fold_0" / "test_predictions.parquet"
        f = pd.read_parquet(p)
        f["date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(f["day_idx"] - 1100, unit="D")
        f["day_idx"] += off
        f.to_parquet(p, index=False)
    r = compare.compare(a, b)
    assert r["matched_on"] == "(symbol, date)" and r["rows"] == 40
    # states: A~night is a sibling run dir, picked with --state
    rs = compare.compare(a, b, state_a="night")
    assert rs["a"].endswith("a~night") and rs["paired"]["diff"] > -1
    with pytest.raises(FileNotFoundError, match="px"):
        compare.compare(a, b, state_a="px")
    assert compare.main([str(a), str(b), "--state", "night"]) == 0


def test_compare_only_pm_keeps_rows_with_the_data(tmp_path, panel_df):
    panel = tmp_path / "panel.parquet"
    panel_df.to_parquet(panel, index=False)
    folds = {0: range(1100, 1140)}
    a = _fake_run(tmp_path, "a", folds, panel=panel, symbols=("S0", "S1", "S2"))
    b = _fake_run(tmp_path, "b", folds, panel=panel, symbols=("S0", "S1", "S2"), shift=0.01)
    r = compare.compare(a, b, only_pm="full")
    known = panel_df[(panel_df["symbol"].isin(["S0", "S1", "S2"]))
                     & panel_df["day_idx"].between(1100, 1139) & (panel_df["pm_full_known"] == 1)]
    assert r["rows"] == len(known) and r["only_pm"] == "full"
    assert compare.compare(a, b, only_pm="px")["rows"] > r["rows"]  # prices exist more often
    assert "only rows with pre-market full" in compare.format_report(r)


def test_overlay_never_crosses_and_never_uses_its_own_fold(tmp_path, panel4_path):
    syms = ("S0", "S1", "S2", "S3")
    folds = {0: range(1100, 1115), 1: range(1115, 1130), 2: range(1130, 1145)}
    base = _fake_run(tmp_path, "base", folds, symbols=syms)
    out = overlay.overlay(base, panel4_path, inputs="px")
    assert out.name == "base-overlay-px"
    cj = json.loads((out / "config.json").read_text())
    assert cj["model"] == "overlay" and cj["inputs"] == "px" and cj["thin_folds"] == [1]
    assert (out / "summary.json").exists()
    assert json.loads((out / "fold_1" / "metrics.json").read_text())["thin"] is True
    assert json.loads((out / "fold_2" / "metrics.json").read_text())["thin"] is False
    assert not (out / "fold_0").exists()  # nothing earlier to learn from
    for k in (1, 2):
        o = pd.read_parquet(out / f"fold_{k}" / "test_predictions.parquet")
        b = pd.read_parquet(base / f"fold_{k}" / "test_predictions.parquet")
        assert (o["q10"] <= o["q50"]).all() and (o["q50"] <= o["q90"]).all()
        assert (o[["symbol", "day_idx", "prev_vwap", "actual"]].values == b[
            ["symbol", "day_idx", "prev_vwap", "actual"]].values).all()
    # R6f uses more columns than R6p, writes elsewhere
    f = overlay.overlay(base, panel4_path, inputs="full", card={"label": "R6f"})
    assert f.name == "base-overlay-full" and overlay.INPUTS["px"] == PM_PX_REALS
    card = model_card.read(f)
    assert (card["label"], card["forecast"], card["kind"]) == ("R6f", "09:45", "Overlay")
    assert model_card.read(out)["forecast"] == "09:30" and "thin" in model_card.read(out)["folds"]

    # fold 1 is scored by a model fitted on fold 0 alone: corrupting fold 2 changes nothing
    # for fold 1, while corrupting fold 0 does
    first = pd.read_parquet(out / "fold_1" / "test_predictions.parquet")
    p2 = base / "fold_2" / "test_predictions.parquet"
    f2 = pd.read_parquet(p2)
    f2["actual"] *= 3
    f2.to_parquet(p2, index=False)
    again = overlay.overlay(base, panel4_path, folds=[1], out=tmp_path / "again", inputs="px")
    assert np.allclose(first["q50"], pd.read_parquet(again / "fold_1" / "test_predictions.parquet")["q50"])
    p0 = base / "fold_0" / "test_predictions.parquet"
    f0 = pd.read_parquet(p0)
    f0["actual"] *= 1.5
    f0.to_parquet(p0, index=False)
    moved = overlay.overlay(base, panel4_path, folds=[1], out=tmp_path / "moved", inputs="px")
    assert not np.allclose(first["q50"], pd.read_parquet(moved / "fold_1" / "test_predictions.parquet")["q50"])


def test_recal_fixes_biased_levels_without_seeing_its_own_fold(tmp_path, panel4_path):
    from tft_vwap import recal
    folds = {k: range(1100 + 40 * k, 1140 + 40 * k) for k in range(3)}
    syms = tuple(f"S{i}" for i in range(6))
    # every quantile 1% too high: about 69% of actuals fall below q50, not 50%
    base = _fake_run(tmp_path, "base", folds, shift=0.01, symbols=syms, panel=panel4_path)
    raw = pd.read_parquet(base / "fold_2" / "test_predictions.parquet")
    out = recal.run(base, window=2, by=(), card={"purpose": "test"})
    assert out.name == "base-cal" and not (out / "fold_0").exists()
    cal = pd.read_parquet(out / "fold_2" / "test_predictions.parquet")
    below = lambda f, c: (f["actual"] <= f[c]).mean()
    assert abs(below(raw, "q50") - 0.5) > 0.1 and abs(below(cal, "q50") - 0.5) < 0.08
    assert (cal["q10"] <= cal["q50"]).all() and (cal["q50"] <= cal["q90"]).all()
    assert (cal[["symbol", "day_idx", "actual"]].values == raw[["symbol", "day_idx", "actual"]].values).all()
    m = json.loads((out / "fold_1" / "metrics.json").read_text())
    assert m["calibrated_on_folds"] == [0] and m["thin"] is True
    assert json.loads((out / "fold_2" / "metrics.json").read_text())["calibrated_on_folds"] == [0, 1]
    card = model_card.read(out)
    assert card["label"] == "base-cal" and card["kind"] == "Recal" and card["status"] == "done"
    # fold 2's own actuals never move its shifts
    p2 = base / "fold_2" / "test_predictions.parquet"
    f2 = raw.copy()
    f2["actual"] *= 1.5
    f2.to_parquet(p2, index=False)
    again = recal.run(base, window=2, by=(), out=tmp_path / "again")
    assert np.allclose(cal["q50"], pd.read_parquet(again / "fold_2" / "test_predictions.parquet")["q50"])
    # by pm and width: groups below MIN_ROWS fall back to the all-rows shift, so it still runs
    grouped = recal.run(base, window=2, by=("pm", "width"), out=tmp_path / "grouped")
    g = pd.read_parquet(grouped / "fold_2" / "test_predictions.parquet")
    assert np.allclose(g["q50"], pd.read_parquet(again / "fold_2" / "test_predictions.parquet")["q50"])


def test_recal_groups_get_their_own_shift(monkeypatch):
    from tft_vwap import recal
    monkeypatch.setattr(recal, "MIN_ROWS", 10)
    rng = np.random.default_rng(0)
    frames, pm = {}, {}
    for k in range(3):
        n = 400
        flag = np.arange(n) % 2 == 0
        prev = np.full(n, 100.0)
        # pm rows run 2% above their forecast, the others 2% below
        act = prev * (1 + np.where(flag, 0.02, -0.02) + rng.normal(0, 0.005, n))
        f = pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "day_idx": k,
                          "prev_vwap": prev, "actual": act, "q50": prev})
        frames[k], pm[k] = f, flag
    out, used = recal.recalibrate(frames, [0.5], ["q50"], window=2, by=("pm",), pm=pm)
    f = out[2]
    assert used[2] == [0, 1]
    assert np.allclose(f.loc[pm[2], "q50"], 102, atol=0.2)
    assert np.allclose(f.loc[~pm[2], "q50"], 98, atol=0.2)


def test_recal_rolling_uses_only_the_sessions_before_each_day(tmp_path, panel4_path, monkeypatch):
    from tft_vwap import recal
    monkeypatch.setattr(recal, "MIN_ROWS", 12)
    folds = {k: range(1100 + 40 * k, 1140 + 40 * k) for k in range(2)}
    syms = tuple(f"S{i}" for i in range(6))
    base = _fake_run(tmp_path, "base", folds, shift=0.01, symbols=syms, panel=panel4_path)
    raw = pd.concat([pd.read_parquet(base / f"fold_{k}" / "test_predictions.parquet") for k in folds])
    out = recal.run(base, by=(), sessions=5, card={"purpose": "test"})
    assert out.name == "base-cal5s"
    cal = pd.concat([pd.read_parquet(out / f"fold_{k}" / "test_predictions.parquet") for k in folds])
    # the first days lack 12 calibration rows (2 sessions x 6 symbols), so they are left out
    assert cal["day_idx"].min() == 1102 and json.loads((out / "config.json").read_text())["sessions"] == 5
    below = lambda f, c: (f["actual"] <= f[c]).mean()
    assert abs(below(raw, "q50") - 0.5) > 0.1 and abs(below(cal, "q50") - 0.5) < 0.1
    assert (cal["q10"] <= cal["q50"]).all() and (cal["q50"] <= cal["q90"]).all()
    assert json.loads((out / "fold_1" / "metrics.json").read_text())["calibrated_on_sessions"] == 5
    assert model_card.read(out)["label"] == "base-cal5s"
    # day d's own actuals (and later ones) never move day d; the 5 days before it do
    d = 1160
    for k, f in ((k, pd.read_parquet(base / f"fold_{k}" / "test_predictions.parquet")) for k in folds):
        f.loc[f["day_idx"] >= d, "actual"] *= 1.5
        f.to_parquet(base / f"fold_{k}" / "test_predictions.parquet", index=False)
    again = pd.read_parquet(recal.run(base, by=(), sessions=5, out=tmp_path / "again")
                            / "fold_1" / "test_predictions.parquet")
    cal = pd.read_parquet(out / "fold_1" / "test_predictions.parquet")
    before = cal[cal["day_idx"] <= d].sort_values(["day_idx", "symbol"])["q50"].to_numpy()
    assert np.allclose(before, again[again["day_idx"] <= d].sort_values(["day_idx", "symbol"])["q50"])
    after = cal[cal["day_idx"] == d + 1].sort_values("symbol")["q50"].to_numpy()
    assert not np.allclose(after, again[again["day_idx"] == d + 1].sort_values("symbol")["q50"])


def test_recal_rolling_groups_get_their_own_shift(monkeypatch):
    from tft_vwap import recal
    monkeypatch.setattr(recal, "MIN_ROWS", 10)
    rng = np.random.default_rng(0)
    n, flag = 200, np.arange(200) % 2 == 0
    frames = {0: [], 1: []}
    for d in range(10):
        prev = np.full(n, 100.0)
        act = prev * (1 + np.where(flag, 0.02, -0.02) + rng.normal(0, 0.005, n))
        frames[d // 5].append(pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "day_idx": d,
                                            "prev_vwap": prev, "actual": act, "q50": prev}))
    frames = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    pm = {k: np.tile(flag, len(v) // n) for k, v in frames.items()}
    out = recal.recalibrate_rolling(frames, [0.5], ["q50"], sessions=3, by=("pm",), pm=pm)
    f, m = out[1], np.tile(flag, 5)
    assert sorted(out) == [0, 1] and out[0]["day_idx"].min() == 1
    assert np.allclose(f.loc[m, "q50"], 102, atol=0.2) and np.allclose(f.loc[~m, "q50"], 98, atol=0.2)


def test_gbm_runs_the_contract_and_loads_in_compare(tmp_path, panel4_path):
    from tft_vwap import gbm
    cfg = _cfg(panel4_path, tmp_path, n_folds=2, fold_len=10, early_stop_len=5,
               premarket="on", market=True, after_hours=True)
    out = gbm.run(cfg, max_iter=10, card={"label": "R7f", "summary": "HGB + full pre-market"})
    card = model_card.read(out)
    assert (card["label"], card["forecast"], card["kind"], card["status"]) == (
        "R7f", "09:45", "GBM", "done")
    assert card["summary"] == "HGB + full pre-market" and card["result"].startswith("Skill ")
    cj = json.loads((out / "config.json").read_text())
    assert cj["model"] == "gbm" and len(cj["folds"]) == 2 and (out / "summary.json").exists()
    for k in (0, 1):
        f = pd.read_parquet(out / f"fold_{k}" / "test_predictions.parquet")
        q = f[list(compare.qlevels(f).values())].to_numpy()
        assert (np.diff(q, axis=1) >= 0).all()  # sorted: quantiles never cross
        assert {"symbol", "day_idx", "prev_vwap", "actual", "date"} <= set(f.columns)
        assert (out / f"fold_{k}" / "metrics.json").exists()
    # same folds and rows as a TFT-style fake run on the same days: compare pairs them
    r = compare.compare(out, out)
    assert r["rows"] > 0 and r["paired"]["diff"] == 0

    # features: lags of vwap_pct_prev_day are the past values; pm flags come from row d
    df = dataset.prepare(pd.read_parquet(panel4_path), extra=cfg.extra_reals)
    feat = gbm.build_features(df, cfg)
    row = feat.iloc[200]
    s = df[df["symbol"] == row["symbol"]].set_index("day_idx")
    d = int(row["day_idx"])
    assert row["f_pm_full_known"] == s.loc[d, "pm_full_known"]
    assert row["f_vwap_pct_prev_day"] == s.loc[d - 1, "vwap_pct_prev_day"]
    if (d - 2) in s.index:
        assert row["f_ret_lag1"] == s.loc[d - 2, "vwap_pct_prev_day"] or np.isnan(row["f_ret_lag1"])
    assert row["actual"] == s.loc[d - 1, "target_tomorrow_vwap"]


def test_gbm_optional_scores_both_states_from_one_model(tmp_path, panel4_path):
    from tft_vwap import gbm
    cfg = _cfg(panel4_path, tmp_path, n_folds=2, fold_len=10, early_stop_len=5,
               premarket="optional", pm_groups="px,full", after_hours=True)
    out = gbm.run(cfg, max_iter=10, tail_iter=15, open_all=True, vol=True,
                  card={"label": "R7o"})
    px = out.with_name(out.name + "~px")
    assert (model_card.read(out)["label"], model_card.read(out)["forecast"]) == ("R7o", "09:45")
    assert (model_card.read(px)["label"], model_card.read(px)["forecast"]) == ("R7o~px", "09:30")
    assert compare.state_dir(out, "px") == px and (px / "summary.json").exists()
    cj = json.loads((out / "config.json").read_text())
    assert (cj["tail_iter"], cj["open_all"], cj["vol"]) == (15, True, True)
    full = pd.read_parquet(out / "fold_1" / "test_predictions.parquet")
    pxf = pd.read_parquet(px / "fold_1" / "test_predictions.parquet")
    assert full[["symbol", "day_idx"]].equals(pxf[["symbol", "day_idx"]])

    # one fit per fold: pre-market training rows twice (as is + full blank), scored twice
    df = dataset.prepare(pd.read_parquet(panel4_path), extra=cfg.extra_reals)
    feat = gbm.build_features(df, cfg, open_all=True, vol=True)
    calls = []
    real = gbm.fit_predict
    gbm.fit_predict = lambda tr, tg, *a, **k: calls.append((tr, tg)) or real(tr, tg, *a, **k)
    try:
        fold = dataset.make_folds(df, cfg)[1]
        gbm.run_fold(feat, df, cfg, fold, 5, None)
    finally:
        gbm.fit_predict = real
    (tr, (t_full, t_px)), = calls
    n_train = (feat["day_idx"] < fold.es_start).sum()
    n_pm = ((feat["day_idx"] < fold.es_start) & (feat["f_pm_full_known"] == 1)).sum()
    assert n_pm > 0 and len(tr) == n_train + n_pm
    assert (t_px[gbm.FULL_COLS] == 0).all().all() and t_full["f_pm_full_known"].sum() > 0

    # the 09:30 view blanks the full group only; the open and vol features are past/09:30 info
    blank = gbm.blank_full(feat)
    assert (blank[gbm.FULL_COLS] == 0).all().all()
    assert blank["f_pm_last_pct_prev_vwap"].equals(feat["f_pm_last_pct_prev_vwap"])
    row = feat.iloc[200]
    s = df[df["symbol"] == row["symbol"]].set_index("day_idx")
    d = int(row["day_idx"])
    assert row["f_open_all"] == pytest.approx(
        np.clip(s.loc[d, "gap_open_vs_prev_vwap_pct"], -0.3, 0.3))
    past = s.loc[:d - 1, "vwap_pct_prev_day"].tail(5)
    assert row["f_vol5"] == pytest.approx(past.std())  # through d-1 only


def test_gbm_tail_iter_only_for_outer_levels(monkeypatch):
    from tft_vwap import gbm
    seen = {}

    class Fake:
        def __init__(self, loss, quantile, max_iter, **kw):
            seen[quantile] = max_iter

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(gbm, "HistGradientBoostingRegressor", Fake)
    t = pd.DataFrame({"f_a": [1.0, 2.0], "actual": [1.0, 1.0], "prev_vwap": [1.0, 1.0]})
    gbm.fit_predict(t, t, [0.05, 0.15, 0.2, 0.5, 0.8, 0.85, 0.95], max_iter=200, tail_iter=600)
    assert seen == {0.05: 600, 0.15: 600, 0.2: 200, 0.5: 200, 0.8: 200, 0.85: 600, 0.95: 600}


def test_write_panel_helper_schema4_meta(tmp_path, panel_df):
    d = tmp_path / "x"
    d.mkdir()
    path = write_panel(d, panel_df, 4)
    meta = json.loads(path.with_name("panel.meta.json").read_text())
    assert meta["schema_version"] == 4 and meta["export_id"] == "exp-1"
    assert dataset.read_meta(path)["premarket_cutoff"] == "09:25"


def test_time_left_range():
    from tft_vwap.progress import estimate_left

    # epoch 3 of 15 done, wait 1 of patience 3, 100 s/epoch, 2 folds after, none finished yet
    left = estimate_left(epoch_secs=100, completed=3, frac_left=0, max_epochs=15, wait=1,
                         patience=3, folds_after=2, prior_epochs=[], prior_secs=[])
    assert left["fold"] == [200, 700, 1200]  # earliest: 2 more epochs without improvement
    assert left["run"] == [200 + 2 * 400, 700 + 2 * 950, 1200 + 2 * 1500]
    # finished folds took 8 epochs / 900 s on average: the estimate follows them
    left = estimate_left(epoch_secs=100, completed=3, frac_left=0, max_epochs=15, wait=0,
                         patience=3, folds_after=1, prior_epochs=[8], prior_secs=[900])
    assert left["fold"][1] == 500 and left["run"][1] == 500 + 900
    # mid-epoch: half of the current epoch remains
    left = estimate_left(epoch_secs=100, completed=14, frac_left=0.5, max_epochs=15, wait=0,
                         patience=3, folds_after=0, prior_epochs=[], prior_secs=[])
    assert left["fold"] == [50, 50, 50]


def test_progress_written_and_status(panel_path, tmp_path, capsys):
    from tft_vwap import progress

    cfg = _cfg(panel_path, tmp_path, only_folds=[0, 1], max_epochs=3)
    run_dir = train.run(cfg, max_hours=1e-9)  # pauses after fold 0's first epoch
    p = json.loads((run_dir / "progress.json").read_text())
    assert p["fold"] == 0 and p["fold_pos"] == 1 and p["folds"] == 2 and p["epoch"] == 1
    lo, mid, hi = p["run_left_seconds"]
    assert 0 < lo <= mid <= hi and p["fold_left_seconds"][2] <= hi
    assert "[progress] fold 0 (1/2) epoch 1/3" in capsys.readouterr().out
    text = progress.status(run_dir)
    assert "paused" in text and "run left" in text
    progress.main([str(cfg.out_dir)])
    assert run_dir.name in capsys.readouterr().out
    train.run(from_run(run_dir), resume=run_dir)
    assert not (run_dir / "progress.json").exists()
    assert progress.status(run_dir).endswith(": done")


def test_constant_statics_dropped(panel_df, tmp_path):
    df = dataset.prepare(panel_df)  # conftest: every stock is issue_type CS
    cfg = _cfg(tmp_path, tmp_path)
    fold = dataset.make_folds(df, cfg)[0]
    train = dataset.build_datasets(df, cfg, fold)[0]
    assert "issue_type" not in train.static_categoricals and "symbol" in train.static_categoricals
    kept = dataset.build_datasets(df, _cfg(tmp_path, tmp_path, drop_constant_statics=False), fold)[0]
    assert "issue_type" in kept.static_categoricals


def test_run_saved_before_drop_constant_statics_kept_them(tmp_path):
    saved = {k: v for k, v in TrainConfig().to_dict().items() if k != "drop_constant_statics"}
    (tmp_path / "config.json").write_text(json.dumps({"config": saved}))
    assert from_run(tmp_path).drop_constant_statics is False
