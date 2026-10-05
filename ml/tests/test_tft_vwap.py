import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tft_vwap import calibration, dataset, evaluate, predict, report, train
from tft_vwap.config import TrainConfig, from_run


def _cfg(panel, tmp_path, **kw):
    base = {"panel": panel, "out_dir": tmp_path / "runs", "n_folds": 2, "fold_len": 10,
            "early_stop_len": 5, "encoder_length": 20, "min_encoder_length": 5,
            "batch_size": 32, "hidden_size": 8, "attention_head_size": 1,
            "hidden_continuous_size": 4, "max_epochs": 1, "limit_train_batches": 2,
            "accelerator": "cpu"}
    return TrainConfig(**{**base, **kw})


def test_prepare_types_fills_and_subsamples(panel_df):
    raw = panel_df.copy()
    raw.loc[0, "today_vwap"] = 0.0  # unusable for a log target
    raw.loc[1, "industry_code"] = None
    df = dataset.prepare(raw)
    assert len(df) == len(raw) - 1
    assert df["vwap_pct_prev_day"].notna().all()
    assert (df["industry_code"] == "unknown").sum() >= 1
    assert df["volume_velocity"].max() < np.log1p(raw["volume_velocity"].max()) + 1e-9
    sub = dataset.prepare(panel_df, symbols=2, seed=1)
    assert sub["symbol"].nunique() == 2


def test_read_meta_rejects_unknown_schema(panel_path):
    panel_path.with_name("panel.meta.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version"):
        dataset.read_meta(panel_path)


def test_make_folds_tile_the_tail(panel_df, tmp_path):
    cfg = _cfg(tmp_path / "p.parquet", tmp_path, n_folds=3)
    folds = dataset.make_folds(dataset.prepare(panel_df), cfg)
    assert [(f.es_start, f.test_start, f.test_end) for f in folds] == [
        (1115, 1120, 1129), (1125, 1130, 1139), (1135, 1140, 1149)]
    with pytest.raises(ValueError, match="history"):
        dataset.make_folds(dataset.prepare(panel_df), _cfg(tmp_path, tmp_path, n_folds=13))


def _decoder_days(ds):
    days = []
    for x, _ in ds.to_dataloader(train=False, batch_size=256):
        days.append(x["decoder_time_idx"].flatten())
    return np.concatenate([d.numpy() for d in days])


def test_datasets_never_train_on_held_out_targets(panel_df, tmp_path):
    df = dataset.prepare(panel_df)
    cfg = _cfg(tmp_path / "p.parquet", tmp_path)
    fold = dataset.make_folds(df, cfg)[0]
    tr, es, te = dataset.build_datasets(df, cfg, fold)
    assert _decoder_days(tr).max() < fold.es_start
    es_days = _decoder_days(es)
    assert es_days.min() >= fold.es_start and es_days.max() < fold.test_start
    te_days = _decoder_days(te)
    assert te_days.min() >= fold.test_start and te_days.max() <= fold.test_end
    # one forecast per symbol/day (6 symbols x 10 days; S1's gap is outside the fold)
    assert len(te) == len(te_days) == 60


def test_score_frame_keeps_real_rows_and_converts_returns(panel_df, tmp_path):
    df = dataset.prepare(panel_df)
    # S1 has no bar on 1050 (so row 1049 was dropped too): forecasts for 1050 (gap-filled
    # decoder) and 1051 (no prior row for the baseline) must be discarded.
    index = pd.DataFrame({"symbol": ["S1", "S1", "S0"], "day_idx": [1050, 1051, 1060]})
    q = np.tile(np.array([[0.01, 0.0, -0.01]]), (3, 1))  # crossed: sorted on scoring
    cfg = _cfg(tmp_path / "p.parquet", tmp_path, target="return", quantiles=[0.1, 0.5, 0.9])
    frame = evaluate.score_frame(index, q, df, cfg)
    assert frame[["symbol", "day_idx"]].values.tolist() == [["S0", 1060]]
    row = df[(df.symbol == "S0") & (df.day_idx == 1059)].iloc[0]
    assert frame.loc[0, "prev_vwap"] == pytest.approx(row.today_vwap)
    assert frame.loc[0, "actual"] == pytest.approx(row.target_tomorrow_vwap)
    assert frame.loc[0, "q50"] == pytest.approx(row.today_vwap)
    assert frame.loc[0, "q90"] == pytest.approx(row.today_vwap * 1.01)


def test_metrics_perfect_and_naive():
    prev = np.array([10.0, 20.0, 30.0, 40.0])
    actual = np.array([11.0, 19.0, 33.0, 38.0])
    base = pd.DataFrame({"symbol": list("abab"), "prev_vwap": prev, "actual": actual})
    perfect = base.assign(q10=actual * 0.9, q50=actual, q90=actual * 1.1)
    m = evaluate.metrics(perfect, [0.1, 0.5, 0.9])
    assert m["mape"] == 0 and m["directional_accuracy"] == 1 and m["interval_coverage"] == 1
    assert m["mape_skill"] == pytest.approx(1.0) and m["up_rate"] == 0.5
    naive = base.assign(q10=prev, q50=prev, q90=prev)
    m = evaluate.metrics(naive, [0.1, 0.5, 0.9])
    assert m["mape_skill"] == pytest.approx(0.0) and m["directional_accuracy"] is None
    assert m["quantile_loss"] == pytest.approx(m["naive_quantile_loss"])


@pytest.mark.parametrize(("target", "normalizer"),
                         [("vwap", "encoder"), ("vwap", "group"), ("return", "encoder")])
def test_train_smoke_writes_results(panel_path, tmp_path, target, normalizer):
    run_dir = train.run(_cfg(panel_path, tmp_path, target=target, normalizer=normalizer,
                             only_folds=[1]))
    summary = json.loads((run_dir / "summary.json").read_text())
    test = summary["folds"]["1"]["test"]
    assert test["rows"] > 0 and 0 <= test["directional_accuracy"] <= 1
    assert (run_dir / "fold_1" / "best.ckpt").exists()
    preds = pd.read_parquet(run_dir / "fold_1" / "test_predictions.parquet")
    assert (preds["q10"] <= preds["q90"]).all() and preds["day_idx"].min() >= 1140
    assert preds["date"].notna().all() and preds["industry_code"].notna().all()
    assert not (run_dir / "fold_0").exists()
    assert {"quantiles", "signals", "grid"} <= set(summary["pooled_calibration"])
    signals = pd.read_csv(run_dir / "calibration_signals.csv")
    assert len(signals) == 12 * 4  # default moves x confidences
    assert (run_dir / "report" / "grid_by_symbol.csv").exists()  # run() builds the report

    # frozen-model forecast for the session after the panel: checkpoint is only read
    ckpt = run_dir / "fold_1" / "best.ckpt"
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    fc, nxt = predict.forecast(run_dir)
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == digest
    assert sorted(fc["symbol"]) == [f"S{i}" for i in range(6)]
    assert (fc["date"] == nxt["date"]).all() and (fc["fold"] == 1).all()
    qs = fc[[evaluate.qcol(lv) for lv in TrainConfig().quantiles]].to_numpy()
    assert (np.diff(qs, axis=1) >= 0).all() and np.abs(qs).max() < 1  # fractional moves
    assert np.allclose(fc["up_70"], fc["q30"]) and np.allclose(fc["down_70"], fc["q70"])
    assert predict.main([str(run_dir), "--side", "up", "--move", "0.005",
                         "--confidence", "0.0"]) == 0
    moves = pd.read_csv(run_dir / f"forecast_{nxt['date']}_moves.csv")
    assert len(moves) == 6 * 12 and {"hist_hit_rate", "all_hit_rate"} <= set(moves.columns)


def test_prob_below_interpolates_and_clamps():
    r_q = np.array([[-0.02, 0.0, 0.02]])
    levels = [0.1, 0.5, 0.9]
    got = [evaluate.prob_below(r_q, levels, x)[0] for x in (-0.05, -0.01, 0.0, 0.01, 0.05)]
    assert got == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])


def _dist_frame(rows):
    """rows: (symbol, return quantiles at P10/P50/P90, actual return); prev_vwap = 100."""
    return pd.DataFrame([
        {"symbol": s, "prev_vwap": 100.0, "actual": 100 * (1 + a),
         "q10": 100 * (1 + q[0]), "q50": 100 * (1 + q[1]), "q90": 100 * (1 + q[2])}
        for s, q, a in rows])


def test_signal_table_hit_rate_against_base_rate():
    bullish, bearish = (0.0, 0.04, 0.08), (-0.04, 0.0, 0.02)  # P(r >= 2%) = 0.7 / 0.1
    frame = _dist_frame([("A", bullish, 0.03), ("B", bullish, 0.01),
                         ("C", bearish, 0.05), ("D", bearish, -0.01)])
    rows = evaluate.signal_table(frame, [0.1, 0.5, 0.9], [0.02, -0.02], [0.6, 0.8])
    by = {(r["move"], r["confidence"]): r for r in rows}
    up60 = by[(0.02, 0.6)]
    assert up60["signals"] == 2 and up60["symbols"] == 2
    assert up60["mean_prob"] == pytest.approx(0.7)
    assert up60["hit_rate"] == 0.5 and up60["base_rate"] == 0.5
    assert by[(0.02, 0.8)]["signals"] == 0 and by[(0.02, 0.8)]["hit_rate"] is None
    # P(r <= -2%) is 0.1 (bullish: clamped at P10 = 0%) or 0.3 (bearish): never a call
    assert by[(-0.02, 0.6)]["signals"] == 0 and by[(-0.02, 0.6)]["base_rate"] == 0


def test_quantile_calibration_counts_actuals_below_each_quantile():
    q = (-0.01, 0.0, 0.01)
    frame = _dist_frame([("A", q, -0.02), ("A", q, -0.005), ("A", q, 0.005), ("A", q, 0.02)])
    cal = evaluate.quantile_calibration(frame, [0.1, 0.5, 0.9])
    assert [c["observed"] for c in cal] == [0.25, 0.5, 0.75]


LEVELS = [0.1, 0.5, 0.9]


def _small_cfg(**kw):
    return TrainConfig(**{"quantiles": LEVELS, "moves": [0.02, -0.02],
                          "prob_buckets": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], **kw})


def test_bucket_of_edges():
    edges = [0.5, 0.6, 0.7, 0.95]
    got = calibration.bucket_of(np.array([0.49, 0.5, 0.65, 0.7, 0.95]), edges)
    assert got.tolist() == [-1, 0, 1, 2, 2]  # below the grid; top edge inclusive
    assert calibration.bucket_labels(edges) == ["50%-60%", "60%-70%", "70%-95%"]


def test_grid_scores_each_probability_bucket():
    bullish = (0.0, 0.0625, 0.1)  # P(r >= 2%) = 1 - (0.1 + 0.4 * 0.32) = 0.772
    bearish = (-0.1, -0.0625, 0.0)  # P(r <= -2%) = 0.772, P(r >= 2%) = 0.1 (outside the grid)
    frame = _dist_frame([("A", bullish, 0.03), ("A", bullish, 0.01), ("B", bullish, 0.025),
                         ("B", bearish, -0.03)])
    g = calibration.grid(frame, _small_cfg())
    up = g[g["move"] == 0.02].iloc[0]
    assert (up["side"], up["bucket"], up["n"]) == ("up", "70%-80%", 3)
    assert up["mean_prob"] == pytest.approx(0.772)
    assert up["hit_rate"] == pytest.approx(2 / 3)
    assert up["deviation"] == pytest.approx(2 / 3 - 0.772)
    assert up["z"] == pytest.approx((2 / 3 - 0.772) / np.sqrt(0.772 * 0.228 / 3))
    down = g[g["move"] == -0.02].iloc[0]
    assert (down["side"], down["n"], down["hit_rate"]) == ("down", 1, 1.0)
    by = calibration.grid(frame, _small_cfg(), ["symbol"])
    a = by[(by["symbol"] == "A") & (by["move"] == 0.02)].iloc[0]
    assert (a["n"], a["hit_rate"]) == (2, 0.5)


def test_ladder_reads_upside_low_quantiles_and_downside_high():
    r_q = np.array([[-0.02, 0.0, 0.02]])
    lad = calibration.ladder(r_q, LEVELS, [0.7, 0.9])
    assert lad["up_70"][0] == pytest.approx(-0.01)  # q30: halfway between P10 and P50
    assert lad["down_70"][0] == pytest.approx(0.01)  # q70
    assert lad["up_90"][0] == pytest.approx(-0.02) and lad["down_90"][0] == pytest.approx(0.02)
    q = (-0.02, 0.0, 0.02)
    frame = _dist_frame([("A", q, 0.0), ("A", q, -0.015), ("A", q, -0.03), ("A", q, 0.02)])
    hits = calibration.ladder_hits(frame, _small_cfg(confidences=[0.7]))
    up = hits[hits["side"] == "up"].iloc[0]
    assert up["hit_rate"] == 0.5 and up["deviation"] == pytest.approx(-0.2)  # r >= -1%: 2/4


@pytest.mark.parametrize(("shift", "label"),
                         [(0.005, "pessimistic"), (-0.005, "optimistic"), (0.0, "calibrated")])
def test_bias_sign_follows_where_actuals_land(shift, label):
    q = (-0.01, 0.0, 0.01)
    actual = np.array([-0.004, 0.004] * 20) + shift  # symmetric around the median, then shifted
    frame = _dist_frame([("A", q, a) for a in actual])
    b = calibration.bias(frame, _small_cfg(), ["symbol"]).iloc[0]
    assert b["n"] == 40 and b["label"] == label
    assert np.sign(b["bias"]) == -np.sign(shift)  # actuals above forecast -> negative bias


def test_report_cli_writes_every_table(panel_path, tmp_path, capsys):
    run_dir = tmp_path / "run"
    (run_dir / "fold_0").mkdir(parents=True)
    saved = TrainConfig(panel=panel_path, quantiles=LEVELS, moves=[0.5])  # moves: not reused
    (run_dir / "config.json").write_text(json.dumps({"config": saved.to_dict()}))
    q = (0.0, 0.02, 0.04)  # P(r >= 1%) = 0.7
    frame = _dist_frame([(s, q, a) for s in ("S0", "S2") for a in (0.02, -0.005, 0.01)])
    frame["day_idx"] = [1100, 1101, 1102] * 2
    frame.to_parquet(run_dir / "fold_0" / "test_predictions.parquet", index=False)

    assert from_run(run_dir).moves == TrainConfig().moves
    assert report.main([str(run_dir), "--symbol", "S0", "--moves", "0.01,-0.01"]) == 0
    names = {p.stem for p in (run_dir / "report").glob("*.csv")}
    assert names == {"grid_overall", "grid_by_symbol", "grid_by_industry", "grid_by_day",
                     "ladder_overall", "ladder_by_symbol", "bias_by_symbol", "bias_by_day"}
    g = pd.read_csv(run_dir / "report" / "grid_by_day.csv")
    assert g["date"].notna().all() and set(g["move"]) <= {0.01, -0.01}
    out = capsys.readouterr().out
    assert "S0: calibration grid" in out and ">= +1.0%" in out
