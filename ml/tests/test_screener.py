import json
import time

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_panel

from screener.evaluate import daily_picks, paired_bootstrap
from screener.features import (
    COMMON_FEATURES,
    bars,
    feature_names,
    history_features,
    targets,
    tft_features,
    training_frame,
)
from screener.rules import Rules, load_rules, rank_passing, rule_pass
from screener.score import score
from screener.train import load_base, train, walk_forward

LEVELS = [round(k * 0.05, 2) for k in range(1, 20)]
FOLDS = [{"es_start": 1070, "test_start": 1090, "test_end": 1109},
         {"es_start": 1090, "test_start": 1110, "test_end": 1129},
         {"es_start": 1110, "test_start": 1130, "test_end": 1148}]


def panel_with_bars(seed=0):
    df = synthetic_panel(n_symbols=12, seed=seed)
    rng = np.random.default_rng(seed + 1)
    v = df["today_vwap"]
    df["open"] = v * (1 + rng.normal(0, 0.01, len(df)))
    df["close"] = v * (1 + rng.normal(0, 0.01, len(df)))
    top, bot = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)
    sym_scale = 0.01 + 0.004 * df["symbol"].str[1:].astype(int)  # a persistent ranking signal
    df["high"] = top * (1 + rng.uniform(0, 1, len(df)) * sym_scale)
    df["low"] = bot * (1 - rng.uniform(0, 1, len(df)) * sym_scale)
    df["volume"] = rng.uniform(1e6, 2e6, len(df))
    df["split_day"] = False
    return df


def write_base_run(root, panel):
    run = root / "base"
    for k, f in enumerate(FOLDS):
        d = panel[panel["day_idx"].between(f["test_start"], f["test_end"])]
        prev = d["today_vwap"].to_numpy()
        out = pd.DataFrame({"symbol": d["symbol"].to_numpy(), "day_idx": d["day_idx"].to_numpy(),
                            "prev_vwap": prev, "actual": prev})
        w = 0.01 * (1 + d["symbol"].str[1:].astype(int).to_numpy() / 4)  # width tracks range
        for lv in LEVELS:
            z = (lv - 0.5) * 3
            out[f"q{round(lv * 100):02d}"] = prev * (1 + w * z)
        (run / f"fold_{k}").mkdir(parents=True, exist_ok=True)
        out.to_parquet(run / f"fold_{k}" / "test_predictions.parquet", index=False)
    (run / "config.json").write_text(json.dumps({"config": {}, "folds": FOLDS}))
    return run


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("scr")
    panel = panel_with_bars()
    path = root / "panel.parquet"
    panel.to_parquet(path, index=False)
    return root, panel, path, write_base_run(root, panel)


def test_target_values():
    b = pd.DataFrame({"symbol": ["A", "A"], "day_idx": [1, 2], "open": [100.0, 100.0],
                      "high": [110.0, 103.0], "low": [99.0, 90.0]})
    t = targets(b)
    assert t["range_from_open"].tolist() == pytest.approx([0.10, 0.10])
    assert t["hl_range_open"].tolist() == pytest.approx([0.11, 0.13])


def test_bars_fallback_and_incoherent_rows():
    df = pd.DataFrame({"symbol": "A", "day_idx": [1, 2], "today_vwap": 10.0,
                       "open_position_pct": 0.0, "close_position_pct": 0.0,
                       "high_position_pct": [0.02, 0.0], "low_position_pct": [-0.02, 0.01]})
    b = bars(df)
    assert b.loc[0, "high"] == pytest.approx(10.2)
    assert b.loc[1, ["open", "high", "low", "close"]].isna().all()  # low above open


def test_no_leak_day_d_high_low_only_in_target(world):
    _, panel, _, run = world
    _, oof = load_base(run, None)
    base = training_frame(panel, tft_features(oof, prices=True).assign(fold=oof["fold"].to_numpy()))
    d = int(base["day_idx"].median())
    bumped = panel.copy()
    hit = bumped["day_idx"] == d
    bumped.loc[hit, "high"] *= 1.5
    bumped.loc[hit, "low"] *= 0.7
    again = training_frame(
        bumped, tft_features(oof, prices=True).assign(fold=oof["fold"].to_numpy()))
    a = base[base["day_idx"] == d].set_index("symbol")
    b = again[again["day_idx"] == d].set_index("symbol")
    feats = [c for c in feature_names("full")]
    pd.testing.assert_frame_equal(a[feats], b.loc[a.index, feats])
    assert not np.allclose(a["range_from_open"], b.loc[a.index, "range_from_open"])
    # and day d's own bar does not enter day d+1 features except as yesterday's
    nxt = base[base["day_idx"] == d + 1]
    assert nxt["prev_range"].notna().all()


def test_history_features_shift(world):
    _, panel, _, _ = world
    h = history_features(panel)
    assert {"prev_range", "atr14_pct", "log_dollar_volume", "vol_10d_mean"} <= set(h.columns)
    assert set(COMMON_FEATURES) - set(h.columns) <= {
        "gap_prev_close", "gap_prev_vwap", "tft_width90", "tft_width50", "tft_median",
        "earn_near", "day_of_week"}


def test_no_crossing_in_tft_features():
    fc = pd.DataFrame({"symbol": ["A"], "q05": [0.02], "q25": [-0.01], "q50": [0.0],
                       "q75": [0.01], "q95": [-0.02]})  # deliberately crossed
    f = tft_features(fc, prices=False)
    assert f["tft_width90"].iloc[0] >= 0 and f["tft_width50"].iloc[0] >= 0


def test_fold_discipline(world):
    _, panel, _, run = world
    bounds, oof = load_base(run, None)
    frame = training_frame(panel, tft_features(oof, prices=True)
                           .assign(fold=oof["fold"].to_numpy()))
    assert set(frame["fold"]) == {0, 1, 2}
    scored = walk_forward(frame, bounds, "px")
    assert set(scored["fold"]) == {1, 2}  # fold 0 has nothing earlier to train on
    seen = []
    import screener.train as tr
    orig = tr.fit
    tr.fit = lambda df, state: (seen.append((df["day_idx"].max(), df["fold"].max())), orig(df, state))[1]
    try:
        walk_forward(frame, bounds, "px")
    finally:
        tr.fit = orig
    assert seen[0][0] < FOLDS[1]["test_start"] and seen[0][1] == 0
    assert seen[1][0] < FOLDS[2]["test_start"] and seen[1][1] == 1


def test_rules_filter():
    df = pd.DataFrame({"open": [10.0, 3.0, 10, 10, 10, 10], "last": 10.0,
                       "dollar_volume_20d": [1e7, 1e7, 1e5, 1e7, 1e7, 1e7],
                       "gap_prev_close": [0.05, 0.05, 0.05, 0.001, 0.05, 0.05],
                       "pm_vol_pct_10d": [2.0] * 6, "split_day": [False] * 4 + [True, False],
                       "earn_near": [0.0] * 5 + [1.0]})
    r = Rules(min_price=5, min_dollar_volume=1e6, min_abs_gap_pct=0.01)
    assert rule_pass(df, r).tolist() == [True, False, False, False, False, True]
    assert rule_pass(df, Rules(min_price=0, min_dollar_volume=0, exclude_split_day=False,
                               earnings="only")).tolist() == [False] * 5 + [True]
    assert rule_pass(df, Rules(earnings="exclude", min_price=0, min_dollar_volume=0)
                     ).tolist() == [True] * 4 + [False, False]
    assert rule_pass(df, Rules(min_pm_vol_pct=3, min_price=0, min_dollar_volume=0)).sum() == 0
    with pytest.raises(ValueError):
        Rules(earnings="maybe")
    ranks = rank_passing(pd.Series([1.0, 3.0, 2.0]), pd.Series([True, False, True]))
    assert ranks.tolist()[0] == 2 and ranks.tolist()[2] == 1 and pd.isna(ranks.tolist()[1])


def test_load_rules(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[screener]\nmin_price = 12.5\nearnings = "exclude"\n')
    r = load_rules(p)
    assert r.min_price == 12.5 and r.earnings == "exclude" and r.exclude_split_day
    assert load_rules(tmp_path / "none.toml") == Rules()


def test_paired_bootstrap_and_picks():
    ci = paired_bootstrap(np.full(50, 0.01))
    assert ci["lo"] == pytest.approx(0.01) and ci["days"] == 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"day_idx": np.repeat([1, 2], 30), "range_from_open": rng.uniform(0, .1, 60),
                       "score": rng.uniform(0, 1, 60), "atr14_pct": rng.uniform(0, 1, 60),
                       "tft_width90": rng.uniform(0, 1, 60)})
    d = daily_picks(df, 5)
    assert len(d) == 2 and {"screener", "random", "naive_atr", "tft_width"} <= set(d.columns)


def test_train_and_score(world, tmp_path):
    root, panel, path, run = world
    model_dir = train(path, run, tmp_path / "models", None,
                      Rules(min_price=0, min_dollar_volume=0, top_n=3))
    summary = json.loads((model_dir / "summary.json").read_text())
    assert summary["folds"] == [0, 1, 2]
    px = summary["states"]["px"]
    assert px["days"] > 0 and "screener_minus_naive_atr" in px
    # the persistent per-symbol range signal must beat a random pick
    assert px["screener_minus_random"]["mean"] > 0

    last = int(panel["day_idx"].max())
    syms = sorted(panel.loc[panel["day_idx"] == last, "symbol"].unique())
    prev = panel[panel["day_idx"] == last].set_index("symbol")["today_vwap"]
    fc = pd.DataFrame({"symbol": syms})
    for lv in LEVELS:
        fc[f"q{round(lv * 100):02d}"] = (lv - 0.5) * 0.04
    fc.to_parquet(tmp_path / "fc.parquet", index=False)
    (tmp_path / "prices.jsonl").write_text("\n".join(json.dumps(
        {"symbol": s, "date": "2026-10-28", "last": prev[s] * 1.01, "open": prev[s] * 1.012,
         "source": "t"}) for s in syms[:-1]))  # the last symbol has no open yet
    (tmp_path / "pm.jsonl").write_text("\n".join(json.dumps(
        {"symbol": s, "date": "2026-10-28", "cutoff": "09:30", "pm_vwap": prev[s] * 1.005,
         "pm_volume": 1e5, "pm_last": prev[s], "pm_high": prev[s] * 1.02,
         "pm_low": prev[s] * 0.99, "complete": True}) for s in syms))
    for state in ("px", "full"):
        t0 = time.perf_counter()
        out = score("2026-10-28", model_dir, tmp_path / "prices.jsonl", tmp_path / "pm.jsonl",
                    tmp_path / "fc.parquet", state, out_dir=tmp_path / state,
                    rules=Rules(min_price=0, min_dollar_volume=0, top_n=3))
        assert time.perf_counter() - t0 < 5
        assert len(out) == len(syms)
        assert {"symbol", "score", "pred_range", "rule_pass", "rank", "lean"} <= set(out.columns)
        assert out["lean"].isin(["up", "down"]).all()
        assert not out.loc[out["symbol"] == syms[-1], "rule_pass"].iloc[0]  # no open
        ranked = out[out["rule_pass"]]
        assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
        assert ranked["score"].is_monotonic_decreasing
        assert (tmp_path / state / "screener-2026-10-28.parquet").exists()
        assert (tmp_path / state / "screener-2026-10-28-top3.csv").exists()
