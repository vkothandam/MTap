"""R6c event overlay: night-before alignment, pass-through on ordinary days, own fold unused."""

import json

import numpy as np
import pandas as pd

import model_card
from tft_vwap import compare, event_overlay as ev

Z = np.array([-1.645, -1.28, 0.0, 1.28, 1.645])
LEVELS = (0.05, 0.10, 0.50, 0.90, 0.95)
SYMS = ("A", "B", "C", "D")


def _panel(days, spike_days=(), seed=0):
    """Day-level series quiet (std 1%) except a 6% market move on `spike_days`."""
    rng = np.random.default_rng(seed)
    mkt = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    for d in spike_days:
        mkt[d] = 0.06
    rows = []
    for s in SYMS:
        for d in days:
            rows.append({"symbol": s, "day_idx": d,
                         "industry_code": "unknown" if s == "D" else ("531010" if s < "C" else "451020"),
                         "mkt_cap_ret": mkt[d], "own_sector_cap_ret": mkt[d],
                         "ust_10y_chg_bp": 0.0 if d == days[0] else rng.normal(0, 3),
                         "ust_2y_chg_bp": rng.normal(0, 3)})
    return pd.DataFrame(rows)


def _run(root, panel, folds, wide_after=(), seed=1):
    """A base run with a 2% band; on the day after a spike the actual moves 3x as much."""
    rng = np.random.default_rng(seed)
    run = root / "base"
    after = {d + 1 for d in wide_after}
    for k, days in folds.items():
        rows = []
        for s in SYMS:
            ic = panel.loc[panel["symbol"] == s, "industry_code"].iloc[0]
            for d in days:
                sd = 0.06 if d in after else 0.02
                rows.append({"symbol": s, "day_idx": d, "prev_vwap": 100.0, "industry_code": ic,
                             "actual": 100.0 * (1 + rng.normal(0, sd))})
        f = pd.DataFrame(rows)
        for q, z in zip(LEVELS, Z):
            f[f"q{round(q * 100):02d}"] = 100.0 * (1 + z * 0.02)
        (run / f"fold_{k}").mkdir(parents=True, exist_ok=True)
        f.to_parquet(run / f"fold_{k}" / "test_predictions.parquet", index=False)
    (run / "config.json").write_text(json.dumps({"config": {}}))
    return run


def test_features_use_the_night_before():
    days = list(range(100, 200))
    by_day, by_sec = ev.event_features(_panel(days, spike_days=(150,)))
    f = by_day.set_index("day_idx")["f_mkt"]
    assert f[151] > 0          # forecast day D = spike + 1 sees it
    assert f[150] == 0         # the spike day itself does not (not known the night before)
    assert by_day.set_index("day_idx")["g_mkt"][151] > 0  # an up move
    assert f.loc[110:149].eq(0).mean() > 0.85  # quiet days are mostly exactly 0
    assert f.loc[100:119].eq(0).all()          # no normal range yet: no event
    # sector vs market is 0 here (sector = market), and "unknown" has no sector rows at all
    assert by_sec["f_sec"].eq(0).all() and set(by_sec["sec"]) == {"53", "45"}


def test_pass_through_widen_and_own_fold(tmp_path):
    days = list(range(100, 400))
    spikes = tuple(range(130, 400, 9))
    panel = _panel(days, spike_days=spikes)
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    folds = {0: range(100, 200), 1: range(200, 300), 2: range(300, 400)}
    base = _run(tmp_path, panel, folds, wide_after=spikes)

    out = ev.overlay(base, path, rows_per_day=10)
    assert out.name == "base-event" and not (out / "fold_0").exists()
    cj = json.loads((out / "config.json").read_text())
    assert cj["model"] == "event_overlay" and cj["folds"] == [1, 2] and cj["thin_folds"] == [1]
    card = model_card.read(out)
    assert card["kind"] == "Event" and card["status"] == "done" and "`base`" in card["inputs"]
    for k in (1, 2):
        m = json.loads((out / f"fold_{k}" / "metrics.json").read_text())
        assert m["coef"]["b_mkt"] > 0.1                   # widens after a big market day
        assert m["train_pinball_after"] < m["train_pinball_before"]
        o = pd.read_parquet(out / f"fold_{k}" / "test_predictions.parquet")
        b = pd.read_parquet(base / f"fold_{k}" / "test_predictions.parquet")
        q = list(compare.qlevels(o).values())
        assert (np.diff(o[q].to_numpy(), axis=1) >= 0).all()   # never crosses
        assert np.allclose(o["q50"], b["q50"])                 # no shift asked for
        feats = ev.row_features(b, *ev.event_features(panel))
        quiet = (feats[list(ev.SCALE_FEATURES)] == 0).all(axis=1).to_numpy()
        assert quiet.any() and np.allclose(o.loc[quiet, q], b.loc[quiet, q])  # unchanged
        spike = (feats["f_mkt"] > 0.5).to_numpy()  # the day after a 6% market move
        assert spike.any() and ((o.loc[spike, "q90"] - o.loc[spike, "q10"])
                                > 1.1 * (b.loc[spike, "q90"] - b.loc[spike, "q10"])).all()

    # fold 1 is fitted on fold 0 alone: corrupting fold 2 changes nothing for fold 1
    first = pd.read_parquet(out / "fold_1" / "test_predictions.parquet")
    p2 = base / "fold_2" / "test_predictions.parquet"
    f2 = pd.read_parquet(p2)
    f2["actual"] = 100.0
    f2.to_parquet(p2, index=False)
    again = ev.overlay(base, path, folds=[1], out=tmp_path / "again", rows_per_day=10)
    assert np.allclose(first["q90"], pd.read_parquet(again / "fold_1" / "test_predictions.parquet")["q90"])

    # calibrate + shift: other output dir, intercept and shift coefficients recorded
    cs = ev.overlay(base, path, folds=[2], calibrate=True, shift=True, rows_per_day=10)
    assert cs.name == "base-event-cal-shift"
    coef = json.loads((cs / "fold_2" / "metrics.json").read_text())["coef"]
    assert {"a", "b_mkt", "c_mkt", "c_sec"} <= set(coef)
    assert compare.compare(base, out, folds=[2])["rows"] == 400
