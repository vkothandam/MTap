import numpy as np
import pandas as pd
import pytest

from forecast_dash import calib

LV = [0.1, 0.5, 0.9]


def test_prob_below_interpolates_and_clamps():
    r_q = np.array([[-0.02, 0.0, 0.02]])
    got = [calib.prob_below(r_q, LV, x)[0] for x in (-0.05, -0.01, 0.0, 0.01, 0.05)]
    assert got == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])


def test_event_sides():
    r_q = np.array([[-0.02, 0.0, 0.02]] * 2)
    p, h = calib.event(r_q, np.array([0.015, -0.015]), LV, 0.01)
    assert p == pytest.approx([0.3, 0.3]) and h.tolist() == [True, False]
    p, h = calib.event(r_q, np.array([0.015, -0.015]), LV, -0.01)
    assert p == pytest.approx([0.3, 0.3]) and h.tolist() == [False, True]


def test_bucket_edges_and_membership():
    assert calib.bucket_edges(0.1) == pytest.approx(
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    assert len(calib.bucket_edges(0.05)) == 19
    edges = [0.5, 0.6, 0.7, 0.95]
    got = calib.bucket_of(np.array([0.49, 0.5, 0.65, 0.7, 0.95]), edges)
    assert got.tolist() == [-1, 0, 1, 2, 2]  # top edge inclusive
    assert calib.bucket_labels(edges) == ["50%-60%", "60%-70%", "70%-95%"]


def test_wilson_interval():
    lo, hi = calib.wilson(5, 10)
    assert (float(lo), float(hi)) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    lo, hi = calib.wilson(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def _rq(rows):
    return np.array([q for q, _ in rows]), np.array([a for _, a in rows])


def test_grid_cell_values():
    bullish = (0.0, 0.0625, 0.1)  # P(r >= 2%) = 1 - (0.1 + 0.4 * 0.32) = 0.772
    r_q, r_act = _rq([(bullish, 0.03), (bullish, 0.01), (bullish, 0.025)])
    ev = {0.02: calib.event(r_q, r_act, LV, 0.02)}
    g = calib.grid(ev, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    c = g.iloc[0]
    assert (c["side"], c["event"], c["bucket"], c["n"]) == ("up", ">= +2.0%", "70%-80%", 3)
    assert c["mean_prob"] == pytest.approx(0.772) and c["hit_rate"] == pytest.approx(2 / 3)
    assert c["deviation"] == pytest.approx(2 / 3 - 0.772)
    assert c["z"] == pytest.approx((2 / 3 - 0.772) / np.sqrt(0.772 * 0.228 / 3))
    keys = pd.DataFrame({"symbol": ["A", "A", "B"]})
    by = calib.grid(ev, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], keys)
    assert by.set_index("symbol")["hit_rate"].to_dict() == {"A": 0.5, "B": 1.0}


def test_ladder_and_hits():
    r_q = np.array([[-0.02, 0.0, 0.02]] * 4)
    lad = calib.ladder(r_q, LV, [0.7])
    assert lad["up_70"][0] == pytest.approx(-0.01) and lad["down_70"][0] == pytest.approx(0.01)
    lh = calib.ladder_hits(r_q, np.array([0.0, -0.015, -0.03, 0.02]), LV, [0.7])
    up = lh[lh["side"] == "up"].iloc[0]
    assert up["hit_rate"] == 0.5 and up["deviation"] == pytest.approx(-0.2)


@pytest.mark.parametrize(("shift", "label"),
                         [(0.005, "pessimistic"), (-0.005, "optimistic"), (0.0, "calibrated")])
def test_bias_sign(shift, label):
    r_q = np.array([[-0.01, 0.0, 0.01]] * 40)
    r_act = np.array([-0.004, 0.004] * 20) + shift
    b = calib.bias(calib.pit(r_q, r_act, LV), pd.DataFrame({"symbol": ["A"] * 40})).iloc[0]
    assert b["label"] == label and np.sign(b["bias"]) == -np.sign(shift)


def test_threshold_and_move_curves():
    rng = np.random.default_rng(0)
    prob = rng.uniform(0.05, 0.95, 2000)
    hit = rng.uniform(size=2000) < prob  # calibrated by construction
    t = calib.threshold_curve(prob, hit, [0.1, 0.5, 0.9])
    assert t["n"].is_monotonic_decreasing and (t["base_rate"] == hit.mean()).all()
    assert (t["hit_rate"] - t["mean_prob"]).abs().max() < 0.05
    m = calib.move_curve({0.01: (prob, hit), -0.01: (prob, hit)}, 0.5)
    assert m["move"].tolist() == [-0.01, 0.01] and m["side"].tolist() == ["down", "up"]


def test_daily_and_summary_on_calibrated_data():
    from conftest import LEVELS, synthetic_predictions

    df = synthetic_predictions(n_symbols=40, n_days=100)
    prev = df["prev_vwap"].to_numpy()
    r_q = df[[f"q{round(lv * 100):02d}" for lv in LEVELS]].to_numpy() / prev[:, None] - 1
    r_act = df["actual"].to_numpy() / prev - 1
    ev = {m: calib.event(r_q, r_act, LEVELS, m) for m in (0.01, -0.01)}
    s = calib.summary(r_q, r_act, LEVELS, ev, calib.bucket_edges(0.1))
    assert s["rows"] == 4000 and abs(s["coverage"] - 0.9) < 0.03 and abs(s["bias"]) < 0.02
    assert s["pinball"] < s["naive_pinball"] and s["ece"] < 0.05
    d = calib.daily(calib.pit(r_q, r_act, LEVELS), df["day_idx"], *ev[0.01])
    assert len(d) == 100 and (d["n"] == 40).all() and "deviation" in d
