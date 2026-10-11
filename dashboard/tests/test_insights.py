import numpy as np
import pandas as pd
from conftest import LEVELS, synthetic_predictions

from forecast_dash import calib, insights

MOVES = [-0.02, -0.01, 0.01, 0.02]
EDGES = calib.bucket_edges(0.1)


def _arrays(df):
    prev = df["prev_vwap"].to_numpy()
    r_q = df[[f"q{round(lv * 100):02d}" for lv in LEVELS]].to_numpy() / prev[:, None] - 1
    return r_q, df["actual"].to_numpy() / prev - 1


def _run(**kw):
    df = synthetic_predictions(n_symbols=40, n_days=200, **kw)
    df["date"] = df["day_idx"]
    r_q, r_act = _arrays(df)
    ev = {m: calib.event(r_q, r_act, LEVELS, m) for m in MOVES}
    return df, r_q, r_act, ev


def _levels(items):
    return [i.level for i in items]


def test_call_table_and_rollup():
    _, _, _, ev = _run()
    calls = insights.call_table(ev, [0.3, 0.4])
    assert len(calls) == len(MOVES) * 2
    assert set(calls["side"]) == {"up", "down"}
    r = calls.iloc[0]
    assert r["deviation"] == r["hit_rate"] - r["mean_prob"]
    assert np.isclose(r["lift"], r["hit_rate"] / r["base_rate"])
    roll = insights.side_rollup(calls)
    assert len(roll) == 4  # 2 sides x 2 levels
    up40 = calls[(calls["side"] == "up") & (calls["confidence"] == 0.4)]
    row = roll[(roll["side"] == "up") & (roll["confidence"] == 0.4)].iloc[0]
    assert row["n"] == up40["n"].sum()
    assert np.isclose(row["hit_rate"], (up40["hit_rate"] * up40["n"]).sum() / up40["n"].sum())


def test_fold_and_quantile_tables():
    df, r_q, r_act, ev = _run(folds=3)
    f = insights.fold_table(df, r_q, r_act, LEVELS, ev, 0.3)
    assert list(f["fold"]) == [0, 1, 2] and f["rows"].sum() == len(df)
    assert {"up_calls", "down_deviation", "pinball_skill", "coverage"} <= set(f.columns)
    q = insights.quantile_table(r_q, r_act, LEVELS)
    assert np.allclose(q["observed"], q["level"], atol=0.02)  # calibrated by construction


def test_calibrated_run_reads_as_calibrated():
    df, r_q, r_act, ev = _run()
    summ = calib.summary(r_q, r_act, LEVELS, ev, EDGES)
    calls = insights.call_table(ev, [0.3, 0.4])
    folds = insights.fold_table(df, r_q, r_act, LEVELS, ev, 0.3)
    found = insights.run_insights(summ, calls, folds, 0.3)
    text = " ".join(i.text for i in found)
    assert "about the right width" in text and "No material overall lean" in text
    assert insights.BAD not in _levels(found[:3])
    b = calib.bias(calib.pit(r_q, r_act, LEVELS), df[["symbol", "industry_code"]]
                   if "industry_code" in df else df[["symbol"]].assign(industry_code="x"))
    assert insights.stock_insights(b)[0].level == insights.GOOD


def test_shifted_run_is_flagged():
    shift = {f"S{i}": -0.02 for i in range(40)}  # actuals 2% below every forecast
    df, r_q, r_act, ev = _run(shift=shift)
    summ = calib.summary(r_q, r_act, LEVELS, ev, EDGES)
    found = insights.run_insights(summ, insights.call_table(ev, [0.3]), None, 0.3)
    lean = next(i for i in found if "optimistic" in i.text)
    assert lean.level == insights.BAD
    up = next(i for i in found if i.text.startswith("Upside calls"))
    assert "overconfident" in up.text and up.level == insights.BAD
    b = calib.bias(calib.pit(r_q, r_act, LEVELS), df[["symbol"]].assign(industry_code="x"))
    assert insights.stock_insights(b)[0].level == insights.WARN


def test_fold_instability_is_reported():
    df, r_q, r_act, _ = _run(folds=2)
    late = (df["fold"] == 1).to_numpy()
    r_act = np.where(late, r_act - 0.02, r_act)  # second period: the market falls short
    ev = {m: calib.event(r_q, r_act, LEVELS, m) for m in MOVES}
    f = insights.fold_table(df, r_q, r_act, LEVELS, ev, 0.3)
    found = insights._fold_stability(f, 0.3)
    assert any("not stable across time" in i.text and "fold 1" in i.text for i in found)


def test_threshold_flags_confidence_that_does_not_help():
    rng = np.random.default_rng(0)
    prob = rng.uniform(0.05, 0.95, 20_000)
    hit = rng.random(20_000) < np.where(prob > 0.8, 0.3, prob)  # extreme calls go wrong
    t = calib.threshold_curve(prob, hit, np.round(np.arange(0.05, 0.951, 0.01), 2))
    found = insights.threshold_insights(t, ">= +1.0%", 0.7)
    assert any("stops helping" in i.text for i in found)


def test_compare_names_the_better_run():
    _, r_q, r_act, ev = _run()
    _, r_q2, r_act2, ev2 = _run(shift={f"S{i}": 0.01 for i in range(40)})
    summ = {"good": calib.summary(r_q, r_act, LEVELS, ev, EDGES),
            "bad": calib.summary(r_q2, r_act2, LEVELS, ev2, EDGES)}
    calls = {"good": insights.call_table(ev, [0.3]), "bad": insights.call_table(ev2, [0.3])}
    found = insights.compare_insights(summ, calls, 0.3, same_rows=True)
    assert found[0].text.startswith("**good** wins")
    assert any("held best in **good**" in i.text for i in found)
    base = {"rows": 100_000, "coverage_nominal": 0.9}
    tie = insights.compare_insights(  # a: better skill + coverage, b: better bias + ECE
        {"a": {**base, "pinball_skill": 0.2, "coverage": 0.9, "bias": 0.05, "ece": 0.05},
         "b": {**base, "pinball_skill": 0.1, "coverage": 0.8, "bias": 0.0, "ece": 0.01}},
        {}, 0.3, True)
    assert tie[0].text.startswith("No clear winner")
    assert any("Only 8,000 forecasts" in i.text for i in found)
    assert "different rows" in insights.compare_insights(summ, calls, 0.3, False)[0].text
    assert insights.compare_insights({"good": summ["good"]}, {}, 0.7, True)[0].level == "info"


def test_day_and_stock_detail_insights():
    df, r_q, r_act, ev = _run()
    u = calib.pit(r_q, r_act, LEVELS)
    d = calib.daily(u, df["date"].to_numpy(), *ev[0.01])
    assert insights.day_insights(d)[0].level == insights.GOOD
    i = (df["symbol"] == "S0").to_numpy()
    b = calib.bias(u[i], df.loc[i, ["symbol"]]).iloc[0]
    found = insights.stock_detail_insights("S0", b, r_q[i], r_act[i], u[i], 0.9)
    assert found[0].text.startswith("S0 has no lean")
    t = pd.DataFrame({"symbol": ["A", "B"], "prob": [0.8, 0.75], "bucket": ["70%-80%"] * 2,
                      "hist_n": [30, 5], "hist_hit_rate": [0.8, 0.9], "all_n": [900, 900],
                      "all_hit_rate": [0.7, 0.7]})
    assert "1 of them" in insights.tomorrow_insights(t, ">= +1.0%", 0.7)[0].text
    graded = insights.tomorrow_insights(t.assign(inside=[True, False]), ">= +1.0%", 0.7,
                                        day="2026-10-05", band=(0.1, 0.9))[-1]
    assert graded.level == insights.WARN and "50% of these 2 stocks" in graded.text


def test_live_insights_flags_narrow_band_and_lean():
    g = {"rows": 500, "coverage": 0.70, "coverage_nominal": 0.8, "above": 0.25, "below": 0.05,
         "direction_hit": 0.55, "bias": -0.05, "pinball_skill": 0.03, "median_width": 0.04,
         "median_actual": 0.01, "median_forecast": 0.0}
    out = insights.live_insights(g, 0.1, 0.9, None, "2026-10-05")
    assert out[0].level == insights.WARN and "too narrow" in out[0].text
    assert any("pessimistic" in i.text for i in out)
    assert "one day" in out[-1].text
    by = pd.DataFrame({"day": ["a", "b"], "coverage": [0.7, 0.9], "coverage_nominal": [0.8, 0.8]})
    assert "2 days graded" in insights.live_headline(by, 0.1, 0.9)[0].text
    assert insights.live_headline(pd.DataFrame(), 0.1, 0.9) == []
