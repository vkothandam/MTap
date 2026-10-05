import numpy as np
import pandas as pd
import plotly.graph_objects as go
from conftest import LEVELS

from forecast_dash import calib, charts, contract


def test_every_chart_builds(run_dir):
    run = contract.load_run(run_dir)
    ev = {m: calib.event(run.r_q, run.r_act, run.levels, m) for m in (-0.01, 0.01, 0.02)}
    g = calib.grid(ev, calib.bucket_edges(0.1))
    u = calib.pit(run.r_q, run.r_act, run.levels)
    b = calib.bias(u, run.frame[["symbol", "industry_code"]])
    figs = [
        charts.reliability(g), charts.heatmap(g),
        charts.ladder_chart(calib.ladder_hits(run.r_q, run.r_act, run.levels, [0.6, 0.8])),
        charts.threshold_chart(calib.threshold_curve(*ev[0.01], [0.1, 0.5]), "t"),
        charts.move_chart(calib.move_curve(ev, 0.3), 0.3),
        charts.funnel(b), charts.bias_bars(calib.bias(u, run.frame[["industry_code"]]),
                                           "industry_code", "x"),
        charts.daily_chart(calib.daily(u, run.frame["date"], *ev[0.01]), ">= +1.0%"),
        charts.fan(run.frame.iloc[:60], run.r_q[:60], run.r_act[:60], LEVELS, "S0"),
        charts.pit_hist(u),
        charts.spread_chart(pd.DataFrame({
            "move": [0.01, -0.01], "side": ["up", "down"], "prob": [0.4, 0.3],
            "hist_hit_rate": [0.5, np.nan], "hist_n": [3, np.nan], "all_hit_rate": [0.4, 0.3]}),
            "S0"),
    ]
    assert all(isinstance(f, go.Figure) and len(f.data) > 0 for f in figs)
    rel = figs[0]
    assert {t.name for t in rel.data} == {"ideal", ">= +1.0%", "<= -1.0%", ">= +2.0%"}
    assert len(figs[1].data[0].y) == 3  # heatmap: one row per event


def test_charts_tolerate_empty_tables():
    g = calib.grid({}, calib.bucket_edges(0.1))
    assert isinstance(charts.reliability(g), go.Figure)
    assert isinstance(charts.funnel(pd.DataFrame(columns=["n", "bias", "label", "symbol",
                                                          "industry_code", "bias_z"])), go.Figure)
