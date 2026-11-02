import numpy as np
import pandas as pd
from mtap_space import pockets as K
from mtap_space.library import Library
from mtap_space.match import Query

from forecast_dash import charts, insights
from forecast_dash import space_calc as sc


def _space(space_lib, dials=(1.5, 3.0), params=None):
    lib = Library(space_lib)
    tk = lib.stocks().tk.iloc[0]
    e = lib.stockday(tk, lib.days_of(tk).date.iloc[0])
    row = lib.moment(e, 120)
    q = Query(row=row, params=params if params is not None else lib.meta["default_params"],
              window=20)
    pool = lib.pool(120, int(lib.m["s"][row]), 20)
    clim = K.climatology(lib, pool)
    return lib, q, pool, clim, sc.evaluate(lib, q, dials, pool, clim)


def test_evidence_rows_and_blending(space_lib):
    lib, q, pool, _, sp = _space(space_lib)
    ev = sc.evidence(sp)
    assert list(ev.index) == list(sc.VIEWS)
    assert ev.loc["clim", "matches"] == len(pool)
    assert ev.loc["loose", "matches"] >= ev.loc["close", "matches"] > 0
    lo = sp.M["loose"]
    assert np.isclose(ev.loc["loose", "trust"], lo.nd / (lo.nd + 10))
    assert np.isclose(ev.loc["clim", "log_score"], 0)
    a = ev.loc["actual"]
    assert a["p_up"] in (0.0, 1.0) and 0 <= a["above_now"] <= 1
    assert np.isclose(a["rod_p50"], lib.m["rod"][q.row])
    for k in ("close", "loose", "clim"):
        assert ev.loc[k, "in_50"] <= ev.loc[k, "in_80"] <= 1


def test_no_parameters_matches_the_whole_pool(space_lib):
    _, _, pool, clim, sp = _space(space_lib, params=[])
    assert sp.M["loose"].n == len(pool)
    assert np.isclose(sc.evidence(sp).loc["loose", "p_up"],
                      sp.M["loose"].shrink * clim.pu + (1 - sp.M["loose"].shrink) * clim.pu)


def test_matches_cells_dial_and_day(space_lib):
    lib, q, pool, clim, sp = _space(space_lib)
    t, rows = sc.matches(sp, n=20)
    assert len(t) == len(rows) <= 20 and t["weight"].is_monotonic_decreasing
    assert set(q.params) <= set(t.columns) and t["above_now"].between(0, 1).all()
    assert t["close"].sum() == np.isin(rows, sp.M["close"].rows).sum()
    i, j = np.unravel_index(np.argmax(sp.R["loose"].Q), sp.R["loose"].Q.shape)
    ct = sc.cell_table(sp, int(i), int(j))
    assert list(ct["view"]) == list(sc.VIEWS.values())
    assert ct.loc[ct.view == "everything", "vs_everything"].iloc[0] == 1.0
    cm = sc.cell_matches(sp, int(i), int(j))
    assert cm.empty or cm["of_cell"].sum() <= 1 + 1e-9
    dc = sc.dial_curve(lib, q, pool, clim)
    assert dc["matches"].is_monotonic_increasing and len(dc) == len(sc.CURVE_DIALS)
    dp = sc.day_path(lib, q.row)
    assert len(dp) == 390 and dp["after"].sum() == 390 - 120
    assert sc.to_price(10.0, 0.02, 0) == 10.0


def test_space_charts_build(space_lib):
    lib, q, pool, clim, sp = _space(space_lib)
    info = sc.moment_info(lib, q.row)
    maps = sp.maps()
    regions = {k: K.region(Q) for k, Q in maps.items()}
    figs = [
        charts.pocket_maps(maps, regions, K.x_edges(), K.y_edges(), info["now"], info["sig"],
                           sc.REGIONS, "t"),
        charts.pocket_diff(sp.R["loose"].Q, clim.Q, K.x_edges(), K.y_edges(), info["now"],
                           info["sig"], "d"),
        charts.dial_curve(sc.dial_curve(lib, q, pool, clim), (1.5, 3.0), clim.pu, clim.above),
        charts.space_day_path(sc.day_path(lib, q.row), 120, info["prev"], "day"),
    ]
    assert all(len(f.data) for f in figs)
    assert charts.span_label(0.4) == "0.4m" and charts.span_label(150) == "2.5h"


def test_space_insights():
    def ev(close_n, close_days, loose_up, clim_up, log_close=0.1):
        cols = ("matches", "days", "trust", "p_up", "above_now", "blobs_50", "in_50", "log_score")
        rows = {
            "close": (close_n, close_days, close_days / (close_days + 10), 0.5, 0.5, 1, 0.6,
                      log_close),
            "loose": (500, 30.0, 0.75, loose_up, loose_up, 2, 0.5, -0.1),
            "clim": (9000, 60.0, np.nan, clim_up, clim_up, 1, 0.55, 0.0),
            "actual": (np.nan, np.nan, np.nan, 1.0, 0.7, np.nan, np.nan, np.nan)}
        return pd.DataFrame.from_dict(rows, orient="index", columns=list(cols))

    out = insights.space_insights(ev(3, 1.5, 0.62, 0.5), past=True)
    text = " ".join(i.text for i in out)
    assert out[0].level == insights.WARN and "1.5 independent days" in out[0].text
    assert "leans higher" in text and "2 separate blobs" in text
    assert any(i.level == insights.GOOD and "close" in i.text for i in out)
    assert any(i.level == insights.WARN and "loose pocket was no closer" in i.text for i in out)
    quiet = insights.space_insights(ev(0, 0.0, 0.51, 0.5), past=False)
    assert "found nothing" in quiet[0].text and "looks like everything" in quiet[1].text
    assert not any("What it did" in i.text for i in quiet)


def test_space3d_data(space_lib):
    from forecast_dash.space_page import S3D_JS, space3d_data

    lib, q, _, _, sp = _space(space_lib)
    info = sc.moment_info(lib, q.row)
    maps = sp.maps()
    regions = {k: K.region(Q) for k, Q in maps.items()}
    d = space3d_data(maps, regions, info["now"], info["sig"], (3, 4))
    assert (d["nx"], d["ny"]) == (K.NX, K.NY) and d["cell"] == [3, 4]
    assert len(d["xe"]) == K.NX + 1 and len(d["ye"]) == K.NY + 1
    assert [t for _, t in d["yt"]][3] == "now" and d["yt"][3][0] == K.NY / 2
    for v, (k, Q) in zip(d["views"], maps.items()):
        assert v["title"] == k and len(v["q"]) == len(v["region"]) == K.NX * K.NY
        assert np.allclose(np.reshape(v["q"], Q.shape), Q, atol=1e-8)   # time-ahead major
        assert v["levels"] == sorted(v["levels"], reverse=True)
    import json
    assert json.dumps(d)                                     # plain JSON for the component
    assert S3D_JS.exists(), "build it: cd dashboard/js && npm install && npm run build"


def test_a_click_in_3d_picks_the_cell(monkeypatch):
    from forecast_dash import space_page

    state = {"sp_cell_x": 1, "sp_cell_y": 2}
    monkeypatch.setattr(space_page.st, "session_state", state)
    space_page._pick_cell()                                  # no click yet: unchanged
    assert (state["sp_cell_x"], state["sp_cell_y"]) == (1, 2)
    state["sp_space3d"] = {"cell": {"i": 7, "j": 9, "at": 1}}
    space_page._pick_cell()
    assert (state["sp_cell_x"], state["sp_cell_y"]) == (7, 9)
