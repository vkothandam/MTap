"""Intraday → Pockets model: the contract readers and the scores (pockets_contract, pockets_calc)."""

import numpy as np
import pandas as pd
from conftest import PK_TARGETS, write_pockets_run

from forecast_dash import pockets_calc as P
from forecast_dash import pockets_contract as C


def test_discover_finds_only_intraday_ft_runs(tmp_path):
    a = write_pockets_run(tmp_path, "20261107-000000-ft-B")
    b = write_pockets_run(tmp_path, "20261108-000000-ft-A", fold="A")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "config.json").write_text('{"model": "tft_intraday"}')
    assert C.discover_pockets(tmp_path) == [b, a]
    assert C.discover_pockets(a) == [a]
    assert C.discover_pockets(tmp_path / "missing") == []
    assert C.folds_of(a) == ["B"] and C.targets_of(a, "B") == PK_TARGETS


def test_quantiles_read_any_row_order(tmp_path):
    run = write_pockets_run(tmp_path)
    Q = np.load(run / "fold_B" / "quantiles.npy")
    j = PK_TARGETS.index("cm_up_b4")
    rows = np.array([500, 3, 77, 3])
    got = C.quantiles(run, "B", rows, ["cm_up_b4"])[:, 0]
    assert np.allclose(got, Q[rows, j])
    allrows = np.arange(len(Q))[::-1]
    assert np.allclose(C.quantiles(run, "B", allrows, ["cm_up_b4"])[:, 0], Q[allrows, j])


def test_load_pred_skips_missing_columns(tmp_path):
    run = write_pockets_run(tmp_path)
    f = C.load_pred(run, "B", ["date", "symbol", "no_such_column"])
    assert list(f.columns) == ["date", "symbol"]


def test_recalibrate_adds_the_day_offsets_and_keeps_order(tmp_path):
    run = write_pockets_run(tmp_path, sessions=24)
    rc = C.load_recal(run, "B")
    f = C.load_pred(run, "B", ["date", "dec_row"])
    q = C.quantiles(run, "B", f.dec_row.to_numpy(), ["sag_b2"])[:, 0]
    out = C.recalibrate(q, rc, "sag_b2", f.date)
    late = f.date.isin(sorted(f.date.unique())[20:]).to_numpy()
    assert np.allclose(out[~late], np.clip(q[~late], 0, None))       # no offset before the window
    assert np.allclose(out[late], np.clip(q[late] + 0.06, 0, None))
    assert (np.diff(out, axis=1) >= 0).all()
    assert C.recalibrate(q, None, "sag_b2", f.date) is q


def test_climatology_cells_match_the_trainer_formula(tmp_path):
    run = write_pockets_run(tmp_path)
    cl = C.load_clim(run, "B")
    f = C.load_pred(run, "B", C.KEYS)
    cell = P.cells(f.minute, f.sigma, f.log_dvol, cl)
    want = (f.minute // 5) * 9 + np.digitize(f.sigma, [0.013, 0.016]) * 3 + np.digitize(
        f.log_dvol, [18.0, 19.0])
    assert (cell == want.to_numpy()).all()
    qc = P.clim_quantiles(cl, "cm_up_b4", cell)
    allq = cl[(cl.cell == -1) & (cl.target == "cm_up_b4")][["q10", "q25", "q50", "q75", "q90"]]
    have = np.isin(cell, cl.cell)
    assert np.isfinite(qc).all()
    assert np.allclose(qc[~have], allq.to_numpy()[0])                # empty cells: overall
    assert P.clim_quantiles(cl, "no_target", cell) is None
    r = P.clim_rate(C.load_clim_rates(run, "B"), "p_bulge60", cell)
    assert ((r >= 0) & (r <= 1)).all()


def test_score_matches_brute_force_and_recal_fixes_the_shift(tmp_path):
    run = write_pockets_run(tmp_path, sessions=30)
    f = C.load_pred(run, "B")
    t = "cm_dn_b3"
    q = C.quantiles(run, "B", f.dec_row.to_numpy(), [t])[:, 0]
    y = f[f"real_{t}"].to_numpy()
    s = P.score(f, q, y, q)                                        # vs itself: skill 0
    assert s["n"] == len(f) and abs(s["skill_clim"]) < 1e-12
    assert np.isclose(s["p50_hit"], (y <= q[:, 2]).mean())
    assert np.isclose(s["cover80"], ((y >= q[:, 0]) & (y <= q[:, 4])).mean())
    assert s["p50_hit"] < 0.42                                     # outcomes +0.3 sd above: P50 too low
    late = f.date.isin(sorted(f.date.unique())[20:]).to_numpy()
    qc = C.recalibrate(q, C.load_recal(run, "B"), t, f.date)
    s2 = P.score(f[late], qc[late], y[late], None)
    assert abs(s2["p50_hit"] - 0.5) < 0.05
    assert s["rank_ic"] > 0.1                                      # the per-stock centre ranks


def test_rank_ic_is_spearman_per_minute():
    d = pd.DataFrame({"date": ["a"] * 4 + ["b"] * 4, "minute": 0,
                      "p": [1, 2, 3, 4, 1, 2, 3, 4], "y": [1, 3, 2, 4, 4, 3, 2, 1]})
    ic = P.rank_ic(d.date, d.minute, d.p, d.y, min_n=4)
    assert np.isclose(ic, (0.8 + -1.0) / 2)
    assert np.isnan(P.rank_ic(d.date, d.minute, d.p, d.y, min_n=5))


def test_by_group_and_the_go_rule(tmp_path):
    run = write_pockets_run(tmp_path)
    f = C.load_pred(run, "B")
    t = "rvol_b4"
    q = C.quantiles(run, "B", f.dec_row.to_numpy(), [t])[:, 0]
    cl = C.load_clim(run, "B")
    qc = P.clim_quantiles(cl, t, P.cells(f.minute, f.sigma, f.log_dvol, cl))
    g = P.by_group(f, q, f[f"real_{t}"].to_numpy(), qc, "cap_band")
    assert set(g.cap_band) == {"mega", "large", "mid", "small"} and g.n.sum() == len(f)
    assert g.rank_ic.isna().all()                                  # 3 stocks per band and minute
    go = P.go_rule(P.report_shape(C.load_report(run), "B"))
    assert len(go) == 25 and go.passes.all()
    assert not go.calibrated.any() and go.calibrated_after_recal.all()
    assert P.report_shape(C.load_report(run), "Z").empty and P.report_shape(None, "B").empty
    h = P.headline(C.load_report(run))
    assert h["B"]["cm_up_b4_skill"] == 0.04 and h["B"]["net60_rank_ic"] == 0.001


def test_cdf_at_and_the_overlay_scores(tmp_path):
    q = np.array([[0.0, 0.1, 0.2, 0.3, 0.4]])
    assert np.isclose(P.cdf_at(q, 0.2)[0], 0.5) and np.isclose(P.cdf_at(q, 0.4)[0], 0.9)
    assert P.cdf_at(q, 2.0)[0] > 0.999 and P.cdf_at(q, -1.0)[0] < 0.01
    run = write_pockets_run(tmp_path)
    f = C.load_pred(run, "B")
    hit = P.event_hit(f, "p_bulge60")
    probs = {v: f[f"p_bulge60__{v}"].to_numpy(np.float64) for v in ("none", "all")}
    rate = np.full(len(f), np.nanmean(hit))
    t = P.overlay_scores(f, probs, hit, rate)
    assert t.n.iloc[0] == len(f) and t.skill_none.iloc[0] > 0.1    # informative synthetic probs
    assert t.skill_none_lo.iloc[0] <= t.skill_none.iloc[0] <= t.skill_none_hi.iloc[0]
    gt = P.overlay_scores(f, probs, hit, rate, "sector")
    assert len(gt) == 3 and gt.n.sum() == len(f)
    g = P.event_grid(probs, hit)
    assert set(g.overlay) == {"none", "+ symbol"} and g.groupby("overlay").n.sum().eq(len(f)).all()
    assert P.overlay_insights(t, "p_bulge60", ["none", "all"])


def test_replay_and_fan_in_dollars(tmp_path):
    run = write_pockets_run(tmp_path)
    f = C.load_pred(run, "B")
    sel = f[(f.date == f.date.iloc[0]) & (f.symbol == "S3")].sort_values("minute")
    rows = sel.dec_row.to_numpy()
    q = {t: C.quantiles(run, "B", rows, [t])[:, 0] for t in PK_TARGETS if not t.endswith("60")}
    r = P.replay(sel, q, 4)
    assert len(r) == len(sel) and r.time.iloc[0] == "09:34"
    assert (r.up_p10 <= r.up_p90).all() and (r.dn_p90 <= r.dn_p10).all()
    assert (r.up_p50 > r.p0).all() and (r.dn_p50 < r.p0).all()
    fan = P.pocket_fan(sel.iloc[0], q, 0)
    assert list(fan.window) == list(C.BINS.values())
    p0, s = sel.real_p0.iloc[0], sel.sigma.iloc[0]
    assert np.isclose(fan.up_p50_usd.iloc[3], p0 * np.exp(s * q["cm_up_b4"][0, 2]))


def test_sector_labels():
    assert C.sector_label("57") == "57 Technology" and C.sector_label(50) == "50 Energy"
    assert C.sector_label("") == "unknown" and C.sector_label("99") == "99"
