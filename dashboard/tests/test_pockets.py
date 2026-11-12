"""Intraday → Pockets model: the contract readers and the scores (pockets_contract, pockets_calc)."""

import numpy as np
import pandas as pd
from conftest import PK_TARGETS, write_pockets_run

from forecast_dash import charts
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


def test_move_prob_reads_the_centre_quantiles():
    q = np.array([[0.5, 0.75, 1.0, 1.25, 1.5]])                    # cm in σ
    sig = np.array([0.01])
    x = np.expm1(1.0 * 0.01)                                       # the target at the P50: 0.5
    r = P.move_prob(q, None, sig, x, "up")
    assert np.isclose(r["p_beyond"][0], 0.5) and np.isclose(r["p"][0], 0.5)
    xd = -np.expm1(-1.0 * 0.01)                                    # down: −X is ln(1 − X) away
    assert np.isclose(P.move_prob(q, None, sig, xd, "dn")["p"][0], 0.5)
    m = np.array([[0.0, 0.02, 0.05, 0.4, 0.6]])                    # mass_up: P(< 5%) = 0.5
    up = P.move_prob(q, m, sig, x, "up")
    assert np.isclose(up["p_forms"][0], 0.5) and up["p"][0] <= up["p_forms"][0]
    dn = P.move_prob(q, m, sig, xd, "dn")                          # mass_up ≤ 0.95 nearly always
    assert dn["p_forms"][0] > 0.9 and dn["p"][0] <= dn["p_forms"][0]


def test_move_outcome_and_extremes_in_dollars():
    f = pd.DataFrame({"sigma": [0.01, 0.01, 0.01], "real_cm_up_b2": [2.0, 0.5, np.nan],
                      "real_cm_dn_b2": [0.3, np.nan, np.nan], "real_mass_up_b2": [0.6, 1.0, np.nan]})
    ext = pd.DataFrame({"hi_b2": np.log([310 / 300, 301 / 300, np.nan]),
                        "lo_b2": np.log([299.1 / 300, 300 / 300, np.nan]), "t_hi_b2": [11, 3, -1],
                        "t_lo_b2": [2, 1, -1]})
    o = P.move_outcome(f, "up", 2, 0.01, ext)
    assert list(o.reached.iloc[:2]) == [1.0, 0.0] and np.isnan(o.reached.iloc[2])   # no trade: unknown
    assert np.isclose(300 * (1 + o.centre_pct.iloc[0]), 300 * np.exp(0.02))       # the TSLA example
    assert list(o.touched.iloc[:2]) == [1.0, 0.0] and np.isclose(o.best_pct.iloc[0], 310 / 300 - 1)
    assert o.best_min.iloc[0] == 11 and np.isclose(o.worst_pct.iloc[0], 299.1 / 300 - 1)
    assert o.vwap_pct.isna().all()                                   # an old extremes.parquet: no averages
    e2 = ext.assign(vw_b2=np.log([303 / 300, 1, np.nan]), tw_b2=np.log([302 / 300, 1, np.nan]),
                    cl_b2=np.log([305 / 300, 1, np.nan]), cvw_b2=np.log([302.5 / 300, 1, np.nan]),
                    ctw_b2=np.log([301.5 / 300, 1, np.nan]))
    o = P.move_outcome(f, "up", 2, 0.01, e2)
    assert np.isclose(300 * (1 + o.vwap_pct.iloc[0]), 303) and np.isclose(300 * (1 + o.twap_pct.iloc[0]), 302)
    assert np.isclose(300 * (1 + o.end_pct.iloc[0]), 305) and np.isclose(300 * (1 + o.cvwap_pct.iloc[0]), 302.5)
    assert np.isclose(o.high_pct.iloc[0], 310 / 300 - 1) and o.low_min.iloc[0] == 2
    d = P.move_outcome(f, "dn", 2, 0.01, None)
    assert d.formed.iloc[1] == 0 and d.reached.iloc[1] == 0 and d.touched.isna().all()
    d = P.move_outcome(f, "dn", 2, 0.01, ext)
    assert np.isclose(d.best_pct.iloc[0], 299.1 / 300 - 1) and d.best_min.iloc[0] == 2 and d.touched.iloc[0] == 0


def test_move_recal_uses_only_earlier_days():
    rng = np.random.default_rng(0)
    days = np.repeat([f"d{i:02d}" for i in range(30)], 400)
    p = rng.uniform(0, 1, len(days))
    hit = (rng.uniform(0, 1, len(days)) < p * 0.7).astype(float)      # P too bold by 30%
    cal, warm = P.move_recal(p, hit, days)
    assert not warm[days < "d20"].any() and warm[days >= "d20"].all()
    assert np.allclose(cal[~warm], p[~warm])
    late = warm
    assert abs(cal[late].mean() - hit[late].mean()) < 0.03                         # fixed
    assert abs(p[late].mean() - hit[late].mean()) > 0.1
    h2 = hit.copy()
    h2[days == "d29"] = 1 - h2[days == "d29"]                       # a later day's outcomes change
    cal2, _ = P.move_recal(p, h2, days)
    assert np.allclose(cal2[days < "d29"], cal[days < "d29"])      # earlier days unchanged


def test_move_runs_merge_consecutive_decisions():
    date = ["a"] * 6 + ["b"] * 2
    sym = ["X", "X", "X", "X", "Y", "Y", "X", "X"]
    minute = [3, 0, 1, 2, 0, 1, 0, 1]                              # out of order on purpose
    match = [True, True, True, False, True, False, True, True]
    start, n = P.move_runs(date, sym, minute, match)
    assert list(start) == [True, True, False, False, True, False, True, False]
    assert list(n) == [1, 2, 0, 0, 1, 0, 2, 0]


def test_move_tables_add_up(tmp_path):
    rng = np.random.default_rng(1)
    k = 300
    mt = pd.DataFrame({"date": rng.choice(["2026-01-05", "2026-01-06", "2026-01-07"], k),
                       "side": rng.choice(["up", "dn"], k), "symbol": rng.choice(list("ABCDE"), k),
                       "minute": rng.integers(0, 390, k), "p": rng.uniform(0.6, 0.9, k),
                       "reached": rng.choice([0.0, 1.0, np.nan], k, p=[0.3, 0.6, 0.1]),
                       "touched": rng.choice([0.0, 1.0], k), "formed": 1.0, "centre_pct": 0.01,
                       "best_pct": 0.02, "best_x": rng.uniform(0, 6, k), "vwap_move": 0.005, "captured": 1.0, "warm": True, "p_up": rng.uniform(0, 1, k),
                       "p_dn": rng.uniform(0, 1, k), "reached_up": 1.0, "reached_dn": 0.0,
                       "net_p50": 0.1, "real_net": rng.normal(0, 1, k)})
    s = P.move_summary(mt, {"up": 1000, "dn": 1000})
    assert list(s.side) == ["up", "dn", "both"]
    assert s.predictions.iloc[:2].sum() == s.predictions.iloc[2] == k
    sd = pd.DataFrame([(d, x, 333) for d in ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
                       for x in ("up", "dn")], columns=["date", "side", "decisions"])
    d = P.move_by_day(mt, sd)
    assert d.predictions.sum() == k and (d.happened + d.touched_only + d.did_not <= d.predictions).all()
    assert d.did_not.sum() == int(((mt.reached == 0) & (mt.touched != 1)).sum())   # neither
    assert d[d.date == "2026-01-08"].predictions.eq(0).all()        # a day without a match shows 0
    assert d.happened.sum() == int((mt.reached == 1).sum())
    assert d.touched_only.sum() == int(((mt.reached != 1) & (mt.touched == 1)).sum())
    css = P.move_day_css(d)
    live = d.predictions > 0
    assert (css.loc[live, "happened"] == P.HIT_BG).all() and (css.loc[live, "touched_only"] == P.TOUCH_BG).all()
    assert (css.loc[~live, "happened"] == P.MISS_FG).all() and (css.loc[live, "did_not"] == P.MISS_FG).all()
    assert css.loc[live & (d.mean_best_x >= 1), "mean_best_x"].str.contains("44, 160, 44").all()
    top = mt.groupby(["date", "side"]).p.max()
    assert np.allclose(d.dropna(subset=["max_said"]).set_index(["date", "side"]).max_said, top.reindex(
        d.dropna(subset=["max_said"]).set_index(["date", "side"]).index))      # the day's highest P
    assert (d.max_said.dropna() >= d.said.dropna()).all()
    t = P.move_times(mt)
    assert t.predictions.sum() == k and (t.happened + t.touched_only + t.did_not + t.unknown == t.predictions).all()
    assert t.touched_only.sum() == d.touched_only.sum() and t.happened.sum() == d.happened.sum()
    assert np.isclose(t.mean_best_x.max(), mt.assign(b=mt.minute // 15).query("touched == 1")
                      .groupby(["b", "side"]).best_x.mean().max())       # the overshoot: touched ones only
    fig = charts.pk_mv_times(t, "t")
    assert {tr.name for tr in fig.data} >= {"happened", "price touched only", "didn't",
                                            "how far it went (median best_x)"}
    lean = P.move_lean(mt)
    assert lean.predictions.sum() == k and np.isclose(lean.share.sum(), 1)
    assert P.move_insights(s, lean, 0.01, 2, 0.6, True, 5, 1, True)
    assert "No decision" in P.move_insights(s.iloc[:0], lean, 0.01, 2, 0.6, True, 0, 0, True)[0].text
    # best_x: only the predictions whose price touched the target count
    up = mt[mt.side == "up"]
    assert np.isclose(s.mean_best_x.iloc[0], up.best_x[up.touched == 1].mean())
    assert np.isclose(s.median_best_x.iloc[2], mt.best_x[mt.touched == 1].median())
    one = mt[(mt.date == "2026-01-05") & (mt.side == "dn")]
    assert np.isclose(d.set_index(["date", "side"]).mean_best_x.loc[("2026-01-05", "dn")],
                      one.best_x[one.touched == 1].mean())


def test_move_list_css_marks_hits_misses_and_the_overshoot():
    lst = pd.DataFrame({"side": ["up", "dn", "up", "dn", "up"], "reached": [1.0, 0.0, np.nan, 0.0, 0.0],
                        "touched": [1.0, 1.0, 0.0, 1.0, 0.0], "best_x": [5.0, 3.0, 0.5, 1.0, 0.4],
                        "high": 1.0, "high_pct": 1.0, "low": 1.0, "low_pct": 1.0, "p": 0.5})
    css = P.move_list_css(lst)
    assert css.loc[0, "p"] == P.HIT_BG and css.loc[2, "p"] == ""                  # centre reached; no outcome
    assert css.loc[1, "p"] == css.loc[3, "p"] == P.TOUCH_BG                       # the price only: blue
    assert css.loc[4, "p"] == P.MISS_FG and css.loc[4, "high"] == P.MISS_FG       # neither: dimmed, no shade
    assert "0.55)" in css.loc[0, "high"] and css.loc[0, "low"] == P.HIT_BG        # 5×: the deepest, up side only
    assert "0.35)" in css.loc[1, "low"] and css.loc[1, "high"] == P.TOUCH_BG      # 3× low on a blue row
    assert css.loc[2, "high"] == "" and css.loc[2, "best_x"] == ""                # never touched: no shade
    assert "0.15)" in css.loc[3, "best_x"]                                        # 1×: the lightest


def test_load_extremes(tmp_path):
    run = write_pockets_run(tmp_path)
    e = C.load_extremes(run, "B")
    assert e.index.name == "dec_row" and {"hi_b2", "t_lo_b5"} <= set(e.columns)
    assert C.load_extremes(write_pockets_run(tmp_path, "x", extremes=False), "B") is None


def test_move_cum_combines_the_windows_from_the_entry():
    p = [np.array([0.1, 0.0, 0.5]), np.array([0.2, 0.0, 0.5])]
    assert np.allclose(P.move_prob_cum(p), [1 - 0.9 * 0.8, 0.0, 0.75])
    assert np.allclose(P.move_prob_cum(p[:1]), p[0])                # one window = itself
    nan = np.nan
    o1 = pd.DataFrame({"formed": [1.0, nan, 0.0], "reached": [0.0, nan, 0.0], "touched": [1.0, nan, 0.0],
                       "centre_pct": [0.005, nan, nan], "high_pct": [0.012, nan, 0.002],
                       "high_min": [4.0, nan, 2.0], "low_pct": [-0.003, nan, -0.02], "low_min": [3.0, nan, 1.0],
                       "cvwap_pct": [0.004, nan, 0.0], "ctwap_pct": [0.003, nan, 0.0], "end_pct": [0.01, nan, 0.0]})
    o2 = pd.DataFrame({"formed": [1.0, nan, 1.0], "reached": [1.0, nan, 0.0], "touched": [1.0, nan, 0.0],
                       "centre_pct": [0.011, nan, 0.004], "high_pct": [0.010, nan, 0.006],
                       "high_min": [9.0, nan, 12.0], "low_pct": [-0.008, nan, -0.001], "low_min": [8.0, nan, 7.0],
                       "cvwap_pct": [0.007, nan, 0.002], "ctwap_pct": [0.006, nan, 0.001], "end_pct": [0.009, nan, 0.003]})
    up = P.move_outcome_cum([o1, o2], "up")
    assert list(up.reached.iloc[[0, 2]]) == [1.0, 0.0] and np.isnan(up.reached.iloc[1])   # no trade: unknown
    assert np.isclose(up.centre_pct.iloc[0], 0.011) and np.isclose(up.best_pct.iloc[0], 0.012)
    assert up.best_min.iloc[0] == 4 and up.best_min.iloc[2] == 12 and np.isnan(up.best_min.iloc[1])
    assert np.isclose(up.worst_pct.iloc[0], -0.008) and up.formed.iloc[2] == 1
    # the VWAP / TWAP from the entry and the last price: the last window's from-the-entry numbers
    assert np.isclose(up.vwap_pct.iloc[0], 0.007) and np.isclose(up.twap_pct.iloc[0], 0.006)
    assert np.isclose(up.end_pct.iloc[0], 0.009)
    dn = P.move_outcome_cum([o1, o2], "dn")
    assert np.isclose(dn.best_pct.iloc[0], -0.008) and dn.best_min.iloc[0] == 8
    assert np.isclose(dn.best_pct.iloc[2], -0.02) and dn.best_min.iloc[2] == 1
    assert np.isclose(dn.worst_pct.iloc[0], 0.012) and np.isclose(dn.low_pct.iloc[0], up.low_pct.iloc[0])
    # the predicted centre: the window with the highest P, and that window's real centre
    c = P.move_centre_cum([np.array([0.3, 0.1]), np.array([0.2, 0.4])], [np.array([1.0, 1.1]), np.array([2.0, 2.1])],
                          [np.array([3.0, 3.1]), np.array([4.0, 4.1])], [np.array([0.01, 0.02]), np.array([0.03, 0.04])])
    assert list(c["win"]) == [1, 2] and list(c["p50"]) == [1.0, 2.1] and list(c["p90"]) == [3.0, 4.1]
    assert list(c["centre_pct"]) == [0.01, 0.04]
    assert P.move_window(4, True) == "0-60 min" and P.move_window(4, False) == "30-60 min"
