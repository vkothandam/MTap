import numpy as np
import pandas as pd
from conftest import ID_SYMBOLS, intraday_frame

from forecast_dash import intraday_calib as ic

DATES = ["2025-03-03", "2025-03-04"]


def frame(target="r_5m", **kw):
    f = intraday_frame(target, DATES, **kw)
    return f[f["mask"] == 1].reset_index(drop=True)


def test_shift_moves_quantiles_and_actual_to_the_reference():
    f = frame()
    assert np.allclose(ic.shift(f, "last"), np.sort(f[ic.QCOLS].to_numpy(), axis=1))
    assert np.allclose(ic.shift(f, "trail")[:, 9], f["q50"] - f["ref_trail"])
    assert np.allclose(ic.shift_actual(f, "session"), f["actual"] - f["ref_session"])


def test_calibrated_forecasts_cover_80_and_have_no_bias():
    big = pd.concat([frame(seed=s) for s in range(20)], ignore_index=True)
    s = ic.score(big, "last")
    assert abs(s["coverage80"] - 0.8) < 0.03 and abs(s["pit_bias"]) < 0.02
    assert s["skill_mom"] > 0  # the momentum baseline is wider than the truth
    assert 0 < s["skill_clim"] < s["skill_mom"]  # climatology is blind to mu, sigma: harder
    assert np.isclose(ic.score(big, "trail")["coverage80"], s["coverage80"])  # shift-invariant


def test_biased_forecasts_are_flagged():
    s = ic.score(frame(bias=0.01), "last")
    assert s["coverage80"] < 0.5 and s["pit_bias"] < -0.1


def test_skill_against_random_walk_follows_the_reference():
    f = frame()
    s_last, s_trail = ic.score(f, "last"), ic.score(f, "trail")
    assert np.isclose(s_last["pinball"], s_trail["pinball"])
    assert s_last["skill_rw"] != s_trail["skill_rw"]


def test_direction_hit_is_a_rate_and_ignores_zero_moves():
    f = frame()
    f.loc[f.index[:5], "actual"] = 0.0  # actual == last price: no direction
    d = ic._direction(f, "last")
    assert np.isnan(d[:5]).all() and np.isin(d[5:], [0, 1]).all()
    s = ic.score(f, "last")
    assert 0.3 < s["dir_last"] < 0.9 and 0 <= s["dir_trail"] <= 1


def test_summary_has_a_row_per_target():
    frames = {t: frame(t) for t in ("r_1m", "r_close")}
    s = ic.summary(frames, "trail")
    assert list(s["target"]) == ["r_1m", "r_close"]
    assert {"coverage80", "skill_clim", "skill_mom", "skill_rw", "dir_trail", "pit_bias",
            "width80_med"} <= set(s)
    assert s["width80_med"].iloc[1] > s["width80_med"].iloc[0]  # longer horizon, wider band


def test_runs_without_climatology_fall_back_to_momentum():
    old = frame(with_clim=False)
    s = ic.summary({"r_5m": old}, "trail")
    assert s["skill_clim"].isna().all() and ic.skill_col(s) == "skill_mom"
    assert "momentum baseline" in ic.summary_insights(s, "trail")[0].text
    new = ic.summary({"r_5m": frame()}, "trail")
    assert ic.skill_col(new) == "skill_clim"
    assert "climatology baseline" in ic.summary_insights(new, "trail")[0].text
    assert ic.by_symbol(frame()).equals(
        ic.by_symbol(frame()).sort_values("skill_clim", ascending=False, ignore_index=True))


def test_groupings():
    f = intraday_frame("r_15m", DATES)
    f["sector_code"] = f["sector_code"].mask(f["symbol"] == "S5")  # one stock without a sector
    f = f[f["mask"] == 1]
    tod = ic.by_tod(f)
    assert tod["n"].sum() == len(f) and tod["time"].iloc[0] == "09:30"
    sym = ic.by_symbol(f)
    assert sorted(sym["symbol"]) == ID_SYMBOLS
    assert sym.set_index("symbol").loc["S5", "sector_code"] == "unknown"
    assert set(ic.by_sector(f)["sector_code"]) == {"tech", "energy", "unknown"}
    assert list(ic.by_day(f)["date"]) == DATES
    assert ic.by_symbol(f, min_n=10**6).empty
    assert ic.by_group(f.iloc[:0], "symbol").empty


def test_quantile_calibration_and_prob_grid():
    big = pd.concat([frame(seed=s) for s in range(20)], ignore_index=True)
    qc = ic.quantile_calibration(big, "last")
    assert np.allclose(qc["observed"], qc["level"], atol=0.04)
    g = ic.prob_grid(big, "last")
    assert (g["event"] == "above last price").all() and g["n"].sum() == len(big)


def test_vs_daily():
    f = frame("r_close", with_daily=True)
    vd = ic.vs_daily(f)
    assert vd["n"].sum() == int((f["has_daily"] == 1).sum())
    assert ic.vs_daily_total(vd) > 0  # the daily bands are wider than the truth
    assert ic.vs_daily(frame("r_5m")).empty and np.isnan(ic.vs_daily_total(pd.DataFrame()))


def test_replay_prices_and_choices():
    f = frame("r_close", with_daily=True)
    assert ic.replay_choices(f)["2025-03-03"] == ID_SYMBOLS
    r = ic.replay(f, "2025-03-03", "S1")
    assert r["t_s"].is_monotonic_increasing and len(r) > 10
    row = f[(f["date"] == "2025-03-03") & (f["symbol"] == "S1")].sort_values("t_s").iloc[3]
    assert np.isclose(r["actual_vwap"][3], row["last"] * np.exp(row["actual"]))
    assert np.isclose(r["p90"][3], row["last"] * np.exp(row["q90"]))
    assert (r["p10"] <= r["p50"]).all() and (r["p50"] <= r["p90"]).all()
    assert "daily_p10" in r and r["daily_p10"].notna().any()
    assert "daily_p10" not in ic.replay(frame("r_5m"), "2025-03-03", "S1")


def test_headline_prefers_the_report_else_computes():
    frames = {t: frame(t) for t in ("r_1m", "r_5m")}
    computed = ic.headline(None, frames)
    assert list(computed["target"]) == ["r_1m", "r_5m"]
    rep = pd.DataFrame({"target": ["r_1m", "r_1m"], "model": ["tft", "momentum"],
                        "n": [9, 9], "coverage80": [0.8, 0.5], "skill_mom": [0.2, None],
                        "dir_last": [0.6, 0.5]})
    h = ic.headline({"by_target": rep}, frames)
    assert list(h["target"]) == ["r_1m"] and h["skill_mom"].iloc[0] == 0.2
    assert list(h.columns) == ic.HEADLINE and h["skill_clim"].isna().all()  # an older report


def test_insights_return_text_for_every_view():
    f = frame("r_close", with_daily=True)
    s = ic.summary({"r_close": f, "r_5m": frame()}, "trail")
    items = (ic.summary_insights(s, "trail")
             + ic.calibration_insights(ic.quantile_calibration(f), ic.prob_grid(f), len(f))
             + ic.tod_insights(ic.by_tod(f)) + ic.stocks_insights(ic.by_symbol(f))
             + ic.replay_insights(ic.replay(f, "2025-03-03", "S0"), "2025-03-03", "S0")
             + ic.vs_daily_insights(ic.vs_daily(f)) + ic.vs_daily_insights(pd.DataFrame()))
    assert items and all(i.text for i in items)
    assert ic.summary_insights(pd.DataFrame(), "trail") == []
