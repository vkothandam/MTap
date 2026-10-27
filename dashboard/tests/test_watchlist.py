import ast
import json

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from test_app import PKG, _assert_explained

from forecast_dash.watchlist_contract import Rules, apply_rules, grade_lists, load_rules


def write_screener(root, dates=("2026-10-26", "2026-10-27"), n=40, seed=0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = []
    for date in dates:
        syms = [f"S{i}" for i in range(n)]
        score = rng.uniform(0.01, 0.1, n)
        lst = pd.DataFrame({
            "symbol": syms, "date": date, "state": "px", "score": score,
            "pred_range": score * 0.7, "lean": np.where(rng.random(n) > 0.5, "up", "down"),
            "open": rng.uniform(3, 100, n), "last": 10.0, "prev_close": 10.0,
            "dollar_volume_20d": rng.uniform(1e5, 1e8, n), "gap_prev_close": rng.normal(0, .02, n),
            "tft_width90": 0.05, "prev_range": 0.03, "atr14_pct": 0.03,
            "earn_near": (rng.random(n) > 0.8).astype(float), "split_day": False})
        lst["rule_pass"], lst["rank"] = True, 1
        lst.to_parquet(root / f"screener-{date}.parquet", index=False)
        # realised range follows the score, so the picks beat the median
        out.append(pd.DataFrame({"date": date, "symbol": syms, "open": 10.0, "high": 11.0,
                                 "low": 9.0, "range_from_open": score * rng.uniform(.8, 1.2, n),
                                 "hl_range_open": score}))
    pd.concat(out).to_parquet(root / "outcomes.parquet", index=False)


def test_apply_rules_and_grade(tmp_path):
    write_screener(tmp_path)
    lst = pd.read_parquet(tmp_path / "screener-2026-10-27.parquet")
    r = apply_rules(lst, Rules(min_price=50, min_dollar_volume=0))
    assert (lst.loc[lst["symbol"].isin(r.loc[r["rule_pass"], "symbol"]), "open"] >= 50).all()
    passing = r[r["rule_pass"]]
    assert passing["rank"].tolist() == list(range(1, len(passing) + 1))
    assert passing["score"].is_monotonic_decreasing
    only = apply_rules(lst, Rules(earnings="only", min_price=0, min_dollar_volume=0))
    assert (only.loc[only["rule_pass"], "earn_near"] == 1).all()
    g = grade_lists(tmp_path, pd.read_parquet(tmp_path / "outcomes.parquet"),
                    Rules(min_price=0, min_dollar_volume=0))
    assert len(g) == 2 and (g["top_range"] > g["universe_median"]).all()
    assert (g["hit_rate"] > 0.7).all()


def test_load_rules(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[screener]\nmin_price = 9\nearnings = "exclude"\n')
    assert load_rules(p).min_price == 9 and load_rules(None) == Rules()


def test_watchlist_page_renders(tmp_path, monkeypatch):
    write_screener(tmp_path / "scr")
    monkeypatch.setenv("FORECAST_DASH_SCREENER", str(tmp_path / "scr"))
    monkeypatch.setenv("FORECAST_DASH_RUNS", str(tmp_path / "runs"))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120)
    at.query_params["section"] = "watchlist"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Today's list", "Screener: picks vs outcome"]
    assert len(at.tabs[0].dataframe[0].value) == 20
    assert any("graded lists" in m.value for m in at.tabs[1].markdown)
    _assert_explained(at)
    at.sidebar.selectbox(key="wl_earn").set_value("only")
    at.sidebar.number_input(key="wl_price").set_value(50.0)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    shown = at.tabs[0].dataframe[0].value
    assert len(shown) < 20 and (shown["earn_near"] == 1).all()


def test_watchlist_without_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECAST_DASH_SCREENER", str(tmp_path))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=60)
    at.query_params["section"] = "watchlist"
    at.run()
    assert not at.exception and at.info


def test_watchlist_panels_have_help():
    from forecast_dash.help_text import HELP, PANELS, column_help

    tree = ast.parse((PKG / "watchlist_page.py").read_text())
    used = {n.args[1].value for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) in ("_chart", "_table")}
    assert used and used <= set(PANELS)
    assert {"wl_today", "wl_outcome"} <= set(HELP)
    assert column_help("hit_rate") and column_help("rank") and json
