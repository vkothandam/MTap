"""The Watchlist section: today's screener list (stocks likely to travel far from the open) with
sliders over the rule filters, and a tab grading past lists against what happened."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_dash.charts import _layout
from forecast_dash.ui import _chart, _header, _pct_cols, _table
from forecast_dash.watchlist_contract import (
    EARNINGS,
    Rules,
    apply_rules,
    grade_lists,
    list_dates,
    load_list,
    load_outcomes,
    load_rules,
)

SHOW = ["rank", "symbol", "score", "pred_range", "lean", "open", "gap_prev_close", "tft_width90",
        "prev_range", "atr14_pct", "dollar_volume_20d", "pm_vol_pct_10d", "earn_near"]


def _settings() -> Path | None:
    env = os.environ.get("FORECAST_DASH_SETTINGS")
    if env:
        return Path(env)
    for p in ("../shared/config/settings.toml", "../shared/config/settings.example.toml"):
        if Path(p).exists():
            return Path(p)
    return None


def render(root: str):
    st.title("Watchlist")
    sb = st.sidebar
    root = sb.text_input("Screener directory", root, key="wl_root",
                         help="holds screener-<date>.parquet (and outcomes.parquet)")
    dates = list_dates(Path(root))
    if not dates:
        st.info(f"No screener lists under `{root}` (looking for screener-<date>.parquet). "
                "Make one with `screener-score`.")
        return
    d = Rules() if _settings() is None else load_rules(_settings())
    sb.markdown("**Rule filters** (start at the `[screener]` settings)")
    rules = Rules(
        min_price=sb.number_input("Min price ($)", 0.0, 10_000.0, float(d.min_price), 1.0,
                                  key="wl_price", help="open (or last) price must be at least this"),
        min_dollar_volume=sb.number_input(
            "Min 20-day dollar volume ($M)", 0.0, 100_000.0, d.min_dollar_volume / 1e6, 1.0,
            key="wl_dv", help="mean of volume x VWAP over 20 sessions, in millions") * 1e6,
        min_pm_vol_pct=sb.slider("Min pre-market volume ratio", 0.0, 6.0, float(d.min_pm_vol_pct),
                                 0.1, key="wl_pmv", help="log1p(100 x pre-market volume / 10-day "
                                 "mean volume); only the 09:45 list has it"),
        min_abs_gap_pct=sb.slider("Min |gap| at the open (%)", 0.0, 20.0,
                                  float(d.min_abs_gap_pct) * 100, 0.1, key="wl_gap",
                                  help="open vs previous close, either direction") / 100,
        exclude_split_day=sb.checkbox("Exclude split days", d.exclude_split_day, key="wl_split",
                                      help="prices around a split are not comparable"),
        earnings=sb.selectbox("Earnings", EARNINGS, EARNINGS.index(d.earnings), key="wl_earn",
                              help="any, only the days around a results release, or exclude them"),
        top_n=int(sb.number_input("Show top", 1, 200, int(d.top_n), key="wl_top")),
    )
    tabs = st.tabs(["Today's list", "Screener: picks vs outcome"])
    with tabs[0]:
        tab_today(Path(root), dates, rules)
    with tabs[1]:
        tab_outcomes(Path(root), rules)


def tab_today(root: Path, dates: list[str], rules: Rules):
    _header("wl_today")
    date = st.selectbox("List", dates, key="wl_date", help="a morning's list, newest first")
    lst = apply_rules(load_list(root, date), rules)
    state = lst["state"].iloc[0] if "state" in lst else "?"
    passing = lst[lst["rule_pass"]]
    st.caption(f"{date} · state **{state}** · {len(lst):,} stocks scored · "
               f"{len(passing):,} pass the rules")
    top = passing.head(rules.top_n)
    if top.empty:
        st.info("No stock passes the rules; loosen a filter in the sidebar.")
        return
    t = top[[c for c in SHOW if c in top.columns]]
    _table(t, "wl_table", {**_pct_cols(t, ["pred_range", "gap_prev_close", "tft_width90",
                                           "prev_range", "atr14_pct"]),
                           "score": st.column_config.NumberColumn("score", format="percent"),
                           "dollar_volume_20d": st.column_config.NumberColumn(
                               "dollar_volume_20d", format="compact")})
    fig = go.Figure(go.Bar(x=top["symbol"], y=top["score"], marker_color="#1f77b4"))
    fig.update_yaxes(tickformat=".1%", title="score (80th-percentile range)")
    _chart(_layout(fig, "Score of the top picks", height=340), "wl_chart")


def tab_outcomes(root: Path, rules: Rules):
    _header("wl_outcome")
    out = load_outcomes(root)
    if out is None:
        st.info("No `outcomes.parquet` yet: `screener-outcomes --panel ... --out-dir ...` writes "
                "what each stock did.")
        return
    g = grade_lists(root, out, rules)
    if g.empty:
        st.info("No past list has outcomes yet (a list is graded the day after it, once the "
                "panel has that day's bars).")
        return
    st.markdown(f"**{len(g)} graded lists** · mean top range {g['top_range'].mean():.2%} vs "
                f"universe median {g['universe_median'].mean():.2%} · hit rate "
                f"{g['hit_rate'].mean():.0%} (picks above the universe median; 50% = chance)")
    fig = go.Figure()
    fig.add_bar(x=g["date"], y=g["top_range"], name="top picks (mean)")
    fig.add_scatter(x=g["date"], y=g["universe_median"], name="universe median", mode="lines+markers")
    fig.update_yaxes(tickformat=".1%", title="range from the open")
    _chart(_layout(fig, "Realised range from the open: picks vs universe", height=380),
           "wl_outcome_chart")
    _table(g, "wl_outcome_table", _pct_cols(g, ["top_range", "universe_median", "hit_rate",
                                                "predicted"]))
