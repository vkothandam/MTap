"""Shared Streamlit helpers: the page header, the ⓘ popovers, and the chart / table / In short
wrappers every page uses (explanations live in help_text.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from forecast_dash import insights
from forecast_dash.contract import Live, Run, load_live, load_run, read_card
from forecast_dash.help_text import HELP, PANELS, column_help

ICONS = {insights.GOOD: "✅", insights.WARN: "⚠️", insights.BAD: "❌", insights.INFO: "ℹ️"}


def _pct_cols(df: pd.DataFrame, cols) -> dict:
    return {c: st.column_config.NumberColumn(c, format="percent") for c in cols if c in df}


def _header(key: str):
    """Page title with its "About this page" popover (help_text.HELP)."""
    title, body = HELP[key]
    a, b = st.columns([6, 1], vertical_alignment="bottom")
    a.subheader(title)
    with b.popover("ℹ️ About this page"):
        st.markdown(body)


def _about(panel: str, columns=()):
    """The ⓘ popover above one chart / table / sub-panel: what it shows and how to read it
    (help_text.PANELS), plus the meaning of each of its columns."""
    title, body = PANELS[panel]
    cols = [(c, h) for c in columns if (h := column_help(c, panel))]
    if cols:
        body += "\n\n**Columns**\n" + "\n".join(f"- `{c}`: {h}" for c, h in cols)
    with st.popover(title, icon=":material/info:", type="tertiary", key=f"about_{panel}",
                    help="What this shows and how to read it"):
        st.markdown(body)


def _model_popover(where, run: Path):
    """ⓘ with the run's MODEL.md (purpose, inputs, folds, result), when it has one."""
    card = read_card(run)
    if card is None:
        where.caption("No MODEL.md in this run: no description.")
        return
    with where.popover(f"ℹ️ {card.label}: about this model"):
        st.markdown(f"**{card.label}** · {card.forecast} forecast  \n{card.summary}\n\n"
                    f"{card.body}\n\n`{run.name}`")


def _chart(fig, panel: str, **kw):
    _about(panel)
    return st.plotly_chart(fig, width="stretch", **kw)


def _table(df: pd.DataFrame, panel: str, column_config: dict | None = None, **kw):
    """st.dataframe with the ⓘ above it and every known column's meaning as a header tooltip."""
    _about(panel, df.columns)
    cfg = dict(column_config or {})
    for c in df.columns:
        if h := column_help(c, panel):
            cfg[c] = {**(cfg.get(c) or st.column_config.Column()), "help": h}
    kw.setdefault("hide_index", True)
    st.dataframe(df, column_config=cfg, **kw)


def _in_short(items: list[insights.Insight], title: str = "In short"):
    if not items:
        return
    with st.container(border=True):
        st.markdown(f"**{title}**\n\n" + "\n".join(f"- {ICONS[i.level]} {i.text}"
                                                    for i in items),
                    help="Plain-English findings from the current filters. ✅ fine · ⚠️ watch · "
                         "❌ problem · ℹ️ context")


# ---- cached run loading, shared by Home and Cross-day ---------------------------------------

@st.cache_resource(show_spinner="Loading run...", max_entries=6)
def get_run(path: str) -> Run:
    return load_run(Path(path))


def get_live(path: str) -> Live | None:
    """Graded live forecasts; re-read when actuals.parquet or a forecast file changes."""
    run = Path(path)
    stamp = tuple(sorted((f.name, f.stat().st_mtime) for f in
                         [*run.glob("forecast_*.parquet"), *run.glob("actuals.parquet")]))
    return _load_live(path, stamp)


@st.cache_resource(max_entries=6)
def _load_live(path: str, stamp: tuple) -> Live | None:
    return load_live(Path(path))


# ---- sections ------------------------------------------------------------------------------

SECTIONS = {"home": "Home", "cross-day": "Cross-day", "intraday": "Intraday",
            "watchlist": "Watchlist"}


def section_radio() -> str:
    """The sidebar Section radio, seeded from ?section= and written back to it. Returns the
    section slug ("home", "cross-day" or "intraday")."""
    if "section" not in st.session_state:
        slug = st.query_params.get("section", "home")
        st.session_state["section"] = SECTIONS.get(slug, SECTIONS["home"])
    label = st.sidebar.radio("Section", list(SECTIONS.values()), key="section",
                             help="Home: the newest runs at a glance. Cross-day: next-day VWAP "
                                  "forecasts. Intraday: VWAP forecasts through the session. Watchlist: the day's screener picks")
    slug = next(k for k, v in SECTIONS.items() if v == label)
    if st.query_params.get("section") != slug:
        st.query_params["section"] = slug
    return slug


def goto_section(slug: str):
    """Button callback: switch the Section radio (callbacks may set widget state)."""
    st.session_state["section"] = SECTIONS[slug]
    st.query_params["section"] = slug
