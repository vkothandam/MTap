"""Home: one card per section with the newest run's headline numbers and a button to open it."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from forecast_dash import calib, insights
from forecast_dash import intraday_calib as ic
from forecast_dash.contract import discover, run_label
from forecast_dash.intraday_contract import discover_intraday, load_config
from forecast_dash.intraday_page import get_report, get_summary
from forecast_dash.ui import ICONS, _pct_cols, _table, get_live, get_run, goto_section

SHORT = ["r_1m", "r_5m", "r_15m", "r_60m"]


def render(daily_root: str, intraday_root: str):
    st.title("Forecast dashboard")
    st.markdown("How good are the VWAP forecasts? **Cross-day** grades the next-day forecasts; "
                "**Intraday** grades forecasts made through the session. Both are graded on "
                "days the model never trained on.")
    daily = discover(Path(daily_root))
    if not daily:
        st.warning(f"No runs under `{daily_root}` (looking for "
                   "fold_*/test_predictions.parquet).")
    a, b = st.columns(2)
    with a, st.container(border=True):
        _daily_card(daily[0] if daily else None)
    with b, st.container(border=True):
        _intraday_card(discover_intraday(Path(intraday_root)), intraday_root)


def _daily_card(run: Path | None):
    st.subheader("Cross-day")
    st.caption("Next-day VWAP: 19 quantiles per stock and day")
    if run is None:
        st.info("No cross-day run yet.")
    else:
        r = get_run(str(run))
        g = calib.live_grade(r.r_q, r.r_act, r.levels, 0.1, 0.9)
        st.markdown(f"Newest run: `{run_label(run)}` · {g['rows']:,} test forecasts")
        m = st.columns(3)
        m[0].metric("P10–P90 coverage", f"{g['coverage']:.1%}",
                    f"{g['coverage'] - g['coverage_nominal']:+.1%} vs 80%", delta_color="off",
                    help="share of actual next-day VWAPs inside the P10–P90 band; target 80%")
        m[1].metric("Pinball skill", f"{g['pinball_skill']:.1%}",
                    help="1 − pinball loss / the loss of forecasting no move. Above 0 beats it")
        m[2].metric("Direction hit", f"{g['direction_hit']:.1%}",
                    help="how often the median had the right sign of the move (50% = coin flip)")
        fc = r.forecasts()
        if fc:
            st.markdown(f"Latest forecast day: **{fc[0].stem.removeprefix('forecast_')}**")
        live = get_live(str(run))
        if live is not None:
            by_day = calib.live_by_day(live.keys["date"], live.r_q, live.r_act, live.levels,
                                       0.1, 0.9)
            for i in insights.live_headline(by_day, 0.1, 0.9):
                st.markdown(f"{ICONS[i.level]} {i.text}")
    st.button("Open Cross-day", key="open_cross_day", on_click=goto_section,
              args=("cross-day",), disabled=run is None)


def _intraday_card(runs: list[Path], root: str):
    st.subheader("Intraday")
    st.caption("VWAP of the next minutes, the rest of the day and the whole day")
    if not runs:
        st.info(f"No intraday run under `{root}`: the model isn't trained yet. "
                "Train one with `tft-intraday`.")
    else:
        path = str(runs[0])
        h = ic.headline(get_report(path), {})
        if h.empty:
            s = get_summary(path, "trail")
            h = s.reindex(columns=ic.HEADLINE)
        h = h[h["target"].isin(SHORT)].copy()
        days = _test_days(path)
        st.markdown(f"Newest run: `{runs[0].name}`" + (f" · {days} test days" if days else ""))
        if h.empty:
            st.info("This run has no short-horizon scores yet.")
        else:
            h["target"] = h["target"].map(ic.TARGET_SHORT)
            h = h.rename(columns={"target": "horizon"})
            if h["skill_clim"].isna().all():  # a run from before the climatology baseline
                h = h.drop(columns="skill_clim")
            _table(h, "home_intraday", _pct_cols(h, ["coverage80", "skill_clim", "skill_mom",
                                                     "dir_last"]))
    st.button("Open Intraday", key="open_intraday", on_click=goto_section,
              args=("intraday",))


def _test_days(path: str) -> int:
    """Test days in the run's config (0 if it does not list them)."""
    folds = load_config(Path(path)).get("folds", [])
    return len({d for f in folds for d in f.get("test_dates", [])})
