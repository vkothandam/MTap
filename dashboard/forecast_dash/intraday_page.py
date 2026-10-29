"""The Intraday section: six tabs grading the intraday model (VWAP of the next minutes, the
rest of the day, the whole day) on the days it never trained on. Run `tft-intraday` to make a run
(see intraday_contract.py for the files)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from forecast_dash import charts
from forecast_dash import intraday_calib as ic
from forecast_dash.intraday_contract import (
    discover_intraday,
    load_config,
    load_intraday,
    load_report,
    targets_of,
)
from forecast_dash.contract import run_label
from forecast_dash.ui import _chart, _header, _in_short, _model_popover, _pct_cols, _table

# columns each page needs, so a big run loads only what it grades
_NEED = ["symbol", "date", "t_s", "tod_bucket", "sector_code", "actual", "ref_trail",
         "ref_session", "last", *ic.QCOLS, *ic.MOMCOLS, *ic.CLIMCOLS]
_DAILY = ["has_daily", *ic.DAILYCOLS]


# ---- cached data (cache_resource: shared, never copied; treat as read-only) -----------------

@st.cache_resource(show_spinner="Loading intraday run...", max_entries=24)
def get_frame(path: str, target: str) -> pd.DataFrame:
    cols = _NEED + (_DAILY if target == "r_close" else [])
    return load_intraday(Path(path), target, columns=cols)


@st.cache_resource(max_entries=6)
def get_report(path: str) -> dict:
    return load_report(Path(path))


def get_frames(path: str) -> dict[str, pd.DataFrame]:
    return {t: get_frame(path, t) for t in targets_of(Path(path))}


@st.cache_resource(max_entries=12)
def get_summary(path: str, ref: str) -> pd.DataFrame:
    return ic.summary(get_frames(path), ref)


# ---- page -----------------------------------------------------------------------------------

def render(root: str):
    st.title("Intraday forecasts")
    sb = st.sidebar
    root = sb.text_input("Intraday runs directory", root, key="id_root")
    runs = discover_intraday(Path(root))
    if not runs:
        st.info(f"No intraday runs under `{root}` (looking for config.json with model "
                "`tft_intraday` and fold_*/intraday_<target>.parquet). Training hasn't run yet: "
                "make one with `tft-intraday`.")
        return
    names = [r.name for r in runs]
    labels = {r.name: run_label(r) for r in runs}
    name = sb.selectbox("Intraday run", names, key="id_run", format_func=labels.get)
    path = str(runs[names.index(name)])
    _model_popover(sb, Path(path))
    cfg = load_config(Path(path))
    targets = targets_of(Path(path))
    target = sb.selectbox("Target", targets, key="id_target", format_func=ic.target_label,
                          help="what is forecast: the VWAP of the next minutes, the rest of "
                               "the day, or the whole day")
    ref = sb.selectbox("Reference", list(ic.REF_LABELS), key="id_ref",
                       format_func=ic.REF_LABELS.get,
                       help="what the move is measured against. It changes direction and "
                            "skill vs the random walk, not coverage")
    df = get_frame(path, target)
    folds = cfg.get("folds", [])
    sb.caption(f"{len(df):,} decision points · {df['symbol'].nunique():,} stocks · "
               f"{df['date'].nunique():,} test days · {len(folds) or df['fold'].nunique()} folds")
    with sb.expander("Run info"):
        st.json({k: cfg[k] for k in ("model", "step_s", "eval_stride_s", "daily_run")
                 if k in cfg} or "n/a")
    if df.empty:
        st.info("This target has no valid decision points.")
        return
    tabs = st.tabs(["Summary", "Calibration", "Time of day", "Session replay", "Across stocks",
                    "Intraday vs daily"])
    with tabs[0]:
        tab_summary(path, target, ref, df)
    with tabs[1]:
        tab_calibration(target, ref, df)
    with tabs[2]:
        tab_tod(target, ref, df)
    with tabs[3]:
        tab_replay(target, df)
    with tabs[4]:
        tab_stocks(target, ref, df)
    with tabs[5]:
        tab_vs_daily(path)


def _tgt_caption(target: str, ref: str):
    st.caption(f"Target: **{ic.target_label(target)}** · reference: "
               f"**{ic.REF_LABELS[ref]}** (both set in the sidebar)")


def _score_cfg(t: pd.DataFrame) -> dict:
    """Display formats: rates as %, bias with a sign, loss to 5 decimals."""
    fmt = {"pit_bias": "%+.3f", "pinball": "%.5f", "width80_med": "percent"}
    return {**_pct_cols(t, ["coverage80", "skill_clim", "skill_rw", "skill_mom", "dir_last",
                            "dir_trail",
                            "dir_session"]),
            **{c: st.column_config.NumberColumn(c, format=f) for c, f in fmt.items() if c in t}}


def tab_summary(path: str, target: str, ref: str, df: pd.DataFrame):
    _header("id_summary")
    s = get_summary(path, ref)
    _in_short(ic.summary_insights(s, ref))
    _chart(charts.id_skill_bars(s, ic.TARGET_LABELS), "id_sum_chart")
    t = s.assign(target=s["target"].map(ic.target_label))
    _table(t, "id_sum_table", _score_cfg(t))
    _tgt_caption(target, ref)
    d = ic.by_day(df, ref)
    _chart(charts.id_days(d), "id_sum_days_chart")
    with st.expander("Scores per test day"):
        _table(d, "id_sum_days", _score_cfg(d))


def tab_calibration(target: str, ref: str, df: pd.DataFrame):
    _header("id_calibration")
    _tgt_caption(target, ref)
    qc = ic.quantile_calibration(df, ref)
    g = ic.prob_grid(df, ref)
    _in_short(ic.calibration_insights(qc, g, len(df)))
    _chart(charts.quantile_chart(qc), "id_cal_quantiles")
    _chart(charts.reliability(g, "P(above reference): said vs came true"), "id_cal_grid")


def tab_tod(target: str, ref: str, df: pd.DataFrame):
    _header("id_tod")
    _tgt_caption(target, ref)
    t = ic.by_tod(df, ref)
    _in_short(ic.tod_insights(t))
    _chart(charts.id_tod(t), "id_tod_chart")
    with st.expander("Time-of-day table"):
        _table(t, "id_tod_table", _score_cfg(t))


def tab_replay(target: str, df: pd.DataFrame):
    _header("id_replay")
    days = ic.replay_choices(df)
    a, b = st.columns(2)
    date = a.selectbox("Day", list(days)[::-1], key="id_day",
                       help="a test day, newest first")
    symbol = b.selectbox("Stock", days[date], key="id_sym", help="a stock with decision "
                         "points that day")
    r = ic.replay(df, date, symbol)
    _in_short(ic.replay_insights(r, date, symbol))
    _chart(charts.id_replay(r, f"{symbol} · {date} · {ic.target_label(target)}"), "id_replay")


def tab_stocks(target: str, ref: str, df: pd.DataFrame):
    _header("id_stocks")
    _tgt_caption(target, ref)
    min_n = st.number_input("Min decision points per stock", 1, 100_000, 1, key="id_min_n",
                            help="hide stocks with fewer decision points: their rates are noise")
    t = ic.by_symbol(df, ref, int(min_n))
    if t.empty:
        st.info("No stocks have that many decision points.")
        return
    _in_short(ic.stocks_insights(t))
    _chart(charts.id_stocks(t), "id_stocks_chart")
    _table(t, "id_stocks_table", _score_cfg(t))
    sec = ic.by_sector(df, ref)
    _table(sec, "id_stocks_sectors", _score_cfg(sec))


def tab_vs_daily(path: str):
    _header("id_vs_daily")
    st.caption("Always the **Whole-day VWAP** target: the one number the daily model also "
               "forecasts.")
    if "r_close" not in targets_of(Path(path)):
        st.info("This run has no whole-day VWAP (`r_close`) predictions.")
        return
    vd = ic.vs_daily(get_frame(path, "r_close"))
    _in_short(ic.vs_daily_insights(vd))
    if vd.empty:
        return
    _chart(charts.id_vs_daily(vd), "id_vs_chart")
    _table(vd, "id_vs_table", {**_pct_cols(vd, ["skill"]),
                               **{c: st.column_config.NumberColumn(c, format="%.5f")
                                  for c in ("intraday_pinball", "daily_pinball")}})
