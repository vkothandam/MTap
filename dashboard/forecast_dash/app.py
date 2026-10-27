"""Streamlit UI. Run with `forecast-dash --runs <dir>` (or `streamlit run app.py -- --runs <dir>`).

Every view is computed live from the run's saved test forecasts, filtered by the sidebar.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit re-executes this script on every interaction but keeps the modules it imported: an
# edit to charts.py etc. would otherwise need a server restart (`from forecast_dash import charts`
# keeps returning the old module). Drop them so every run uses the code on disk.
for _name in [m for m in sys.modules if m == "forecast_dash" or m.startswith("forecast_dash.")]:
    del sys.modules[_name]

from forecast_dash import calib, charts, home, insights, intraday_page
from forecast_dash.contract import Run, discover, load_forecast
from forecast_dash.help_text import HELP
from forecast_dash.ui import (
    _chart,
    _header,
    _in_short,
    _pct_cols,
    _table,
    get_live,
    get_run,
    section_radio,
)

PRESET_MOVES = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
DEFAULT_MOVES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
CALL_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DEFAULT_CALLS = [0.6, 0.7, 0.8, 0.9]


def _default_root() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--runs", default=os.environ.get("FORECAST_DASH_RUNS", "../ml/runs"))
    return p.parse_known_args(sys.argv[1:])[0].runs


def _default_intraday_root() -> str:
    """--intraday-runs, else $FORECAST_DASH_INTRADAY_RUNS, else <daily runs>/intraday."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--intraday-runs", default=os.environ.get("FORECAST_DASH_INTRADAY_RUNS"))
    given = p.parse_known_args(sys.argv[1:])[0].intraday_runs
    return given or str(Path(_default_root()) / "intraday")


# ---- cached data (cache_resource: shared, never copied; treat as read-only) -----------------

@st.cache_resource(max_entries=96)
def get_event(path: str, move: float) -> tuple[np.ndarray, np.ndarray]:
    r = get_run(path)
    return calib.event(r.r_q, r.r_act, r.levels, move)


@st.cache_resource(max_entries=6)
def get_pit(path: str) -> np.ndarray:
    r = get_run(path)
    return calib.pit(r.r_q, r.r_act, r.levels)


@st.cache_resource(max_entries=12)
def get_forecast(path: str):
    return load_forecast(Path(path))


class View:
    """The sidebar-filtered slice of a run: row indices plus derived arrays."""

    def __init__(self, path: str, run: Run, idx: np.ndarray, moves: list[float], edges):
        self.path, self.run, self.idx, self.moves, self.edges = path, run, idx, moves, edges
        self.keys = run.frame.iloc[idx].reset_index(drop=True)
        self.r_q, self.r_act = run.r_q[idx], run.r_act[idx]

    def events(self, moves=None) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for m in moves or self.moves:
            p, h = get_event(self.path, m)
            out[m] = (p[self.idx], h[self.idx])
        return out

    def pit(self) -> np.ndarray:
        return get_pit(self.path)[self.idx]


# ---- sidebar --------------------------------------------------------------------------------

def sidebar() -> View | None:
    sb = st.sidebar
    root = sb.text_input("Runs directory", _default_root())
    runs = discover(Path(root))
    if not runs:
        st.warning(f"No runs under `{root}` (looking for fold_*/test_predictions.parquet).")
        return None
    names = [r.name for r in runs]
    name = sb.selectbox("Run", names, key="run")
    path = str(runs[names.index(name)])
    run = get_run(path)
    f = run.frame

    folds = sorted(f["fold"].unique())
    sel_folds = sb.multiselect("Folds (test blocks)", folds, default=folds,
                               help="consecutive blocks of test days the model never trained "
                                    "on; one fold = one market period")
    side = sb.radio("Side", ["both", "up", "down"], horizontal=True, key="side",
                    help="score upside events (≥ +m), downside events (≤ −m), or both")
    mags = sb.multiselect("Move sizes", PRESET_MOVES, default=DEFAULT_MOVES,
                          format_func=lambda m: f"{m:.2%}",
                          help="the event sizes m to score, as next-day VWAP moves")
    custom = sb.number_input("Extra move size (%)", 0.0, 20.0, 0.0, 0.05,
                             help="Any size, e.g. 1.3; 0 = none")
    if custom > 0:
        mags = sorted({*mags, round(custom / 100, 6)})
    signs = {"both": (-1, 1), "up": (1,), "down": (-1,)}[side]
    moves = sorted(s * m for m in mags for s in signs)
    width = sb.select_slider("Probability bucket width", [0.05, 0.1], value=0.1,
                             format_func=lambda w: f"{w:.0%}",
                             help="how finely forecasts are grouped by the probability the "
                                  "model said, e.g. 70–80%")
    edges = calib.bucket_edges(width)

    inds = sorted(f["industry_code"].unique())
    sel_inds = sb.multiselect("Industries", inds, help="empty = all")
    syms = sorted(f["symbol"].unique())
    sel_syms = sb.multiselect("Stocks", syms, help="empty = all")

    mask = f["fold"].isin(sel_folds).to_numpy()
    if sel_inds:
        mask = mask & f["industry_code"].isin(sel_inds).to_numpy()
    if sel_syms:
        mask = mask & f["symbol"].isin(sel_syms).to_numpy()
    if run.has_dates:
        lo, hi = f["date"].min().date(), f["date"].max().date()
        rng = sb.date_input("Test days", (lo, hi), min_value=lo, max_value=hi,
                            help="only score forecasts for days in this range")
        if isinstance(rng, tuple) and len(rng) == 2:
            mask = mask & f["date"].between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1])).to_numpy()
    else:
        lo, hi = int(f["date"].min()), int(f["date"].max())
        a, b = sb.slider("Test days (day_idx)", lo, hi, (lo, hi))
        mask = mask & f["date"].between(a, b).to_numpy()

    sb.caption(f"{int(mask.sum()):,} of {len(f):,} test forecasts · "
               f"{f.loc[mask, 'symbol'].nunique():,} stocks · quantiles "
               f"P{run.levels[0] * 100:.0f}..P{run.levels[-1] * 100:.0f} ({len(run.levels)})")
    with sb.expander("Run info"):
        cfg = run.config.get("config", {})
        st.json({k: cfg[k] for k in ("target", "normalizer", "n_folds", "fold_len",
                                     "encoder_length", "hidden_size") if k in cfg} or "n/a")
    return View(path, run, np.flatnonzero(mask), moves, edges)


def _need(v: View) -> bool:
    if len(v.idx) == 0:
        st.info("No test forecasts match the filters.")
        return False
    if not v.moves:
        st.info("Pick at least one move size.")
        return False
    return True


def _fmt_pct(x) -> str:
    return f"{x:.0%}"


def _default_band(levels) -> tuple[int, int]:
    pct = [round(lv * 100) for lv in levels]
    return (10, 90) if {10, 90} <= set(pct) else (pct[0], pct[-1])


def _band_slider(levels, key: str) -> tuple[float, float]:
    pct = [round(lv * 100) for lv in levels]
    lo, hi = st.select_slider("Percentile band", options=pct, value=_default_band(levels),
                              key=key, format_func=lambda x: f"P{x}",
                              help="the low and high percentiles of the range. P10–P90: the "
                                   "model gives 80% chance the VWAP lands inside")
    if lo >= hi:
        lo, hi = pct[0], pct[-1]
    st.caption(f"P{lo}–P{hi} = a {hi - lo}% central range")
    return lo / 100, hi / 100


def _live_headline(v: View) -> list[insights.Insight]:
    live = get_live(v.path)
    if live is None:
        return []
    lo, hi = (x / 100 for x in _default_band(live.levels))
    by_day = calib.live_by_day(live.keys["date"], live.r_q, live.r_act, live.levels, lo, hi)
    return insights.live_headline(by_day, lo, hi)


def _usd(cols) -> dict:
    return {c: st.column_config.NumberColumn(c, format="$%.2f") for c in cols}


# ---- tabs -----------------------------------------------------------------------------------

def tab_summary(v: View, min_n: int):
    _header("summary")
    if not _need(v):
        return
    a, b = st.columns([3, 1])
    confs = sorted(a.multiselect("Call levels", CALL_LEVELS, default=DEFAULT_CALLS,
                                 format_func=_fmt_pct, key="sum_confs"))
    if not confs:
        st.info("Pick at least one call level.")
        return
    head = b.selectbox("Headline call level", confs, format_func=_fmt_pct, key="sum_head",
                       index=confs.index(0.7) if 0.7 in confs else 0)
    ev = v.events()
    summ = calib.summary(v.r_q, v.r_act, v.run.levels, ev, v.edges)
    calls = insights.call_table(ev, confs)
    folds = insights.fold_table(v.keys, v.r_q, v.r_act, v.run.levels, ev, head)

    m = st.columns(6)
    m[0].metric("forecasts", f"{summ['rows']:,}", help="(stock, test day) forecasts that pass "
                "the sidebar filters")
    m[1].metric("stocks", f"{v.keys['symbol'].nunique():,}", help="distinct stocks among them")
    m[2].metric("quantile skill", f"{summ['pinball_skill']:.1%}", help="1 − pinball loss / "
                "naive pinball loss, where naive forecasts a 0% move at every quantile. > 0 "
                "beats it")
    m[3].metric("P5–P95 coverage", f"{summ['coverage']:.1%}",
                f"{summ['coverage'] - summ['coverage_nominal']:+.1%} vs "
                f"{summ['coverage_nominal']:.0%}", delta_color="off",
                help="share of actual moves inside the outer band. It should be near the "
                     "band's width (90%): below means the bands are too narrow, above too wide")
    m[4].metric("bias", f"{summ['bias']:+.3f}", help="0.5 − mean PIT: + optimistic (actuals "
                "landed below the forecast), − pessimistic. 0 is ideal")
    m[5].metric("probability error (ECE)", f"{summ.get('ece', np.nan):.1%}",
                help="mean |came true − said| over the grid of the selected moves, weighted "
                     "by n. Lower is better")
    _in_short(insights.run_insights(summ, calls, folds, head) + _live_headline(v))

    _chart(charts.call_heatmap(calls, min_n=min_n), "sum_heatmap")
    c1, c2 = st.columns(2)
    with c1:
        _chart(charts.call_chart(insights.side_rollup(calls)), "sum_sides")
    with c2:
        _chart(charts.quantile_chart(insights.quantile_table(v.r_q, v.r_act, v.run.levels)),
               "sum_quantiles")
    st.subheader(f"Fold stability (calls at P ≥ {head:.0%})",
                 help="the headline numbers per test block (time period)")
    _table(folds, "sum_folds", _pct_cols(folds, [
        "pinball_skill", "coverage", *(f"{s}_{c}" for s in (calib.SIDE_UP, calib.SIDE_DOWN)
                                       for c in ("said", "came_true", "deviation"))]))
    with st.expander("Calls table"):
        _table(calls, "sum_calls", _pct_cols(
            calls, ["share", "mean_prob", "hit_rate", "ci_lo", "ci_hi", "base_rate",
                    "deviation"]))
        st.download_button("Download CSV", calls.to_csv(index=False), "calls.csv")


def tab_calibration(v: View, min_n: int):
    _header("calibration")
    if not _need(v):
        return
    confs = st.multiselect("Ladder confidences", CALL_LEVELS, default=DEFAULT_CALLS,
                           format_func=_fmt_pct, help="confidences c for the ladder chart: "
                           "the actual should clear each forecast's own c-level move c of "
                           "the time")
    g = calib.grid(v.events(), v.edges)
    lh = calib.ladder_hits(v.r_q, v.r_act, v.run.levels, sorted(confs)) if confs else None
    _in_short(insights.grid_insights(g, min_n)
              + (insights.ladder_insights(lh) if lh is not None else []))
    _chart(charts.reliability(g, min_n=min_n), "cal_reliability")
    _chart(charts.heatmap(g, min_n=min_n), "cal_heatmap")
    if lh is not None:
        _chart(charts.ladder_chart(lh), "cal_ladder")
    with st.expander("Grid table"):
        _table(g, "cal_grid", _pct_cols(
            g, ["mean_prob", "hit_rate", "deviation", "ci_lo", "ci_hi"]))
        st.download_button("Download CSV", g.to_csv(index=False), "grid.csv")


def tab_threshold(v: View):
    _header("threshold")
    if not _need(v):
        return
    c1, c2 = st.columns(2)
    move = c1.selectbox("Event", v.moves, index=len(v.moves) - 1 if v.moves[-1] > 0 else 0,
                        format_func=calib.event_label, key="thr_move")
    conf = c2.slider("Call level: P(event) >=", 0.05, 0.95, 0.7, 0.01, format="%.2f")
    ev = v.events()
    prob, hit = ev[move]
    sel = prob >= conf - 1e-9
    n = int(sel.sum())
    lo, hi = calib.wilson(hit[sel].sum(), n)
    t = calib.threshold_curve(prob, hit, np.round(np.arange(0.05, 0.951, 0.01), 2))
    _in_short(insights.threshold_insights(t, calib.event_label(move), conf))
    m = st.columns(4)
    m[0].metric("calls", f"{n:,}", f"{n / len(prob):.1%} of rows", delta_color="off",
                help="forecasts that gave the event at least the call level")
    m[1].metric("came true", f"{hit[sel].mean():.1%}" if n else "–",
                f"95% CI {lo:.0%}–{hi:.0%}" if n else None, delta_color="off",
                help="share of those calls where the event happened. The 95% CI is the range "
                     "the true rate very likely lies in, given this many calls")
    m[2].metric("model said (mean)", f"{prob[sel].mean():.1%}" if n else "–",
                help="the average probability the model gave on those calls. Honest calls "
                     "have came true ≈ said")
    m[3].metric("base rate (all rows)", f"{hit.mean():.1%}", help="how often the event "
                "happened across all forecasts. Calls are only useful if they beat this")
    _chart(charts.threshold_chart(t, f"{calib.event_label(move)}: hit rate as the call level "
                                     "rises"), "thr_curve")
    _chart(charts.move_chart(calib.move_curve(ev, conf), conf), "thr_move")

    st.subheader(f"Stocks: calls with P({calib.event_label(move)}) >= {conf:.0%}",
                 help="the calls above, split by stock")
    if not n:
        st.info("No calls at this level.")
        return
    d = v.keys.loc[sel, ["symbol", "industry_code"]].assign(prob=prob[sel], hit=hit[sel])
    per = (d.groupby(["symbol", "industry_code"])
           .agg(calls=("hit", "size"), came_true=("hit", "mean"), said=("prob", "mean"))
           .reset_index())
    per["deviation"] = per["came_true"] - per["said"]
    a, b = st.columns(2)
    min_calls = a.slider("Min calls per stock", 1, max(1, int(per["calls"].max())), 1,
                         help="hide stocks with fewer calls: a high hit rate on 2 calls is luck")
    min_acc = b.slider("Min came-true rate (accuracy)", 0.0, 1.0, 0.0, 0.05, format="%.2f",
                       help="keep stocks whose calls came true at least this often")
    per = per[(per["calls"] >= min_calls) & (per["came_true"] >= min_acc - 1e-9)]
    st.caption(f"{len(per):,} stocks")
    _table(per.sort_values(["came_true", "calls"], ascending=False), "thr_stocks",
           _pct_cols(per, ["came_true", "said", "deviation"]))


def tab_stocks(v: View):
    _header("stocks")
    if len(v.idx) == 0:
        st.info("No test forecasts match the filters.")
        return
    u = v.pit()
    b = calib.bias(u, v.keys[["symbol", "industry_code"]])
    min_n = st.slider("Min forecasts per stock", 1, max(1, int(b["n"].max())),
                      min(40, int(b["n"].max())), help="hide stocks with fewer test forecasts "
                      "(their bias is mostly noise)")
    b = b[b["n"] >= min_n]
    bi = calib.bias(u, v.keys[["industry_code"]])
    _in_short(insights.stock_insights(b, bi))
    ev = _chart(charts.funnel(b), "st_funnel", on_select="rerun", key="funnel",
                selection_mode=("points", "box", "lasso"))
    picked = [p["customdata"][0] for p in (ev.selection.points if ev else []) if "customdata" in p]
    if picked:
        st.write(f"Selected: {', '.join(picked[:20])}{' …' if len(picked) > 20 else ''}")
        if st.session_state.get("_last_pick") != picked[0]:
            st.session_state["_last_pick"] = picked[0]
            st.session_state["detail_symbol"] = picked[0]
        st.caption(f"{picked[0]} is now selected in the Stock detail tab.")
    _chart(charts.bias_bars(bi, "industry_code", "Bias by industry"), "st_industry")
    _table(b.sort_values("bias_z"), "st_table", _pct_cols(b, ["mean_pit"]))


def tab_days(v: View):
    _header("days")
    if not _need(v):
        return
    move = st.selectbox("Overlay event (mean said vs came true, per day)", [None, *v.moves],
                        format_func=lambda m: "none" if m is None else calib.event_label(m),
                        help="also draw this event's came true − said for each day")
    prob = hit = None
    if move is not None:
        prob, hit = v.events([move])[move]
    d = calib.daily(v.pit(), v.keys["date"].to_numpy(), prob, hit)
    _in_short(insights.day_insights(d))
    _chart(charts.daily_chart(d, None if move is None else calib.event_label(move)),
           "day_chart")
    st.subheader("Regime days (|z| ≥ 2: the whole cross-section missed together)",
                 help="days where the market as a whole moved against the forecast")
    reg = d[d["bias_z"].abs() >= 2].sort_values("bias_z")
    _table(reg, "day_regimes", _pct_cols(reg, ["mean_prob", "hit_rate", "deviation"]))


def tab_stock(v: View, min_n: int):
    _header("stock")
    syms = sorted(v.keys["symbol"].unique())
    if not syms:
        st.info("No test forecasts match the filters.")
        return
    if st.session_state.get("detail_symbol") not in syms:
        st.session_state["detail_symbol"] = syms[0]
    sym = st.selectbox("Stock", syms, key="detail_symbol")
    rows = np.flatnonzero(v.keys["symbol"].to_numpy() == sym)
    keys, r_q, r_act = v.keys.iloc[rows], v.r_q[rows], v.r_act[rows]
    u = calib.pit(r_q, r_act, v.run.levels)
    b = calib.bias(u, keys[["symbol"]]).iloc[0]
    st.markdown(f"**{sym}** ({keys['industry_code'].iloc[0]}) · {len(rows)} test forecasts · "
                f"bias {b['bias']:+.3f} (z {b['bias_z']:+.1f}) → **{b['label']}**")
    _in_short(insights.stock_detail_insights(sym, b, r_q, r_act, u,
                                             v.run.levels[-1] - v.run.levels[0]))
    _chart(charts.fan(keys, r_q, r_act, v.run.levels, sym), "sd_fan")
    c1, c2 = st.columns(2)
    with c1:
        _chart(charts.pit_hist(u), "sd_pit")
    lh = calib.ladder_hits(r_q, r_act, v.run.levels, [0.6, 0.7, 0.8, 0.9])
    with c2:
        _chart(charts.ladder_chart(lh, f"{sym}: ladder"), "sd_ladder")
    if v.moves:
        ev = {m: (p[rows], h[rows]) for m, (p, h) in v.events().items()}
        g = calib.grid(ev, v.edges)
        _chart(charts.heatmap(g, f"{sym}: deviation", min_n=min_n), "sd_heatmap")
        _chart(charts.reliability(g, f"{sym}: said vs came true", min_n=min_n),
               "sd_reliability")


def _history(v: View, move: float, prob_now: np.ndarray, symbols: pd.Series) -> pd.DataFrame:
    """For tomorrow's prob per stock: the stock's and all stocks' test hit rate in that bucket."""
    p, h = v.events([move])[move]
    labels = np.asarray(calib.bucket_labels(v.edges))
    b = calib.bucket_of(prob_now, v.edges)
    out = pd.DataFrame({"symbol": symbols.to_numpy(), "prob": prob_now,
                        "bucket": np.where(b >= 0, labels[np.clip(b, 0, None)], "outside")})
    cols = ["n", "hit_rate", "deviation"]
    per = calib.cells(p, h, v.edges, v.keys[["symbol"]])[["symbol", "bucket", *cols]]
    out = out.merge(per.rename(columns={c: f"hist_{c}" for c in cols}),
                    on=["symbol", "bucket"], how="left")
    allc = calib.cells(p, h, v.edges)[["bucket", *cols]]
    return out.merge(allc.rename(columns={c: f"all_{c}" for c in cols}), on="bucket", how="left")


def tab_forecast_page(v: View):
    fc, act = st.tabs(["Price ranges", "Forecast vs actual"])
    with fc:
        tab_forecast(v)
    with act:
        tab_live(v)


def tab_forecast(v: View):
    _header("tomorrow")
    files = v.run.forecasts()
    if not files:
        st.info("No forecast_<date>.parquet in this run. Produce one with the model's predict "
                "command (for ml/: `uv run tft-vwap-predict <run>`).")
        return
    by_day = {p.stem.removeprefix("forecast_"): p for p in files}  # newest first
    live = get_live(v.path)
    graded = live.keys[["date", "symbol", "actual_vwap"]] if live is not None else None
    done = set(graded["date"]) if graded is not None else set()
    day = st.selectbox("Forecast day", list(by_day), key="tm_day",
                       format_func=lambda d: f"{d} · {'graded' if d in done else 'open'}",
                       help="the day being forecast, newest first. graded = its actuals are "
                            "in; open = not yet")
    f = by_day[day]
    keys, r_q, levels = get_forecast(str(f))
    made = f" from the {keys['asof'].iloc[0]} session's bars" if "asof" in keys else ""
    st.caption(f"Forecast for {day}{made}. "
               + ("It has happened: actuals are shown, and the full grade is in Forecast vs "
                  "actual." if day in done else "Its actuals are not in yet."))

    st.markdown("**Price range**", help="how the $ range in the table is drawn: the "
                "central percentile band, a one-sided floor/cap, and filters on width and price")
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        lo, hi = _band_slider(levels, "tm_band")
    one = c2.slider("One-sided confidence", 0.55, 0.95, 0.8, 0.05, key="tm_one",
                    format="%.2f", help="floor: VWAP ≥ floor with this confidence; "
                    "cap: VWAP ≤ cap with this confidence")
    max_w = c3.number_input("Max band width (%)", 0.0, 100.0, 0.0, 0.5, key="tm_maxw",
                            help="keep stocks whose low–high range is at most this wide, as a "
                                 "% of the price. 0 = no limit") / 100
    min_px = c4.number_input("Min price ($)", 0.0, 10_000.0, 0.0, 1.0, key="tm_minpx",
                             help="keep stocks whose reference VWAP is at least this")

    st.markdown("**% deviation**", help="the event to screen on: a VWAP move of at least "
                "+m (up) or at most −m (down) vs the reference VWAP, and the minimum "
                "probability the model must give it")
    c1, c2, c3 = st.columns(3)
    side = c1.radio("Side", ["up", "down"], horizontal=True, key="tm_side",
                    help="up: VWAP ≥ +m. down: VWAP ≤ −m")
    mag = c2.number_input("Move (%)", 0.05, 20.0, 1.0, 0.05, key="tm_move",
                          help="the move size m") / 100
    conf = c3.slider("P(event) >=", 0.0, 0.95, 0.6, 0.01, key="tm_conf", format="%.2f",
                     help="keep stocks where the model gives the event at least this "
                          "probability")
    move = round(mag if side == "up" else -mag, 6)
    prob = calib.event_prob(r_q, levels, move)
    ref = (keys["today_vwap"].to_numpy(dtype=float) if "today_vwap" in keys
           else np.full(len(keys), np.nan))
    px = calib.price_table(ref, r_q, levels, lo, hi, one)
    t = _history(v, move, prob, keys["symbol"])
    if "industry_code" in keys:
        t.insert(1, "industry_code", keys["industry_code"].to_numpy())
    t = pd.concat([t.iloc[:, :2 if "industry_code" in keys else 1], px,
                   t.iloc[:, 2 if "industry_code" in keys else 1:]], axis=1)
    actual = np.full(len(keys), np.nan)
    if day in done:
        a = graded[graded["date"] == day].set_index("symbol")["actual_vwap"]
        actual = keys["symbol"].map(a).to_numpy(dtype=float)
        pos = t.columns.get_loc("width") + 1
        t.insert(pos, "actual_vwap", actual)
        t.insert(pos + 1, "inside", pd.Series((px["low"] <= actual) & (actual <= px["high"]))
                 .where(~np.isnan(actual)))
    a, b = st.columns(2)
    min_hn = a.slider("Min historical n (stock, same bucket)", 0, 100, 0,
                      help="keep stocks with at least this many past test forecasts in the "
                           "same probability bucket (hist_n)")
    min_hr = b.slider("Min historical came-true (stock)", 0.0, 1.0, 0.0, 0.05, format="%.2f",
                      help="keep stocks whose past forecasts in this bucket came true at "
                           "least this often (hist_hit_rate)")
    keep = ((t["prob"] >= conf - 1e-9) & (t["hist_n"].fillna(0) >= min_hn)
            & ((t["hist_hit_rate"].fillna(0) >= min_hr - 1e-9) | (min_hr == 0)))
    if max_w > 0:
        keep &= t["width"] <= max_w + 1e-12
    if min_px > 0:
        keep &= t["ref_vwap"] >= min_px
    s = t[keep]
    st.subheader(f"{len(s):,} stocks with P({calib.event_label(move)}) >= {conf:.0%}",
                 help="the stocks that pass every filter above")
    _in_short(insights.tomorrow_insights(s, calib.event_label(move), conf, day=day,
                                         band=(lo, hi)))
    pct_lo, pct_hi = round(lo * 100), round(hi * 100)
    shown = s.rename(columns={"low": f"low (P{pct_lo})", "high": f"high (P{pct_hi})",
                              "median": "median (P50)", "floor": f"floor ({one:.0%})",
                              "cap": f"cap ({one:.0%})"}).sort_values("prob", ascending=False)
    cfg = {**_usd(["ref_vwap", f"low (P{pct_lo})", "median (P50)", f"high (P{pct_hi})",
                   f"floor ({one:.0%})", f"cap ({one:.0%})", "actual_vwap"]),
           **_pct_cols(shown, ["width", "prob", "hist_hit_rate", "hist_deviation",
                               "all_hit_rate", "all_deviation"])}
    _table(shown, "tm_table", cfg)
    st.download_button("Download CSV", shown.to_csv(index=False), f"{f.stem}_ranges.csv",
                       key="tm_dl")

    st.subheader("One stock: forecast VWAP range", help="the full forecast for one stock, in $ "
                 "and %, and how its probabilities have held up in the past")
    syms = sorted(keys["symbol"])
    default = st.session_state.get("detail_symbol")
    sym = st.selectbox("Stock", syms, index=syms.index(default) if default in syms else 0,
                       key="tm_sym")
    i = int(np.flatnonzero(keys["symbol"].to_numpy() == sym)[0])
    row = px.iloc[i]
    has_act = bool(np.isfinite(actual[i]))
    m = st.columns(5 if has_act else 4)
    m[0].metric("reference VWAP", f"${row['ref_vwap']:,.2f}", help="the VWAP of the session "
                "the forecast was made from; every forecast price is this × (1 + move)")
    m[1].metric(f"P{pct_lo}–P{pct_hi} range", f"${row['low']:,.2f} – ${row['high']:,.2f}",
                f"{row['width']:.2%} wide", delta_color="off",
                help=f"the model gives a {pct_hi - pct_lo}% chance the VWAP ends inside this "
                     "range")
    m[2].metric("median (P50)", f"${row['median']:,.2f}",
                f"{row['median'] / row['ref_vwap'] - 1:+.2%}", delta_color="off",
                help="an even chance of ending above or below it; the % is vs the reference "
                     "VWAP")
    m[3].metric(f"{one:.0%} floor / cap", f"${row['floor']:,.2f} / ${row['cap']:,.2f}",
                help=f"VWAP ≥ floor with {one:.0%} confidence; VWAP ≤ cap with {one:.0%} "
                     "confidence")
    if has_act:
        inside = row["low"] <= actual[i] <= row["high"]
        m[4].metric("actual VWAP", f"${actual[i]:,.2f}",
                    f"{actual[i] / row['ref_vwap'] - 1:+.2%}, "
                    + ("inside" if inside else "outside") + " the range", delta_color="off",
                    help="the VWAP that printed on the forecast day, and its move vs the "
                         "reference VWAP")
    prices = ref[i] * (1 + r_q[i])
    _chart(charts.price_range(levels, prices, ref[i], sym, lo, hi,
                              actual=actual[i] if has_act else None), "tm_range", key="tm_range")
    c1, c2 = st.columns(2)
    with c1:
        _table(pd.DataFrame({"percentile": [f"P{round(lv * 100)}" for lv in levels],
                             "move": r_q[i], "VWAP": prices}), "tm_pct",
               {**_usd(["VWAP"]), "move": st.column_config.NumberColumn(format="percent")})
    confs = sorted({0.6, 0.7, 0.8, 0.9, round(one, 2)})
    lad = calib.ladder(r_q[i:i + 1], levels, confs)
    with c2:
        _table(pd.DataFrame({
            "confidence": [f"{c:.0%}" for c in confs],
            "VWAP ≥ (floor)": [ref[i] * (1 + lad[f"up_{round(c * 100)}"][0]) for c in confs],
            "floor move": [lad[f"up_{round(c * 100)}"][0] for c in confs],
            "VWAP ≤ (cap)": [ref[i] * (1 + lad[f"down_{round(c * 100)}"][0]) for c in confs],
            "cap move": [lad[f"down_{round(c * 100)}"][0] for c in confs],
        }), "tm_ladder", {
            **_usd(["VWAP ≥ (floor)", "VWAP ≤ (cap)"]),
            **{c: st.column_config.NumberColumn(format="percent")
               for c in ("floor move", "cap move")},
        })
    mags = sorted({abs(m) for m in v.moves} | {mag})
    parts = []
    for mv in [-x for x in mags] + mags:
        p_now = calib.event_prob(r_q[i:i + 1], levels, mv)
        h = _history(v, mv, p_now, keys["symbol"].iloc[i:i + 1])
        parts.append(h.assign(move=mv, side=calib.side_of(mv), event=calib.event_label(mv)))
    sp = pd.concat(parts, ignore_index=True)
    _chart(charts.spread_chart(sp, sym), "tm_spread")
    _table(sp[["event", "prob", "bucket", "hist_hit_rate", "hist_n", "all_hit_rate", "all_n"]],
           "tm_spread_table", _pct_cols(sp, ["prob", "hist_hit_rate", "all_hit_rate"]))


def tab_live(v: View):
    _header("tomorrow_actual")
    live = get_live(v.path)
    if live is None:
        st.info("No graded forecast yet. Once a forecast day has closed and the panel is "
                "refreshed, run the model's grade step (for ml/: `uv run tft-vwap-grade <run>`, "
                "also run by every `tft-vwap-predict`) to write actuals.parquet.")
        return
    k, levels = live.keys, live.levels
    days = sorted(k["date"].unique(), reverse=True)
    c1, c2 = st.columns([1, 3])
    picked = st.session_state.get("tm_day")
    day = c1.selectbox("Graded day", days, index=days.index(picked) if picked in days else 0,
                       key="lv_day", help="a forecast day whose actuals are in")
    with c2:
        lo, hi = _band_slider(levels, "lv_band")
    c1, c2, c3 = st.columns(3)
    side = c1.radio("Side", ["up", "down"], horizontal=True, key="lv_side",
                    help="up: VWAP ≥ +m. down: VWAP ≤ −m")
    mag = c2.number_input("Move (%)", 0.05, 20.0, 1.0, 0.05, key="lv_move",
                          help="the move size m of the event to grade") / 100
    conf = c3.slider("P(event) >=", 0.05, 0.95, 0.6, 0.01, key="lv_conf", format="%.2f",
                     help="the call level: grade the stocks the model gave the event at least "
                          "this probability")
    move = round(mag if side == "up" else -mag, 6)

    rows = np.flatnonzero(k["date"].to_numpy() == day)
    d, r_q, r_act = k.iloc[rows].reset_index(drop=True), live.r_q[rows], live.r_act[rows]
    g = calib.live_grade(r_q, r_act, levels, lo, hi)
    by_day = calib.live_by_day(k["date"], live.r_q, live.r_act, levels, lo, hi)
    prob, hit = calib.event(r_q, r_act, levels, move)
    called = prob >= conf - 1e-9
    if "asof" in d:
        st.caption(f"Forecast for {day}, made from the {d['asof'].iloc[0]} session's bars.")

    m = st.columns(5)
    m[0].metric("stocks graded", f"{g['rows']:,}", help="stocks with both a forecast and an "
                "actual for this day")
    m[1].metric(f"P{round(lo * 100)}–P{round(hi * 100)} held", f"{g['coverage']:.1%}",
                f"{g['coverage'] - g['coverage_nominal']:+.1%} vs {g['coverage_nominal']:.0%}",
                delta_color="off", help="share of stocks whose actual VWAP landed inside the "
                "band, vs the band's width (its target)")
    m[2].metric("direction right", f"{g['direction_hit']:.1%}", help="share of stocks where "
                "the forecast median's sign (up / down) matched the actual move")
    m[3].metric("bias", f"{g['bias']:+.3f}", help="0.5 − mean PIT: + optimistic (actuals "
                "landed below the forecast), − pessimistic. 0 is ideal")
    m[4].metric("quantile skill", f"{g['pinball_skill']:.1%}", help="1 − pinball loss / "
                "naive pinball loss, where naive forecasts a 0% move. > 0 beats it")
    st.caption(f"P({calib.event_label(move)}) ≥ {conf:.0%}: {int(called.sum()):,} calls, "
               + (f"said {prob[called].mean():.0%} on average, came true {hit[called].mean():.0%}."
                  if called.any() else "none on this day."))
    _in_short(insights.live_insights(g, lo, hi, by_day, day))

    q_lo, q_hi = (calib.quantile_at(r_q, levels, t) for t in (lo, hi))
    med = calib.quantile_at(r_q, levels, 0.5)
    inside = (q_lo <= r_act) & (r_act <= q_hi)
    c1, c2 = st.columns(2)
    with c1:
        _chart(charts.live_scatter(med, r_act, inside, d["symbol"],
                                   f"{day}: forecast median vs actual move"), "lv_scatter")
    with c2:
        _chart(charts.pit_hist(calib.pit(r_q, r_act, levels),
                               title=f"{day}: PIT histogram (flat = calibrated)"), "lv_pit")
    if len(by_day) > 1:
        _chart(charts.live_days(by_day), "lv_days")

    ref = d["today_vwap"].to_numpy(dtype=float)
    px = calib.price_table(ref, r_q, levels, lo, hi, 0.8)
    t = pd.DataFrame({"symbol": d["symbol"]})
    if "industry_code" in d:
        t["industry_code"] = d["industry_code"]
    t = t.assign(ref_vwap=ref, low=px["low"], median=px["median"], high=px["high"],
                 actual_vwap=d["actual_vwap"].to_numpy(dtype=float), actual_move=r_act,
                 inside=inside, pit=calib.pit(r_q, r_act, levels),
                 outside_by=np.where(r_act > q_hi, r_act - q_hi,
                                     np.where(r_act < q_lo, r_act - q_lo, 0.0)),
                 prob=prob, came_true=hit)
    pl, ph = round(lo * 100), round(hi * 100)
    t = t.rename(columns={"low": f"low (P{pl})", "high": f"high (P{ph})", "median": "median (P50)",
                          "prob": f"P({calib.event_label(move)})"})
    _table(t.sort_values("outside_by", key=np.abs, ascending=False), "lv_table",
           {**_usd(["ref_vwap", f"low (P{pl})", "median (P50)", f"high (P{ph})", "actual_vwap"]),
            **_pct_cols(t, ["actual_move", "outside_by", f"P({calib.event_label(move)})"])})
    st.download_button("Download CSV", t.to_csv(index=False), f"graded_{day}.csv", key="lv_dl")

    st.subheader("One stock: forecast range vs actual, every graded day",
                 help="one stock's forecast and actual VWAP, in $")
    syms = sorted(d["symbol"])
    default = st.session_state.get("tm_sym") or st.session_state.get("detail_symbol")
    sym = st.selectbox("Stock", syms, index=syms.index(default) if default in syms else 0,
                       key="lv_sym")
    j = np.flatnonzero(k["symbol"].to_numpy() == sym)
    ks = k.iloc[j].reset_index(drop=True)
    _chart(charts.fan(ks.assign(date=pd.to_datetime(ks["date"])), live.r_q[j], live.r_act[j],
                      levels, sym, ref=ks["today_vwap"].to_numpy(dtype=float)), "lv_fan")
    i = int(np.flatnonzero(d["symbol"].to_numpy() == sym)[0])
    _chart(charts.price_range(levels, ref[i] * (1 + r_q[i]), ref[i], sym, lo, hi,
                              actual=float(t["actual_vwap"].iloc[i])), "lv_range", key="lv_range")


def tab_compare(v: View, min_n: int):
    _header("compare")
    root = Path(v.path).parent
    names = [p.name for p in discover(root)] or [v.run.name]
    others = [n for n in names if n != v.run.name]
    chosen = st.multiselect("Runs to compare", names, default=[v.run.name, *others[:1]],
                            help="Newest first. Each run is scored on all its rows (or the common "
                                 "ones); the sidebar's fold/stock/industry/day filters do not "
                                 "apply here, its moves and bucket width do.")
    if not chosen or not v.moves:
        st.info("Pick runs and at least one move size.")
        return
    paths = {n: str(root / n) if n != v.run.name else v.path for n in chosen}
    runs = {n: get_run(p) for n, p in paths.items()}
    # real dates line runs up across panels (a 2-year and a 5-year panel number days differently)
    by_date = all(r.has_dates for r in runs.values())
    dkey = "date" if by_date else "day_idx"
    common = st.checkbox(
        "Only rows present in every chosen run", value=True,
        help="Fair comparison when runs cover different stocks or days. Rows match on "
             + ("(stock, date): every chosen run has real dates." if by_date else
                "(stock, day_idx): some chosen run has no dates, so the panels must number "
                "days the same way."))
    a, b = st.columns(2)
    move = a.selectbox("Event for the reliability overlay", v.moves,
                       format_func=calib.event_label, key="cmp_move",
                       help="the event whose said-vs-came-true curve is drawn per run")
    conf = b.selectbox("Call level to compare", CALL_LEVELS, index=CALL_LEVELS.index(0.7),
                       format_func=_fmt_pct, key="cmp_conf",
                       help="the call level the In short box compares the runs' calls at")
    sets = None
    if common and len(runs) > 1:
        sets = set.intersection(*(set(zip(r.frame["symbol"], r.frame[dkey], strict=True))
                                  for r in runs.values()))
        if not sets:
            st.info("These runs share no (stock, day) test rows, e.g. they tested different "
                    "folds. Untick the checkbox to compare each run on all of its own rows.")
            return
    grids, rows, calls = [], {}, {}
    for n, r in runs.items():
        if sets is not None:
            k = pd.Series(list(zip(r.frame["symbol"], r.frame[dkey], strict=True)))
            idx = np.flatnonzero(k.isin(sets).to_numpy())
        else:
            idx = np.arange(len(r.frame))
        ev = {}
        for m in v.moves:
            p, h = get_event(paths[n], m)
            ev[m] = (p[idx], h[idx])
        grids.append(calib.cells(*ev[move], v.edges).assign(run=n))
        rows[n] = calib.summary(r.r_q[idx], r.r_act[idx], r.levels, ev, v.edges)
        calls[n] = insights.call_table(ev, CALL_LEVELS)
    _in_short(insights.compare_insights(rows, calls, conf, same_rows=sets is not None
                                        or len(runs) == 1))
    _chart(charts.reliability(pd.concat(grids, ignore_index=True),
                              f"{calib.event_label(move)}: said vs came true by run",
                              min_n=min_n, group="run"), "cmp_reliability")
    st.caption("pinball: quantile loss on the move (lower is better; naive = 0% move at every "
               "quantile). coverage: share inside the outer quantiles. ece: n-weighted mean "
               "|came true − said| over the grid for the selected moves.")
    _table(pd.DataFrame(rows).T, "cmp_table", hide_index=False)
    roll = pd.concat([insights.side_rollup(c).assign(run=n) for n, c in calls.items()],
                     ignore_index=True)
    _chart(charts.call_chart(roll, "Calls by run and side: came true vs said"),
           "cmp_calls_chart")
    with st.expander("Calls by run (every event and call level)"):
        allc = pd.concat([c.assign(run=n) for n, c in calls.items()], ignore_index=True)
        _table(allc, "cmp_calls", _pct_cols(
            allc, ["share", "mean_prob", "hit_rate", "ci_lo", "ci_hi", "base_rate",
                   "deviation"]))


def main():
    st.set_page_config(page_title="Forecast calibration", layout="wide")
    section = section_radio()
    if section == "home":
        return home.render(_default_root(), _default_intraday_root())
    if section == "intraday":
        return intraday_page.render(_default_intraday_root())
    title, body = HELP["overview"]
    a, b = st.columns([6, 1], vertical_alignment="bottom")
    a.title(title)
    with b.popover("ℹ️ How to read this"):
        st.markdown(body)

    v = sidebar()
    if v is None:
        return
    min_n = st.sidebar.slider("Min n per grid point", 1, 500, 30,
                              help="hide grid cells and curve points with fewer forecasts "
                                   "than this; they are too noisy to read")
    tabs = st.tabs(["Summary", "Calibration", "Threshold explorer", "Across stocks", "By day",
                    "Stock detail", "Forecast", "Compare runs"])
    with tabs[0]:
        tab_summary(v, min_n)
    with tabs[1]:
        tab_calibration(v, min_n)
    with tabs[2]:
        tab_threshold(v)
    with tabs[3]:
        tab_stocks(v)
    with tabs[4]:
        tab_days(v)
    with tabs[5]:
        tab_stock(v, min_n)
    with tabs[6]:
        tab_forecast_page(v)
    with tabs[7]:
        tab_compare(v, min_n)


main()
