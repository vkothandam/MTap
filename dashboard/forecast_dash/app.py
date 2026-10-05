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

from forecast_dash import calib, charts
from forecast_dash.contract import Run, discover, load_forecast, load_run

PRESET_MOVES = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
DEFAULT_MOVES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
CONFIDENCES = [0.5, 0.6, 0.7, 0.8, 0.9]


def _default_root() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--runs", default=os.environ.get("FORECAST_DASH_RUNS", "../ml/runs"))
    return p.parse_known_args(sys.argv[1:])[0].runs


# ---- cached data (cache_resource: shared, never copied; treat as read-only) -----------------

@st.cache_resource(show_spinner="Loading run...", max_entries=6)
def get_run(path: str) -> Run:
    return load_run(Path(path))


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
    sel_folds = sb.multiselect("Folds (test blocks)", folds, default=folds)
    side = sb.radio("Side", ["both", "up", "down"], horizontal=True)
    mags = sb.multiselect("Move sizes", PRESET_MOVES, default=DEFAULT_MOVES,
                          format_func=lambda m: f"{m:.2%}")
    custom = sb.number_input("Extra move size (%)", 0.0, 20.0, 0.0, 0.05,
                             help="Any size, e.g. 1.3; 0 = none")
    if custom > 0:
        mags = sorted({*mags, round(custom / 100, 6)})
    signs = {"both": (-1, 1), "up": (1,), "down": (-1,)}[side]
    moves = sorted(s * m for m in mags for s in signs)
    width = sb.select_slider("Probability bucket width", [0.05, 0.1], value=0.1,
                             format_func=lambda w: f"{w:.0%}")
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
        rng = sb.date_input("Test days", (lo, hi), min_value=lo, max_value=hi)
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


def _pct_cols(df: pd.DataFrame, cols) -> dict:
    return {c: st.column_config.NumberColumn(c, format="percent") for c in cols if c in df}


# ---- tabs -----------------------------------------------------------------------------------

def tab_calibration(v: View, min_n: int):
    if not _need(v):
        return
    g = calib.grid(v.events(), v.edges)
    st.plotly_chart(charts.reliability(g, min_n=min_n), width="stretch")
    st.plotly_chart(charts.heatmap(g, min_n=min_n), width="stretch")
    confs = st.multiselect("Ladder confidences", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                           default=[0.6, 0.7, 0.8, 0.9], format_func="{:.0%}".format)
    if confs:
        lh = calib.ladder_hits(v.r_q, v.r_act, v.run.levels, sorted(confs))
        st.plotly_chart(charts.ladder_chart(lh), width="stretch")
    with st.expander("Grid table"):
        st.dataframe(g, hide_index=True, column_config=_pct_cols(
            g, ["mean_prob", "hit_rate", "deviation", "ci_lo", "ci_hi"]))
        st.download_button("Download CSV", g.to_csv(index=False), "grid.csv")


def tab_threshold(v: View):
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
    m = st.columns(4)
    m[0].metric("calls", f"{n:,}", f"{n / len(prob):.1%} of rows", delta_color="off")
    m[1].metric("came true", f"{hit[sel].mean():.1%}" if n else "–",
                f"95% CI {lo:.0%}–{hi:.0%}" if n else None, delta_color="off")
    m[2].metric("model said (mean)", f"{prob[sel].mean():.1%}" if n else "–")
    m[3].metric("base rate (all rows)", f"{hit.mean():.1%}")

    t = calib.threshold_curve(prob, hit, np.round(np.arange(0.05, 0.951, 0.01), 2))
    st.plotly_chart(charts.threshold_chart(t, f"{calib.event_label(move)}: hit rate as the "
                                              "call level rises"), width="stretch")
    st.plotly_chart(charts.move_chart(calib.move_curve(ev, conf), conf), width="stretch")

    st.subheader(f"Stocks: calls with P({calib.event_label(move)}) >= {conf:.0%}")
    if not n:
        st.info("No calls at this level.")
        return
    d = v.keys.loc[sel, ["symbol", "industry_code"]].assign(prob=prob[sel], hit=hit[sel])
    per = (d.groupby(["symbol", "industry_code"])
           .agg(calls=("hit", "size"), came_true=("hit", "mean"), said=("prob", "mean"))
           .reset_index())
    per["deviation"] = per["came_true"] - per["said"]
    a, b = st.columns(2)
    min_calls = a.slider("Min calls per stock", 1, max(1, int(per["calls"].max())), 1)
    min_acc = b.slider("Min came-true rate (accuracy)", 0.0, 1.0, 0.0, 0.05, format="%.2f")
    per = per[(per["calls"] >= min_calls) & (per["came_true"] >= min_acc - 1e-9)]
    st.caption(f"{len(per):,} stocks")
    st.dataframe(per.sort_values(["came_true", "calls"], ascending=False), hide_index=True,
                 column_config=_pct_cols(per, ["came_true", "said", "deviation"]))


def tab_stocks(v: View):
    if len(v.idx) == 0:
        st.info("No test forecasts match the filters.")
        return
    u = v.pit()
    b = calib.bias(u, v.keys[["symbol", "industry_code"]])
    min_n = st.slider("Min forecasts per stock", 1, max(1, int(b["n"].max())),
                      min(40, int(b["n"].max())))
    b = b[b["n"] >= min_n]
    counts = b["label"].value_counts()
    flagged = int(counts.get("optimistic", 0) + counts.get("pessimistic", 0))
    st.markdown(f"**{len(b):,} stocks** · optimistic {counts.get('optimistic', 0):,} · "
                f"pessimistic {counts.get('pessimistic', 0):,} · flagged {flagged:,} vs "
                f"~{0.0455 * len(b):,.0f} expected from noise alone (|z| ≥ 2)")
    ev = st.plotly_chart(charts.funnel(b), width="stretch", on_select="rerun", key="funnel",
                         selection_mode=("points", "box", "lasso"))
    picked = [p["customdata"][0] for p in (ev.selection.points if ev else []) if "customdata" in p]
    if picked:
        st.write(f"Selected: {', '.join(picked[:20])}{' …' if len(picked) > 20 else ''}")
        if st.session_state.get("_last_pick") != picked[0]:
            st.session_state["_last_pick"] = picked[0]
            st.session_state["detail_symbol"] = picked[0]
        st.caption(f"{picked[0]} is now selected in the Stock detail tab.")
    bi = calib.bias(u, v.keys[["industry_code"]])
    st.plotly_chart(charts.bias_bars(bi, "industry_code", "Bias by industry"), width="stretch")
    st.dataframe(b.sort_values("bias_z"), hide_index=True,
                 column_config=_pct_cols(b, ["mean_pit"]))


def tab_days(v: View):
    if not _need(v):
        return
    move = st.selectbox("Overlay event (mean said vs came true, per day)", [None, *v.moves],
                        format_func=lambda m: "none" if m is None else calib.event_label(m))
    prob = hit = None
    if move is not None:
        prob, hit = v.events([move])[move]
    d = calib.daily(v.pit(), v.keys["date"].to_numpy(), prob, hit)
    st.plotly_chart(charts.daily_chart(d, None if move is None else calib.event_label(move)),
                    width="stretch")
    st.subheader("Regime days (|z| ≥ 2: the whole cross-section missed together)")
    st.dataframe(d[d["bias_z"].abs() >= 2].sort_values("bias_z"), hide_index=True)


def tab_stock(v: View, min_n: int):
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
    st.plotly_chart(charts.fan(keys, r_q, r_act, v.run.levels, sym), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.pit_hist(u), width="stretch")
    lh = calib.ladder_hits(r_q, r_act, v.run.levels, [0.6, 0.7, 0.8, 0.9])
    c2.plotly_chart(charts.ladder_chart(lh, f"{sym}: ladder"), width="stretch")
    if v.moves:
        ev = {m: (p[rows], h[rows]) for m, (p, h) in v.events().items()}
        g = calib.grid(ev, v.edges)
        st.plotly_chart(charts.heatmap(g, f"{sym}: deviation", min_n=min_n), width="stretch")
        st.plotly_chart(charts.reliability(g, f"{sym}: said vs came true", min_n=min_n),
                        width="stretch")


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


def tab_tomorrow(v: View):
    files = v.run.forecasts()
    if not files:
        st.info("No forecast_<date>.parquet in this run. Produce one with the model's predict "
                "command (for ml/: `uv run tft-vwap-predict <run>`).")
        return
    f = st.selectbox("Forecast", files, format_func=lambda p: p.stem.removeprefix("forecast_"))
    keys, r_q, levels = get_forecast(str(f))
    c1, c2, c3 = st.columns(3)
    side = c1.radio("Side", ["up", "down"], horizontal=True, key="tm_side")
    mag = c2.number_input("Move (%)", 0.05, 20.0, 1.0, 0.05, key="tm_move") / 100
    conf = c3.slider("P(event) >=", 0.05, 0.95, 0.6, 0.01, key="tm_conf", format="%.2f")
    move = round(mag if side == "up" else -mag, 6)
    prob = calib.event_prob(r_q, levels, move)
    t = _history(v, move, prob, keys["symbol"])
    if "industry_code" in keys:
        t.insert(1, "industry_code", keys["industry_code"].to_numpy())
    a, b = st.columns(2)
    min_hn = a.slider("Min historical n (stock, same bucket)", 0, 100, 0)
    min_hr = b.slider("Min historical came-true (stock)", 0.0, 1.0, 0.0, 0.05, format="%.2f")
    s = t[(t["prob"] >= conf - 1e-9) & (t["hist_n"].fillna(0) >= min_hn)
          & ((t["hist_hit_rate"].fillna(0) >= min_hr - 1e-9) | (min_hr == 0))]
    st.subheader(f"{len(s):,} stocks with P({calib.event_label(move)}) >= {conf:.0%}")
    st.dataframe(s.sort_values("prob", ascending=False), hide_index=True,
                 column_config=_pct_cols(s, ["prob", "hist_hit_rate", "hist_deviation",
                                             "all_hit_rate", "all_deviation"]))

    st.subheader("One stock: tomorrow's spread vs its history")
    syms = sorted(keys["symbol"])
    default = st.session_state.get("detail_symbol")
    sym = st.selectbox("Stock", syms, index=syms.index(default) if default in syms else 0,
                       key="tm_sym")
    i = int(np.flatnonzero(keys["symbol"].to_numpy() == sym)[0])
    mags = sorted({abs(m) for m in v.moves} | {mag})
    parts = []
    for m in [-x for x in mags] + mags:
        p_now = calib.event_prob(r_q[i:i + 1], levels, m)
        h = _history(v, m, p_now, keys["symbol"].iloc[i:i + 1])
        parts.append(h.assign(move=m, side=calib.side_of(m), event=calib.event_label(m)))
    sp = pd.concat(parts, ignore_index=True)
    st.plotly_chart(charts.spread_chart(sp, sym), width="stretch")
    lad = calib.ladder(r_q[i:i + 1], levels, [0.6, 0.7, 0.8, 0.9])
    st.dataframe(pd.DataFrame({
        "confidence": [f"{c}%" for c in (60, 70, 80, 90)],
        "upside: move >= x": [f"{lad[f'up_{c}'][0]:+.2%}" for c in (60, 70, 80, 90)],
        "downside: move <= x": [f"{lad[f'down_{c}'][0]:+.2%}" for c in (60, 70, 80, 90)],
    }), hide_index=True)
    st.dataframe(sp[["event", "prob", "bucket", "hist_hit_rate", "hist_n", "all_hit_rate",
                     "all_n"]], hide_index=True,
                 column_config=_pct_cols(sp, ["prob", "hist_hit_rate", "all_hit_rate"]))


def tab_compare(v: View, min_n: int):
    root = Path(v.path).parent
    names = [p.name for p in discover(root)] or [v.run.name]
    chosen = st.multiselect("Runs to compare", names, default=[v.run.name])
    if not chosen or not v.moves:
        st.info("Pick runs and at least one move size.")
        return
    common = st.checkbox("Only (stock, day) rows present in every chosen run", value=True,
                         help="Fair comparison when runs cover different stocks or days")
    move = st.selectbox("Event for the reliability overlay", v.moves,
                        format_func=calib.event_label, key="cmp_move")
    paths = {n: str(root / n) if n != v.run.name else v.path for n in chosen}
    runs = {n: get_run(p) for n, p in paths.items()}
    sets = None
    if common and len(runs) > 1:
        sets = set.intersection(*(set(zip(r.frame["symbol"], r.frame["day_idx"], strict=True))
                                  for r in runs.values()))
        if not sets:
            st.info("These runs share no (stock, day) test rows, e.g. they tested different "
                    "folds. Untick the checkbox to compare each run on all of its own rows.")
            return
    grids, rows = [], {}
    for n, r in runs.items():
        if sets is not None:
            k = pd.Series(list(zip(r.frame["symbol"], r.frame["day_idx"], strict=True)))
            idx = np.flatnonzero(k.isin(sets).to_numpy())
        else:
            idx = np.arange(len(r.frame))
        ev = {}
        for m in v.moves:
            p, h = get_event(paths[n], m)
            ev[m] = (p[idx], h[idx])
        grids.append(calib.cells(*ev[move], v.edges).assign(run=n))
        rows[n] = calib.summary(r.r_q[idx], r.r_act[idx], r.levels, ev, v.edges)
    st.plotly_chart(charts.reliability(pd.concat(grids, ignore_index=True),
                                       f"{calib.event_label(move)}: said vs came true by run",
                                       min_n=min_n, group="run"), width="stretch")
    st.caption("pinball: quantile loss on the move (lower is better; naive = 0% move at every "
               "quantile). coverage: share inside the outer quantiles. ece: n-weighted mean "
               "|came true − said| over the grid for the selected moves.")
    st.dataframe(pd.DataFrame(rows).T)


def main():
    st.set_page_config(page_title="Forecast calibration", layout="wide")
    st.title("Forecast calibration")
    v = sidebar()
    if v is None:
        return
    min_n = st.sidebar.slider("Min n per grid point", 1, 500, 30)
    tabs = st.tabs(["Calibration", "Threshold explorer", "Across stocks", "By day",
                    "Stock detail", "Tomorrow", "Compare runs"])
    with tabs[0]:
        tab_calibration(v, min_n)
    with tabs[1]:
        tab_threshold(v)
    with tabs[2]:
        tab_stocks(v)
    with tabs[3]:
        tab_days(v)
    with tabs[4]:
        tab_stock(v, min_n)
    with tabs[5]:
        tab_tomorrow(v)
    with tabs[6]:
        tab_compare(v, min_n)


main()
