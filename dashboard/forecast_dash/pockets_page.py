"""Intraday → Pockets: the intraday-ft model (intraday v2) graded on its test months.

Every decision (stock, day, minute) has P10 … P90 of the pocket shape over five duration bins after
the entry price P0 (where the volume above / below P0 concentrates, how much of it, how busy the bin
is), plus event probabilities under four identity overlays. The page grades them against what
happened, on the rows the sidebar filters keep, raw or after the rolling recalibration. Reads only
the run files (pockets_contract.py); `intraday-ft-train` + `intraday-ft-report` make them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from forecast_dash import charts
from forecast_dash import pockets_calc as P
from forecast_dash import pockets_contract as C
from forecast_dash.contract import run_label
from forecast_dash.ui import _chart, _header, _in_short, _model_popover, _pct_cols, _table

MODES = {"raw": "Raw (as the model said)", "recal": "Recalibrated (rolling 20 sessions)"}
MIN_N = 2000
SIG = "%+.3f"
EVENT_T = [t for t, _, _ in P.EVENTS.values()]


# ---- cached data (cache_resource: shared, never copied; treat as read-only) -----------------

def _columns() -> list[str]:
    real = [f"real_{f}_b{b}" for f in P.SHAPE for b in C.BINS]
    reads = [f"{e}__{v}" for e in P.EVENTS for v in C.OVERLAYS]
    return [*C.KEYS, "real_p0", *real, *[f"real_{t}" for t in EVENT_T], *reads]


@st.cache_resource(show_spinner="Loading the pockets run...", max_entries=2)
def get_frame(path: str, fold: str) -> pd.DataFrame:
    f = C.load_pred(Path(path), fold, _columns())
    f["date"] = f["date"].astype(str)
    f["minute_bucket"] = P.minute_bucket(f["minute"])
    for c in ("industry", "cap_band"):
        if c in f:
            f[c] = f[c].astype(str).replace({"": "unknown", "None": "unknown", "nan": "unknown"})
    if "sector" in f:
        f["sector"] = f["sector"].astype(str).map({s: C.sector_label(s)
                                                   for s in f["sector"].astype(str).unique()})
    return f


@st.cache_resource(show_spinner="Reading quantiles...", max_entries=16)
def get_q(path: str, fold: str, target: str) -> np.ndarray:
    """(rows, 5) float32 quantiles of one target for every run-file row (zero-mass dust read as 0)."""
    f = get_frame(path, fold)
    q = C.quantiles(Path(path), fold, f["dec_row"].to_numpy(), [target])[:, 0]
    return P.dust(q, target).astype(np.float32)


@st.cache_resource(max_entries=4)
def get_side(path: str, fold: str) -> dict:
    return {"recal": C.load_recal(Path(path), fold), "clim": C.load_clim(Path(path), fold),
            "rates": C.load_clim_rates(Path(path), fold)}


@st.cache_resource(max_entries=4)
def get_cells(path: str, fold: str) -> np.ndarray | None:
    cl = get_side(path, fold)["clim"]
    if cl is None:
        return None
    f = get_frame(path, fold)
    return P.cells(f["minute"].to_numpy(), f["sigma"].to_numpy(), f["log_dvol"].to_numpy(), cl)


def get_report(path: str) -> dict | None:
    return C.load_report(Path(path))


@st.cache_resource(max_entries=8)
def get_idx(path: str, fold: str, key: tuple) -> np.ndarray:
    """Row positions the filters keep."""
    f = get_frame(path, fold)
    (d0, d1), sectors, industries, caps, stocks, live = key
    m = (f["date"] >= d0) & (f["date"] <= d1)
    for col, vals in (("sector", sectors), ("industry", industries), ("cap_band", caps),
                      ("symbol", stocks)):
        if vals:
            m &= f[col].isin(vals)
    if live:
        m &= f["live_set"]
    return np.flatnonzero(m.to_numpy())


class View:
    """What every tab reads: the run, the filtered rows, and quantiles in the chosen mode."""

    def __init__(self, path: str, fold: str, idx: np.ndarray, mode: str, min_n: int):
        self.path, self.fold, self.idx, self.mode, self.min_n = path, fold, idx, mode, min_n
        self.full = get_frame(path, fold)
        self.f = self.full.iloc[idx]
        side = get_side(path, fold)
        self.rc, self.cl, self.rates = side["recal"], side["clim"], side["rates"]
        self.cell = get_cells(path, fold)
        self.targets = C.targets_of(Path(path), fold)

    @property
    def recal(self) -> bool:
        return self.mode == "recal" and self.rc is not None

    def warm(self) -> np.ndarray:
        """Rows whose day has a full recalibration window (all rows when raw)."""
        if not self.recal:
            return np.ones(len(self.f), bool)
        days = self.rc.loc[self.rc["n_days"] >= 20, "date"].unique()
        return self.f["date"].isin(days).to_numpy()

    def q(self, target: str, rows: np.ndarray | None = None, mode: str | None = None) -> np.ndarray:
        idx = self.idx if rows is None else rows
        q = np.asarray(get_q(self.path, self.fold, target)[idx], np.float64)
        if (mode or self.mode) == "recal" and self.rc is not None:
            q = C.recalibrate(q, self.rc, target, self.full["date"].to_numpy()[idx])
        return q

    def qc(self, target: str, rows: np.ndarray | None = None) -> np.ndarray | None:
        if self.cl is None:
            return None
        return P.clim_quantiles(self.cl, target, self.cell[self.idx if rows is None else rows])

    def y(self, target: str, rows: np.ndarray | None = None) -> np.ndarray:
        c = self.full[f"real_{target}"].to_numpy(np.float64)
        return c[self.idx if rows is None else rows]


# ---- sidebar ----------------------------------------------------------------------------------

def _sidebar(root: str) -> View | None:
    sb = st.sidebar
    root = sb.text_input("Pockets runs directory", root, key="pk_root",
                         help="a directory of intraday-ft runs (ml/runs/intraday_ft)")
    runs = C.discover_pockets(Path(root))
    if not runs:
        st.info(f"No intraday-ft runs under `{root}` (looking for config.json with model "
                f"`{C.MODEL}` and fold_*/{C.PRED}). Make one with `intraday-ft-train`.")
        return None
    names = [r.name for r in runs]
    labels = {r.name: run_label(r) for r in runs}
    name = sb.selectbox("Pockets run", names, key="pk_run", format_func=labels.get)
    path = str(runs[names.index(name)])
    _model_popover(sb, Path(path))
    folds = C.folds_of(Path(path))
    fold = sb.selectbox("Fold", folds, key="pk_fold", help="the test block (B = 2026, A = 2025)") \
        if len(folds) > 1 else folds[0]
    f = get_frame(path, fold)
    mode = sb.radio("Quantiles", list(MODES), key="pk_mode", format_func=MODES.get,
                    help="Recalibrated: each day's quantiles shifted by what the 20 test sessions "
                         "before it got wrong (only past outcomes, as live). The first 20 days have "
                         "no shift and are left out of the recalibrated scores")
    if mode == "recal" and get_side(path, fold)["recal"] is None:
        sb.warning("This run has no recal.parquet: re-run `intraday-ft-report` on it. Showing raw.")
    with sb.expander("Filters", expanded=False):
        dates = sorted(f["date"].unique())
        d = st.select_slider("Test days", dates, value=(dates[0], dates[-1]), key="pk_dates") \
            if len(dates) > 1 else (dates[0], dates[0])
        sectors = st.multiselect("Sector", sorted(f["sector"].unique()), key="pk_sector")
        ind = f["industry"][f["sector"].isin(sectors)] if sectors else f["industry"]
        industries = st.multiselect("Industry", sorted(ind.unique()), key="pk_industry")
        caps = st.multiselect("Cap band", sorted(f["cap_band"].unique()), key="pk_cap")
        stocks = st.multiselect("Stocks", sorted(f["symbol"].unique()), key="pk_stocks",
                                help="type to search; empty = every stock")
        live = st.checkbox("Live set only (the E*TRADE lists)", key="pk_live")
    min_n = int(sb.number_input("Min rows per group", 1, 1_000_000, MIN_N, key="pk_min_n",
                                help="groups with fewer rows are greyed and left out of In short"))
    idx = get_idx(path, fold, (tuple(d), tuple(sectors), tuple(industries), tuple(caps),
                               tuple(stocks), live))
    v = View(path, fold, idx, mode, min_n)
    sb.caption(f"{len(idx):,} of {len(f):,} decisions · {v.f['symbol'].nunique():,} stocks · "
               f"{v.f['date'].nunique():,} test days")
    if v.cl is None:
        sb.warning("No clim.parquet in this fold: skill vs climatology is blank. Re-run "
                   "`intraday-ft-report` on the run.")
    return v


# ---- tabs -------------------------------------------------------------------------------------

def _score_cfg(t: pd.DataFrame) -> dict:
    return {**_pct_cols(t, ["skill_clim", "skill_lo", "skill_hi", "p50_hit", "cover80",
                            "cal_skill_clim", "cal_p50_hit", "cal_cover80", "came_true",
                            *[c for c in t.columns if c.startswith(("said_", "skill_"))]]),
            **{c: st.column_config.NumberColumn(c, format=SIG)
               for c in ("rank_ic", "clim_rank_ic") if c in t}}


def tab_go(v: View, runs_root: str):
    _header("pk_go")
    st.caption("From the run's own report (all test rows; the sidebar filters don't apply here).")
    t = P.go_rule(P.report_shape(get_report(v.path), v.fold))
    _in_short(P.go_insights(t))
    if t.empty:
        return
    _chart(charts.pk_go(t), "pk_go_chart")
    cols = ["target", "family", "bin", "passes", "skill_clim", "skill_lo", "skill_hi", "rank_ic",
            "clim_rank_ic", "p50_hit", "cover80", "calibrated", "cal_skill_clim", "cal_p50_hit",
            "cal_cover80", "calibrated_after_recal"]
    t = t[[c for c in cols if c in t]]
    _table(t, "pk_go_table", _score_cfg(t))
    rows = []
    for r in C.discover_pockets(Path(runs_root)):
        for fold, h in P.headline(get_report(str(r))).items():
            rows.append({"run": run_label(r), "fold": fold, **h})
    if rows:
        h = pd.DataFrame(rows)
        _table(h, "pk_runs", {**_pct_cols(h, [c for c in h if c.endswith("_skill")]),
                              **{c: st.column_config.NumberColumn(c, format=SIG)
                                 for c in h if c.endswith(("_ic", "_auc"))}})


def _target_pick(key: str, v: View) -> tuple[str, int]:
    a, b = st.columns(2)
    fam = a.selectbox("Pocket-shape target", P.SHAPE, key=f"{key}_family",
                      format_func=lambda x: f"{x}: {P.FAMILIES[x]}")
    b_ = b.selectbox("Duration bin", list(C.BINS), index=3, key=f"{key}_bin",
                     format_func=lambda k: f"b{k}: {C.BINS[k]} after the entry")
    return fam, b_


def tab_shape(v: View):
    _header("pk_shape")
    fam, b = _target_pick("pk_shape", v)
    target = f"{fam}_b{b}"
    w = np.flatnonzero(v.warm())
    rows = v.idx[w]
    if not len(rows):
        st.info("No rows under these filters.")
        return
    f = v.f.iloc[w]
    s = P.score(f, v.q(target, rows), v.y(target, rows), v.qc(target, rows))
    _in_short(P.shape_insights(s, target, v.recal, f["date"].nunique()))
    raw = P.hits_table(P.score(f, v.q(target, rows, "raw"), v.y(target, rows), None, ic=False)["hits"])
    cal = (P.hits_table(P.score(f, v.q(target, rows, "recal"), v.y(target, rows), None,
                                ic=False)["hits"]) if v.rc is not None else None)
    _chart(charts.pk_hits(raw, cal, f"{target}: share of outcomes at or below each quantile"),
           "pk_shape_hits")
    out = []
    for k in C.BINS:
        t = f"{fam}_b{k}"
        sc = P.score(f, v.q(t, rows), v.y(t, rows), v.qc(t, rows))
        out.append({"bin": k, "window": C.BINS[k], **{c: x for c, x in sc.items() if c != "hits"}})
    t = pd.DataFrame(out)
    _table(t, "pk_shape_bins", _score_cfg(t))


def tab_groups(v: View):
    _header("pk_groups")
    fam, b = _target_pick("pk_groups", v)
    key = st.selectbox("Group by", list(P.GROUPS), key="pk_group_by", format_func=P.GROUPS.get)
    target = f"{fam}_b{b}"
    w = np.flatnonzero(v.warm())
    rows = v.idx[w]
    if not len(rows):
        st.info("No rows under these filters.")
        return
    f = v.f.iloc[w]
    t = P.by_group(f, v.q(target, rows), v.y(target, rows), v.qc(target, rows), key)
    if key == "symbol":
        t = t[t["n"] >= max(1, v.min_n // 10)].head(60)
    t["enough_n"] = t["n"] >= v.min_n
    _in_short(P.group_insights(t, key, v.min_n))
    if t["skill_clim"].notna().any():
        _chart(charts.pk_groups(t, key, v.min_n), "pk_groups_chart")
    _table(t, "pk_groups_table", _score_cfg(t))


def _event_probs(v: View, event: str, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Stored readouts per overlay; `all` re-read from its quantiles with the current tail rule
    (as the run report does), so runs scored before the rare-event fix compare like for like."""
    t, thr, _ = P.EVENTS[event]
    out = {o: v.full[f"{event}__{o}"].to_numpy(np.float64)[rows] for o in C.OVERLAYS
           if f"{event}__{o}" in v.full}
    if t in v.targets and "all" in out:
        out["all"] = 1 - P.cdf_at(v.q(t, rows, "raw"), thr)
    return out


def tab_overlays(v: View):
    _header("pk_overlays")
    a, b = st.columns(2)
    event = a.selectbox("Event", list(P.EVENTS), key="pk_event", format_func=lambda e: P.EVENTS[e][2])
    key = b.selectbox("Group by", list(P.GROUPS), key="pk_ov_by", format_func=P.GROUPS.get)
    rows = v.idx
    if not len(rows):
        st.info("No rows under these filters.")
        return
    f = v.f
    probs = _event_probs(v, event, rows)
    hit = P.event_hit(f, event)
    if v.rates is None:
        st.info("This fold has no clim_rates.parquet: re-run `intraday-ft-report` for the skill.")
        rate = np.full(len(f), np.nanmean(hit))
    else:
        rate = P.clim_rate(v.rates, event, v.cell[rows])
    variants = list(probs)
    t = P.overlay_scores(f, probs, hit, rate)
    _in_short(P.overlay_insights(t, event, variants))
    g = P.event_grid(probs, hit)
    _chart(charts.reliability(g, f"{P.EVENTS[event][2]}: said vs came true", min_n=v.min_n // 10 or 1,
                              group="overlay"), "pk_ov_reliab")
    _table(t, "pk_ov_table", _score_cfg(t))
    gt = P.overlay_scores(f, probs, hit, rate, key)
    if key == "symbol":
        gt = gt.head(60)
    gt["enough_n"] = gt["n"] >= v.min_n
    _table(gt, "pk_ov_groups", _score_cfg(gt))


def tab_day(v: View):
    _header("pk_day")
    f = v.f
    if f.empty:
        st.info("No rows under these filters.")
        return
    a, b, c = st.columns(3)
    days = sorted(f["date"].unique())[::-1]
    date = a.selectbox("Day", days, key="pk_day", help="a test day, newest first")
    on = f[f["date"] == date]
    syms = on["symbol"].value_counts().index.tolist()
    sym = b.selectbox("Stock", syms, key="pk_sym", help="stocks with decisions that day, most first")
    bin_ = c.selectbox("Duration bin", list(C.BINS), index=3, key="pk_day_bin",
                       format_func=lambda k: f"b{k}: {C.BINS[k]}")
    sel = on[on["symbol"] == sym].sort_values("minute")
    rows = v.idx[np.flatnonzero((f["date"] == date).to_numpy() & (f["symbol"] == sym).to_numpy())]
    rows = rows[np.argsort(v.full["minute"].to_numpy()[rows], kind="stable")]
    q = {t: v.q(t, rows) for t in [f"{s}_b{k}" for s in ("cm_up", "cm_dn", "rvol") for k in C.BINS]}
    r = P.replay(sel, q, bin_)
    st.caption(f"{sym} · {date} · {len(sel)} decisions · σ = {sel['sigma'].iloc[0]:.2%} · "
               f"{'recalibrated' if v.recal else 'raw'} quantiles")
    _chart(charts.pk_replay(r, f"{sym} {date}: where the volume above / below the entry "
                               f"concentrates in b{bin_} ({C.BINS[bin_]})"), "pk_replay")
    event = st.selectbox("Event for the timeline", list(P.EVENTS), key="pk_day_event",
                         format_func=lambda e: P.EVENTS[e][2])
    probs = _event_probs(v, event, rows)
    tl = pd.DataFrame({"time": r["time"], **probs})
    _chart(charts.pk_timeline(tl, {o: C.OVERLAYS[o] for o in probs},
                              pd.Series(P.event_hit(sel, event)),
                              f"{sym} {date}: {P.EVENTS[event][2]}"), "pk_timeline")
    times = r["time"].tolist()
    when = st.select_slider("Decision time", times, value=times[len(times) // 2], key="pk_when") \
        if len(times) > 1 else times[0]
    i = times.index(when)
    fan = P.pocket_fan(sel.iloc[i], q, i)
    _chart(charts.pk_fan(fan, float(sel["real_p0"].iloc[i]),
                         f"{sym} {date} {when}: the pocket over the bins"), "pk_fan")
    cols = ["window", "up_p10_usd", "up_p50_usd", "up_p90_usd", "up_real_usd", "dn_p10_usd",
            "dn_p50_usd", "dn_p90_usd", "dn_real_usd", "rvol_p50", "rvol_real"]
    ft = fan[cols]
    _table(ft, "pk_fan_table", {
        **{c: st.column_config.NumberColumn(c, format="$%.2f") for c in cols if c.endswith("_usd")},
        **{c: st.column_config.NumberColumn(c, format="%+.2f") for c in ("rvol_p50", "rvol_real")}})


# ---- page -------------------------------------------------------------------------------------

def render(root: str):
    st.title("Intraday pockets")
    v = _sidebar(root)
    if v is None:
        return
    tabs = st.tabs(["Go rule", "Pocket shape", "By group", "Overlays", "Stock day"])
    with tabs[0]:
        tab_go(v, st.session_state.get("pk_root", root))
    with tabs[1]:
        tab_shape(v)
    with tabs[2]:
        tab_groups(v)
    with tabs[3]:
        tab_overlays(v)
    with tabs[4]:
        tab_day(v)
