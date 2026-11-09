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
    moves = [f"real_{t}_b{b}" for t in ("mass_up", "net") for b in C.BINS]
    return [*C.KEYS, "real_p0", *real, *moves, *[f"real_{t}" for t in EVENT_T], *reads]


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


@st.cache_resource(show_spinner="Reading the highs and lows...", max_entries=2)
def get_extremes(path: str, fold: str) -> pd.DataFrame | None:
    """extremes.parquet aligned to the run-file rows (None if the fold has none)."""
    e = C.load_extremes(Path(path), fold)
    return None if e is None else e.reindex(get_frame(path, fold)["dec_row"].to_numpy()).reset_index(drop=True)


@st.cache_resource(show_spinner="Reading the move probabilities...", max_entries=16)
def get_move_bin(path: str, fold: str, x: float, b: int) -> dict:
    """Per run-file row, both sides at ±x in bin b alone: raw P, the centre's P50 / P90 and the outcome."""
    f = get_frame(path, fold)
    ext = get_extremes(path, fold)
    targets = C.targets_of(Path(path), fold)
    out = {}
    for side in P.SIDES:
        q_cm = np.asarray(get_q(path, fold, f"cm_{side}_b{b}"), np.float64)
        mass = f"mass_up_b{b}"
        q_m = np.asarray(get_q(path, fold, mass), np.float64) if mass in targets else None
        out[side] = {"raw": P.move_prob(q_cm, q_m, f["sigma"].to_numpy(), x, side)["p"],
                     "p50": q_cm[:, 2], "p90": q_cm[:, 4], "outcome": P.move_outcome(f, side, b, x, ext)}
    return out


@st.cache_resource(show_spinner="Calibrating the move probabilities...", max_entries=16)
def get_moves(path: str, fold: str, x: float, b: int, cum: bool = True) -> dict:
    """Per run-file row, both sides at ±x: raw P, P recalibrated on the previous 20 sessions (and
    whether the day had them), and the outcome. cum: in any window from the entry to b's end (the
    windows' P combined, the outcomes OR-ed); else in bin b alone. Every row, so a day's
    calibration always sees its 20 earlier sessions whatever the filters."""
    f = get_frame(path, fold)
    bins = [get_move_bin(path, fold, x, k) for k in (range(1, b + 1) if cum else [b])]
    out = {}
    for side in P.SIDES:
        sb = [m[side] for m in bins]
        raw = P.move_prob_cum([m["raw"] for m in sb]) if cum else sb[-1]["raw"]
        oc = P.move_outcome_cum([m["outcome"] for m in sb], side) if cum else sb[-1]["outcome"]
        cal, warm = P.move_recal(raw, oc["reached"].to_numpy(), f["date"].to_numpy())
        # each window has its own centre: show the one of the window the model gave the highest P
        c = P.move_centre_cum([m["raw"] for m in sb], [m["p50"] for m in sb], [m["p90"] for m in sb],
                              [m["outcome"]["centre_pct"].to_numpy() for m in sb])
        if not cum:
            c["win"] = np.full(len(raw), b)
        out[side] = {"raw": raw, "cal": cal, "warm": warm, "win": c["win"], "p50": c["p50"], "p90": c["p90"],
                     "win_centre_pct": c["centre_pct"], "outcome": oc}
    return out


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


# ---- moves: the ±X% filter ---------------------------------------------------------------------

X_CHOICES = {0.005: "0.5%", 0.01: "1%", 0.015: "1.5%", 0.02: "2%", 0.03: "3%"}
MV_SIDES = {"both": "both", "up": "up only", "dn": "down only"}


def _usd(cols) -> dict:
    return {c: st.column_config.NumberColumn(c, format="$%.2f") for c in cols}


def _signed(cols) -> dict:
    return {c: st.column_config.NumberColumn(c, format="%+.2f%%") for c in cols}


def _moves_controls(v: View):
    dates = sorted(v.f["date"].unique())
    a, b, c = st.columns(3)
    d = a.select_slider("Days", dates, value=(dates[max(0, len(dates) - 10)], dates[-1]), key="pk_mv_dates",
                        help="the test days to report on (default: the last 10 the sidebar keeps)") \
        if len(dates) > 1 else (dates[0], dates[0])
    x = b.selectbox("Move X", list(X_CHOICES), index=1, key="pk_mv_x", format_func=X_CHOICES.get,
                    help="the target: P0 × (1 ± X)")
    side = c.selectbox("Side", list(MV_SIDES), key="pk_mv_side", format_func=MV_SIDES.get)
    a, b, c = st.columns(3)
    bin_ = a.selectbox("Window", list(C.BINS), index=1, key="pk_mv_bin",
                       format_func=lambda k: f"b{k}: {C.BINS[k]} (by its end: {P.CUM_BINS[k]})",
                       help="with the toggle on, everything from the entry to the window's end; off, the "
                            "window alone")
    min_p = b.slider("Minimum P", 0.05, 0.95, 0.4, 0.05, key="pk_mv_min_p",
                     help="show the decisions whose probability is at least this. The centre "
                          "reaches ±1% in 30-60 min only ~6% of the time, so a calibrated P "
                          "rarely goes above 0.6")
    cal = c.radio("P", ["cal", "raw"], key="pk_mv_mode", horizontal=True,
                  format_func={"cal": "calibrated", "raw": "raw"}.get,
                  help="calibrated: each day's P mapped through what happened at each P over the 20 "
                       "sessions before it (only past outcomes, as live). raw: as the model's quantiles say")
    a, b = st.columns(2)
    cum = a.toggle("By the window's end (any time from the entry)", value=True, key="pk_mv_cum",
                   help="on: P that it happens in any window from the entry to the chosen one's end (b4 = "
                        "0-60 min), outcomes over all of them. off: the chosen window alone (b4 = 30-60 "
                        "min only)")
    first = b.checkbox("One row per stock run (the first decision of consecutive matches)", key="pk_mv_first",
                       help="a stock often matches at several decisions in a row; this keeps the first of each")
    return d, x, side, bin_, min_p, cal == "cal", first, cum


def tab_moves(v: View):
    _header("pk_moves")
    if v.f.empty:
        st.info("No rows under these filters.")
        return
    (d0, d1), x, side, b, min_p, cal, first, cum = _moves_controls(v)
    mv = get_moves(v.path, v.fold, x, b, cum)
    window = P.move_window(b, cum)
    ext = get_extremes(v.path, v.fold)
    keep = (v.f["date"] >= d0).to_numpy() & (v.f["date"] <= d1).to_numpy()
    rows = v.idx[keep]
    f = v.full.iloc[rows]
    sides = ["up", "dn"] if side == "both" else [side]
    p = {s: (mv[s]["cal"] if cal else mv[s]["raw"])[rows] for s in ("up", "dn")}
    parts, scored = [], {}
    both = (p["up"] >= min_p) & (p["dn"] >= min_p)
    for s in sides:
        m = p[s] >= min_p
        o = mv[s]["outcome"].iloc[rows]
        if first:
            start, _ = P.move_runs(f["date"], f["symbol"], f["minute"], m)
            m &= start
        scored[s] = len(rows)
        if not m.any():
            continue
        g = f[m]
        oth = "dn" if s == "up" else "up"
        sig, p0 = g["sigma"].to_numpy(np.float64), g["real_p0"].to_numpy(np.float64)
        sign = 1.0 if s == "up" else -1.0
        p50, p90 = mv[s]["p50"][rows][m], mv[s]["p90"][rows][m]
        win, wcp = mv[s]["win"][rows][m], mv[s]["win_centre_pct"][rows][m]
        oc = o[m]
        ooc = mv[oth]["outcome"].iloc[rows][m]
        parts.append(pd.DataFrame({
            "date": g["date"].to_numpy(), "time": [P.hhmm(t) for t in g["minute"]], "minute": g["minute"].to_numpy(),
            "symbol": g["symbol"].to_numpy(), "sector": g["sector"].to_numpy(), "side": s,
            "p0": p0, "target": p0 * (1 + sign * x), "p": p[s][m], "p_raw": mv[s]["raw"][rows][m],
            "warm": mv[s]["warm"][rows][m],
            "window": [f"b{k}: {C.BINS[k]}" for k in win],
            "centre_p50": p0 * np.exp(sign * sig * p50), "centre_p90": p0 * np.exp(sign * sig * p90),
            "real_centre": p0 * (1 + wcp), "far_centre": p0 * (1 + oc["centre_pct"].to_numpy()),
            "reached": oc["reached"].to_numpy(), "formed": oc["formed"].to_numpy(),
            "touched": oc["touched"].to_numpy(),
            "best_pct": oc["best_pct"].to_numpy(), "best_min": oc["best_min"].to_numpy(),
            "worst_pct": oc["worst_pct"].to_numpy(),
            **{c: oc[c].to_numpy() for c in P.AFTER},
            "vwap_move": sign * oc["vwap_pct"].to_numpy(),
            "centre_pct": oc["centre_pct"].to_numpy(),
            "captured": wcp / np.expm1(sign * sig * p50),
            "other_p": p[oth][m], "other_reached": ooc["reached"].to_numpy(),
            "other_centre_pct": ooc["centre_pct"].to_numpy(),
            "p_up": p["up"][m], "p_dn": p["dn"][m],
            "reached_up": (oc if s == "up" else ooc)["reached"].to_numpy(),
            "reached_dn": (ooc if s == "up" else oc)["reached"].to_numpy(),
            "net_p50": np.asarray(get_q(v.path, v.fold, f"net_b{b}"), np.float64)[rows][m, 2]
            if f"net_b{b}" in v.targets else np.nan,
            "real_net": g[f"real_net_b{b}"].to_numpy(np.float64) if f"real_net_b{b}" in g else np.nan,
        }))
    mt = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["date", "side", "symbol", "minute", "p", "reached", "touched", "formed", "centre_pct",
                 "best_pct", "vwap_move", "captured", "warm", "p_up", "p_dn", "reached_up", "reached_dn", "net_p50",
                 "real_net"])
    days_in = sorted(f["date"].unique())
    cold = int((~pd.Series(mv["up"]["warm"][rows]).groupby(f["date"].to_numpy()).all()).sum()) if cal else 0
    summ = P.move_summary(mt, scored)
    lean = P.move_lean(mt)
    _in_short(P.move_insights(summ, lean, x, window, min_p, cal, int(both.sum()), cold, ext is not None))
    if ext is None:
        st.warning("No extremes.parquet in this fold: touched, the high / low, VWAP / TWAP after the entry "
                   "are blank. Run `intraday-ft-extremes` on the run.")
    elif "vw_b1" not in ext:
        st.warning("This fold's extremes.parquet is from before the VWAP / TWAP columns: those are blank. "
                   "Run `intraday-ft-extremes` on the run again.")
    pct = ["share_flagged", "said", "reached", "gap", "gap_lo", "gap_hi", "touched", "not_formed",
           "median_centre_pct", "median_best_pct", "median_vwap_move"]
    _table(summ, "pk_mv_summary", {**_pct_cols(summ, pct),
                                   "median_captured": st.column_config.NumberColumn("median_captured",
                                                                                    format="%.2f×")})
    sd = pd.DataFrame([(dd, s, int((f["date"] == dd).sum())) for dd in days_in for s in sides],
                      columns=["date", "side", "decisions"])
    by_day = P.move_by_day(mt, sd) if len(mt) else sd.assign(predictions=0, happened=0, did_not=0, stocks=0)
    _table(by_day, "pk_mv_days", _pct_cols(by_day, ["said", "max_said", "reached", "touched", "median_best_pct",
                                                    "median_vwap_move"]))
    tm = P.move_times(mt)
    if len(tm):
        _chart(charts.pk_mv_times(tm, f"When in the day: predictions per 15 min, {d0} → {d1}"), "pk_mv_times")
        _table(tm, "pk_mv_times_table", _pct_cols(tm, ["said", "reached"]))
    if len(lean):
        _table(lean, "pk_mv_lean", _pct_cols(lean, ["share", "mean_p_up", "mean_p_dn", "went_up",
                                                    "lean_reached", "other_reached"]))
    if len(mt):
        day = st.selectbox("List: day", ["every day", *sorted(mt["date"].unique())[::-1]], key="pk_mv_day")
        lst = mt if day == "every day" else mt[mt["date"] == day]
        lst = lst.sort_values("p", ascending=False).head(500)
        moves = ["high_pct", "low_pct", "vwap_pct", "twap_pct", "end_pct", "centre_pct", "other_centre_pct"]
        lst = lst.assign(**{c: lst[c] * 100 for c in moves},
                         **{c.removesuffix("_pct"): lst["p0"] * (1 + lst[c]) for c in moves[:5]})
        cols = ["date", "time", "symbol", "sector", "side", "p0", "target", "p", "p_raw", "window", "centre_p50",
                "centre_p90", "real_centre", "captured", "far_centre", "reached", "touched", "high", "high_pct",
                "high_min", "low", "low_pct", "low_min", "vwap", "vwap_pct", "twap", "twap_pct", "end",
                "end_pct", "other_p", "other_reached", "other_centre_pct"]
        lst = lst[cols]
        _table(lst, "pk_mv_list", {**_usd(["p0", "target", "centre_p50", "centre_p90", "real_centre", "far_centre",
                                           "high", "low", "vwap", "twap", "end"]),
                                   **_pct_cols(lst, ["p", "p_raw", "other_p"]),
                                   **_signed(moves),
                                   "captured": st.column_config.NumberColumn("captured", format="%.2f×"),
                                   **{c: st.column_config.NumberColumn(c, format="+%d min")
                                      for c in ("high_min", "low_min")}})
        st.caption(f"{len(lst):,} shown (highest P first, at most 500) of {len(mt):,}")
    # said vs happened over the range, every decision (not just the matches)
    probs, hits = {}, []
    for s in sides:
        probs[P.SIDES[s]] = p[s]
        hits.append(mv[s]["outcome"]["reached"].to_numpy()[rows])
    g = pd.concat([P.event_grid({k: pr}, h) for (k, pr), h in zip(probs.items(), hits)], ignore_index=True)
    if len(g):
        _chart(charts.reliability(g, f"±{X_CHOICES.get(x, x)} in {window}: said vs happened "
                                     f"({'calibrated' if cal else 'raw'}, every decision in the range)",
                                  min_n=20, group="overlay"), "pk_mv_reliab")
        _table(g, "pk_mv_cal", _pct_cols(g, ["mean_prob", "hit_rate", "deviation", "ci_lo", "ci_hi"]))


# ---- page -------------------------------------------------------------------------------------

def render(root: str):
    st.title("Intraday pockets")
    v = _sidebar(root)
    if v is None:
        return
    tabs = st.tabs(["Go rule", "Pocket shape", "By group", "Overlays", "Stock day", "Moves"])
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
    with tabs[5]:
        tab_moves(v)
