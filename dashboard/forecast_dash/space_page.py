"""Intraday → Space: where the later volume traded after moments like this one.

Pick a stock, a day and a time (a **moment**). Its parameters (the day so far, in the stock's σ)
are matched against the same time of day on earlier sessions of every stock, at two tolerances.
The **pockets** show where the later volume of those matches traded, next to everything at that
time of day (climatology) and what the stock then did. Reads a moment library built by
`space-build` (the mtap_space package; see docs/analog-pockets.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from mtap_space import pockets as K
from mtap_space.library import Library, find, hhmm
from mtap_space.match import Query
from streamlit.errors import StreamlitAPIException

from forecast_dash import charts, insights
from forecast_dash import space_calc as sc
from forecast_dash.ui import _about, _chart, _header, _in_short, _pct_cols, _table

WINDOWS = (30, 60, 90)
TODS = (0, 15, 30)
DEFAULT_TIME = 90                                  # 11:00
BUILD = "cd space && uv run space-build --sessions 90"
SIG = "%+.2f"
S3D_JS = Path(__file__).with_name("space3d") / "space3d.js"     # built from dashboard/js
S3D_BUILD = "cd dashboard/js && npm install && npm run build"
SIGMA_TICKS = (-3, -2, -1, 0, 1, 2, 3)


# ---- cached library and readings (cache_resource: shared, never copied; read-only) ----------

@st.cache_resource(show_spinner="Opening the moment library...", max_entries=2)
def get_library(path: str) -> Library:
    return Library(path)


@st.cache_resource(show_spinner=False, max_entries=32)
def get_pool(path: str, p: int, s: int, window: int, tod: int, min_dollar: float,
             min_price: float, tk: str | None) -> tuple[np.ndarray, K.Reading | None]:
    """The moments a query at (p, s) is matched against, and their climatology. The same for
    every stock at that time and day, unless the pool is the stock's own history."""
    lib = get_library(path)
    rows = lib.pool(p, s, window, tod, min_dollar, min_price, tk)
    return rows, (K.climatology(lib, rows) if len(rows) else None)


def _query(key: tuple) -> tuple[Library, Query, np.ndarray, K.Reading | None]:
    path, row, params, tol, window, tod, min_dollar, min_price, own = key
    lib = get_library(path)
    q = Query(row=row, params=list(params), window=window, tod=tod, min_dollar=min_dollar,
              min_price=min_price, own_stock=own, tol=dict(tol))
    e = int(lib.m["e"][row])
    pool, clim = get_pool(path, int(lib.m["p"][row]), int(lib.m["s"][row]), window, tod,
                          min_dollar, min_price, lib.tk[e] if own else None)
    return lib, q, pool, clim


@st.cache_resource(show_spinner="Matching...", max_entries=16)
def get_space(key: tuple, dials: tuple[float, float]) -> sc.Space | None:
    lib, q, pool, clim = _query(key)
    return None if clim is None else sc.evaluate(lib, q, dials, pool, clim)


@st.cache_resource(show_spinner="Sweeping the tolerance...", max_entries=8)
def get_curve(key: tuple) -> pd.DataFrame:
    lib, q, pool, clim = _query(key)
    return sc.dial_curve(lib, q, pool, clim)


# ---- URL: ?sp_stock=MU&sp_date=2026-10-02&sp_time=11:00 opens a moment -----------------------

def _seed(key: str, options, parse=str):
    """Seed a widget from ?<key>= on its first run, if the value is one of its options."""
    if key in st.session_state or key not in st.query_params:
        return
    try:
        v = parse(st.query_params[key])
    except ValueError:
        return
    if v in options:
        st.session_state[key] = v


def _parse_time(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m) - 570


def _usd(x) -> str:
    """$ for st.markdown / st.caption: escaped, else two $ signs open LaTeX."""
    return f"\\${float(x):,.2f}"


def _write_url(values: dict[str, str]):
    for k, v in values.items():
        if st.query_params.get(k) != v:
            st.query_params[k] = v


# ---- sidebar ----------------------------------------------------------------------------------

def _moment(sb, lib: Library) -> int | None:
    """Stock, date and time pickers. Returns the moment's row in the library."""
    sb.markdown("**The moment**")
    stocks = lib.stocks()
    tks = stocks["tk"].tolist()
    ndol = dict(zip(stocks["tk"], stocks["ndol"]))
    _seed("sp_stock", tks)
    tk = sb.selectbox("Stock", tks, key="sp_stock",
                      format_func=lambda t: f"{t} (${ndol[t] / 1e6:,.0f}M a day)",
                      help="every stock in the library, most traded first (median dollar volume "
                           "a day over its library days). Type to search")
    dates = lib.days_of(tk)["date"].tolist()
    _seed("sp_date", dates)
    date = sb.selectbox("Date", dates, key="sp_date",
                        help="the stock's sessions in the library, newest first")
    e = lib.stockday(tk, date)
    times = lib.times_of(e)
    if not times:
        return None
    _seed("sp_time", times, _parse_time)
    if "sp_time" not in st.session_state and DEFAULT_TIME in times:
        st.session_state["sp_time"] = DEFAULT_TIME
    if len(times) == 1:
        p = times[0]
        sb.caption(f"Time: {hhmm(p)} ET (the only one that day)")
    else:
        p = sb.select_slider("Time (ET)", times, key="sp_time", format_func=hhmm,
                             help="the decision time. The parameters use only the bars before "
                                  "it; the pockets show the bars from it to the close")
    _write_url({"sp_stock": tk, "sp_date": date, "sp_time": hhmm(p)})
    return lib.moment(e, p)


def _matching(sb, lib: Library, row: int) -> tuple:
    """Parameter, tolerance, history and pool controls. Returns the query key and the dials."""
    meta = lib.meta["params"]
    x0 = sc.query_values(lib, row)
    sb.markdown("**Matching**")
    params = sb.multiselect(
        "Parameters", lib.params, default=[k for k in lib.meta["default_params"] if k in lib.params],
        key="sp_params", format_func=lambda k: meta[k]["label"],
        help="what must be similar for a past moment to count as a match. None: every moment "
             "in the pool counts (that is the 'everything' view)")
    if params:
        sb.caption("This moment: " + " · ".join(
            f"`{k}` {x0[k]:+.2f}{'σ' if meta[k]['unit'] == 'σ' else ''}" for k in params))
    with sb.expander("Tolerance per parameter"):
        st.caption("A multiplier on the dial for one parameter: 2 = twice as loose for it.")
        tol = {k: st.number_input(meta[k]["label"], 0.25, 4.0, 1.0, 0.25, key=f"sp_tol_{k}")
               for k in params}
    dials = sb.slider("Tolerance: close – loose", 0.1, 3.0, (0.5, 1.0), 0.05, key="sp_dials",
                      help="how far a past moment's parameters may be from this one's, in units "
                           "of each parameter's spread over the library (IQR / 1.349). An exact "
                           "match gets weight 1, one at the edge 0. Left handle: the close "
                           "match; right handle: the loose match")
    n = len(lib.sessions)
    wins = sorted({min(w, n) for w in WINDOWS})
    window = sb.radio("History (sessions)", wins, index=wins.index(min(60, n)), key="sp_window",
                      horizontal=True,
                      help="how many sessions before the moment's day the matches come from. "
                           "The moment's own day and later days are never used")
    with sb.expander("Pool filters"):
        min_dollar = st.number_input(
            "Min dollar volume a day ($M)", 0.0, 100_000.0, lib.meta["min_dollar"] / 1e6, 1.0,
            key="sp_mindol", help="only stocks that traded at least this much a day over the "
                                  "20 sessions before") * 1e6
        min_price = st.number_input("Min price ($)", 0.0, 10_000.0, float(lib.meta["min_price"]),
                                    1.0, key="sp_minpx", help="the previous day's VWAP")
        own = st.checkbox("This stock only", key="sp_own",
                          help="match only against the stock's own earlier days")
        tod = st.radio("Time of day ±", TODS, key="sp_tod", horizontal=True,
                       format_func=lambda m: f"{m} min",
                       help="also match decision times this far from this one: ±15 min makes "
                            "the pool three times bigger")
    key = (str(lib.path), int(row), tuple(params), tuple(sorted(tol.items())), int(window),
           int(tod), float(min_dollar), float(min_price), bool(own))
    return key, (float(dials[0]), float(dials[1]))


# ---- page -------------------------------------------------------------------------------------

def render(root: str):
    st.title("Intraday space")
    sb = st.sidebar
    root = sb.text_input("Moment library directory", root, key="sp_root",
                         help="holds lib-<last session>/ folders made by space-build")
    libs = [str(p) for p in find(root)] if Path(root).is_dir() else []
    if not libs:
        st.info(f"No moment library under `{root}` (looking for lib-*/meta.json). Build one "
                f"from the minute bars with `{BUILD}` (about a minute for 90 sessions).")
        return
    path = sb.selectbox("Library", libs, key="sp_lib", format_func=lambda s: Path(s).name,
                        help="newest first; each is named after its last session")
    lib = get_library(path)
    s = lib.sessions
    sb.caption(f"{len(s)} sessions ({s[0]} … {s[-1]}) · {lib.meta['n_moments']:,} moments · "
               f"{len(lib.sd):,} stock-days")
    row = _moment(sb, lib)
    if row is None:
        st.info("This stock has no moment on that day.")
        return
    key, dials = _matching(sb, lib, row)
    sp = get_space(key, dials)
    if sp is None:
        st.info("No earlier moments in the pool: pick a later date, more history or looser pool "
                "filters.")
        return
    info = sc.moment_info(lib, row)
    _header("sp_space")
    st.markdown(
        f"**{info['symbol']}** · {info['date']} · {info['time']} ET · price now "
        f"**{_usd(info['now'])}** · σ = {info['sig']:.2%} a day (1σ ≈ "
        f"{_usd(info['now'] * info['sig'])}) · previous VWAP {_usd(info['prev'])} · open "
        f"{_usd(info['open'])} · pool {len(sp.pool):,} moments")
    ev = sc.evidence(sp)
    past = bool(sp.A.sum() > 0)
    _in_short(insights.space_insights(ev, past))
    maps, regions = _titled(sp)
    _space3d(maps, regions, info, _cell(sp))
    _evidence(ev, past)
    tabs = st.tabs(["Compare", "Matches", "Dial curve", "Day path"])
    with tabs[0]:
        tab_compare(sp, info, maps, regions)
    with tabs[1]:
        tab_matches(sp, info)
    with tabs[2]:
        tab_dial(key, dials, sp)
    with tabs[3]:
        tab_day(sp, info)


def _evidence(ev: pd.DataFrame, past: bool):
    t = ev.reset_index(drop=True)
    weak = (t["days"] < insights.SPACE_MIN_DAYS) & t["view"].isin(
        [sc.VIEWS["close"], sc.VIEWS["loose"]])
    t.loc[weak, "view"] = "⚠️ " + t.loc[weak, "view"]
    if not past:
        t = t.drop(columns=["in_50", "in_80", "log_score"])
    cfg = {**_pct_cols(t, ["trust", "p_up", "above_now", "in_50", "in_80"]),
           **{c: st.column_config.NumberColumn(c, format=SIG) for c in
              ("rod_p10", "rod_p50", "rod_p90", "bulge_p50", "sag_p50")},
           "matches": st.column_config.NumberColumn("matches", format="localized"),
           "stocks": st.column_config.NumberColumn("stocks", format="localized"),
           "days": st.column_config.NumberColumn("days", format="%.1f"),
           "log_score": st.column_config.NumberColumn("log_score", format="%+.3f")}
    _table(t, "sp_evidence", cfg)


def _titled(sp: sc.Space) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The four maps under their chart titles, and each one's pocket regions."""
    maps, regions = {}, {}
    for k, Q in sp.maps().items():
        if k == "actual" and Q.sum() == 0:
            continue
        reg = K.region(Q)
        n = f"{sp.M[k].n:,} matches · " if k in sp.M else ""
        nb = K.blobs(reg == 0)
        title = f"{sc.VIEWS[k]} · {n}{nb} blob{'s' if nb != 1 else ''} in 50%"
        maps[title], regions[title] = Q, reg
    return maps, regions


# ---- the 3-D view (three.js, st.components.v2; source in dashboard/js) ----------------------

def _cell(sp: sc.Space) -> tuple[int, int]:
    """The picked cell (Compare tab sliders, or a click on the 3-D view). It starts at the loose
    match's densest cell."""
    if "sp_cell_x" not in st.session_state or "sp_cell_y" not in st.session_state:
        i, j = np.unravel_index(np.argmax(sp.R["loose"].Q), sp.R["loose"].Q.shape)
        st.session_state["sp_cell_x"], st.session_state["sp_cell_y"] = int(i), int(j)
    return st.session_state["sp_cell_x"], st.session_state["sp_cell_y"]


def _pick_cell():
    """A click on the 3-D view picks that cell (runs before the rerun, so before the sliders)."""
    c = (st.session_state.get("sp_space3d") or {}).get("cell")
    if c:
        st.session_state["sp_cell_x"], st.session_state["sp_cell_y"] = int(c["i"]), int(c["j"])


def space3d_data(maps: dict[str, np.ndarray], regions: dict[str, np.ndarray], now: float,
                 sig: float, cell: tuple[int, int] | None) -> dict:
    """What space3d.js draws: per view its title, the share of the later volume per cell and
    the cell's pocket (both flat, time-ahead major) and the 50 / 80 / 95% levels; the grid edges
    and axis ticks (in cell units) and the price now and σ for the hover's $."""
    xe, ye = K.x_edges(), K.y_edges()
    xt = [[float(np.interp(m, xe, np.arange(len(xe)))), t] for m, t in charts.MIN_TICKS]
    yt = [[float(np.interp(s, ye, np.arange(len(ye)))), "now" if s == 0 else f"{s:+d}σ"]
          for s in SIGMA_TICKS]
    return {"nx": K.NX, "ny": K.NY, "xe": xe.tolist(), "ye": ye.tolist(), "xt": xt, "yt": yt,
            "now": now, "sig": sig, "cell": list(cell) if cell else None,
            "views": [{"title": k, "q": np.round(Q.ravel(), 8).tolist(),
                       "region": regions[k].ravel().tolist(), "levels": sc.thresholds(Q)}
                      for k, Q in maps.items()]}


def _space3d_component():
    try:   # the bundle as a file the browser caches (pyproject.toml [tool.streamlit.component])
        return st.components.v2.component("forecast-dash.space3d", js="space3d.js")
    except StreamlitAPIException:   # no manifest found (not an editable install): inline it
        return st.components.v2.component("forecast-dash.space3d", js=S3D_JS.read_text())


def _space3d(maps: dict[str, np.ndarray], regions: dict[str, np.ndarray], info: dict,
             cell: tuple[int, int]):
    if not S3D_JS.exists():
        st.info(f"The 3-D view is not built yet: `{S3D_BUILD}`. The maps below show the same.")
        return
    _about("sp_space3d")
    _space3d_component()(key="sp_space3d", on_cell_change=_pick_cell,
                         data=space3d_data(maps, regions, info["now"], info["sig"], cell))


def tab_compare(sp: sc.Space, info: dict, maps: dict[str, np.ndarray],
                regions: dict[str, np.ndarray]):
    _header("sp_compare")
    xe, ye = K.x_edges(), K.y_edges()
    _chart(charts.pocket_maps(maps, regions, xe, ye, info["now"], info["sig"], sc.REGIONS,
                              "Pockets: where the later volume traded"), "sp_maps")
    which = st.radio("Difference of", ["loose", "close"], key="sp_diff_of", horizontal=True,
                     format_func=sc.VIEWS.get, help="which match to set against everything")
    _chart(charts.pocket_diff(sp.R[which].Q, sp.R["clim"].Q, xe, ye, info["now"], info["sig"],
                              f"{sc.VIEWS[which]} − everything: share of the later volume"),
           "sp_diff")
    st.markdown("**Pick a cell** (or click one in the 3-D view)")
    a, b = st.columns(2)
    i = a.select_slider("Time ahead", list(range(K.NX)), key="sp_cell_x",
                        format_func=lambda i: f"{charts.span_label(xe[i])}–"
                                              f"{charts.span_label(xe[i + 1])}",
                        help="the cell's column: how long after the decision time")
    j = b.select_slider("Price vs now", list(range(K.NY)), key="sp_cell_y",
                        format_func=lambda j: f"{ye[j]:+.2f}σ to {ye[j + 1]:+.2f}σ",
                        help="the cell's row: the minute VWAP vs the price at the decision time")
    lo, hi = sc.to_price(info["now"], info["sig"], [ye[j], ye[j + 1]])
    st.caption(f"{charts.span_label(xe[i])}–{charts.span_label(xe[i + 1])} after "
               f"{info['time']}, at {_usd(lo)}–{_usd(hi)}")
    ct = sc.cell_table(sp, i, j)
    _table(ct, "sp_cell", {**_pct_cols(ct, ["share_here"]), "vs_everything":
                           st.column_config.NumberColumn("vs_everything", format="%.2f×")})
    cm = sc.cell_matches(sp, i, j)
    if cm.empty:
        st.caption("No loose match traded volume in this cell.")
    else:
        _table(cm, "sp_cell_matches", {**_pct_cols(cm, ["of_cell"]), "weight":
                                       st.column_config.NumberColumn("weight", format="%.2f")})


def tab_matches(sp: sc.Space, info: dict):
    _header("sp_matches")
    lib, meta = sp.lib, sp.lib.meta["params"]
    x0 = sc.query_values(lib, sp.q.row)
    tol = {k: [d * lib.spread[k] * sp.q.tol.get(k, 1.0) for d in sp.dials] for k in lib.params}
    qt = pd.DataFrame({"parameter": [meta[k]["label"] for k in lib.params],
                       "value": [x0[k] for k in lib.params],
                       "close_tol": [tol[k][0] for k in lib.params],
                       "loose_tol": [tol[k][1] for k in lib.params],
                       "used": [k in sp.q.params for k in lib.params]})
    _table(qt, "sp_query", {c: st.column_config.NumberColumn(c, format=f) for c, f in
                            (("value", SIG), ("close_tol", "±%.2f"), ("loose_tol", "±%.2f"))})
    t, rows = sc.matches(sp)
    if t.empty:
        st.info("The loose match found no moments: widen its dial or drop a parameter.")
        return
    cfg = {k: st.column_config.NumberColumn(k, format=SIG, help=meta[k]["label"] + " (σ)")
           for k in sp.q.params}
    cfg.update({c: st.column_config.NumberColumn(c, format=SIG)
                for c in ("rest_of_day", "bulge", "sag")})
    cfg.update({**_pct_cols(t, ["above_now"]),
                "weight": st.column_config.NumberColumn("weight", format="%.2f")})
    st.caption(f"The {len(t)} heaviest of {sp.M['loose'].n:,} loose matches. Pick a row to "
               f"see that moment's own pocket.")
    ev = _table(t, "sp_match_table", cfg, on_select="rerun", selection_mode="single-row",
                key="sp_match_pick")
    picked = [r for r in (ev.selection.rows if ev is not None else []) if r < len(rows)]
    if picked:
        r = int(rows[picked[0]])
        mine = sp.maps()["actual"]
        maps = {f"{lib.label(r)}: what it did": K.density(K.actual(lib, r)),
                f"{info['symbol']} {info['date']} {info['time']}: what it did": mine}
        regions = {k: K.region(Q) for k, Q in maps.items()}
        _chart(charts.pocket_maps(maps, regions, K.x_edges(), K.y_edges(), info["now"],
                                  info["sig"], sc.REGIONS, "The picked match next to this moment"),
               "sp_match_maps")


def tab_dial(key: tuple, dials: tuple[float, float], sp: sc.Space):
    _header("sp_dial")
    t = get_curve(key)
    clim = sp.R["clim"]
    _chart(charts.dial_curve(t, dials, clim.pu, clim.above), "sp_dial_chart")
    with st.expander("Dial curve table"):
        _table(t, "sp_dial_table", {**_pct_cols(t, ["p_up", "above_now"]),
                                    "days": st.column_config.NumberColumn("days", format="%.1f"),
                                    "dial": st.column_config.NumberColumn("dial", format="%.2f")})


def tab_day(sp: sc.Space, info: dict):
    _header("sp_day")
    d = sc.day_path(sp.lib, sp.q.row)
    _chart(charts.space_day_path(d, sp.p, info["prev"],
                                 f"{info['symbol']} · {info['date']}: decision at {info['time']}"),
           "sp_day_chart")
    m = sp.lib.m
    rod = sc.to_price(info["now"], info["sig"], m["rod"][sp.q.row])
    st.caption(f"After {info['time']} the stock's VWAP to the close was {_usd(rod)} "
               f"({float(m['rod'][sp.q.row]):+.2f}σ vs now); its high after was "
               f"{float(m['bul'][sp.q.row]):+.2f}σ and its low {float(m['sag'][sp.q.row]):+.2f}σ.")
