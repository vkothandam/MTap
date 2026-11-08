"""Plotly figures from the tables in calib.py. Pure functions: tables in, Figure out."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .calib import SIDE_UP

PCT = ".0%"
IDEAL = {"dash": "dash", "color": "grey", "width": 1}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _layout(fig: go.Figure, title: str, height: int = 460, **kw) -> go.Figure:
    fig.update_layout(title=title, height=height, margin={"l": 50, "r": 20, "t": 50, "b": 40},
                      legend={"orientation": "h", "y": -0.18}, **kw)
    return fig


def _sizes(n: pd.Series, lo: float = 6, hi: float = 28) -> np.ndarray:
    s = np.sqrt(n.to_numpy(dtype=float))
    return lo + (hi - lo) * s / s.max() if len(s) and s.max() > 0 else np.full(len(s), lo)


def reliability(g: pd.DataFrame, title: str = "Said vs came true", min_n: int = 1,
                group: str = "event") -> go.Figure:
    """One line per `group` (event or run): x = mean forecast prob per bucket, y = hit rate,
    Wilson 95% error bars, marker size ~ sqrt(n). Below the diagonal = overconfident."""
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=IDEAL, name="ideal",
                               hoverinfo="skip"))
    g = g[g["n"] >= min_n]
    sizes = pd.Series(_sizes(g["n"]), index=g.index) if len(g) else pd.Series(dtype=float)
    for name, d in g.groupby(group, sort=False):
        fig.add_trace(go.Scatter(
            x=d["mean_prob"], y=d["hit_rate"], mode="lines+markers", name=str(name),
            marker={"size": sizes[d.index]},
            error_y={"type": "data", "symmetric": False, "array": d["ci_hi"] - d["hit_rate"],
                     "arrayminus": d["hit_rate"] - d["ci_lo"], "thickness": 1},
            customdata=np.stack([d["bucket"], d["n"], d["deviation"]], axis=1),
            hovertemplate=("%{customdata[0]}: said %{x:.1%}, came true %{y:.1%}"
                           "<br>n=%{customdata[1]:,}  deviation %{customdata[2]:+.1%}"),
        ))
    fig.update_xaxes(title="model probability", tickformat=PCT, range=[0, 1])
    fig.update_yaxes(title="came true", tickformat=PCT, range=[0, 1])
    return _layout(fig, title, height=520)


def _num(label) -> float:
    return float(re.search(r"\d+(\.\d+)?", str(label)).group())


def heatmap(g: pd.DataFrame, title: str = "Deviation (came true - said)", min_n: int = 1,
            x: str = "bucket", xtitle: str = "model probability bucket") -> go.Figure:
    """Events x `x` (probability bucket, or call level), coloured by came true - said."""
    g = g[g["n"] >= min_n]
    order = g.sort_values("move", key=lambda m: m.where(m < 0, 1 + m))["event"].unique()
    buckets = sorted(g[x].unique(), key=_num)
    piv = lambda col: g.pivot(index="event", columns=x, values=col).reindex(
        index=order, columns=buckets)
    dev, hit, said, n = piv("deviation"), piv("hit_rate"), piv("mean_prob"), piv("n")
    text = (hit.map(lambda v: "" if pd.isna(v) else f"{v:.0%}") + " / "
            + said.map(lambda v: "" if pd.isna(v) else f"{v:.0%}"))
    text = text.where(n.notna(), "")
    lim = float(np.nanmax(np.abs(dev.to_numpy()))) if dev.notna().any().any() else 0.1
    fig = go.Figure(go.Heatmap(
        z=dev.to_numpy(), x=buckets, y=list(order), zmid=0, zmin=-lim, zmax=lim,
        colorscale="RdBu", text=text.to_numpy(), texttemplate="%{text}",
        customdata=n.to_numpy(), colorbar={"title": "dev", "tickformat": "+.0%"},
        hovertemplate="%{y} at %{x}<br>came true / said: %{text}<br>n=%{customdata:,}"
                      "<br>deviation %{z:+.1%}<extra></extra>",
    ))
    fig.update_xaxes(title=xtitle, type="category")
    return _layout(fig, title + "  —  red: overconfident, blue: too cautious",
                   height=max(320, 34 * len(order) + 120))


def ladder_chart(lh: pd.DataFrame, title: str = "Ladder: own move at each confidence"
                 ) -> go.Figure:
    fig = go.Figure()
    for side, d in lh.groupby("side", sort=False):
        fig.add_trace(go.Bar(
            x=d["confidence"], y=d["hit_rate"], name=f"{side}side cleared",
            error_y={"type": "data", "symmetric": False, "array": d["ci_hi"] - d["hit_rate"],
                     "arrayminus": d["hit_rate"] - d["ci_lo"]},
            customdata=np.stack([d["mean_move"], d["n"]], axis=1),
            hovertemplate=("conf %{x:.0%}: cleared %{y:.1%}<br>avg ladder move "
                           "%{customdata[0]:+.2%}, n=%{customdata[1]:,}"),
        ))
    c = sorted(lh["confidence"].unique())
    fig.add_trace(go.Scatter(x=c, y=c, mode="lines+markers", line=IDEAL, name="ideal"))
    fig.update_xaxes(title="confidence", tickformat=PCT, tickvals=c)
    fig.update_yaxes(title="actual cleared the move", tickformat=PCT, range=[0, 1])
    return _layout(fig, title, barmode="group")


def threshold_chart(t: pd.DataFrame, title: str) -> go.Figure:
    """Hit rate of calls with P >= c as c sweeps, with the number of calls on a 2nd axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t["confidence"], y=t["ci_hi"], mode="lines", line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["confidence"], y=t["ci_lo"], mode="lines", line={"width": 0},
                             fill="tonexty", fillcolor="rgba(31,119,180,0.15)", name="95% CI",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["confidence"], y=t["hit_rate"], mode="lines", name="came true",
                             line={"color": "#1f77b4"},
                             hovertemplate="P >= %{x:.0%}: came true %{y:.1%}"))
    fig.add_trace(go.Scatter(x=t["confidence"], y=t["mean_prob"], mode="lines",
                             name="model said (mean)", line={"color": "#ff7f0e", "dash": "dot"}))
    fig.add_trace(go.Scatter(x=t["confidence"], y=t["base_rate"], mode="lines",
                             name="base rate (all rows)", line=IDEAL))
    fig.add_trace(go.Bar(x=t["confidence"], y=t["n"], name="calls", opacity=0.25,
                         marker_color="grey"), secondary_y=True)
    fig.update_xaxes(title="call level: P(event) >= c", tickformat=PCT)
    fig.update_yaxes(title="rate", tickformat=PCT, range=[0, 1], secondary_y=False)
    fig.update_yaxes(title="calls", type="log", secondary_y=True, showgrid=False)
    return _layout(fig, title)


def move_chart(m: pd.DataFrame, confidence: float) -> go.Figure:
    fig = go.Figure()
    for side, d in m.groupby("side", sort=False):
        d = d.sort_values("move", key=abs)
        fig.add_trace(go.Scatter(
            x=d["move"].abs(), y=d["hit_rate"], mode="lines+markers", name=f"{side}side",
            error_y={"type": "data", "symmetric": False, "array": d["ci_hi"] - d["hit_rate"],
                     "arrayminus": d["hit_rate"] - d["ci_lo"], "thickness": 1},
            customdata=np.stack([d["event"], d["n"], d["mean_prob"]], axis=1),
            hovertemplate=("%{customdata[0]}: came true %{y:.1%} (said %{customdata[2]:.1%})"
                           "<br>calls %{customdata[1]:,}"),
        ))
    fig.add_hline(y=confidence, line=IDEAL, annotation_text=f"call level {confidence:.0%}")
    fig.update_xaxes(title="move size |m|", tickformat=".1%")
    fig.update_yaxes(title="came true", tickformat=PCT, range=[0, 1])
    return _layout(fig, f"Calls with P >= {confidence:.0%}: hit rate by move size")


def funnel(b: pd.DataFrame, title: str = "Per-stock bias vs sample size") -> go.Figure:
    """bias = 0.5 - mean PIT per stock against n, with the +-2 sigma noise funnel."""
    fig = go.Figure()
    colors = {"calibrated": "#9e9e9e", "optimistic": "#d62728", "pessimistic": "#1f77b4"}
    for label, d in b.groupby("label"):
        fig.add_trace(go.Scatter(
            x=d["n"], y=d["bias"], mode="markers", name=f"{label} ({len(d):,})",
            marker={"color": colors.get(label, "black"), "size": 6, "opacity": 0.7},
            customdata=np.stack([d["symbol"], d["industry_code"], d["bias_z"]], axis=1),
            hovertemplate=("%{customdata[0]} (%{customdata[1]})<br>n=%{x}  bias %{y:+.3f}"
                           "  z=%{customdata[2]:+.1f}<extra></extra>"),
        ))
    if len(b):
        n = np.linspace(max(1, b["n"].min()), b["n"].max(), 100)
        band = 2 * np.sqrt(1 / 12 / n)
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=n, y=sign * band, mode="lines", line=IDEAL,
                                     name="±2σ (noise)", showlegend=sign == 1, hoverinfo="skip"))
    fig.update_xaxes(title="test forecasts for the stock")
    fig.update_yaxes(title="bias (+ optimistic, − pessimistic)", zeroline=True)
    return _layout(fig, title, height=540, dragmode="select")


def bias_bars(b: pd.DataFrame, key: str, title: str) -> go.Figure:
    b = b.sort_values("bias")
    err = 2 * np.sqrt(1 / 12 / b["n"])
    fig = go.Figure(go.Bar(
        x=b["bias"], y=b[key].astype(str), orientation="h",
        error_x={"type": "data", "array": err}, customdata=b["n"],
        marker_color=np.where(b["bias"] > 0, "#d62728", "#1f77b4"),
        hovertemplate="%{y}: bias %{x:+.3f} (n=%{customdata:,})<extra></extra>",
    ))
    fig.update_xaxes(title="bias (+ optimistic, − pessimistic); bars ±2σ")
    return _layout(fig, title, height=max(320, 18 * len(b) + 120))


def daily_chart(d: pd.DataFrame, event: str | None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.06)
    band = 2 * np.sqrt(1 / 12 / d["n"])
    fig.add_trace(go.Scatter(x=d["day"], y=band, mode="lines", line={"width": 0},
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["day"], y=-band, mode="lines", line={"width": 0},
                             fill="tonexty", fillcolor="rgba(128,128,128,0.2)",
                             name="±2σ (noise)", hoverinfo="skip"), row=1, col=1)
    out = d["bias_z"].abs() >= 2
    fig.add_trace(go.Scatter(x=d["day"], y=d["bias"], mode="lines", name="PIT bias",
                             line={"color": "#444"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.loc[out, "day"], y=d.loc[out, "bias"], mode="markers",
                             name="regime day (|z| >= 2)",
                             marker={"color": np.where(d.loc[out, "bias"] > 0, "#d62728",
                                                       "#1f77b4"), "size": 7}), row=1, col=1)
    if event and "deviation" in d.columns:
        fig.add_trace(go.Scatter(x=d["day"], y=d["deviation"], mode="lines",
                                 name=f"{event}: came true − said", line={"color": "#2ca02c"}),
                      row=1, col=1)
    fig.add_trace(go.Bar(x=d["day"], y=d["n"], name="forecasts", marker_color="grey",
                         opacity=0.5), row=2, col=1)
    fig.update_yaxes(title="bias / deviation", tickformat="+.0%", row=1, col=1)
    fig.update_yaxes(title="n", row=2, col=1)
    return _layout(fig, "Across stocks, per day (+ optimistic, − pessimistic)", height=560)


def fan(d: pd.DataFrame, r_q: np.ndarray, r_act: np.ndarray, levels, symbol: str,
        ref: np.ndarray | None = None) -> go.Figure:
    """Forecast bands over the days, in % move, with the actual move as dots. With `ref` (each
    row's reference price) the bands and dots are VWAP prices instead."""
    order = np.argsort(d["date"].to_numpy(), kind="stable")
    x = d["date"].to_numpy()[order]
    q, a = r_q[order], r_act[order]
    if ref is not None:
        r = np.asarray(ref, dtype=float)[order]
        q, a = r[:, None] * (1 + q), r * (1 + a)
    lv = np.asarray(levels)
    fig = go.Figure()
    pairs = [(lv[0], lv[-1], "rgba(31,119,180,0.15)"), (0.25, 0.75, "rgba(31,119,180,0.3)")]
    for lo, hi, color in pairs:
        i, j = int(np.argmin(np.abs(lv - lo))), int(np.argmin(np.abs(lv - hi)))
        fig.add_trace(go.Scatter(x=x, y=q[:, j], mode="lines", line={"width": 0},
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=q[:, i], mode="lines", line={"width": 0}, fill="tonexty",
                                 fillcolor=color, name=f"P{lv[i] * 100:.0f}–P{lv[j] * 100:.0f}",
                                 hoverinfo="skip"))
    m = int(np.argmin(np.abs(lv - 0.5)))
    fig.add_trace(go.Scatter(x=x, y=q[:, m], mode="lines", name="P50", line={"color": "#1f77b4"}))
    inside = (q[:, 0] <= a) & (a <= q[:, -1])
    fig.add_trace(go.Scatter(x=x, y=a, mode="markers", name="actual",
                             marker={"color": np.where(inside, "black", "#d62728"), "size": 6},
                             hovertemplate="%{x}: actual %{y:" + (
                                 "$,.2f" if ref is not None else "+.2%") + "}<extra></extra>"))
    if ref is not None:
        fig.update_yaxes(title="VWAP", tickprefix="$")
    else:
        fig.update_yaxes(title="next-day move", tickformat="+.1%")
    return _layout(fig, f"{symbol}: forecast range vs actual (red = outside the outer band)")


def pit_hist(u: np.ndarray, bins: int = 10, title: str = "PIT histogram (flat = calibrated)"
             ) -> go.Figure:
    counts, edges = np.histogram(u, bins=bins, range=(0, 1))
    share = counts / max(1, counts.sum())
    fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=share, width=1 / bins * 0.95,
                           hovertemplate="u≈%{x:.2f}: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=1 / bins, line=IDEAL, annotation_text="calibrated")
    fig.update_xaxes(title="u = forecast CDF at the actual (left: actual below forecast)",
                     range=[0, 1])
    fig.update_yaxes(title="share", tickformat=PCT)
    return _layout(fig, title, height=380)


def spread_chart(t: pd.DataFrame, symbol: str) -> go.Figure:
    """Forecast, one stock: model probability per move next to its historical hit rate in the
    same bucket (stock and all stocks)."""
    fig = go.Figure()
    for side in t["side"].unique():
        d = t[t["side"] == side].sort_values("move", key=abs)
        x = d["move"].abs()
        dash = None if side == SIDE_UP else "dot"
        fig.add_trace(go.Scatter(x=x, y=d["prob"], mode="lines+markers",
                                 name=f"{side}: model says", line={"dash": dash, "width": 3}))
        fig.add_trace(go.Scatter(x=x, y=d["hist_hit_rate"], mode="markers",
                                 name=f"{side}: {symbol} came true (same bucket)",
                                 marker={"symbol": "diamond", "size": 9},
                                 customdata=d["hist_n"],
                                 hovertemplate="%{y:.0%} (n=%{customdata})<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=d["all_hit_rate"], mode="markers",
                                 name=f"{side}: all stocks came true",
                                 marker={"symbol": "x", "size": 8}))
    fig.update_xaxes(title="move size |m| (up: >= +m, down: <= −m)", tickformat=".1%")
    fig.update_yaxes(title="probability", tickformat=PCT, range=[0, 1])
    return _layout(fig, f"{symbol}: forecast probability per move vs history", height=500)


def call_heatmap(calls: pd.DataFrame, min_n: int = 1) -> go.Figure:
    """insights.call_table as event x call level: came true / said per cell."""
    c = calls.assign(call=[f"P ≥ {v:.0%}" for v in calls["confidence"]])
    return heatmap(c, "Calls: came true / said", min_n=min_n, x="call", xtitle="call level")


def call_chart(roll: pd.DataFrame, title: str = "Calls by side: came true vs said") -> go.Figure:
    """insights.side_rollup (optionally with a `run` column): one solid came-true line and one
    dotted said line per side (and run), plus the base rate."""
    fig = go.Figure()
    keys = ["run", "side"] if "run" in roll.columns else ["side"]
    for i, (k, d) in enumerate(roll.groupby(keys, sort=False)):
        name = " · ".join(f"{x}side" if x in ("up", "down") else str(x)
                          for x in (k if isinstance(k, tuple) else (k,)))
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=d["confidence"], y=d["hit_rate"], mode="lines+markers", name=f"{name}: came true",
            line={"color": color}, customdata=d["n"],
            hovertemplate="P ≥ %{x:.0%}: came true %{y:.1%} (n=%{customdata:,})<extra></extra>"))
        fig.add_trace(go.Scatter(x=d["confidence"], y=d["mean_prob"], mode="lines",
                                 name=f"{name}: said", line={"color": color, "dash": "dot"}))
        fig.add_trace(go.Scatter(x=d["confidence"], y=d["base_rate"], mode="lines",
                                 name=f"{name}: base rate",
                                 line={"color": color, "dash": "dash", "width": 1}))
    fig.update_xaxes(title="call level: P(event) >= c", tickformat=PCT)
    fig.update_yaxes(title="rate", tickformat=PCT, range=[0, 1])
    return _layout(fig, title, height=440)


def quantile_chart(qt: pd.DataFrame, title: str = "Quantile calibration: share of actual "
                                                  "moves below each forecast quantile"
                   ) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=IDEAL, name="ideal",
                               hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=qt["level"], y=qt["observed"], mode="lines+markers", name="observed",
        customdata=qt["gap"],
        hovertemplate="P%{x:.0%}: %{y:.1%} below (gap %{customdata:+.1%})<extra></extra>"))
    fig.update_xaxes(title="forecast quantile", tickformat=PCT, range=[0, 1])
    fig.update_yaxes(title="actual moves below it", tickformat=PCT, range=[0, 1])
    return _layout(fig, title, height=420)


def price_range(levels, prices: np.ndarray, ref: float, symbol: str, lo: float, hi: float,
                actual: float | None = None) -> go.Figure:
    """One stock's forecast VWAP at every percentile, the chosen lo..hi band shaded, the
    reference VWAP (and the actual, once known) as lines."""
    pct = np.asarray(levels, dtype=float) * 100
    fig = go.Figure(go.Scatter(x=pct, y=prices, mode="lines+markers", name="forecast VWAP",
                               hovertemplate="P%{x:.0f}: $%{y:,.2f}<extra></extra>"))
    inside = (pct >= lo * 100 - 1e-9) & (pct <= hi * 100 + 1e-9)
    fig.add_trace(go.Scatter(x=pct[inside], y=prices[inside], mode="markers",
                             name=f"P{lo * 100:.0f}–P{hi * 100:.0f} band",
                             marker={"size": 11, "color": "#2ca02c"}, hoverinfo="skip"))
    fig.add_hrect(y0=float(np.interp(lo * 100, pct, prices)),
                  y1=float(np.interp(hi * 100, pct, prices)),
                  fillcolor="rgba(44,160,44,0.12)", line_width=0)
    fig.add_hline(y=ref, line=IDEAL, annotation_text=f"reference VWAP ${ref:,.2f}")
    if actual is not None and np.isfinite(actual):
        fig.add_hline(y=actual, line={"color": "black", "width": 2},
                      annotation_text=f"actual ${actual:,.2f}", annotation_position="bottom right")
    fig.update_xaxes(title="percentile (P(VWAP ≤ price))", ticksuffix="")
    fig.update_yaxes(title="VWAP", tickprefix="$")
    return _layout(fig, f"{symbol}: forecast VWAP by percentile", height=420)


def live_scatter(med: np.ndarray, act: np.ndarray, inside: np.ndarray, symbols,
                 title: str) -> go.Figure:
    """Graded day: forecast median move vs the actual move per stock (red = outside band)."""
    lim = float(np.nanquantile(np.abs(np.r_[med, act]), 0.99)) if len(act) else 0.05
    fig = go.Figure(go.Scattergl(
        x=med, y=act, mode="markers", text=np.asarray(symbols),
        marker={"size": 5, "opacity": 0.6, "color": np.where(inside, "#1f77b4", "#d62728")},
        hovertemplate="%{text}: median %{x:+.2%}, actual %{y:+.2%}<extra></extra>"))
    fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line=IDEAL)
    fig.add_hline(y=0, line={"color": "lightgrey", "width": 1})
    fig.add_vline(x=0, line={"color": "lightgrey", "width": 1})
    fig.update_xaxes(title="forecast median move", tickformat="+.1%", range=[-lim, lim])
    fig.update_yaxes(title="actual move", tickformat="+.1%", range=[-lim, lim])
    return _layout(fig, title, height=440)


def live_days(by_day: pd.DataFrame) -> go.Figure:
    """Graded days: band coverage against its nominal, and the PIT bias."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=by_day["day"], y=by_day["coverage"], mode="lines+markers",
                             name="band held"), row=1, col=1)
    fig.add_trace(go.Scatter(x=by_day["day"], y=by_day["coverage_nominal"], mode="lines",
                             name="target", line=IDEAL), row=1, col=1)
    fig.add_trace(go.Bar(x=by_day["day"], y=by_day["bias"], name="bias (+ optimistic)",
                         marker_color=np.where(by_day["bias"] > 0, "#d62728", "#2ca02c")),
                  row=2, col=1)
    fig.update_yaxes(title="share inside band", tickformat=PCT, row=1, col=1)
    fig.update_yaxes(title="bias", tickformat="+.0%", row=2, col=1)
    return _layout(fig, "Graded days: band coverage and bias", height=480)


# ---- intraday --------------------------------------------------------------------------------

def _id_skills(t: pd.DataFrame, prefix: str = "") -> list[tuple[str, str, str]]:
    """(column, name, colour) of the skill series a table has: climatology (runs from before
    it have none), momentum, random walk."""
    out = [("skill_clim", "vs climatology", COLORS[0]), ("skill_mom", "vs momentum", COLORS[3]),
           ("skill_rw", "vs random walk", COLORS[4])]
    return [(c, prefix + n, k) for c, n, k in out if c in t and t[c].notna().any()]


def id_skill_bars(s: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """Per target: quantile skill vs each baseline (> 0 wins)."""
    x = [labels.get(t, t) for t in s["target"]]
    fig = go.Figure()
    for col, name, color in _id_skills(s):
        fig.add_bar(x=x, y=s[col], name=name, marker_color=color,
                    hovertemplate="%{x}: %{y:+.1%}<extra>" + name + "</extra>")
    fig.add_hline(y=0, line=IDEAL)
    fig.update_yaxes(title="pinball skill", tickformat=PCT)
    fig.update_layout(barmode="group")
    return _layout(fig, "Quantile skill by horizon (above 0 = beats the baseline)", height=380)


def id_days(d: pd.DataFrame) -> go.Figure:
    """Per test day: skill vs each baseline."""
    fig = go.Figure()
    for col, name, color in _id_skills(d):
        fig.add_scatter(x=d["date"], y=d[col], mode="lines+markers", name=name,
                        line={"color": color})
    fig.add_hline(y=0, line=IDEAL)
    fig.update_yaxes(title="pinball skill", tickformat=PCT)
    return _layout(fig, "Skill by test day", height=360)


def id_tod(t: pd.DataFrame) -> go.Figure:
    """Four small panels over the trading day: coverage, band width, skill, direction."""
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=("P10–P90 coverage", "Median P10–P90 width",
                                        "Skill", "Direction hit rate"))
    x = t["time"]
    fig.add_scatter(x=x, y=t["coverage80"], name="coverage", mode="lines+markers",
                    line={"color": COLORS[0]}, row=1, col=1)
    fig.add_hline(y=0.8, line=IDEAL, row=1, col=1)
    fig.add_scatter(x=x, y=t["width80_med"], name="width", mode="lines+markers",
                    line={"color": COLORS[3]}, row=1, col=2)
    for col, name, color in _id_skills(t, "skill "):
        fig.add_scatter(x=x, y=t[col], name=name, mode="lines+markers", line={"color": color},
                        row=2, col=1)
    fig.add_hline(y=0, line=IDEAL, row=2, col=1)
    for col, name, color in (("dir_trail", "vs trailing VWAP", COLORS[2]),
                             ("dir_session", "vs session VWAP", COLORS[5]),
                             ("dir_last", "vs last price", COLORS[1])):
        fig.add_scatter(x=x, y=t[col], name=name, mode="lines+markers", line={"color": color},
                        row=2, col=2)
    fig.add_hline(y=0.5, line=IDEAL, row=2, col=2)
    for r, c in ((1, 1), (2, 1), (2, 2)):
        fig.update_yaxes(tickformat=PCT, row=r, col=c)
    fig.update_yaxes(tickformat=".2%", row=1, col=2)
    return _layout(fig, "By time of day", height=560)


def id_replay(r: pd.DataFrame, title: str) -> go.Figure:
    """One session: last price, session VWAP, the VWAP that followed, and the predicted
    P10–P90 band with its median at each decision time (prices in $)."""
    fig = go.Figure()
    if "daily_p10" in r and r["daily_p10"].notna().any():
        d = r[r["daily_p10"].notna()]
        fig.add_scatter(x=d["time"], y=d["daily_p90"], mode="lines", line={"width": 0},
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=d["time"], y=d["daily_p10"], mode="lines", line={"width": 0},
                        fill="tonexty", fillcolor="rgba(148,103,189,0.15)",
                        name="morning daily P10–P90")
    fig.add_scatter(x=r["time"], y=r["p90"], mode="lines", line={"width": 0},
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=r["time"], y=r["p10"], mode="lines", line={"width": 0}, fill="tonexty",
                    fillcolor="rgba(31,119,180,0.2)", name="predicted P10–P90")
    fig.add_scatter(x=r["time"], y=r["p50"], mode="lines", name="predicted median",
                    line={"color": COLORS[0], "dash": "dot"})
    fig.add_scatter(x=r["time"], y=r["actual_vwap"], mode="lines", name="VWAP that followed",
                    line={"color": COLORS[1], "width": 2})
    fig.add_scatter(x=r["time"], y=r["session_vwap"], mode="lines", name="session VWAP so far",
                    line={"color": COLORS[2]})
    fig.add_scatter(x=r["time"], y=r["last"], mode="lines", name="last price",
                    line={"color": "grey", "width": 1})
    fig.update_xaxes(title="decision time (ET)")
    fig.update_yaxes(title="price ($)", tickprefix="$")
    return _layout(fig, title, height=520)


def id_stocks(t: pd.DataFrame) -> go.Figure:
    """Per stock: coverage (x) vs skill (y: vs climatology, else momentum), bubble size ~ sqrt(n)."""
    col, name, _ = _id_skills(t)[0]
    fig = go.Figure(go.Scatter(
        x=t["coverage80"], y=t[col], mode="markers+text", text=t["symbol"],
        textposition="top center", marker={"size": _sizes(t["n"]), "color": COLORS[0],
                                           "opacity": 0.7},
        customdata=t["n"], hovertemplate="%{text}: coverage %{x:.0%}, skill %{y:+.1%}"
                                         "<br>n=%{customdata:,}<extra></extra>"))
    fig.add_vline(x=0.8, line=IDEAL)
    fig.add_hline(y=0, line=IDEAL)
    fig.update_xaxes(title="P10–P90 coverage (target 80%)", tickformat=PCT)
    fig.update_yaxes(title="skill " + name, tickformat=PCT)
    return _layout(fig, "Per stock: band coverage vs skill", height=460)


def id_vs_daily(vd: pd.DataFrame) -> go.Figure:
    """Whole-day VWAP pinball by time of day, intraday vs the morning daily forecast."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("Pinball loss (lower is better)",
                                        "Intraday skill vs the daily forecast"))
    for col, name, color in (("intraday_pinball", "intraday", COLORS[0]),
                             ("daily_pinball", "morning daily", COLORS[3])):
        fig.add_scatter(x=vd["time"], y=vd[col], mode="lines+markers", name=name,
                        line={"color": color}, row=1, col=1)
    fig.add_bar(x=vd["time"], y=vd["skill"], name="skill", marker_color=COLORS[2],
                row=2, col=1)
    fig.add_hline(y=0, line=IDEAL, row=2, col=1)
    fig.update_yaxes(tickformat=".2%", row=1, col=1)
    fig.update_yaxes(tickformat=PCT, row=2, col=1)
    return _layout(fig, "Intraday vs daily: whole-day VWAP", height=540)


# ---- Intraday → Space: pockets of later volume (maps are NX x NY: time ahead x σ vs now) ------

MIN_TICKS = [(1, "1m"), (5, "5m"), (15, "15m"), (30, "30m"), (60, "1h"), (120, "2h"),
             (240, "4h"), (390, "6.5h")]


def span_label(m: float) -> str:
    """Minutes ahead as a short label: 0.4m, 25m, 2.5h."""
    return f"{m:.1f}m" if m < 10 else f"{m:.0f}m" if m < 90 else f"{m / 60:.1f}h"


def _cell_text(Q: np.ndarray, xe: np.ndarray, ye: np.ndarray, now: float, sig: float,
               region: np.ndarray | None, names) -> np.ndarray:
    """Hover text per cell, shaped like the heatmap (NY x NX)."""
    out = np.empty(Q.T.shape, dtype=object)
    px = now * np.exp(ye * sig)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            r = f"<br>{names[region[i, j]]}" if region is not None else ""
            out[j, i] = (f"{span_label(xe[i])}–{span_label(xe[i + 1])} ahead<br>"
                         f"{ye[j]:+.2f}σ to {ye[j + 1]:+.2f}σ (${px[j]:,.2f}–${px[j + 1]:,.2f})"
                         f"<br>{Q[i, j]:.2%} of the later volume{r}")
    return out


def _map_axes(fig: go.Figure, xe: np.ndarray, n: int, cols: int):
    pos = [float(np.interp(m, xe, np.arange(len(xe)))) for m, _ in MIN_TICKS]
    for k in range(n):
        r, c = k // cols + 1, k % cols + 1
        fig.update_xaxes(tickvals=pos, ticktext=[t for _, t in MIN_TICKS], range=[0, len(xe) - 1],
                         row=r, col=c, title="time ahead" if r == (n - 1) // cols + 1 else None)
        fig.update_yaxes(title="price vs now (σ)" if c == 1 else None, row=r, col=c)
        fig.add_hline(y=0, line={"color": "white", "width": 1, "dash": "dot"}, row=r, col=c)


def pocket_maps(maps: dict[str, np.ndarray], regions: dict[str, np.ndarray | None],
                xe: np.ndarray, ye: np.ndarray, now: float, sig: float, region_names,
                title: str, cols: int = 2) -> go.Figure:
    """One heatmap per map (each scaled to its own peak), with the 50 / 80 / 95% pocket edges
    as white lines. The dotted line is the price now."""
    names = list(maps)
    rows = (len(names) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, horizontal_spacing=0.06,
                        vertical_spacing=0.12, shared_yaxes=True)
    xc = np.arange(len(xe) - 1) + 0.5
    yc = (ye[:-1] + ye[1:]) / 2
    for k, name in enumerate(names):
        Q, reg = maps[name], regions.get(name)
        r, c = k // cols + 1, k % cols + 1
        peak = Q.max() if Q.max() > 0 else 1.0
        fig.add_trace(go.Heatmap(
            x=xc, y=yc, z=(Q / peak).T, colorscale="Viridis", showscale=False, zmin=0, zmax=1,
            text=_cell_text(Q, xe, ye, now, sig, reg, region_names),
            hovertemplate="%{text}<extra>" + name + "</extra>"), row=r, col=c)
        if reg is not None:
            fig.add_trace(go.Contour(
                x=xc, y=yc, z=reg.T.astype(float), showscale=False, hoverinfo="skip",
                contours={"start": 0.5, "end": 2.5, "size": 1, "coloring": "lines"},
                line={"color": "white", "width": 1.2}, colorscale=[[0, "white"], [1, "white"]]),
                row=r, col=c)
    _map_axes(fig, xe, len(names), cols)
    return _layout(fig, title, height=300 * rows + 80)


def pocket_diff(Qm: np.ndarray, Qc: np.ndarray, xe: np.ndarray, ye: np.ndarray, now: float,
                sig: float, title: str) -> go.Figure:
    """Match − everything, per cell, in points of the later volume: red = more volume there
    after the matches than usual, blue = less."""
    D = Qm - Qc
    lim = float(np.abs(D).max()) or 1e-6
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Heatmap(
        x=np.arange(len(xe) - 1) + 0.5, y=(ye[:-1] + ye[1:]) / 2, z=D.T * 100,
        colorscale="RdBu_r", zmid=0, zmin=-lim * 100, zmax=lim * 100,
        colorbar={"title": "pts"}, text=_cell_text(D, xe, ye, now, sig, None, None),
        hovertemplate="%{text}<extra>difference</extra>"))
    _map_axes(fig, xe, 1, 1)
    return _layout(fig, title, height=420)


def dial_curve(t: pd.DataFrame, dials: tuple[float, float], clim_up: float,
               clim_above: float) -> go.Figure:
    """Against the match tolerance: matches and N_eff days (log scale), then P(up) and the share
    of later volume above now, with everything (climatology) dashed."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Matches and independent days",
                                        "Readings (blended to everything)"))
    for col, name, color in (("matches", "matches", COLORS[0]), ("days", "N_eff days", COLORS[3]),
                             ("stocks", "stocks", COLORS[5])):
        fig.add_scatter(x=t["dial"], y=t[col].clip(lower=0.5), name=name, mode="lines+markers",
                        line={"color": color}, row=1, col=1)
    for col, name, color, ref in (("p_up", "P(rest of day above now)", COLORS[2], clim_up),
                                  ("above_now", "later volume above now", COLORS[1], clim_above)):
        fig.add_scatter(x=t["dial"], y=t[col], name=name, mode="lines+markers",
                        line={"color": color}, row=2, col=1)
        fig.add_hline(y=ref, line={**IDEAL, "color": color}, row=2, col=1)
    for d, label in zip(dials, ("close", "loose")):
        fig.add_vline(x=d, line={"color": "grey", "width": 1, "dash": "dot"})
        # labelled at the foot of the top panel: at the top they would cover the panel titles
        fig.add_annotation(x=d, xref="x", y=0, yref="y domain", text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font={"color": "grey", "size": 11})
    fig.update_xaxes(title="tolerance (× each parameter's spread)", row=2, col=1)
    fig.update_yaxes(type="log", title="count", row=1, col=1)
    fig.update_yaxes(tickformat=PCT, row=2, col=1)
    return _layout(fig, "What the tolerance does", height=560)


def space_day_path(t: pd.DataFrame, decision: int, prev: float, title: str) -> go.Figure:
    """The stock's day: each minute's VWAP and the session VWAP so far, volume below, the
    decision time marked and the minutes after it dotted."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.05)
    before, after = t[~t["after"]], t[t["after"]]
    fig.add_scatter(x=before["minute"], y=before["vwap"], mode="lines", name="minute VWAP",
                    text=before["time"], hovertemplate="%{text}: $%{y:,.2f}",
                    line={"color": COLORS[0], "width": 1.5}, row=1, col=1)
    fig.add_scatter(x=after["minute"], y=after["vwap"], mode="lines", name="after the decision",
                    text=after["time"], hovertemplate="%{text}: $%{y:,.2f}",
                    line={"color": COLORS[0], "width": 1, "dash": "dot"}, opacity=0.6,
                    row=1, col=1)
    fig.add_scatter(x=t["minute"], y=t["session_vwap"], mode="lines", name="session VWAP so far",
                    hoverinfo="skip", line={"color": COLORS[2]}, row=1, col=1)
    fig.add_hline(y=prev, line=IDEAL, row=1, col=1)
    fig.add_bar(x=t["minute"], y=t["volume"], name="volume", row=2, col=1, text=t["time"],
                hovertemplate="%{text}: %{y:,.0f} shares",
                marker_color=np.where(t["after"], "rgba(127,127,127,0.35)", COLORS[0]))
    fig.add_vline(x=decision, line={"color": COLORS[1], "width": 1.5})
    ticks = t["minute"][t["minute"] % 30 == 0]
    fig.update_xaxes(tickvals=ticks, ticktext=t["time"][ticks.index], row=2, col=1)
    fig.update_yaxes(title="price ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title="shares", row=2, col=1)
    return _layout(fig, title + " (grey dashed: previous VWAP)", height=520)


# ---- intraday pockets (intraday-ft runs) -------------------------------------------------------

def pk_go(t: pd.DataFrame) -> go.Figure:
    """Pinball skill vs climatology per shape target and bin, raw (bars, 95% CI) and after the
    rolling recalibration (diamonds)."""
    fig = go.Figure()
    for i, (fam, d) in enumerate(t.groupby("family", sort=False)):
        x = [f"b{b}" for b in d["bin"]]
        fig.add_bar(x=x, y=d["skill_clim"], name=fam, marker_color=COLORS[i % len(COLORS)],
                    offsetgroup=fam, legendgroup=fam,
                    error_y={"type": "data", "symmetric": False,
                             "array": d["skill_hi"] - d["skill_clim"],
                             "arrayminus": d["skill_clim"] - d["skill_lo"], "thickness": 1},
                    customdata=np.stack([d["rank_ic"], d["clim_rank_ic"]], axis=1),
                    hovertemplate=f"{fam} %{{x}}: skill %{{y:+.1%}}<br>rank IC %{{customdata[0]:+.3f}} "
                                  "(climatology %{customdata[1]:+.3f})<extra></extra>")
        if "cal_skill_clim" in d and d["cal_skill_clim"].notna().any():
            fig.add_scatter(x=x, y=d["cal_skill_clim"], mode="markers", name=f"{fam} recalibrated",
                            offsetgroup=fam, legendgroup=fam, showlegend=False,
                            marker={"symbol": "diamond", "size": 9, "color": "black"},
                            hovertemplate=f"{fam} %{{x}} recalibrated: %{{y:+.1%}}<extra></extra>")
    fig.add_hline(y=0, line=IDEAL)
    fig.update_layout(barmode="group", scattermode="group")
    fig.update_xaxes(title="duration bin (b1 0-5, b2 5-15, b3 15-30, b4 30-60 min, b5 60 min-close)")
    fig.update_yaxes(title="pinball skill vs climatology", tickformat="+.0%")
    return _layout(fig, "Pocket shape: skill vs climatology (◆ = after the recalibration)", height=440)


def pk_hits(raw: pd.DataFrame, cal: pd.DataFrame | None, title: str) -> go.Figure:
    """Share of outcomes at or below each quantile, raw and recalibrated, against the diagonal."""
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=IDEAL, name="ideal",
                               hoverinfo="skip"))
    for t, name, color in ((raw, "raw", COLORS[0]), (cal, "recalibrated", COLORS[2])):
        if t is None:
            continue
        fig.add_scatter(x=t["level"], y=t["observed"], mode="lines+markers", name=name,
                        line={"color": color}, customdata=t["gap"],
                        hovertemplate="P%{x:.0%}: %{y:.1%} at or below (gap %{customdata:+.1%})"
                                      "<extra></extra>")
    fig.update_xaxes(title="forecast quantile", tickformat=PCT, range=[0, 1])
    fig.update_yaxes(title="outcomes at or below it", tickformat=PCT, range=[0, 1])
    return _layout(fig, title, height=420)


def pk_groups(t: pd.DataFrame, key: str, min_n: int) -> go.Figure:
    """Per group: skill vs climatology with its 95% CI; groups below min_n greyed."""
    t = t.sort_values("skill_clim")
    small = t["n"] < min_n
    fig = go.Figure(go.Bar(
        y=t[key], x=t["skill_clim"], orientation="h",
        marker_color=np.where(small, "lightgrey", COLORS[0]),
        error_x={"type": "data", "symmetric": False, "array": t["skill_hi"] - t["skill_clim"],
                 "arrayminus": t["skill_clim"] - t["skill_lo"], "thickness": 1},
        customdata=np.stack([t["n"], t["p50_hit"], t["cover80"]], axis=1),
        hovertemplate="%{y}: skill %{x:+.1%}<br>n=%{customdata[0]:,}  P50 hit %{customdata[1]:.0%}"
                      "  coverage %{customdata[2]:.0%}<extra></extra>"))
    fig.add_vline(x=0, line=IDEAL)
    fig.update_xaxes(title="pinball skill vs climatology", tickformat="+.0%")
    fig.update_yaxes(type="category")
    return _layout(fig, f"Skill by {key.replace('_', ' ')} (grey: n below {min_n:,})",
                   height=max(320, 26 * len(t) + 120))


def pk_timeline(r: pd.DataFrame, variants: dict[str, str], hit: pd.Series, title: str) -> go.Figure:
    """One stock-day: the event's probability at each decision under each overlay, and whether
    it came true (markers at 0 / 1)."""
    fig = go.Figure()
    for i, (v, name) in enumerate(variants.items()):
        fig.add_scatter(x=r["time"], y=r[v], mode="lines+markers", name=name,
                        line={"color": COLORS[i % len(COLORS)]}, marker={"size": 4},
                        hovertemplate=f"%{{x}} {name}: %{{y:.1%}}<extra></extra>")
    fig.add_scatter(x=r["time"], y=hit, mode="markers", name="came true (1) / not (0)",
                    marker={"symbol": "x", "size": 7, "color": "black"},
                    hovertemplate="%{x}: %{y:.0f}<extra></extra>")
    fig.update_xaxes(title="decision time (ET)")
    fig.update_yaxes(title="probability", tickformat=PCT, range=[-0.05, 1.05])
    return _layout(fig, title, height=420)


def pk_replay(r: pd.DataFrame, title: str) -> go.Figure:
    """One stock-day: the entry price and, per decision, the P10-P90 band of where the volume
    above / below it concentrates in the chosen bin (cm_up / cm_dn in $), with what happened."""
    fig = go.Figure()
    for side, color, fill in (("up", COLORS[2], "rgba(44,160,44,0.18)"),
                              ("dn", COLORS[1], "rgba(214,39,40,0.15)")):
        name = "above the entry" if side == "up" else "below the entry"
        fig.add_scatter(x=r["time"], y=r[f"{side}_p90"], mode="lines", line={"width": 0},
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=r["time"], y=r[f"{side}_p10"], mode="lines", line={"width": 0},
                        fill="tonexty", fillcolor=fill, name=f"{name}: P10–P90")
        fig.add_scatter(x=r["time"], y=r[f"{side}_p50"], mode="lines", name=f"{name}: P50",
                        line={"color": color, "dash": "dot"})
        fig.add_scatter(x=r["time"], y=r[f"{side}_real"], mode="markers", name=f"{name}: happened",
                        marker={"color": color, "size": 5})
    fig.add_scatter(x=r["time"], y=r["p0"], mode="lines", name="entry price P0",
                    line={"color": "grey", "width": 1.5})
    fig.update_xaxes(title="decision time (ET)")
    fig.update_yaxes(title="price ($)", tickprefix="$")
    return _layout(fig, title, height=520)


def pk_fan(t: pd.DataFrame, p0: float, title: str) -> go.Figure:
    """One decision across the bins: the P10-P90 and P25-P75 ranges of the centre of mass above
    and below the entry (in $), the P50, and what happened."""
    fig = go.Figure()
    x = t["window"]
    for side, color in (("up", COLORS[2]), ("dn", COLORS[1])):
        name = "above" if side == "up" else "below"
        for lo, hi, w in (("p10", "p90", 6), ("p25", "p75", 14)):
            fig.add_scatter(x=x, y=t[f"{side}_{hi}_usd"], mode="markers", marker={"size": 0.1},
                            showlegend=False, hoverinfo="skip",
                            error_y={"type": "data", "symmetric": False, "array": [0] * len(t),
                                     "arrayminus": t[f"{side}_{hi}_usd"] - t[f"{side}_{lo}_usd"],
                                     "thickness": w, "width": 0, "color": color})
        fig.add_scatter(x=x, y=t[f"{side}_p50_usd"], mode="markers", name=f"{name}: P50 (bars P10–P90, "
                                                                            "P25–P75)",
                        marker={"color": "white", "size": 8, "line": {"color": color, "width": 2}},
                        hovertemplate=f"%{{x}} {name}: P50 $%{{y:,.2f}}<extra></extra>")
        fig.add_scatter(x=x, y=t[f"{side}_real_usd"], mode="markers", name=f"{name}: happened",
                        marker={"color": "black", "symbol": "x", "size": 8},
                        hovertemplate=f"%{{x}} {name} happened: $%{{y:,.2f}}<extra></extra>")
    fig.add_hline(y=p0, line=IDEAL, annotation_text=f"entry P0 ${p0:,.2f}")
    fig.update_xaxes(title="minutes after the entry", type="category")
    fig.update_yaxes(title="centre of mass of the volume ($)", tickprefix="$")
    return _layout(fig, title, height=480)
