"""Plotly figures from the tables in calib.py. Pure functions: tables in, Figure out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .calib import SIDE_UP

PCT = ".0%"
IDEAL = {"dash": "dash", "color": "grey", "width": 1}


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


def heatmap(g: pd.DataFrame, title: str = "Deviation (came true - said)", min_n: int = 1
            ) -> go.Figure:
    g = g[g["n"] >= min_n]
    order = g.sort_values("move", key=lambda m: m.where(m < 0, 1 + m))["event"].unique()
    buckets = sorted(g["bucket"].unique(), key=lambda b: float(b.split("%")[0]))
    piv = lambda col: g.pivot(index="event", columns="bucket", values=col).reindex(
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
    fig.update_xaxes(title="model probability bucket")
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


def fan(d: pd.DataFrame, r_q: np.ndarray, r_act: np.ndarray, levels, symbol: str) -> go.Figure:
    """Forecast bands over the test days, in % move, with the actual move as dots."""
    order = np.argsort(d["date"].to_numpy(), kind="stable")
    x = d["date"].to_numpy()[order]
    q, a = r_q[order], r_act[order]
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
                             hovertemplate="%{x}: actual %{y:+.2%}<extra></extra>"))
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
    """Tomorrow, one stock: model probability per move next to its historical hit rate in the
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
    return _layout(fig, f"{symbol}: tomorrow's probability per move vs history", height=500)
