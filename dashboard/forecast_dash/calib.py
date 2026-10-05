"""Calibration maths for quantile forecasts of a next-period move. Pure numpy/pandas.

Inputs are arrays: `r_q` (rows x levels) forecast quantiles of the fractional move, sorted
per row; `levels` the quantile levels (e.g. 0.05..0.95); `r_act` the realised move.

- event "move >= m" (m > 0, upside) has forecast prob 1 - F(m); "move <= m" (m < 0) has F(m).
  F is linear between the quantiles and clamped to [levels[0], levels[-1]] outside them.
- grid cell = (move, forecast-probability bucket): n, mean_prob (said), hit_rate (happened),
  deviation = hit_rate - mean_prob (< 0 overconfident), z = deviation / binomial s.e.
- ladder: the move at confidence c, upside q(1-c) and downside q(c).
- PIT u = F(actual); bias = 0.5 - mean(u): > 0 optimistic (actuals below the forecast),
  < 0 pessimistic. A calibrated forecast has u ~ U(0, 1), so bias s.e. = sqrt(1/12/n).
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

import numpy as np
import pandas as pd

SIDE_UP, SIDE_DOWN = "up", "down"
Events = Mapping[float, tuple[np.ndarray, np.ndarray]]  # move -> (prob, hit)


def side_of(move: float) -> str:
    return SIDE_UP if move > 0 else SIDE_DOWN


def event_label(move: float) -> str:
    return f"{'>=' if move > 0 else '<='} {move:+.1%}"


def prob_below(r_q: np.ndarray, levels, x) -> np.ndarray:
    """Per-row forecast CDF F(x); `x` is a scalar or one value per row."""
    lv = np.asarray(levels, dtype=float)
    x = np.broadcast_to(np.asarray(x, dtype=float), (len(r_q),))
    k = (r_q < x[:, None]).sum(axis=1)  # quantiles strictly below x
    lo, hi = np.clip(k - 1, 0, len(lv) - 1), np.clip(k, 0, len(lv) - 1)
    rows = np.arange(len(r_q))
    q_lo, q_hi = r_q[rows, lo], r_q[rows, hi]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(q_hi > q_lo, (x - q_lo) / (q_hi - q_lo), 1.0)
    return np.where(k == 0, lv[0], np.where(k == len(lv), lv[-1],
                                            lv[lo] + np.clip(frac, 0, 1) * (lv[hi] - lv[lo])))


def event_prob(r_q: np.ndarray, levels, move: float) -> np.ndarray:
    below = prob_below(r_q, levels, move)
    return 1 - below if move > 0 else below


def event(r_q: np.ndarray, r_act: np.ndarray, levels, move: float):
    """(forecast probability, outcome) of 'move >= m' (m > 0) or 'move <= m' (m < 0)."""
    hit = r_act >= move if move > 0 else r_act <= move
    return event_prob(r_q, levels, move), hit


def quantile_at(r_q: np.ndarray, levels, tau: float) -> np.ndarray:
    lv = np.asarray(levels, dtype=float)
    if tau <= lv[0]:
        return r_q[:, 0]
    if tau >= lv[-1]:
        return r_q[:, -1]
    j = int(np.searchsorted(lv, tau, side="right") - 1)
    w = (tau - lv[j]) / (lv[j + 1] - lv[j])
    return r_q[:, j] + w * (r_q[:, j + 1] - r_q[:, j])


def ladder(r_q: np.ndarray, levels, confidences) -> dict[str, np.ndarray]:
    """up_c: P(move >= up_c) = c; down_c: P(move <= down_c) = c."""
    out = {}
    for c in confidences:
        out[f"up_{round(c * 100)}"] = quantile_at(r_q, levels, 1 - c)
        out[f"down_{round(c * 100)}"] = quantile_at(r_q, levels, c)
    return out


# ---- buckets ---------------------------------------------------------------------------------

def bucket_edges(width: float, lo: float = 0.05, hi: float = 0.95) -> list[float]:
    """Edges lo, then every multiple of `width`, then hi (e.g. 0.05, 0.1, 0.2, ..., 0.9, 0.95)."""
    inner = np.arange(width, 1, width)
    inner = [round(float(e), 4) for e in inner if lo < e - 1e-9 and e + 1e-9 < hi]
    return [lo, *inner, hi]


def bucket_labels(edges) -> list[str]:
    return [f"{a:.0%}-{b:.0%}" for a, b in pairwise(edges)]


def bucket_of(prob: np.ndarray, edges) -> np.ndarray:
    """Bucket index per probability (-1 outside the edges); the top edge is inclusive."""
    e = np.asarray(edges, dtype=float)
    b = np.digitize(prob, e) - 1
    b = np.where(prob >= e[-1] - 1e-12, len(e) - 2, b)
    return np.where((prob < e[0] - 1e-12) | (prob > e[-1] + 1e-12), -1, b)


# ---- statistics ------------------------------------------------------------------------------

def wilson(hits, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a binomial rate (NaN where n == 0)."""
    hits, n = np.asarray(hits, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = hits / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _z(observed, expected, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (observed - expected) / np.sqrt(expected * (1 - expected) / n)


def _rates(g: pd.DataFrame) -> pd.DataFrame:
    g["deviation"] = g["hit_rate"] - g["mean_prob"]
    g["z"] = _z(g["hit_rate"], g["mean_prob"], g["n"])
    g["ci_lo"], g["ci_hi"] = wilson(g["hit_rate"] * g["n"], g["n"])
    return g


def cells(prob: np.ndarray, hit: np.ndarray, edges, keys: pd.DataFrame | None = None
          ) -> pd.DataFrame:
    """One row per (keys..., bucket): n, mean_prob, hit_rate, deviation, z, ci_lo, ci_hi."""
    by = [] if keys is None else list(keys.columns)
    b = bucket_of(prob, edges)
    ok = b >= 0
    d = pd.DataFrame({"bucket_i": b[ok], "prob": prob[ok], "hit": hit[ok]})
    for c in by:
        d[c] = keys[c].to_numpy()[ok]
    g = (d.groupby([*by, "bucket_i"], observed=True)
         .agg(n=("hit", "size"), mean_prob=("prob", "mean"), hit_rate=("hit", "mean"))
         .reset_index())
    labels = np.asarray(bucket_labels(edges))
    g.insert(len(by), "bucket", labels[g["bucket_i"].to_numpy()] if len(g) else [])
    return _rates(g)


def grid(events: Events, edges, keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """`cells` for every move: adds move, side and the event label."""
    parts = []
    for move, (prob, hit) in events.items():
        g = cells(prob, hit, edges, keys)
        g.insert(0, "event", event_label(move))
        g.insert(0, "side", side_of(move))
        g.insert(0, "move", move)
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["move", "side", "event", "bucket", "n", "mean_prob",
                                     "hit_rate", "deviation", "z", "ci_lo", "ci_hi"])
    return pd.concat(parts, ignore_index=True)


def threshold_curve(prob: np.ndarray, hit: np.ndarray, cs) -> pd.DataFrame:
    """For each call level c: the calls with P >= c, how many, how often they came true."""
    rows = []
    for c in cs:
        sel = prob >= c - 1e-9
        n = int(sel.sum())
        rows.append({"confidence": c, "n": n,
                     "mean_prob": float(prob[sel].mean()) if n else np.nan,
                     "hit_rate": float(hit[sel].mean()) if n else np.nan})
    t = pd.DataFrame(rows)
    t["ci_lo"], t["ci_hi"] = wilson(t["hit_rate"] * t["n"], t["n"])
    t["base_rate"] = float(hit.mean()) if len(hit) else np.nan
    return t


def move_curve(events: Events, confidence: float) -> pd.DataFrame:
    """For a fixed call level: per move, the calls with P >= confidence and their hit rate."""
    rows = []
    for move, (prob, hit) in sorted(events.items()):
        r = threshold_curve(prob, hit, [confidence]).iloc[0].to_dict()
        rows.append({"move": move, "event": event_label(move), "side": side_of(move), **r})
    return pd.DataFrame(rows)


def ladder_hits(r_q: np.ndarray, r_act: np.ndarray, levels, confidences,
                keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """How often the actual move cleared the model's own ladder move (ideal: = confidence)."""
    by = [] if keys is None else list(keys.columns)
    lad = ladder(r_q, levels, confidences)
    parts = []
    for c in confidences:
        for side in (SIDE_UP, SIDE_DOWN):
            move = lad[f"{side}_{round(c * 100)}"]
            hit = r_act >= move if side == SIDE_UP else r_act <= move
            d = pd.DataFrame({"move": move, "hit": hit})
            for k in by:
                d[k] = keys[k].to_numpy()
            g = d.groupby(by, observed=True) if by else d.groupby(np.zeros(len(d)))
            g = g.agg(n=("hit", "size"), mean_move=("move", "mean"), hit_rate=("hit", "mean"))
            g = g.reset_index(drop=not by)
            g.insert(len(by), "confidence", c)
            g.insert(len(by), "side", side)
            parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["deviation"] = out["hit_rate"] - out["confidence"]
    out["z"] = _z(out["hit_rate"], out["confidence"], out["n"])
    out["ci_lo"], out["ci_hi"] = wilson(out["hit_rate"] * out["n"], out["n"])
    return out


def pit(r_q: np.ndarray, r_act: np.ndarray, levels) -> np.ndarray:
    return prob_below(r_q, levels, r_act)


def bias(u: np.ndarray, keys: pd.DataFrame) -> pd.DataFrame:
    """Per group: n, mean PIT, bias = 0.5 - mean PIT, bias_z and a label."""
    by = list(keys.columns)
    d = keys.reset_index(drop=True).assign(u=u)
    out = d.groupby(by, observed=True)["u"].agg(n="size", mean_pit="mean").reset_index()
    out["bias"] = 0.5 - out["mean_pit"]
    out["bias_z"] = out["bias"] / np.sqrt(1 / 12 / out["n"])
    out["label"] = np.select([out["bias_z"].abs() < 2, out["bias"] > 0],
                             ["calibrated", "optimistic"], "pessimistic")
    return out


def daily(u: np.ndarray, days, prob: np.ndarray | None = None,
          hit: np.ndarray | None = None) -> pd.DataFrame:
    """Per day across stocks: PIT bias (+ z) and, for one event, mean prob vs hit rate."""
    d = pd.DataFrame({"day": np.asarray(days), "u": u})
    if prob is not None:
        d["prob"], d["hit"] = prob, hit
    agg = {"n": ("u", "size"), "mean_pit": ("u", "mean")}
    if prob is not None:
        agg |= {"mean_prob": ("prob", "mean"), "hit_rate": ("hit", "mean")}
    out = d.groupby("day").agg(**agg).reset_index()
    out["bias"] = 0.5 - out["mean_pit"]
    out["bias_z"] = out["bias"] / np.sqrt(1 / 12 / out["n"])
    if prob is not None:
        out["deviation"] = out["hit_rate"] - out["mean_prob"]
    return out


def summary(r_q: np.ndarray, r_act: np.ndarray, levels, events: Events | None = None,
            edges=None) -> dict:
    """Run-level scores: pinball loss vs naive (move = 0 at every level), outer-interval
    coverage, mean PIT and the n-weighted mean |deviation| over the grid (ECE)."""
    if len(r_act) == 0:
        return {"rows": 0}
    lv = np.asarray(levels, dtype=float)[None, :]
    err = r_act[:, None] - r_q
    pinball = float(np.maximum(lv * err, (lv - 1) * err).mean())
    a = r_act[:, None]
    naive = float(np.maximum(lv * a, (lv - 1) * a).mean())
    u = pit(r_q, r_act, levels)
    out = {
        "rows": len(r_act),
        "pinball": pinball,
        "naive_pinball": naive,
        "pinball_skill": 1 - pinball / naive if naive else np.nan,
        "coverage": float(((r_q[:, 0] <= r_act) & (r_act <= r_q[:, -1])).mean()),
        "coverage_nominal": float(levels[-1] - levels[0]),
        "mean_pit": float(u.mean()),
        "bias": float(0.5 - u.mean()),
        "median_width": float(np.median(r_q[:, -1] - r_q[:, 0])),
    }
    if events and edges is not None:
        g = grid(events, edges)
        out["ece"] = float((g["deviation"].abs() * g["n"]).sum() / g["n"].sum()) if len(g) else np.nan
    return out
