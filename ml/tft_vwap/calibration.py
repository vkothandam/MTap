"""Calibration of the forecast distribution, overall or per stock / industry / day.

Three views, all scored on saved test forecasts (no retraining):

- grid:   for each move m and forecast-probability bucket, how often the event actually
          happened. "AAPL, said 70-80% for >= +2%: came true 55% (n=31)".
- ladder: the move the model gives at each confidence c (upside: q(1-c), downside: q(c)) and
          how often the actual cleared it. "60% ladder for AAPL = +1.0%; cleared 52%".
- bias:   PIT u = F(actual) per row; mean(u) < 0.5 means actuals ran below the forecast
          (optimistic, positive bias), > 0.5 means pessimistic (negative bias).

deviation = observed - forecast; z scales it by the binomial standard error, so |z| > 2 marks a
deviation that is unlikely to be noise at that sample size.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from .evaluate import _event, _return_quantiles, prob_below

SIDE_UP, SIDE_DOWN = "up", "down"


def _z(observed, expected, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (observed - expected) / np.sqrt(expected * (1 - expected) / n)


def bucket_labels(edges: list[float]) -> list[str]:
    return [f"{lo:.0%}-{hi:.0%}" for lo, hi in pairwise(edges)]


def bucket_of(prob: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bucket index per probability (-1 below the first edge); the top edge is inclusive."""
    e = np.asarray(edges)
    b = np.digitize(prob, e) - 1
    b = np.where(prob >= e[-1] - 1e-12, len(e) - 2, b)
    return np.where(prob < e[0], -1, b)


def grid(frame: pd.DataFrame, cfg, by: list[str] | None = None) -> pd.DataFrame:
    """One row per (by..., move, bucket): n, mean forecast prob, observed hit rate, deviation, z."""
    by = list(by or [])
    r_q, r_act = _return_quantiles(frame, cfg.quantiles)
    labels = bucket_labels(cfg.prob_buckets)
    keys = {c: frame[c].to_numpy() for c in by}
    parts = []
    for move in cfg.moves:
        prob, hit = _event(r_q, r_act, cfg.quantiles, move)
        b = bucket_of(prob, cfg.prob_buckets)
        ok = b >= 0
        d = pd.DataFrame({**{c: v[ok] for c, v in keys.items()},
                          "bucket": b[ok], "prob": prob[ok], "hit": hit[ok]})
        g = (d.groupby([*by, "bucket"], observed=True)
             .agg(n=("hit", "size"), mean_prob=("prob", "mean"), hit_rate=("hit", "mean"))
             .reset_index())
        g.insert(len(by), "move", move)
        g.insert(len(by), "side", SIDE_UP if move > 0 else SIDE_DOWN)
        parts.append(g)
    cols = [*by, "side", "move", "bucket", "n", "mean_prob", "hit_rate", "deviation", "z"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out["bucket"] = np.asarray(labels)[out["bucket"].to_numpy()]
    out["deviation"] = out["hit_rate"] - out["mean_prob"]
    out["z"] = _z(out["hit_rate"], out["mean_prob"], out["n"])
    return out[cols]


def quantile_at(r_q: np.ndarray, levels: list[float], tau: float) -> np.ndarray:
    """Per-row forecast quantile at level tau, linear between the predicted quantiles."""
    lv = np.asarray(levels)
    if tau <= lv[0]:
        return r_q[:, 0]
    if tau >= lv[-1]:
        return r_q[:, -1]
    j = int(np.searchsorted(lv, tau, side="right") - 1)
    w = (tau - lv[j]) / (lv[j + 1] - lv[j])
    return r_q[:, j] + w * (r_q[:, j + 1] - r_q[:, j])


def ladder(r_q: np.ndarray, levels: list[float], confidences: list[float]) -> dict[str, np.ndarray]:
    """Move at each confidence c: up_c with P(move >= up_c) = c, down_c with P(move <= down_c) = c."""
    out = {}
    for c in confidences:
        out[f"up_{round(c * 100)}"] = quantile_at(r_q, levels, 1 - c)
        out[f"down_{round(c * 100)}"] = quantile_at(r_q, levels, c)
    return out


def ladder_hits(frame: pd.DataFrame, cfg, by: list[str] | None = None) -> pd.DataFrame:
    """How often the actual move cleared the model's own ladder move (ideal: = confidence)."""
    by = list(by or [])
    r_q, r_act = _return_quantiles(frame, cfg.quantiles)
    lad = ladder(r_q, cfg.quantiles, cfg.confidences)
    parts = []
    for c in cfg.confidences:
        for side in (SIDE_UP, SIDE_DOWN):
            move = lad[f"{side}_{round(c * 100)}"]
            hit = r_act >= move if side == SIDE_UP else r_act <= move
            d = pd.DataFrame({**{k: frame[k].to_numpy() for k in by}, "move": move, "hit": hit})
            g = (d.groupby(by, observed=True) if by else d.groupby(np.zeros(len(d))))
            g = g.agg(n=("hit", "size"), mean_move=("move", "mean"), hit_rate=("hit", "mean"))
            g = g.reset_index(drop=not by)
            g.insert(len(by), "confidence", c)
            g.insert(len(by), "side", side)
            parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["deviation"] = out["hit_rate"] - out["confidence"]
    out["z"] = _z(out["hit_rate"], out["confidence"], out["n"])
    return out


def pit(frame: pd.DataFrame, cfg) -> np.ndarray:
    """u = forecast CDF at the actual move (clamped to the outer quantile levels)."""
    r_q, r_act = _return_quantiles(frame, cfg.quantiles)
    return prob_below(r_q, cfg.quantiles, r_act)


def bias(frame: pd.DataFrame, cfg, by: list[str]) -> pd.DataFrame:
    """Per group: mean PIT, bias = 0.5 - mean PIT (> 0: forecasts too high, optimistic), its z
    against a calibrated U(0,1) PIT, and the n-weighted grid deviation per side."""
    d = frame[by].copy()
    d["u"] = pit(frame, cfg)
    out = d.groupby(by, observed=True)["u"].agg(n="size", mean_pit="mean").reset_index()
    out["bias"] = 0.5 - out["mean_pit"]
    out["bias_z"] = out["bias"] / np.sqrt(1 / 12 / out["n"])
    g = grid(frame, cfg, by)
    g["wdev"] = g["deviation"] * g["n"]
    side = g.groupby([*by, "side"], observed=True)[["wdev", "n"]].sum()
    side = (side["wdev"] / side["n"]).unstack("side").reindex(columns=[SIDE_UP, SIDE_DOWN])
    side = side.rename(columns={SIDE_UP: "up_deviation", SIDE_DOWN: "down_deviation"})
    out = out.merge(side.reset_index(), on=by, how="left")
    out["label"] = np.select(
        [out["bias_z"].abs() < 2, out["bias"] > 0], ["calibrated", "optimistic"], "pessimistic")
    return out
