"""Forecast frames, point metrics (vs the naive "tomorrow = today" baseline) and calibration
of the forecast distribution (do 70% calls come true 70% of the time?).

Scoring is on panel rows: a forecast for session d is kept only when the panel has both row d
(so the decoder saw real, not gap-filled, known inputs) and row d-1, whose `today_vwap` is the
baseline and whose `target_tomorrow_vwap` is the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_frame(model, dataset, df: pd.DataFrame, cfg, trainer_kwargs: dict) -> pd.DataFrame:
    pred = model.predict(
        dataset.to_dataloader(train=False, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers),
        mode="quantiles",
        return_index=True,
        trainer_kwargs=dict(trainer_kwargs),  # predict() mutates it (adds its callback)
    )
    q = pred.output[:, 0, :].cpu().numpy()  # (n, quantiles); decoder length is 1
    return score_frame(pred.index[["symbol", "day_idx"]], q, df, cfg)


def qcol(level: float) -> str:
    return f"q{round(level * 100):02d}"


def score_frame(index: pd.DataFrame, q: np.ndarray, df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach the label + baseline to raw model outputs (`q`: one column per quantile, in
    target units) and keep only forecasts that land on real panel rows."""
    out = index.reset_index(drop=True).copy()
    # Drop duplicates from encoder windows (pytorch-forecasting keeps all encoder rows)
    out = out.drop_duplicates(subset=["symbol", "day_idx"], keep="last")
    out = out.reset_index(drop=True)
    q = np.sort(q, axis=1)  # QuantileLoss doesn't forbid crossing; rearrange to monotone
    # Re-align q after duplicate drop
    orig_len = len(index)
    q = q[index.drop_duplicates(subset=["symbol", "day_idx"], keep="last").index.to_numpy()]
    for i, level in enumerate(cfg.quantiles):
        out[qcol(level)] = q[:, i]
    prev = df[["symbol", "day_idx", "today_vwap", "target_tomorrow_vwap"]].rename(
        columns={"today_vwap": "prev_vwap", "target_tomorrow_vwap": "actual"}
    )
    prev["day_idx"] = prev["day_idx"] + 1
    out = out.merge(prev, on=["symbol", "day_idx"], how="inner", validate="one_to_one")
    out = out.merge(df[["symbol", "day_idx"]], on=["symbol", "day_idx"], how="inner")
    if cfg.target == "return":  # return forecasts -> prices
        for level in cfg.quantiles:
            out[qcol(level)] = out["prev_vwap"] * (1 + out[qcol(level)])
    return out.dropna(subset=["actual"]).reset_index(drop=True)


def metrics(frame: pd.DataFrame, quantiles: list[float]) -> dict:
    """Scale-free metrics on a forecast frame (prices -> returns vs the previous session)."""
    if frame.empty:
        return {"rows": 0}
    cols = [qcol(level) for level in quantiles]
    prev, actual = frame["prev_vwap"].to_numpy(), frame["actual"].to_numpy()
    p50 = frame["q50"].to_numpy()
    r_act = actual / prev - 1
    r_q = frame[cols].to_numpy() / prev[:, None] - 1
    r_p50 = p50 / prev - 1

    err = r_act[:, None] - r_q
    lv = np.asarray(quantiles)[None, :]
    pinball = np.maximum(lv * err, (lv - 1) * err).mean()
    naive_pinball = np.maximum(lv * r_act[:, None], (lv - 1) * r_act[:, None]).mean()

    ape, naive_ape = np.abs(p50 - actual) / actual, np.abs(prev - actual) / actual
    moved = (r_act != 0) & (r_p50 != 0)
    return {
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
        "quantile_loss": float(pinball),
        "naive_quantile_loss": float(naive_pinball),
        "mape": float(ape.mean()),
        "naive_mape": float(naive_ape.mean()),
        "mape_skill": float(1 - ape.mean() / naive_ape.mean()),
        "median_ape": float(np.median(ape)),
        "naive_median_ape": float(np.median(naive_ape)),
        "directional_accuracy": (
            float((np.sign(r_p50) == np.sign(r_act))[moved].mean()) if moved.any() else None
        ),
        "up_rate": float((r_act > 0).mean()),
        "interval_coverage": float(
            ((frame[cols[0]] <= frame["actual"]) & (frame["actual"] <= frame[cols[-1]])).mean()
        ),
        "interval_nominal": float(quantiles[-1] - quantiles[0]),
    }


def _return_quantiles(frame: pd.DataFrame, quantiles: list[float]) -> tuple[np.ndarray, np.ndarray]:
    prev = frame["prev_vwap"].to_numpy()
    r_q = frame[[qcol(level) for level in quantiles]].to_numpy() / prev[:, None] - 1
    return r_q, frame["actual"].to_numpy() / prev - 1


def prob_below(r_q: np.ndarray, levels: list[float], x) -> np.ndarray:
    """Per-row forecast CDF F(x) = P(return <= x), linear between quantiles; `x` is a scalar
    or one value per row. Outside the quantile grid it is clamped to the end levels, so
    probabilities lie in [P_lo, P_hi]."""
    lv = np.asarray(levels)
    x = np.broadcast_to(np.asarray(x, dtype=float), (len(r_q),))
    k = (r_q < x[:, None]).sum(axis=1)  # quantiles strictly below x
    lo, hi = np.clip(k - 1, 0, len(lv) - 1), np.clip(k, 0, len(lv) - 1)
    rows = np.arange(len(r_q))
    q_lo, q_hi = r_q[rows, lo], r_q[rows, hi]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(q_hi > q_lo, (x - q_lo) / (q_hi - q_lo), 1.0)
    return np.where(k == 0, lv[0], np.where(k == len(lv), lv[-1],
                                            lv[lo] + np.clip(frac, 0, 1) * (lv[hi] - lv[lo])))


def _event(r_q, r_act, levels, move):
    """(forecast probability, outcome) of 'return >= move' (upside) or '<= move' (downside)."""
    if move > 0:
        return 1 - prob_below(r_q, levels, move), r_act >= move
    return prob_below(r_q, levels, move), r_act <= move


def quantile_calibration(frame: pd.DataFrame, quantiles: list[float]) -> list[dict]:
    """For each level tau: the share of actuals at or below the forecast P_tau (ideal: tau)."""
    r_q, r_act = _return_quantiles(frame, quantiles)
    return [{"quantile": tau, "observed": float((r_act <= r_q[:, i]).mean())}
            for i, tau in enumerate(quantiles)]


def signal_table(frame: pd.DataFrame, quantiles: list[float], moves: list[float],
                 confidences: list[float]) -> list[dict]:
    """For each (move, confidence c): rows where the forecast gives P(event) >= c, and how
    often the event then happened. Calibrated when hit_rate is at least about c (precisely: near
    mean_prob); `base_rate` is how often the event happens over all rows."""
    r_q, r_act = _return_quantiles(frame, quantiles)
    out = []
    for move in moves:
        prob, hit = _event(r_q, r_act, quantiles, move)
        for c in confidences:
            sel = prob >= c - 1e-9
            out.append({
                "move": move, "confidence": c,
                "signals": int(sel.sum()), "signal_share": float(sel.mean()),
                "symbols": int(frame.loc[sel, "symbol"].nunique()),
                "mean_prob": float(prob[sel].mean()) if sel.any() else None,
                "hit_rate": float(hit[sel].mean()) if sel.any() else None,
                "base_rate": float(hit.mean()),
            })
    return out


def calibration(frame: pd.DataFrame, cfg) -> dict:
    from .calibration import grid

    if frame.empty:
        return {}
    return {
        "quantiles": quantile_calibration(frame, cfg.quantiles),
        "signals": signal_table(frame, cfg.quantiles, cfg.moves, cfg.confidences),
        "grid": grid(frame, cfg).to_dict("records"),
    }
