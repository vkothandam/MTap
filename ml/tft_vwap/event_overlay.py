"""R6c: a market/macro event overlay that widens (or narrows) a base run's band on event days.

    uv run tft-vwap-event-overlay runs/<BASE> --panel PANEL [--calibrate] [--shift] [--folds 1,3]

The base run's quantiles become, per row,

    move_q' = P50 + shift + s * (move_q - P50),    s = exp(a + sum_i b_i * f_i)

where every move is vs prev_vwap. The event features f_i are 0 on ordinary days, so with no
`--calibrate` (a = 0) and no `--shift` the base forecast passes through unchanged on those days.
Features are day-level (the same for every stock, except own sector) and use only what is known
the night before the forecast day D: the panel row of session t = D - 1.

    f_mkt    yesterday's market move, |mkt_cap_ret| in units of its past 63-session std, beyond Z_CUT
    f_sec    the stock's sector vs the market (own_sector_cap_ret - mkt_cap_ret), the same way
    f_10y    the 10-year yield change (ust_10y_chg_bp), the same way
    f_2y     the 2-year yield change (ust_2y_chg_bp), the same way

Each is max(0, |z| - Z_CUT), capped at F_CAP; a missing value is 0 (no event). `--shift` adds a
median shift  shift = sum_i c_i * sign(z_i) * f_i  for the market and sector features.
`--calibrate` adds the intercept a: one band scale for every day, the base's own calibration
error, kept separate so the event effect can be read on its own.

For each target fold k the few coefficients are fitted by minimising the mean pinball loss
(the one tft-vwap-compare reports) over the base run's test forecasts of the EARLIER folds only
(< k), ROWS_PER_DAY sampled per day so every day weighs the same. Fold 0 has nothing earlier and
is skipped; folds with fewer than MIN_TRAIN_FOLDS earlier ones are flagged `thin`. s > 0, so the
order of the quantiles never changes. The coefficients per fold are in fold_k/metrics.json.
Writes runs/<BASE>-event[-cal][-shift]/ in the run contract, with a MODEL.md card (`--label`,
`--about`, `--purpose`; model_card).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import model_card

from .compare import qlevels
from .overlay import MIN_TRAIN_FOLDS, load_folds
from .runout import write_fold, write_summary

WINDOW = 63           # sessions in the "normal range" of each series (past only)
MIN_PERIODS = 20
Z_CUT = 2.0           # |z| below this is an ordinary day
F_CAP = 4.0
S_RANGE = (0.5, 3.0)  # the band scale never goes outside this
ROWS_PER_DAY = 400
SCALE_FEATURES = ("f_mkt", "f_sec", "f_10y", "f_2y")
SHIFT_FEATURES = ("g_mkt", "g_sec")
PANEL_COLS = ["day_idx", "industry_code", "mkt_cap_ret", "own_sector_cap_ret", "ust_10y_chg_bp",
              "ust_2y_chg_bp"]


def _sector(code: pd.Series) -> pd.Series:
    """industry_code[:2] when numeric, else '' (no sector), as in handoff.py."""
    s = code.astype(str).str[:2]
    return s.where(s.str.fullmatch(r"\d\d"), "")


def _excess(x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(f, signed f) for a series ordered by session: |x| / past std beyond Z_CUT."""
    sd = x.rolling(WINDOW, min_periods=MIN_PERIODS).std().shift(1)
    z = x / sd
    f = (z.abs() - Z_CUT).clip(lower=0, upper=F_CAP).fillna(0.0)
    return f, np.sign(z).fillna(0.0) * f


def event_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(by_day, by_sector_day), both keyed by the FORECAST day D = t + 1, from panel rows t.

    by_day: day_idx, f_mkt, g_mkt, f_10y, f_2y. by_sector_day: sec, day_idx, f_sec, g_sec
    (stocks with no sector get f_sec = g_sec = 0)."""
    day = (panel.groupby("day_idx")[["mkt_cap_ret", "ust_10y_chg_bp", "ust_2y_chg_bp"]].first()
           .sort_index())
    full = pd.RangeIndex(day.index.min(), day.index.max() + 1)
    day = day.reindex(full)  # a session with no rows stays a gap in the rolling window
    out = pd.DataFrame(index=day.index)
    out["f_mkt"], out["g_mkt"] = _excess(day["mkt_cap_ret"])
    out["f_10y"], _ = _excess(day["ust_10y_chg_bp"])
    out["f_2y"], _ = _excess(day["ust_2y_chg_bp"])
    by_day = out.rename_axis("day_idx").reset_index()
    by_day["day_idx"] += 1

    p = panel.assign(sec=_sector(panel["industry_code"]))
    p = p[p["sec"] != ""]
    rel = (p.groupby(["sec", "day_idx"])[["own_sector_cap_ret", "mkt_cap_ret"]].first()
           .eval("own_sector_cap_ret - mkt_cap_ret").rename("rel").reset_index())
    parts = []
    for sec, g in rel.groupby("sec"):
        s = g.set_index("day_idx")["rel"].sort_index().reindex(full)
        f, sg = _excess(s)
        parts.append(pd.DataFrame({"sec": sec, "day_idx": full + 1, "f_sec": f.to_numpy(),
                                   "g_sec": sg.to_numpy()}))
    by_sec = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=["sec", "day_idx", "f_sec", "g_sec"]))
    return by_day, by_sec


def row_features(frame: pd.DataFrame, by_day: pd.DataFrame, by_sec: pd.DataFrame) -> pd.DataFrame:
    """The event features of each forecast row (day_idx = D); 0 where unknown."""
    keys = pd.DataFrame({"day_idx": frame["day_idx"].to_numpy(),
                         "sec": _sector(frame["industry_code"]).to_numpy()})
    got = keys.merge(by_day, on="day_idx", how="left").merge(by_sec, on=["sec", "day_idx"],
                                                             how="left")
    cols = [*SCALE_FEATURES, *SHIFT_FEATURES]
    return got[cols].fillna(0.0).set_index(frame.index)


class Form:
    """Which coefficients are fitted: a (calibrate), b (scale features), c (shift features)."""

    def __init__(self, calibrate: bool, shift: bool):
        self.calibrate, self.shift = calibrate, shift
        self.names = [*(["a"] if calibrate else []), *(f"b_{f[2:]}" for f in SCALE_FEATURES),
                      *((f"c_{f[2:]}" for f in SHIFT_FEATURES) if shift else ())]

    def design(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(X for log s, G for the shift)."""
        n = len(feats)
        x = feats[list(SCALE_FEATURES)].to_numpy(float)
        if self.calibrate:
            x = np.column_stack([np.ones(n), x])
        g = feats[list(SHIFT_FEATURES)].to_numpy(float) if self.shift else np.zeros((n, 0))
        return x, g


def apply(moves: np.ndarray, x: np.ndarray, g: np.ndarray, beta: np.ndarray,
          gamma: np.ndarray, mid: int) -> tuple[np.ndarray, np.ndarray]:
    """(new moves, s) for base moves (rows x levels), P50 at column `mid`."""
    s = np.clip(np.exp(x @ beta), *S_RANGE)
    p50 = moves[:, [mid]]
    shift = (g @ gamma)[:, None] if gamma.size else 0.0
    return p50 + shift + s[:, None] * (moves - p50), s


def fit(moves: np.ndarray, y: np.ndarray, x: np.ndarray, g: np.ndarray, levels: np.ndarray,
        mid: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Coefficients minimising the mean pinball loss; also the loss before and after."""
    tau = levels[None, :]
    w = moves - moves[:, [mid]]
    nx, ng = x.shape[1], g.shape[1]

    def loss_grad(theta):
        beta, gamma = theta[:nx], theta[nx:]
        q, s = apply(moves, x, g, beta, gamma, mid)
        err = y[:, None] - q
        # in bp: in return units the gradient is under L-BFGS's tolerance and it never moves
        loss = 1e4 * np.maximum(tau * err, (tau - 1) * err).mean()
        dq = 1e4 * ((err < 0).astype(float) - tau) / err.size  # d loss / d q
        inside = (s > S_RANGE[0]) & (s < S_RANGE[1])
        ds = (dq * w).sum(axis=1) * s * inside
        grad = np.concatenate([x.T @ ds, g.T @ dq.sum(axis=1)])
        return loss, grad

    theta0 = np.zeros(nx + ng)
    before = loss_grad(theta0)[0]
    bounds = [(-1.0, 1.0)] * nx + [(-0.02, 0.02)] * ng
    r = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    return r.x[:nx], r.x[nx:], float(before) / 1e4, float(r.fun) / 1e4


def _moves(frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    prev = frame["prev_vwap"].to_numpy(float)
    return (frame[cols].to_numpy(float) / prev[:, None] - 1,
            frame["actual"].to_numpy(float) / prev - 1)


def _sample(frame: pd.DataFrame, per_day: int, seed: int) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("day_idx", sort=False).head(per_day).reset_index(drop=True)


def overlay(base: Path, panel_path: Path, folds: list[int] | None = None, out: Path | None = None,
            calibrate: bool = False, shift: bool = False, rows_per_day: int = ROWS_PER_DAY,
            seed: int = 0, card: dict | None = None) -> Path:
    base = Path(base)
    by_fold = load_folds(base)
    if not by_fold:
        raise FileNotFoundError(f"no fold_*/test_predictions.parquet under {base}")
    first = next(iter(by_fold.values()))
    if "industry_code" not in first.columns:
        raise ValueError(f"{base}: test_predictions lack industry_code (needed for f_sec)")
    lv = qlevels(first)
    levels, cols = np.asarray(list(lv)), list(lv.values())
    if 0.5 not in lv:
        raise ValueError(f"{base}: no q50")
    mid = list(lv).index(0.5)
    panel = pd.read_parquet(panel_path, columns=PANEL_COLS)
    by_day, by_sec = event_features(panel)
    form = Form(calibrate, shift)
    tag = "event" + ("-cal" if calibrate else "") + ("-shift" if shift else "")
    out = Path(out) if out else base.with_name(f"{base.name}-{tag}")
    done, thin, scored, coefs = [], [], {}, {}
    for k in sorted(by_fold):
        if folds is not None and k not in folds:
            continue
        earlier = [j for j in sorted(by_fold) if j < k]
        if not earlier:
            print(f"fold {k}: no earlier fold to learn from, skipped")
            continue
        train = _sample(pd.concat([by_fold[j] for j in earlier], ignore_index=True),
                        rows_per_day, seed)
        mt, yt = _moves(train, cols)
        ok = np.isfinite(mt).all(axis=1) & np.isfinite(yt)
        xt, gt = form.design(row_features(train, by_day, by_sec))
        beta, gamma, before, after = fit(mt[ok], yt[ok], xt[ok], gt[ok], levels, mid)

        target = by_fold[k]
        mk, _ = _moves(target, cols)
        feats = row_features(target, by_day, by_sec)
        xk, gk = form.design(feats)
        new, s = apply(mk, xk, gk, beta, gamma, mid)
        res = target.copy()
        res[cols] = target["prev_vwap"].to_numpy()[:, None] * (1 + new)
        is_thin = len(earlier) < MIN_TRAIN_FOLDS
        coef = dict(zip(form.names, map(float, np.concatenate([beta, gamma]))))
        event_rows = (feats[list(SCALE_FEATURES)] > 0).any(axis=1)
        info = {"overlay_trained_on_folds": earlier, "thin": is_thin, "coef": coef,
                "train_rows": int(ok.sum()), "train_pinball_before": before,
                "train_pinball_after": after, "event_row_share": float(event_rows.mean()),
                "scale_mean_event_rows": float(s[event_rows.to_numpy()].mean())
                if event_rows.any() else None,
                "scale_mean": float(s.mean())}
        write_fold(out, k, res, info)
        done.append(k)
        scored[k], coefs[k] = res, coef
        if is_thin:
            thin.append(k)
        print(f"fold {k}: folds {earlier[0]}-{earlier[-1]}, {ok.sum():,} rows; "
              + " ".join(f"{n}={v:+.3f}" for n, v in coef.items())
              + f"; event rows {event_rows.mean():.0%}, mean s {s.mean():.3f}")
    saved = json.loads((base / "config.json").read_text()) if (base / "config.json").exists() else {}
    (out / "config.json").write_text(json.dumps(
        {"model": "event_overlay", "base": str(base), "calibrate": calibrate, "shift": shift,
         "z_cut": Z_CUT, "window": WINDOW, "rows_per_day": rows_per_day, "folds": done,
         "thin_folds": thin, "coef": coefs, "config": saved.get("config", {})},
        indent=2, default=str))
    if scored:
        write_summary(out, scored, {"thin_folds": thin})
    model_card.write(
        out, kind="Event", forecast=model_card.read(base).get("forecast", "night"),
        inputs=f"Event-band overlay on {model_card.label_of(base)} (`{base.name}`): band scale "
               "after big market/sector/yield moves"
               + (", + one scale for every day" if calibrate else "")
               + (", + a median shift" if shift else "") + f"; {rows_per_day} rows per day.",
        folds=f"Folds {done}, each fitted on the base's earlier folds only"
              + (f"; thin (fewer than {MIN_TRAIN_FOLDS} earlier): {thin}." if thin else "."),
        status="done", result=model_card.contract_result(out), **(card or {}))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-event-overlay", description=__doc__.split("\n")[0])
    p.add_argument("base_run", type=Path)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--calibrate", action="store_true",
                   help="also fit one band scale for every day (the intercept a)")
    p.add_argument("--shift", action="store_true", help="also fit a median shift on event days")
    p.add_argument("--rows-per-day", type=int, default=ROWS_PER_DAY)
    p.add_argument("--folds", help="comma list of target folds (default: all with an earlier one)")
    p.add_argument("--out", type=Path, help="output run dir (default: <base>-event[-cal][-shift])")
    model_card.add_args(p)
    a = p.parse_args(argv)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    out = overlay(a.base_run, a.panel, folds, a.out, a.calibrate, a.shift, a.rows_per_day,
                  card=model_card.from_args(a, "tft-vwap-event-overlay", argv))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
