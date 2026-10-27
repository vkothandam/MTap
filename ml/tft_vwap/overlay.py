"""R6p / R6f: a gradient-boosted overlay that adjusts a base run's quantiles with pre-market
features.

    uv run tft-vwap-overlay runs/<BASE> --panel PANEL --inputs px|full [--folds 1,3,5]

--inputs px uses the prices group (R6p, the 09:30 forecast), full the prices and full groups
(R6f, 09:45). BASE should be the no-pre-market forecast (R1, or an optional run's `~night`
sibling). For each target fold k the overlay trains one HistGradientBoostingRegressor(
loss="quantile") per quantile level on the base run's test forecasts of the EARLIER folds only
(< k): features are the base quantiles as moves vs prev_vwap plus the panel's pre-market
features of the chosen groups (with their flags), the target is the actual move. It then
predicts fold k and sorts the quantiles per row so none cross. Training on the base's
out-of-fold forecasts matters: its in-sample outputs are overconfident, and an overlay fitted
to them would be miscalibrated live. Folds with no earlier fold are skipped; folds with fewer
than MIN_TRAIN_FOLDS (2) earlier ones are flagged `thin` in config.json and summary.json and
should be reported separately (the plan: odd folds only, so fold 1 trains on fold 0 alone).
"Earlier" means earlier in the base run's own folds, which are the odd ones only when the base
was run on them: a base on all 16 folds trains fold 9 on 0-8.
Writes runs/<BASE>-overlay-<inputs>/ in the run contract (price units: prev_vwap, actual, qNN).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .compare import qlevels
from .config import PM_FULL_REALS, PM_PX_REALS
from .runout import write_fold, write_summary

MIN_TRAIN_FOLDS = 2
INPUTS = {"px": PM_PX_REALS, "full": (*PM_PX_REALS, *PM_FULL_REALS)}


def load_folds(run_dir: Path) -> dict[int, pd.DataFrame]:
    return {int(p.parent.name.split("_")[1]): pd.read_parquet(p)
            for p in sorted(Path(run_dir).glob("fold_*/test_predictions.parquet"))}


def features(frame: pd.DataFrame, cols: list[str], panel: pd.DataFrame,
             pm: tuple[str, ...]) -> pd.DataFrame:
    """Base quantiles as moves vs prev_vwap, plus the pre-market features (NaN stays NaN:
    the booster handles it, and the flags say whether it is real)."""
    x = pd.DataFrame(frame[cols].to_numpy() / frame["prev_vwap"].to_numpy()[:, None] - 1,
                     columns=[f"base_{c}" for c in cols], index=frame.index)
    keys = pd.DataFrame({"symbol": frame["symbol"].str.split("~").str[0].to_numpy(),
                         "day_idx": frame["day_idx"].to_numpy()})
    got = keys.merge(panel, on=["symbol", "day_idx"], how="left")
    for c in pm:
        x[c] = got[c].to_numpy()
    return x


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, cols: list[str], levels: list[float],
                panel: pd.DataFrame, pm: tuple[str, ...] = INPUTS["full"], seed: int = 0,
                **hgb) -> np.ndarray:
    """Quantile moves for `target` rows from a model fitted on `train` rows only."""
    xt, xp = features(train, cols, panel, pm), features(target, cols, panel, pm)
    y = train["actual"].to_numpy() / train["prev_vwap"].to_numpy() - 1
    kw = {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200,
          "random_state": seed, **hgb}
    out = np.column_stack([
        HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw).fit(xt, y).predict(xp)
        for q in levels])
    return np.sort(out, axis=1)  # no crossing


def overlay(base: Path, panel_path: Path, folds: list[int] | None = None,
            out: Path | None = None, inputs: str = "full") -> Path:
    base = Path(base)
    pm = INPUTS[inputs]
    by_fold = load_folds(base)
    if not by_fold:
        raise FileNotFoundError(f"no fold_*/test_predictions.parquet under {base}")
    cols_by_level = qlevels(next(iter(by_fold.values())))
    levels, cols = list(cols_by_level), list(cols_by_level.values())
    have = set(pd.read_parquet(panel_path).columns)
    if set(pm) - have:
        raise ValueError(f"{panel_path} lacks pre-market columns {sorted(set(pm) - have)}")
    panel = pd.read_parquet(panel_path, columns=["symbol", "day_idx", *pm])
    out = Path(out) if out else base.with_name(f"{base.name}-overlay-{inputs}")
    done, thin, scored = [], [], {}
    for k in sorted(by_fold):
        if folds is not None and k not in folds:
            continue
        earlier = [j for j in sorted(by_fold) if j < k]
        if not earlier:
            print(f"fold {k}: no earlier fold to learn from, skipped")
            continue
        train, target = pd.concat([by_fold[j] for j in earlier], ignore_index=True), by_fold[k]
        moves = fit_predict(train, target, cols, levels, panel, pm)
        res = target.copy()
        res[cols] = target["prev_vwap"].to_numpy()[:, None] * (1 + moves)
        is_thin = len(earlier) < MIN_TRAIN_FOLDS
        write_fold(out, k, res, {"overlay_trained_on_folds": earlier, "thin": is_thin})
        done.append(k)
        scored[k] = res
        if is_thin:
            thin.append(k)
        print(f"fold {k}: trained on {len(train):,} rows of folds {earlier}"
              f"{' (thin: fewer than 2 earlier folds)' if is_thin else ''}, scored {len(res):,}")
    saved = json.loads((base / "config.json").read_text()) if (base / "config.json").exists() else {}
    (out / "config.json").write_text(json.dumps(
        {"model": "overlay", "inputs": inputs, "base": str(base), "folds": done, "thin_folds": thin,
         "config": saved.get("config", {})}, indent=2, default=str))
    if scored:
        write_summary(out, scored, {"thin_folds": thin})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-overlay", description=__doc__.split("\n")[0])
    p.add_argument("base_run", type=Path)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--inputs", choices=list(INPUTS), default="full",
                   help="px = prices group (R6p), full = prices + full (R6f)")
    p.add_argument("--folds", help="comma list of target folds (default: all with an earlier one)")
    p.add_argument("--out", type=Path, help="output run dir (default: <base>-overlay-<inputs>)")
    a = p.parse_args(argv)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    print(f"wrote {overlay(a.base_run, a.panel, folds, a.out, a.inputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
