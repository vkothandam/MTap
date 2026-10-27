"""R6: a gradient-boosted overlay that adjusts a base run's quantiles with pre-market features.

    uv run tft-vwap-overlay runs/<BASE> --panel PANEL [--folds 1,3,5]

For each target fold k the overlay trains one HistGradientBoostingRegressor(loss="quantile")
per quantile level on the base run's test forecasts of the EARLIER folds only (< k): features
are the base quantiles as moves vs prev_vwap plus the panel's pre-market features
(pm_vwap_pct_prev_vwap, pm_vol_pct_10d, pm_known), the target is the actual move. It then
predicts fold k, sorts the quantiles per row so none cross, and writes
runs/<BASE>-overlay/fold_k/test_predictions.parquet in the base run's own contract (price units:
prev_vwap, actual, qNN) plus config.json {"model": "overlay", "base": ...}. Training on the
base's out-of-fold test forecasts matters: its in-sample outputs are overconfident, and an
overlay fitted to them would be miscalibrated live. Fold 0 has no earlier fold and is skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .compare import qlevels

PM_FEATURES = ("pm_vwap_pct_prev_vwap", "pm_vol_pct_10d", "pm_known")


def load_folds(run_dir: Path) -> dict[int, pd.DataFrame]:
    return {int(p.parent.name.split("_")[1]): pd.read_parquet(p)
            for p in sorted(Path(run_dir).glob("fold_*/test_predictions.parquet"))}


def features(frame: pd.DataFrame, cols: list[str], panel: pd.DataFrame) -> pd.DataFrame:
    """Base quantiles as moves vs prev_vwap, plus the pre-market features (NaN stays NaN:
    the booster handles it, and pm_known says whether it is real)."""
    x = pd.DataFrame(frame[cols].to_numpy() / frame["prev_vwap"].to_numpy()[:, None] - 1,
                     columns=[f"base_{c}" for c in cols], index=frame.index)
    pm = frame[["symbol", "day_idx"]].merge(panel, on=["symbol", "day_idx"], how="left")
    for c in PM_FEATURES:
        x[c] = pm[c].to_numpy()
    return x


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, cols: list[str], levels: list[float],
                panel: pd.DataFrame, seed: int = 0, **hgb) -> np.ndarray:
    """Quantile moves for `target` rows from a model fitted on `train` rows only."""
    xt, xp = features(train, cols, panel), features(target, cols, panel)
    y = train["actual"].to_numpy() / train["prev_vwap"].to_numpy() - 1
    kw = {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200,
          "random_state": seed, **hgb}
    out = np.column_stack([
        HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw).fit(xt, y).predict(xp)
        for q in levels])
    return np.sort(out, axis=1)  # no crossing


def overlay(base: Path, panel_path: Path, folds: list[int] | None = None,
            out: Path | None = None) -> Path:
    base = Path(base)
    by_fold = load_folds(base)
    if not by_fold:
        raise FileNotFoundError(f"no fold_*/test_predictions.parquet under {base}")
    cols_by_level = qlevels(next(iter(by_fold.values())))
    levels, cols = list(cols_by_level), list(cols_by_level.values())
    have = [c for c in PM_FEATURES if c in pd.read_parquet(panel_path).columns]
    if len(have) != len(PM_FEATURES):
        raise ValueError(f"{panel_path} lacks pre-market columns {set(PM_FEATURES) - set(have)}")
    panel = pd.read_parquet(panel_path, columns=["symbol", "day_idx", *PM_FEATURES])
    out = Path(out) if out else base.with_name(base.name + "-overlay")
    done = []
    for k in sorted(by_fold):
        if folds is not None and k not in folds:
            continue
        earlier = [by_fold[j] for j in sorted(by_fold) if j < k]
        if not earlier:
            print(f"fold {k}: no earlier fold to learn from, skipped")
            continue
        train, target = pd.concat(earlier, ignore_index=True), by_fold[k]
        moves = fit_predict(train, target, cols, levels, panel)
        res = target.copy()
        res[cols] = target["prev_vwap"].to_numpy()[:, None] * (1 + moves)
        (out / f"fold_{k}").mkdir(parents=True, exist_ok=True)
        res.to_parquet(out / f"fold_{k}" / "test_predictions.parquet", index=False)
        done.append(k)
        print(f"fold {k}: trained on {len(train):,} rows of folds {sorted(j for j in by_fold if j < k)}, "
              f"scored {len(res):,}")
    saved = json.loads((base / "config.json").read_text()) if (base / "config.json").exists() else {}
    (out / "config.json").write_text(json.dumps(
        {"model": "overlay", "base": str(base), "folds": done,
         "config": saved.get("config", {})}, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-overlay", description=__doc__.split("\n")[0])
    p.add_argument("base_run", type=Path)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--folds", help="comma list of target folds (default: all with an earlier one)")
    p.add_argument("--out", type=Path, help="output run dir (default: <base>-overlay)")
    a = p.parse_args(argv)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    print(f"wrote {overlay(a.base_run, a.panel, folds, a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
