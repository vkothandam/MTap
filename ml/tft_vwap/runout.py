"""Write a non-TFT model's out-of-fold forecasts in the run contract the dashboard, compare and
grade read: config.json, fold_k/{test_predictions.parquet, metrics.json}, summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .compare import qlevels
from .evaluate import metrics


def write_fold(out: Path, k: int, frame: pd.DataFrame, extra: dict | None = None) -> dict:
    d = Path(out) / f"fold_{k}"
    d.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(d / "test_predictions.parquet", index=False)
    m = {"test": metrics(frame, list(qlevels(frame))), **(extra or {})}
    (d / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    return m


def write_summary(out: Path, folds: dict[int, pd.DataFrame], extra: dict | None = None) -> None:
    pooled = pd.concat(folds.values(), ignore_index=True)
    levels = list(qlevels(pooled))
    (Path(out) / "summary.json").write_text(json.dumps(
        {"pooled_test": metrics(pooled, levels),
         "folds": {k: {"test": metrics(f, levels)} for k, f in folds.items()}, **(extra or {})},
        indent=2, default=str))
