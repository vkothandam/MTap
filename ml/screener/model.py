"""The screener model: a mean model and an 0.8-quantile model on log(range_from_open)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import feature_names

QUANTILE = 0.8
PARAMS = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 50,
          "l2_regularization": 1.0, "random_state": 0}


@dataclass
class ScreenerModel:
    state: str
    features: list[str]
    blank: list[str]  # features with no value in training (e.g. a panel without that group)
    mean: HistGradientBoostingRegressor
    q80: HistGradientBoostingRegressor

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """predicted range (mean model) and score (0.8-quantile model), both as fractions."""
        x = df[self.features].to_numpy(float).copy()
        x[:, [self.features.index(c) for c in self.blank]] = 0.0
        return pd.DataFrame({"pred_range": np.exp(self.mean.predict(x)),
                             "score": np.exp(self.q80.predict(x))}, index=df.index)


def fit(df: pd.DataFrame, state: str) -> ScreenerModel:
    """Fit on rows with a `range_from_open` target (log scale)."""
    feats = feature_names(state)
    x, y = df[feats].to_numpy(float).copy(), np.log(df["range_from_open"].to_numpy(float))
    blank = [c for c, empty in zip(feats, np.isnan(x).all(axis=0), strict=True) if empty]
    x[:, [feats.index(c) for c in blank]] = 0.0  # HGB cannot bin an all-NaN column
    mean = HistGradientBoostingRegressor(**PARAMS).fit(x, y)
    q80 = HistGradientBoostingRegressor(loss="quantile", quantile=QUANTILE, **PARAMS).fit(x, y)
    return ScreenerModel(state, feats, blank, mean, q80)


def save(models: dict[str, ScreenerModel], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for state, m in models.items():
        joblib.dump(m, directory / f"model_{state}.joblib")


def load(directory: Path, state: str) -> ScreenerModel:
    path = Path(directory) / f"model_{state}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path}: train that state first (screener-train --state {state})")
    return joblib.load(path)


def read_summary(directory: Path) -> dict:
    return json.loads((Path(directory) / "summary.json").read_text())
