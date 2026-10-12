"""Baselines the model has to beat, fitted on the training days only.

Climatology: "the VWAP ahead sits where it usually does from the last price": the training
days' quantiles of each target per 15-minute slot, the same for every stock and second. It knows
nothing about the moment, so beating it is the model's real edge.
Momentum: "the VWAP ahead equals the trailing VWAP" (MOMENTUM_REF), with the spread of the
training-day residuals (actual - reference) per 15-minute slot as its quantiles. Weaker than
climatology: the VWAP ahead stays near the last price while the trailing VWAP lags behind it.
The random walk (no move from the reference, a point forecast) and the morning daily forecast
need no fitting; the report and the dashboard build them from the prediction files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MOMENTUM_REF, QUANTILES, TARGETS

N_BUCKETS = 27  # 15-minute slots 0..25, plus 26 for the closing seconds
MIN_ROWS = 200  # a slot with fewer valid training rows uses the all-day residuals


def valid(df: pd.DataFrame, target: str) -> np.ndarray:
    return ((df["tmask"].to_numpy().astype(np.int64) >> TARGETS.index(target)) & 1) == 1


def _slot_table(res: np.ndarray, bucket: np.ndarray) -> list[list[float]]:
    overall = np.quantile(res, QUANTILES) if len(res) else np.zeros(len(QUANTILES))
    table = np.tile(overall, (N_BUCKETS, 1))
    for b in range(N_BUCKETS):
        r = res[bucket == b]
        if len(r) >= MIN_ROWS:
            table[b] = np.quantile(r, QUANTILES)
    return table.tolist()


def _fit(df: pd.DataFrame, refs: dict[str, str | None]) -> dict[str, list[list[float]]]:
    """Per target: quantiles of (actual - reference) per slot (N_BUCKETS x len(QUANTILES)),
    from panel rows whose targets sit on their decision rows. A None reference is the last
    price (0)."""
    out = {}
    bucket = df["tod_bucket"].to_numpy().astype(int)
    for t in TARGETS:
        ok = valid(df, t)
        res = df[t].to_numpy(float)
        if refs[t] is not None:
            res = res - df[refs[t]].to_numpy(float)
        out[t] = _slot_table(res[ok], bucket[ok])
    return out


def fit_momentum(df: pd.DataFrame) -> dict[str, list[list[float]]]:
    return _fit(df, MOMENTUM_REF)


def fit_climatology(df: pd.DataFrame) -> dict[str, list[list[float]]]:
    return _fit(df, dict.fromkeys(TARGETS))


def _apply(df: pd.DataFrame, target: str, table: dict, ref: str | None) -> np.ndarray:
    tab = np.asarray(table[target])
    b = np.clip(df["tod_bucket"].to_numpy().astype(int), 0, N_BUCKETS - 1)
    base = 0.0 if ref is None else df[ref].to_numpy(float)[:, None]
    return base + tab[b]


def momentum_quantiles(df: pd.DataFrame, target: str, table: dict) -> np.ndarray:
    """(rows, quantiles) momentum forecasts, as log moves from the last price."""
    return _apply(df, target, table, MOMENTUM_REF[target])


def climatology_quantiles(df: pd.DataFrame, target: str, table: dict) -> np.ndarray:
    """(rows, quantiles) climatology forecasts, as log moves from the last price."""
    return _apply(df, target, table, None)
