"""panel.parquet -> walk-forward TimeSeriesDataSets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder

from .config import (
    KNOWN_REALS,
    OBSERVED_REALS,
    SENTIMENT_REALS,
    STATIC_CATEGORICALS,
    SUPPORTED_SCHEMA_VERSIONS,
    TrainConfig,
)


@dataclass(frozen=True)
class Fold:
    """Train targets < es_start; early-stop targets in [es_start, test_start); test targets in
    [test_start, test_end]. Encoders may look back across the boundaries, targets never do."""

    es_start: int
    test_start: int
    test_end: int


def read_meta(panel: Path) -> dict:
    meta_path = panel.with_name(panel.stem + ".meta.json")
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{meta_path}: schema_version {meta.get('schema_version')!r} not in "
            f"{SUPPORTED_SCHEMA_VERSIONS}; re-export the panel or update tft_vwap"
        )
    return meta


def prepare(df: pd.DataFrame, *, symbols: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Type, clean and optionally subsample a raw panel frame (one row per symbol/day_idx)."""
    df = df[df["today_vwap"] > 0].copy()
    if symbols:
        rng = np.random.default_rng(seed)
        keep = rng.choice(np.sort(df["symbol"].unique()), size=symbols, replace=False)
        df = df[df["symbol"].isin(keep)]
    for c in STATIC_CATEGORICALS:
        df[c] = df[c].fillna("unknown").astype(str)
    df["day_idx"] = df["day_idx"].astype("int64")
    reals = [*KNOWN_REALS, *OBSERVED_REALS, *SENTIMENT_REALS]
    df[reals] = df[reals].astype("float64").fillna(0.0)
    # heavy right tail (news days run 100x+ normal volume)
    df["volume_velocity"] = np.log1p(df["volume_velocity"].clip(lower=0))
    # pytorch-forecasting scores the loss in target units, so a raw-price loss is dominated by
    # high-priced stocks; 1/median price makes it roughly a % loss. Loss weight only, not input.
    df["loss_weight"] = 1.0 / df.groupby("symbol")["today_vwap"].transform("median")
    return df.sort_values(["symbol", "day_idx"]).reset_index(drop=True)


def load_panel(cfg: TrainConfig) -> tuple[pd.DataFrame, dict]:
    meta = read_meta(cfg.panel)
    return prepare(pd.read_parquet(cfg.panel), symbols=cfg.symbols, seed=cfg.seed), meta


def make_folds(df: pd.DataFrame, cfg: TrainConfig) -> list[Fold]:
    last = int(df["day_idx"].max())
    folds = []
    for k in range(cfg.n_folds):
        test_end = last - (cfg.n_folds - 1 - k) * cfg.fold_len
        test_start = test_end - cfg.fold_len + 1
        folds.append(Fold(test_start - cfg.early_stop_len, test_start, test_end))
    if folds[0].es_start - int(df["day_idx"].min()) < cfg.encoder_length + cfg.min_encoder_length:
        raise ValueError(f"not enough history before the first fold ({folds[0]})")
    return folds


def _normalizer(cfg: TrainConfig):
    transformation = "log" if cfg.target == "vwap" else None
    if cfg.normalizer == "encoder":
        return EncoderNormalizer(transformation=transformation)
    return GroupNormalizer(groups=["symbol"], transformation=transformation)


def build_datasets(
    df: pd.DataFrame, cfg: TrainConfig, fold: Fold
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """(train, early_stop, test) datasets; the latter two reuse train's encoders/normalizers."""
    train = TimeSeriesDataSet(
        df[df["day_idx"] < fold.es_start],
        time_idx="day_idx",
        target=cfg.target_col,
        group_ids=["symbol"],
        max_encoder_length=cfg.encoder_length,
        min_encoder_length=cfg.min_encoder_length,
        max_prediction_length=1,
        min_prediction_length=1,
        static_categoricals=list(STATIC_CATEGORICALS),
        time_varying_known_reals=list(KNOWN_REALS),
        time_varying_unknown_reals=[cfg.target_col, *cfg.observed_reals],
        target_normalizer=_normalizer(cfg),
        weight="loss_weight" if cfg.target == "vwap" else None,
        categorical_encoders={c: NaNLabelEncoder(add_nan=True) for c in STATIC_CATEGORICALS},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    early_stop = TimeSeriesDataSet.from_dataset(
        train, df[df["day_idx"] < fold.test_start], min_prediction_idx=fold.es_start,
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        train, df[df["day_idx"] <= fold.test_end], min_prediction_idx=fold.test_start,
        stop_randomization=True,
    )
    return train, longest_encoder(early_stop), longest_encoder(test)


def longest_encoder(ds: TimeSeriesDataSet) -> TimeSeriesDataSet:
    """One sample per (symbol, forecast day): the one with the most history. Evaluation sets
    otherwise repeat each day once per encoder length from min_encoder_length up."""
    idx = ds.decoded_index
    first = idx.groupby(["symbol", "time_idx_first_prediction"])["time_idx_first"].transform("min")
    return ds.filter(lambda d: (d["time_idx_first"] == first).to_numpy())
