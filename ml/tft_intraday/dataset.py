"""The intraday panel -> walk-forward TimeSeriesDataSets.

One series per (symbol, date), one step per 15 s decision row (`step`). The panel stores each
decision's targets on the decision row itself. The TFT reads a sample's target from the row
after its encoder, so here every target (and the validity bitmask, the dataset weight) moves one
row down and each series gets one padding row at the end: the sample whose encoder ends at
decision step s predicts the targets of step s, and its encoder never contains them. The
targets are not model inputs at all; they are future values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer, NaNLabelEncoder

from .config import DAILY_REALS, GROUP_REALS, SUPPORTED_SCHEMA_VERSIONS, TARGETS, IntradayConfig

MODEL_TARGETS = [f"tgt_{t}" for t in TARGETS]
WEIGHT = "tmask_next"
KEYS = ["symbol", "date", "step"]


@dataclass(frozen=True)
class Fold:
    train_dates: list[str]
    es_dates: list[str]
    test_dates: list[str]


def read_meta(panel: Path) -> dict:
    meta = json.loads((Path(panel) / "panel.meta.json").read_text())
    if meta.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"{panel}: schema_version {meta.get('schema_version')!r} not in "
                         f"{SUPPORTED_SCHEMA_VERSIONS}; re-export the panel or update tft_intraday")
    return meta


def make_folds(meta: dict, cfg: IntradayConfig) -> list[Fold]:
    """Each test block's early-stop dates are the ones just before it, enough to reach
    es_min_rows; training gets every date before those."""
    dates = list(meta["dates"])
    rows = {d: meta["days"][d]["rows"] for d in dates}
    folds = []
    for block in cfg.test_blocks:
        missing = [d for d in block if d not in rows]
        if missing:
            raise ValueError(f"test dates {missing} are not in the panel")
        before = [d for d in dates if d < min(block)]
        es, n = [], 0
        while before and n < cfg.es_min_rows:
            es.insert(0, before.pop())
            n += rows[es[0]]
        if not before or not es:
            raise ValueError(f"not enough dates before {block} to train and early-stop")
        folds.append(Fold(before, es, sorted(block)))
    return folds


def all_symbols(panel: Path, dates: list[str]) -> list[str]:
    seen: set[str] = set()
    for d in dates:
        seen.update(pq.read_table(Path(panel) / f"date={d}" / "part.parquet",
                                  columns=["symbol"]).column("symbol").to_pylist())
    return sorted(seen)


def pick_symbols(panel: Path, dates: list[str], n: int | None, seed: int) -> list[str] | None:
    if not n:
        return None
    syms = all_symbols(panel, dates)
    return sorted(np.random.default_rng(seed).choice(syms, size=min(n, len(syms)), replace=False))


def load_days(panel: Path, dates: list[str], symbols: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for d in dates:
        filters = None if symbols is None else [("symbol", "in", list(symbols))]
        parts.append(pq.read_table(Path(panel) / f"date={d}" / "part.parquet",
                                   filters=filters).to_pandas())
    return pd.concat(parts, ignore_index=True).sort_values(KEYS, ignore_index=True)


def features(meta: dict, cfg: IntradayConfig) -> dict[str, list[str]]:
    f = meta["features"]
    observed = [c for c in f["observed_reals"] if cfg.use_group or c not in GROUP_REALS]
    static_reals = list(f["static_reals"]) + (list(DAILY_REALS) if cfg.daily_run else [])
    return {"static_categoricals": list(f["static_categoricals"]), "static_reals": static_reals,
            "known_categoricals": list(f["known_categoricals"]),
            "known_reals": list(f["known_reals"]), "observed_reals": observed}


def model_frame(df: pd.DataFrame, feats: dict, step_s: int) -> pd.DataFrame:
    """Panel rows (sorted by KEYS) -> the frame the TFT reads: inputs as they are, targets and
    their bitmask one row down, a padding row per series. Masked targets become 0 (the dataset
    rejects NaN); their weight bit is 0."""
    cats = feats["static_categoricals"] + feats["known_categoricals"]
    reals = feats["static_reals"] + feats["known_reals"] + feats["observed_reals"]
    keep = [*KEYS, "t_s", *cats, *[c for c in reals if c != "t_s"], *TARGETS, "tmask"]
    df = df[list(dict.fromkeys(keep))]
    end = (df["symbol"] != df["symbol"].shift(-1)) | (df["date"] != df["date"].shift(-1))
    pad = df[end.to_numpy()].copy()
    sf, ts = pad["session_frac"].to_numpy(float), pad["t_s"].to_numpy(float)
    pad["step"] += 1
    pad["t_s"] += step_s
    pad["minutes_to_close"] -= step_s / 60
    pad["session_frac"] = np.where(ts > 0, np.minimum(1.0, sf * (ts + step_s) / np.maximum(ts, 1)), sf)
    pad["tod_bucket"] = pad["t_s"] // 900
    out = pd.concat([df, pad], ignore_index=True).sort_values(KEYS, ignore_index=True)
    first = ((out["symbol"] != out["symbol"].shift()) | (out["date"] != out["date"].shift())).to_numpy()
    for t, m in zip(TARGETS, MODEL_TARGETS):
        v = out[t].shift(1).to_numpy(np.float32, copy=True)
        v[first] = np.nan
        out[m] = np.nan_to_num(v, nan=0.0)
    w = out["tmask"].shift(1).to_numpy(np.float32, copy=True)
    w[first] = 0
    out[WEIGHT] = np.nan_to_num(w, nan=0.0)
    out = out.drop(columns=[*TARGETS, "tmask"])
    for c in cats:
        out[c] = out[c].astype(object).where(out[c].notna(), "unknown").astype(str)
    for c in ("symbol", "date"):
        out[c] = out[c].astype(str)
    out[reals] = out[reals].astype("float32").fillna(0.0)
    out["step"] = out["step"].astype("int64")
    return out


def build_train(frame: pd.DataFrame, feats: dict, cfg: IntradayConfig) -> TimeSeriesDataSet:
    cats = feats["static_categoricals"] + feats["known_categoricals"]
    return TimeSeriesDataSet(
        frame,
        time_idx="step",
        target=list(MODEL_TARGETS),
        group_ids=["symbol", "date"],
        max_encoder_length=cfg.encoder_steps,
        min_encoder_length=cfg.min_encoder_steps,
        max_prediction_length=1,
        min_prediction_length=1,
        static_categoricals=feats["static_categoricals"],
        static_reals=feats["static_reals"],
        time_varying_known_categoricals=feats["known_categoricals"],
        time_varying_known_reals=feats["known_reals"],
        time_varying_unknown_reals=feats["observed_reals"],
        # per symbol, never per (symbol, date): that would scale by the day's own targets
        target_normalizer=MultiNormalizer(
            [GroupNormalizer(groups=["symbol"]) for _ in MODEL_TARGETS]),
        weight=WEIGHT,
        # unseen dates and symbols (every test date is unseen) encode as "nan"
        categorical_encoders={c: NaNLabelEncoder(add_nan=True) for c in ["symbol", "date", *cats]},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


def evaluation_set(train: TimeSeriesDataSet, frame: pd.DataFrame, stride_steps: int
                   ) -> TimeSeriesDataSet:
    """One sample per decision on the stride grid (decision step % stride_steps == 0), with
    the longest encoder the series has."""
    ds = TimeSeriesDataSet.from_dataset(train, frame, stop_randomization=True)

    def keep(idx: pd.DataFrame) -> np.ndarray:
        first = idx.groupby(["symbol", "date", "time_idx_first_prediction"], observed=True)[
            "time_idx_first"].transform("min")
        decision = idx["time_idx_first_prediction"] - 1
        return ((idx["time_idx_first"] == first) & (decision % stride_steps == 0)).to_numpy()

    return ds.filter(keep)
