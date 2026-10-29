"""panel.parquet -> walk-forward TimeSeriesDataSets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder

from .config import (
    BREAK_GAP,
    BREAK_JUMP,
    EARNINGS_FAR,
    EARNINGS_OVERDUE,
    EARNINGS_REALS,
    EARNINGS_SOURCES,
    EARNINGS_WINDOWS,
    KNOWN_REALS,
    OBSERVED_REALS,
    PM_GROUPS,
    SENTIMENT_REALS,
    STATIC_CATEGORICALS,
    SUPPORTED_SCHEMA_VERSIONS,
    UNSCALED_REALS,
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


def prepare(df: pd.DataFrame, *, symbols: int | None = None, seed: int = 42,
            series_breaks: bool = True, extra: tuple[str, ...] | list[str] = ()) -> pd.DataFrame:
    """Type, clean and optionally subsample a raw panel frame (one row per symbol/day_idx).
    `extra` names the schema-4 columns of the enabled input groups (TrainConfig.extra_reals):
    only those are read, so a schema-3 panel works with every group off."""
    missing = [c for c in extra if c not in df.columns]
    if missing:
        raise ValueError(f"panel lacks columns {missing} that the enabled input groups need "
                         "(schema 4 panel required)")
    df = df[df["today_vwap"] > 0].copy()
    if symbols:
        rng = np.random.default_rng(seed)
        keep = rng.choice(np.sort(df["symbol"].unique()), size=symbols, replace=False)
        df = df[df["symbol"].isin(keep)]
    if series_breaks:
        df = split_series(df)
    for c in STATIC_CATEGORICALS:
        df[c] = df[c].fillna("unknown").astype(str)
    df["day_idx"] = df["day_idx"].astype("int64")
    # earnings calendar: no release yet (NULL) reads as far from one; one-hot both windows
    df["days_since_results"] = df["days_since_results"].fillna(EARNINGS_FAR).clip(0, EARNINGS_FAR)
    for c in ("days_to_results", "days_to_results_est"):
        df[c] = df[c].fillna(EARNINGS_FAR).clip(EARNINGS_OVERDUE, EARNINGS_FAR)
    for mode, source in EARNINGS_SOURCES.items():
        suffix = "_pit" if mode == "pit" else ""
        for window, col in EARNINGS_WINDOWS.items():
            df[col + suffix] = (df[source] == window).astype("float64")
    earnings = list(dict.fromkeys(c for cols in EARNINGS_REALS.values() for c in cols))
    reals = [*KNOWN_REALS, *OBSERVED_REALS, *earnings, *SENTIMENT_REALS, *extra]
    df[reals] = df[reals].astype("float64").fillna(0.0)
    # heavy right tail (news days run 100x+ normal volume)
    df["volume_velocity"] = np.log1p(df["volume_velocity"].clip(lower=0))
    # pytorch-forecasting scores the loss in target units, so a raw-price loss is dominated by
    # high-priced stocks; 1/median price makes it roughly a % loss. Loss weight only, not input.
    df["loss_weight"] = 1.0 / df.groupby("symbol")["today_vwap"].transform("median")
    return df.sort_values(["symbol", "day_idx"]).reset_index(drop=True)


def split_series(df: pd.DataFrame) -> pd.DataFrame:
    """Split each symbol at its series breaks (BREAK_GAP, BREAK_JUMP): the latest part keeps
    the ticker, earlier ones become SYMBOL~1, ~2 ... counting back, so no encoder spans a
    break. The move across a break is unknown, like a series' first bar."""
    df = df.sort_values(["symbol", "day_idx"])
    by = df.groupby("symbol")
    move = 1 + df["vwap_pct_prev_day"]
    brk = (by["day_idx"].diff() > BREAK_GAP) | (move >= BREAK_JUMP) | (move <= 1 / BREAK_JUMP)
    later = brk.groupby(df["symbol"]).transform("sum") - brk.groupby(df["symbol"]).cumsum()
    df.loc[brk, "vwap_pct_prev_day"] = np.nan
    df["symbol"] = df["symbol"].where(later == 0, df["symbol"] + "~" + later.astype(str))
    return df


def load_panel(cfg: TrainConfig) -> tuple[pd.DataFrame, dict]:
    meta = read_meta(cfg.panel)
    df = prepare(pd.read_parquet(cfg.panel), symbols=cfg.symbols, seed=cfg.seed,
                 series_breaks=cfg.series_breaks, extra=cfg.extra_reals)
    return df, meta


MIN_TRAIN_SESSIONS = 252


def make_folds(df: pd.DataFrame, cfg: TrainConfig) -> list[Fold]:
    last = int(df["day_idx"].max())
    first = int(df["day_idx"].min())
    if cfg.test_end is not None:
        if cfg.test_end > last:
            raise ValueError(f"test_end {cfg.test_end} is past the panel's last session {last}")
        last = cfg.test_end
    n_folds = cfg.n_folds
    if not n_folds:  # as many blocks as fit after the minimum training span + first early-stop
        need = (cfg.min_train_sessions or MIN_TRAIN_SESSIONS) + cfg.early_stop_len
        n_folds = (last - (first + need) + 1) // cfg.fold_len
        if n_folds < 1:
            raise ValueError(f"panel too short for one fold of {cfg.fold_len} sessions after "
                             f"{need} sessions of history")
    folds = []
    for k in range(n_folds):
        test_end = last - (n_folds - 1 - k) * cfg.fold_len
        test_start = test_end - cfg.fold_len + 1
        folds.append(Fold(test_start - cfg.early_stop_len, test_start, test_end))
    if folds[0].es_start - first < cfg.encoder_length + cfg.min_encoder_length:
        raise ValueError(f"not enough history before the first fold ({folds[0]})")
    if cfg.min_train_sessions and folds[0].es_start - first < cfg.min_train_sessions:
        raise ValueError(f"first fold ({folds[0]}) trains on fewer than "
                         f"{cfg.min_train_sessions} sessions")
    return folds


def _normalizer(cfg: TrainConfig):
    transformation = "log" if cfg.target == "vwap" else None
    if cfg.normalizer == "encoder":
        return EncoderNormalizer(transformation=transformation)
    return GroupNormalizer(groups=["symbol"], transformation=transformation)


def static_categoricals(train_df: pd.DataFrame, cfg: TrainConfig) -> list[str]:
    """STATIC_CATEGORICALS, less those with one value in the training rows when
    cfg.drop_constant_statics (symbol always stays: it is the series id)."""
    if not cfg.drop_constant_statics:
        return list(STATIC_CATEGORICALS)
    return [c for c in STATIC_CATEGORICALS if c == "symbol" or train_df[c].nunique() > 1]


def build_datasets(
    df: pd.DataFrame, cfg: TrainConfig, fold: Fold
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """(train, early_stop, test) datasets; the latter two reuse train's encoders/normalizers."""
    train_df = df[df["day_idx"] < fold.es_start]
    statics = static_categoricals(train_df, cfg)
    train = TimeSeriesDataSet(
        train_df,
        time_idx="day_idx",
        target=cfg.target_col,
        group_ids=["symbol"],
        max_encoder_length=cfg.encoder_length,
        min_encoder_length=cfg.min_encoder_length,
        max_prediction_length=1,
        min_prediction_length=1,
        static_categoricals=statics,
        time_varying_known_reals=cfg.known_reals,
        time_varying_unknown_reals=[cfg.target_col, *cfg.observed_reals],
        target_normalizer=_normalizer(cfg),
        weight="loss_weight" if cfg.target == "vwap" else None,
        categorical_encoders={c: NaNLabelEncoder(add_nan=True) for c in statics},
        # pre-market values and flags stay raw: 0 means blank (see PmMaskCollate)
        scalers={c: None for c in UNSCALED_REALS if c in (*cfg.known_reals, *cfg.observed_reals)},
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
    if cfg.dedupe_train:  # else each series end repeats once per encoder length
        train = longest_encoder(train, copy=False)
    return train, longest_encoder(early_stop), longest_encoder(test)


def longest_encoder(ds: TimeSeriesDataSet, copy: bool = True) -> TimeSeriesDataSet:
    """One sample per (symbol, forecast day): the one with the most history. Evaluation sets
    otherwise repeat each day once per encoder length from min_encoder_length up. Next to
    missing timesteps pytorch-forecasting can also emit the same sample twice (seen on the
    5-year panel: WALD, gaps around day 750), so ties keep only the first."""
    idx = ds.decoded_index
    key = ["symbol", "time_idx_first_prediction"]
    first = idx.groupby(key)["time_idx_first"].transform("min")
    keep = (idx["time_idx_first"] == first).to_numpy().copy()
    keep[keep] = ~idx[keep].duplicated(key).to_numpy()
    return ds.filter(lambda d: keep, copy=copy)


STATE_RATES = {"night": {"px": 1.0, "full": 1.0}, "px": {"px": 0.0, "full": 1.0},
               "full": {"px": 0.0, "full": 0.0}}


class PmMaskCollate:
    """Collate function that blanks the pre-market inputs of the decoder step (values 0 and
    the group's flag 0) for a share of the samples, so one model learns to forecast with and
    without them. `rates` maps a group ("px", "full") to its share; a sample whose px group is
    blanked has its full group blanked too (full is never on without prices). The encoder
    history keeps its real values (live, history is always known).
    deterministic=True picks the samples by a hash of (symbol, forecast day) instead of at
    random: the same samples every epoch, for a comparable early-stop loss."""

    def __init__(self, rates: dict[str, float], cols_idx: dict[str, list[int]],
                 deterministic: bool = False):
        self.rates = dict(rates)
        self.cols_idx = {g: list(c) for g, c in cols_idx.items()}
        self.deterministic = deterministic

    @classmethod
    def for_dataset(cls, ds: TimeSeriesDataSet, rates: dict[str, float],
                    deterministic: bool = False):
        idx = {g: [ds.reals.index(c) for c in cols if c in ds.reals]
               for g, cols in PM_GROUPS.items()}
        idx = {g: c for g, c in idx.items() if c}
        return cls({g: r for g, r in rates.items() if g in idx}, idx, deterministic)

    @classmethod
    def for_state(cls, ds: TimeSeriesDataSet, state: str):
        """Forces a scoring state: night blanks both groups, px blanks full, full blanks none."""
        return cls.for_dataset(ds, STATE_RATES[state])

    def _u(self, x) -> torch.Tensor:
        """One uniform [0, 1) draw per sample, shared by the groups so px blank is a subset of
        full blank (full never on without prices)."""
        n = len(x["decoder_cont"])
        if self.deterministic:
            g = x["groups"][:, 0].long()
            t = x["decoder_time_idx"][:, 0].long()
            return (((g * 73856093) ^ (t * 19349663)).abs() % 1000).float() / 1000
        return torch.rand(n)

    def __call__(self, batch):
        x, y = TimeSeriesDataSet._collate_fn(batch)
        dc = x["decoder_cont"]
        u = self._u(x)
        px = self.rates.get("px", 0.0)
        # full is blank for its own share, which includes every px-blank sample
        rate = {"px": px, "full": max(self.rates.get("full", 0.0), px)}
        for g, cols_idx in self.cols_idx.items():
            hit = u < rate[g] if rate[g] < 1 else torch.ones_like(u, dtype=torch.bool)
            cols = torch.zeros(dc.shape[-1], dtype=torch.bool)
            cols[cols_idx] = True
            dc = dc.masked_fill(hit[:, None, None] & cols[None, None, :], 0.0)
        x["decoder_cont"] = dc
        return x, y
