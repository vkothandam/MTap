"""Forecast-run contract: what a run directory must contain for the dashboard to grade it.

    <run>/fold_<k>/test_predictions.parquet   one row per (symbol, forecast day):
        symbol, day_idx, prev_vwap, actual, q05..q95 (forecast quantiles, in price units)
        optional: date, industry_code (else joined from config.json's panel, if present)
    <run>/config.json                         optional; shown as run info
    <run>/forecast_<date>.parquet             optional; tomorrow: symbol, q05..q95 as
                                              fractional moves, optional today_vwap (the
                                              reference price, for price ranges), asof
    <run>/actuals.parquet                     optional; what happened on forecast days:
                                              date, symbol, actual_move (fractional, vs the
                                              forecast's reference), optional actual_vwap

Quantile levels come from the column names (q05 -> 0.05). Nothing here depends on the model
that wrote the run, so any model writing these files is graded the same way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

QCOL = re.compile(r"^q(\d{2})$")
REQUIRED = ("symbol", "day_idx", "prev_vwap", "actual")
TEST_GLOB = "fold_*/test_predictions.parquet"
ACTUALS = "actuals.parquet"


@dataclass
class Run:
    """Test forecasts as fractional moves: r_q (rows x levels) and r_act, plus row keys in
    `frame` (symbol, day_idx, date, industry_code, fold)."""

    path: Path
    frame: pd.DataFrame
    r_q: np.ndarray
    r_act: np.ndarray
    levels: list[float]
    config: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def has_dates(self) -> bool:
        return pd.api.types.is_datetime64_any_dtype(self.frame["date"])

    def forecasts(self) -> list[Path]:
        return sorted(self.path.glob("forecast_*.parquet"), reverse=True)


def quantile_columns(columns) -> tuple[list[str], list[float]]:
    found = sorted((int(m.group(1)), c) for c in columns if (m := QCOL.match(str(c))))
    if len(found) < 3:
        raise ValueError(f"need at least 3 quantile columns named qNN, found {len(found)}")
    return [c for _, c in found], [k / 100 for k, _ in found]


def discover(root: Path) -> list[Path]:
    """Run directories under `root` (or `root` itself), newest name first."""
    root = Path(root)
    if any(root.glob(TEST_GLOB)):
        return [root]
    return sorted({p.parent.parent for p in root.glob(f"*/{TEST_GLOB}")}, reverse=True)


def _read_config(path: Path) -> dict:
    f = path / "config.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _join_panel_keys(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fill missing date / industry_code from the panel recorded in config.json, if any."""
    need = [c for c in ("date", "industry_code") if c not in frame.columns]
    panel = Path(config.get("config", {}).get("panel", "")) if config else None
    if need and panel and panel.is_file():
        keys = pd.read_parquet(panel, columns=["symbol", "day_idx", *need])
        frame = frame.merge(keys, on=["symbol", "day_idx"], how="left")
    if "date" not in frame.columns:
        frame["date"] = frame["day_idx"]
    if "industry_code" not in frame.columns:
        frame["industry_code"] = "unknown"
    frame["industry_code"] = frame["industry_code"].fillna("unknown").astype(str)
    if not pd.api.types.is_integer_dtype(frame["date"]):
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_run(path: Path) -> Run:
    path = Path(path)
    files = sorted(path.glob(TEST_GLOB))
    if not files:
        raise FileNotFoundError(f"no {TEST_GLOB} under {path}")
    parts, qsets = [], set()
    for f in files:
        d = pd.read_parquet(f)
        missing = [c for c in REQUIRED if c not in d.columns]
        if missing:
            raise ValueError(f"{f}: missing columns {missing}")
        qsets.add(tuple(quantile_columns(d.columns)[0]))
        d["fold"] = int(f.parent.name.split("_")[1])
        parts.append(d)
    if len(qsets) > 1:
        raise ValueError(f"{path}: folds disagree on quantile columns")
    raw = pd.concat(parts, ignore_index=True)
    qcols, levels = quantile_columns(raw.columns)
    raw = raw.dropna(subset=["prev_vwap", "actual"])
    raw = raw[raw["prev_vwap"] > 0].reset_index(drop=True)
    prev = raw["prev_vwap"].to_numpy(dtype=float)
    r_q = np.sort(raw[qcols].to_numpy(dtype=float) / prev[:, None] - 1, axis=1)
    r_act = raw["actual"].to_numpy(dtype=float) / prev - 1
    config = _read_config(path)
    keep = [c for c in raw.columns if c not in qcols]
    frame = _join_panel_keys(raw[keep].copy(), config)
    frame["symbol"] = frame["symbol"].astype(str)
    return Run(path=path, frame=frame, r_q=r_q, r_act=r_act, levels=levels, config=config)


def load_forecast(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[float]]:
    """A forecast file: (keys frame, r_q as fractional moves, levels)."""
    df = pd.read_parquet(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path}: forecast has no symbol column")
    qcols, levels = quantile_columns(df.columns)
    r_q = np.sort(df[qcols].to_numpy(dtype=float), axis=1)
    keys = df[[c for c in df.columns if c not in qcols and not c.startswith(("up_", "down_"))]]
    return keys.assign(symbol=keys["symbol"].astype(str)).reset_index(drop=True), r_q, levels


@dataclass
class Live:
    """Graded live forecasts: every forecast_<date> row that has an actual. `keys` holds
    date (str), symbol, today_vwap, actual_vwap and the forecast's other key columns."""

    keys: pd.DataFrame
    r_q: np.ndarray
    r_act: np.ndarray
    levels: list[float]


def load_live(run_path: Path) -> Live | None:
    """Forecast files joined to actuals.parquet, or None if the run has no graded day yet."""
    run_path = Path(run_path)
    f = run_path / ACTUALS
    if not f.is_file():
        return None
    act = pd.read_parquet(f)
    missing = [c for c in ("date", "symbol", "actual_move") if c not in act.columns]
    if missing:
        raise ValueError(f"{f}: missing columns {missing}")
    act = act.assign(date=act["date"].astype(str), symbol=act["symbol"].astype(str))
    parts, qs, levels = [], [], None
    for p in sorted(run_path.glob("forecast_*.parquet")):
        keys, r_q, lv = load_forecast(p)
        if "date" not in keys.columns:
            keys = keys.assign(date=p.stem.removeprefix("forecast_"))
        m = (keys.assign(date=keys["date"].astype(str), _row=np.arange(len(keys)))
             .merge(act, on=["date", "symbol"], how="inner"))
        if m.empty:
            continue
        if levels is not None and lv != levels:
            raise ValueError(f"{p}: quantile levels differ from the run's other forecasts")
        levels = lv
        parts.append(m.drop(columns="_row"))
        qs.append(r_q[m["_row"].to_numpy()])
    if not parts:
        return None
    keys = pd.concat(parts, ignore_index=True)
    r_act = keys["actual_move"].to_numpy(dtype=float)
    if "today_vwap" not in keys.columns:
        keys["today_vwap"] = np.nan
    if "actual_vwap" not in keys.columns:
        keys["actual_vwap"] = keys["today_vwap"] * (1 + r_act)
    return Live(keys=keys, r_q=np.vstack(qs), r_act=r_act, levels=levels)
