"""Intraday-run contract: what a run directory must contain for the Intraday section.

    <run>/config.json                         model == "tft_intraday", targets, horizon_s, folds,
                                              daily_run, step_s, eval_stride_s ...
    <run>/fold_<k>/intraday_<target>.parquet  one row per (symbol, date, t_s) decision point on
                                              the fold's test dates:
        symbol, date (YYYY-MM-DD), t_s (seconds since the 09:30 ET open), tod_bucket (15-minute
        slot 0..25), sector_code (may be null), rvol, mask (1 = valid answer), actual, q05..q95,
        clim_q05..clim_q95 (climatology baseline; older runs lack it), mom_q05..mom_q95
        (momentum baseline), ref_trail, ref_session, y, last, prev_vwap
        (r_close only: daily_q05..daily_q95 and has_daily)
    <run>/report/by_target.parquet            optional; precomputed per-target scores

actual / q* / clim_q* / mom_q* / ref_* are log moves from the last price at t (the last price is 0).
Targets: r_1m, r_5m, r_15m, r_60m (VWAP of the next 1/5/15/60 minutes), r_rod (VWAP from t to the
close) and r_close (the whole day's VWAP, as a move from the last price at t).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

MODEL = "tft_intraday"
PRED_GLOB = "fold_*/intraday_*.parquet"
LEVELS = [round(k * 0.05, 2) for k in range(1, 20)]
QCOLS = [f"q{round(lv * 100):02d}" for lv in LEVELS]
MOMCOLS = [f"mom_{c}" for c in QCOLS]
CLIMCOLS = [f"clim_{c}" for c in QCOLS]
DAILYCOLS = [f"daily_{c}" for c in QCOLS]


def discover_intraday(root: Path) -> list[Path]:
    """Intraday run directories under `root` (or `root` itself), newest name first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for p in [root, *(d for d in root.iterdir() if d.is_dir())]:
        f = p / "config.json"
        if f.is_file() and any(p.glob(PRED_GLOB)):
            try:
                if json.loads(f.read_text()).get("model") == MODEL:
                    found.append(p)
            except (OSError, ValueError):
                continue
    return sorted(set(found), key=lambda p: p.name, reverse=True)


def load_config(run: Path) -> dict:
    return json.loads((Path(run) / "config.json").read_text())


def targets_of(run: Path) -> list[str]:
    """Targets that have predictions, in the config's order."""
    have = {f.stem.removeprefix("intraday_") for f in Path(run).glob(PRED_GLOB)}
    order = load_config(run).get("targets", [])
    return [t for t in order if t in have] + sorted(have - set(order))


def load_intraday(run: Path, target: str, columns: list[str] | None = None,
                  valid_only: bool = True) -> pd.DataFrame:
    """One target's test rows from every fold, with a `fold` column. By default only rows with
    mask == 1 (a valid answer). `columns` limits what is read (mask is always read); names a
    file does not have are skipped."""
    run = Path(run)
    cols = None if columns is None else list(dict.fromkeys([*columns, "mask"]))
    parts = []
    for f in sorted(run.glob(f"fold_*/intraday_{target}.parquet"),
                    key=lambda p: int(p.parent.name.removeprefix("fold_"))):
        have = set(pq.read_schema(f).names)
        d = pd.read_parquet(f, columns=None if cols is None else [c for c in cols if c in have])
        d.insert(0, "fold", int(f.parent.name.removeprefix("fold_")))
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"{run}: no intraday_{target}.parquet under fold_*/")
    df = pd.concat(parts, ignore_index=True)
    if valid_only:
        df = df[df["mask"] == 1].reset_index(drop=True)
    if columns is not None and "mask" not in columns:
        df = df.drop(columns="mask")
    return df


def load_report(run: Path) -> dict[str, pd.DataFrame]:
    """Every report/*.parquet by file stem ({} if there is no report)."""
    d = Path(run) / "report"
    return {f.stem: pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))} if d.is_dir() else {}
