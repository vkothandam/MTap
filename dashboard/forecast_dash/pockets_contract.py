"""Pockets-run contract: what an intraday-ft run directory must contain for Intraday → Pockets.

    <run>/config.json                         model == "intraday_ft", label, folds_def ...
    <run>/MODEL.md                            optional; the run card (label, purpose, result)
    <run>/report.json                         optional; the run's own report (intraday-ft-report)
    <run>/fold_<k>/intraday_ft_pred.parquet   one row per decision (stock, day, minute) on the test months:
        date, minute (0 = 09:30 ET), stock_id, symbol, sector, industry, cap_band, live_set,
        delisted_later, poll_tier, sigma, log_dvol, dec_row (row in quantiles.npy);
        <readout>__{none,sec,ind,all}: p_mass_up60, p_bulge60, p_sag60, net60_p10 / p50 / p90,
        t50_up_p50, t50_dn_p50, p_burst_up / dn, p_race_up / dn (the 4 overlays);
        real_<target>: the realised targets (real_<t>_b<k> per bin, real_<t>60 per row)
    <run>/fold_<k>/quantiles.npy              (decisions, targets, 5) float16: P10 / P25 / P50 / P75 / P90 of
                                              every target (the `all` overlay)
    <run>/fold_<k>/quantiles.json             {"taus": [...], "targets": [...], "row": "dec_row"}
    <run>/fold_<k>/recal.parquet              optional; per (date, target): o10 .. o90 offsets fitted on
                                              the previous sessions' outcomes, lo / hi clip, n_days
    <run>/fold_<k>/clim.parquet               optional; the training span's quantiles q10 .. q90 per
                                              (cell, target), cell -1 = all; sigma_cuts, dvol_cuts
    <run>/fold_<k>/clim_rates.parquet         optional; the event rates per (cell, event), cell -1 = all
    <run>/fold_<k>/extremes.parquet           optional; per dec_row and bin k: hi_b<k> / lo_b<k> = ln(max
                                              high / P0), ln(min low / P0) in the bin, t_hi_b<k> / t_lo_b<k> =
                                              their minute after the decision (-1 = none); `intraday-ft-extremes`
    (cell = (minute // 5) * 9 + digitize(sigma, sigma_cuts) * 3 + digitize(log_dvol, dvol_cuts);
    a run scored before 2026-11-07 23:00 has none of the three: re-run `intraday-ft-report` on it)

Units: bin targets are per duration bin b1 0-5, b2 5-15, b3 15-30, b4 30-60, b5 60 min-close after the
entry; prices in the stock's sigma from the entry price P0 (cm_up / cm_dn: centre of mass of the volume
above / below P0; bulge / sag: volume share x distance; net = bulge - sag), mass_up / occ_* shares 0..1,
rvol = log(bin volume / normal).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "intraday_ft"
PRED = "intraday_ft_pred.parquet"
TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
QLABELS = ("P10", "P25", "P50", "P75", "P90")
OCOLS = [f"o{round(t * 100)}" for t in TAUS]
BINS = {1: "0-5 min", 2: "5-15 min", 3: "15-30 min", 4: "30-60 min", 5: "60 min-close"}
OVERLAYS = {"none": "none", "sec": "+ sector", "ind": "+ industry", "all": "+ symbol"}
# TRBC economic sectors (the run file stores the 2-digit code; display only)
SECTORS = {"50": "Energy", "51": "Basic Materials", "52": "Industrials", "53": "Consumer Cyclicals",
           "54": "Consumer Non-Cyclicals", "55": "Financials", "56": "Healthcare", "57": "Technology",
           "59": "Utilities", "60": "Real Estate", "62": "Government Activity",
           "63": "Academic & Educational Services"}
KEYS = ["date", "minute", "stock_id", "symbol", "sector", "industry", "cap_band", "live_set",
        "delisted_later", "poll_tier", "sigma", "log_dvol", "dec_row"]


def discover_pockets(root: Path) -> list[Path]:
    """intraday-ft run directories under `root` (or `root` itself), newest name first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for p in [root, *(d for d in root.iterdir() if d.is_dir())]:
        f = p / "config.json"
        if f.is_file() and any(p.glob(f"fold_*/{PRED}")):
            try:
                if json.loads(f.read_text()).get("model") == MODEL:
                    found.append(p)
            except (OSError, ValueError):
                continue
    return sorted(set(found), key=lambda p: p.name, reverse=True)


def folds_of(run: Path) -> list[str]:
    return sorted(p.parent.name.removeprefix("fold_") for p in Path(run).glob(f"fold_*/{PRED}"))


def load_config(run: Path) -> dict:
    return json.loads((Path(run) / "config.json").read_text())


def load_report(run: Path) -> dict | None:
    f = Path(run) / "report.json"
    return json.loads(f.read_text()) if f.is_file() else None


def targets_of(run: Path, fold: str) -> list[str]:
    """The stored quantile targets of a fold, in quantiles.npy order."""
    return json.loads((Path(run) / f"fold_{fold}" / "quantiles.json").read_text())["targets"]


def load_pred(run: Path, fold: str, columns: list[str] | None = None) -> pd.DataFrame:
    """A fold's run file; `columns` limits what is read (names the file lacks are skipped)."""
    import pyarrow.parquet as pq

    f = Path(run) / f"fold_{fold}" / PRED
    have = pq.read_schema(f).names
    return pd.read_parquet(f, columns=None if columns is None else [c for c in columns if c in have])


def quantiles(run: Path, fold: str, rows: np.ndarray, targets: list[str]) -> np.ndarray:
    """(len(rows), len(targets), 5) float64 quantiles for the given dec_rows (memory-mapped read)."""
    Q = np.load(Path(run) / f"fold_{fold}" / "quantiles.npy", mmap_mode="r")
    names = targets_of(run, fold)
    idx = [names.index(t) for t in targets]
    rows = np.asarray(rows)
    if len(rows) > len(Q) // 50:      # most rows: read the target columns whole, then pick
        return np.asarray(Q[:, idx], np.float64)[rows]
    order = np.argsort(rows)          # a few rows: read just them, in file order
    out = np.asarray(Q[rows[order]][:, idx], np.float64)
    res = np.empty_like(out)
    res[order] = out
    return res


def _optional(run: Path, fold: str, name: str) -> pd.DataFrame | None:
    f = Path(run) / f"fold_{fold}" / name
    return pd.read_parquet(f) if f.is_file() else None


def load_recal(run: Path, fold: str) -> pd.DataFrame | None:
    return _optional(run, fold, "recal.parquet")


def load_clim(run: Path, fold: str) -> pd.DataFrame | None:
    return _optional(run, fold, "clim.parquet")


def load_clim_rates(run: Path, fold: str) -> pd.DataFrame | None:
    return _optional(run, fold, "clim_rates.parquet")


def load_extremes(run: Path, fold: str) -> pd.DataFrame | None:
    """The realised high / low after each decision, indexed by dec_row (None if not computed)."""
    e = _optional(run, fold, "extremes.parquet")
    return None if e is None else e.set_index("dec_row")


def recalibrate(q: np.ndarray, rc: pd.DataFrame | None, target: str, dates) -> np.ndarray:
    """q (rows, 5) + the run's per-day offsets for `target`, sorted and clipped to its range. Days
    without offsets (the first sessions of a fold, or no recal.parquet) keep the raw quantiles."""
    if rc is None:
        return q
    r = rc[rc.target == target]
    if r.empty:
        return q
    off = r.set_index("date")[OCOLS].reindex(pd.Index(dates).astype(str)).fillna(0.0).to_numpy(np.float64)
    lo, hi = float(r.lo.iloc[0]), float(r.hi.iloc[0])
    return np.clip(np.sort(q + off, axis=-1), lo, hi)


def sector_label(code) -> str:
    """'57' -> '57 Technology'; blank -> 'unknown'."""
    c = str(code).strip()
    if c in ("", "None", "nan"):
        return "unknown"
    return f"{c} {SECTORS[c]}" if c in SECTORS else c
