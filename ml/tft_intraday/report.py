"""Scores of an intraday run's test forecasts.

    uv run tft-intraday-report runs/intraday/<run_id>

Writes <run>/report/{by_target, by_day, by_tod}.parquet, one row per (target, model[, key]):
    n, coverage80 (share inside P10-P90), pinball (mean over P5..P95), skill_clim / skill_rw /
    skill_mom / skill_daily (1 - pinball / the baseline's, on the same rows), dir_last / dir_trail /
    dir_session (the median called the side of that reference right), pit_bias (0.5 - mean
    PIT; > 0: actuals land low in the band), width80_med
by_target adds block-bootstrap 90% intervals for skill_clim, skill_rw and skill_mom. Forecasts made every
minute overlap (a 60-minute window is shared by 60 of them), so rows are resampled in blocks of
one session hour per day, not one by one.

Models: tft (q*), climatology (clim_q*), momentum (mom_q*) and, for r_close, the morning daily
forecast (daily_q*, rows with has_daily). The random walk is the point "no move from the
reference". Climatology is the bar that matters (baselines.py); dir_trail flatters every model
for the same reason momentum is weak, so dir_last is the direction call to read.
Everything is graded against the trailing VWAP (`ref_trail`), like the dashboard's default;
pinball and coverage do not depend on the reference, only the direction hits do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import QUANTILES, TARGETS

QCOLS = [f"q{round(q * 100):02d}" for q in QUANTILES]
MODELS = {"tft": QCOLS, "climatology": [f"clim_{c}" for c in QCOLS],
          "momentum": [f"mom_{c}" for c in QCOLS], "daily": [f"daily_{c}" for c in QCOLS]}
BASELINES = {"pin_clim": "climatology", "pin_mom": "momentum"}
REFS = {"last": None, "trail": "ref_trail", "session": "ref_session"}
BOOT = 1000
LEVELS = np.asarray(QUANTILES)


def load_target(run: Path, target: str) -> pd.DataFrame:
    parts = []
    for f in sorted(Path(run).glob(f"fold_*/intraday_{target}.parquet")):
        d = pd.read_parquet(f)
        d.insert(0, "fold", int(f.parent.name.removeprefix("fold_")))
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df[df["mask"] == 1].reset_index(drop=True)


def pinball(q: np.ndarray, act: np.ndarray) -> np.ndarray:
    err = act[:, None] - q
    return np.maximum(LEVELS * err, (LEVELS - 1) * err).mean(axis=1)


def row_metrics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    q = np.sort(df[cols].to_numpy(float), axis=1)
    act = df["actual"].to_numpy(float)
    trail = df["ref_trail"].to_numpy(float)
    out = {
        "inside80": ((q[:, 1] <= act) & (act <= q[:, 17])).astype(float),
        "pin": pinball(q, act),
        "pin_rw": pinball(np.repeat(trail[:, None], len(cols), axis=1), act),
        "u": ((q < act[:, None]).sum(axis=1) + 0.5) / (len(cols) + 1),
        "width80": q[:, 17] - q[:, 1],
    }
    for key, model in BASELINES.items():  # NaN for runs written before a baseline existed
        cols_b = MODELS[model]
        out[key] = (pinball(np.sort(df[cols_b].to_numpy(float), axis=1), act)
                    if all(c in df for c in cols_b) else np.full(len(df), np.nan))
    for name, col in REFS.items():
        m = np.zeros(len(df)) if col is None else df[col].to_numpy(float)
        a, p = act - m, q[:, 9] - m
        out[f"dir_{name}"] = np.where(a == 0, np.nan, (np.sign(p) == np.sign(a)).astype(float))
    if "has_daily" in df and all(c in df for c in MODELS["daily"]):
        d = np.sort(df[MODELS["daily"]].to_numpy(float), axis=1)
        out["pin_daily"] = np.where(df["has_daily"].to_numpy() == 1, pinball(d, act), np.nan)
    return pd.DataFrame(out, index=df.index)


def _agg(m: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = m.groupby(by, observed=True)
    out = g.agg(n=("pin", "size"), coverage80=("inside80", "mean"), pinball=("pin", "mean"),
                _p=("pin", "sum"), _rw=("pin_rw", "sum"), _mom=("pin_mom", "sum"),
                _clim=("pin_clim", "sum"),
                pit_bias=("u", lambda u: 0.5 - u.mean()), width80_med=("width80", "median"),
                dir_last=("dir_last", "mean"), dir_trail=("dir_trail", "mean"),
                dir_session=("dir_session", "mean"))
    out["skill_clim"] = 1 - out["_p"] / out["_clim"].where(out["_clim"] > 0)
    out["skill_rw"] = 1 - out["_p"] / out["_rw"].where(out["_rw"] > 0)
    out["skill_mom"] = 1 - out["_p"] / out["_mom"].where(out["_mom"] > 0)
    if "pin_daily" in m:
        both = m[m["pin_daily"].notna()].groupby(by, observed=True)
        out["skill_daily"] = 1 - both["pin"].sum() / both["pin_daily"].sum()
    else:
        out["skill_daily"] = np.nan
    return out.drop(columns=["_p", "_rw", "_mom", "_clim"]).reset_index()


def bootstrap(m: pd.DataFrame, blocks: np.ndarray, seed: int = 0) -> dict:
    """90% intervals of skill_clim / skill_rw / skill_mom, resampling whole blocks."""
    codes, inv = np.unique(blocks, return_inverse=True)
    sums = {k: np.bincount(inv, weights=m[k].to_numpy(float), minlength=len(codes))
            for k in ("pin", "pin_rw", "pin_mom", "pin_clim")}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(codes), size=(BOOT, len(codes)))
    p = sums["pin"][pick].sum(axis=1)
    out = {}
    for name, base in (("skill_clim", "pin_clim"), ("skill_rw", "pin_rw"),
                       ("skill_mom", "pin_mom")):
        s = 1 - p / sums[base][pick].sum(axis=1)
        out[f"{name}_lo"], out[f"{name}_hi"] = (float(x) for x in np.quantile(s, [0.05, 0.95]))
    out["blocks"] = len(codes)
    return out


def score_target(df: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """by_target / by_day / by_tod rows of one target, for every model it has columns for."""
    frames: dict[str, list[pd.DataFrame]] = {"by_target": [], "by_day": [], "by_tod": []}
    if df.empty:
        return {k: pd.DataFrame() for k in frames}
    blocks = df["date"].astype(str).to_numpy() + "/" + (df["t_s"] // 3600).astype(str).to_numpy()
    for model, cols in MODELS.items():
        if not all(c in df for c in cols):
            continue
        d = df[df["has_daily"] == 1] if model == "daily" else df
        if d.empty:
            continue
        m = row_metrics(d, cols)
        m["_all"], m["date"], m["tod_bucket"] = 0, d["date"].to_numpy(), d["tod_bucket"].to_numpy()
        top = _agg(m, ["_all"]).drop(columns="_all")
        for k, v in bootstrap(m, blocks[d.index.to_numpy()]).items():
            top[k] = v
        frames["by_target"].append(top.assign(target=target, model=model))
        frames["by_day"].append(_agg(m, ["date"]).assign(target=target, model=model))
        frames["by_tod"].append(_agg(m, ["tod_bucket"]).assign(target=target, model=model))
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in frames.items()}


def build_report(run: Path) -> dict[str, pd.DataFrame]:
    run = Path(run)
    targets = json.loads((run / "config.json").read_text()).get("targets", list(TARGETS))
    tables: dict[str, list[pd.DataFrame]] = {"by_target": [], "by_day": [], "by_tod": []}
    for t in targets:
        for k, v in score_target(load_target(run, t), t).items():
            if not v.empty:
                tables[k].append(v)
    out = {}
    (run / "report").mkdir(exist_ok=True)
    for k, parts in tables.items():
        if not parts:
            continue
        t = pd.concat(parts, ignore_index=True)
        first = ["target", "model"]
        t = t[first + [c for c in t.columns if c not in first]]
        t.to_parquet(run / "report" / f"{k}.parquet", index=False)
        out[k] = t
    return out


def format_summary(by_target: pd.DataFrame) -> str:
    cols = ["target", "model", "n", "coverage80", "pinball", "skill_clim", "skill_clim_lo",
            "skill_clim_hi", "skill_rw", "skill_rw_lo",
            "skill_rw_hi", "skill_mom", "skill_mom_lo", "skill_mom_hi", "skill_daily",
            "dir_last", "dir_trail", "dir_session", "pit_bias", "width80_med"]
    t = by_target[[c for c in cols if c in by_target]].copy()
    t["pinball"] = t["pinball"] * 1e4  # basis points
    t["width80_med"] = t["width80_med"] * 1e4
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        return ("test scores (pinball, width80_med in bp; skills are 1 - pinball / baseline's, "
                "with 90% block-bootstrap intervals)\n" + t.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-intraday-report", description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path)
    a = p.parse_args(argv)
    out = build_report(a.run)
    if "by_target" in out:
        print(format_summary(out["by_target"]))
    print(f"report: {a.run / 'report'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
