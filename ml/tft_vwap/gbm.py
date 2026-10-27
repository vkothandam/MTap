"""R7: plain gradient-boosted quantile baseline, no neural net (does the TFT earn its cost?).

    uv run tft-vwap-gbm --panel PANEL --n-folds 0 --min-train-sessions 252 --fold-len 63 \
        --early-stop-len 21 --market --macro --after-hours --session-shape [--folds 1,3,5]
        [--premarket on --pm-groups px,full]

Same panel, input-group flags and fold boundaries as `tft-vwap` (make_folds), same run contract.
One sample per (symbol, forecast day d): the features are the last session's inputs (row d-1:
the enabled observed groups, incl. earnings) plus lags 0, 1, 2, 5, 20 of vwap_pct_prev_day, and
the known-future inputs of d (calendar and, when enabled, the pre-market groups with their flags
from row d). The target is the move vs the last VWAP. One HistGradientBoostingRegressor(
loss="quantile") per quantile, fitted on forecast days before the fold's early-stop block
(no early stopping: a fixed `--max-iter`), outputs sorted so none cross. Writes
runs/<stamp>-gbm-<n>sym/{config.json, summary.json, fold_k/{test_predictions,metrics}} with the
TFT runs' columns, so compare, the dashboard and overlay read it like a TFT run.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import EARNINGS_MODES, KNOWN_REALS, PM_GROUP_CHOICES, PREMARKET_MODES, TrainConfig
from .dataset import Fold, load_panel, make_folds
from .evaluate import qcol
from .runout import write_fold, write_summary

LAGS = (0, 1, 2, 5, 20)  # of vwap_pct_prev_day, relative to the last session


def build_features(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """One row per (symbol, forecast day d) with a real row at d-1 and at d: key columns
    (symbol, day_idx = d), the label inputs (prev_vwap, actual), `X` columns f_*."""
    obs = [c for c in dict.fromkeys([*cfg.observed_reals, "vwap_pct_prev_day"])]
    known = [*KNOWN_REALS, *cfg.pm_reals]
    last = df[["symbol", "day_idx", "today_vwap", "target_tomorrow_vwap", *obs]].copy()
    ret = df.set_index(["symbol", "day_idx"])["vwap_pct_prev_day"]
    out = last.rename(columns={c: f"f_{c}" for c in obs})
    for lag in LAGS[1:]:
        idx = pd.MultiIndex.from_arrays([last["symbol"], last["day_idx"] - lag])
        out[f"f_ret_lag{lag}"] = ret.reindex(idx).to_numpy()
    nxt = df[["symbol", "day_idx", *known]].copy()
    nxt["day_idx"] -= 1  # row d's known inputs attach to the session before it
    out = out.merge(nxt.rename(columns={c: f"f_{c}" for c in known}), on=["symbol", "day_idx"],
                    how="inner", validate="one_to_one")
    out["day_idx"] += 1
    out = out.rename(columns={"today_vwap": "prev_vwap", "target_tomorrow_vwap": "actual"})
    return out.dropna(subset=["actual"]).reset_index(drop=True)


def _xcols(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns if c.startswith("f_")]


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, levels: list[float], seed: int = 0,
                max_iter: int = 200, **hgb) -> np.ndarray:
    """Sorted quantile moves (rows x levels) for `target`, fitted on `train` rows only."""
    cols = _xcols(train)
    y = train["actual"].to_numpy() / train["prev_vwap"].to_numpy() - 1
    kw = {"max_iter": max_iter, "learning_rate": 0.05, "max_leaf_nodes": 31,
          "min_samples_leaf": 200, "random_state": seed, **hgb}
    out = np.column_stack([
        HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw)
        .fit(train[cols], y).predict(target[cols]) for q in levels])
    return np.sort(out, axis=1)  # no crossing


def run_fold(feat: pd.DataFrame, df: pd.DataFrame, cfg: TrainConfig, fold: Fold,
             max_iter: int, max_train_rows: int | None) -> pd.DataFrame:
    train = feat[feat["day_idx"] < fold.es_start]
    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=cfg.seed)
    test = feat[feat["day_idx"].between(fold.test_start, fold.test_end)].reset_index(drop=True)
    moves = fit_predict(train, test, cfg.quantiles, cfg.seed, max_iter)
    res = test[["symbol", "day_idx"]].copy()
    for i, level in enumerate(cfg.quantiles):
        res[qcol(level)] = test["prev_vwap"].to_numpy() * (1 + moves[:, i])
    res["prev_vwap"], res["actual"] = test["prev_vwap"], test["actual"]
    keys = [c for c in ("date", "industry_code") if c in df.columns]
    return res.merge(df[["symbol", "day_idx", *keys]], on=["symbol", "day_idx"], how="left",
                     validate="one_to_one")


def run(cfg: TrainConfig, *, max_iter: int = 200, max_train_rows: int | None = None) -> Path:
    df, meta = load_panel(cfg)
    folds = make_folds(df, cfg)
    cfg.test_end, cfg.n_folds = folds[-1].test_end, len(folds)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    out = cfg.out_dir / f"{stamp}-gbm-{df['symbol'].nunique()}sym"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"model": "gbm", "config": cfg.to_dict(), "panel_meta": meta,
         "folds": [asdict(f) for f in folds], "max_iter": max_iter,
         "lags": list(LAGS)}, indent=2, default=str))
    feat = build_features(df, cfg)
    print(f"run {out.name}: {len(feat):,} samples, {len(_xcols(feat))} features")
    done = {}
    for k, fold in enumerate(folds):
        if cfg.only_folds is not None and k not in cfg.only_folds:
            continue
        res = run_fold(feat, df, cfg, fold, max_iter, max_train_rows)
        m = write_fold(out, k, res, {"fold": asdict(fold)})
        done[k] = res
        print(f"fold {k}: {len(res):,} test rows, pinball {m['test']['quantile_loss']:.5f} "
              f"(naive {m['test']['naive_quantile_loss']:.5f})")
    if done:
        write_summary(out, done)
    (out / "state.json").write_text(json.dumps(
        {"status": "done", "path": os.path.realpath(out)}, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    d = TrainConfig()
    p = argparse.ArgumentParser(prog="tft-vwap-gbm", description=__doc__.split("\n")[0])
    p.add_argument("--panel", type=Path, default=d.panel)
    p.add_argument("--out", type=Path, default=d.out_dir)
    p.add_argument("--target", choices=["return", "vwap"], default="return")
    p.add_argument("--symbols", type=int, default=None)
    p.add_argument("--folds", default=None, help="comma list of 0-based folds to run")
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--fold-len", type=int, default=d.fold_len)
    p.add_argument("--early-stop-len", type=int, default=d.early_stop_len)
    p.add_argument("--min-train-sessions", type=int, default=None)
    p.add_argument("--test-end", type=int, default=None)
    p.add_argument("--earnings", choices=EARNINGS_MODES, default=d.earnings)
    p.add_argument("--market", action="store_true")
    p.add_argument("--macro", action="store_true")
    p.add_argument("--session-shape", action="store_true")
    p.add_argument("--after-hours", action="store_true")
    p.add_argument("--premarket", choices=PREMARKET_MODES[:2], default="off",
                   help="on = pre-market groups are features (no masking: trees see 0/flag 0)")
    p.add_argument("--pm-groups", choices=PM_GROUP_CHOICES, default=d.pm_groups)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--max-train-rows", type=int, default=None, help="subsample the training rows")
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args(argv)
    n_folds = a.n_folds if a.n_folds is not None else (0 if a.min_train_sessions else d.n_folds)
    out = run(TrainConfig(
        panel=a.panel, out_dir=a.out, target=a.target, symbols=a.symbols,
        only_folds=[int(x) for x in a.folds.split(",")] if a.folds else None,
        n_folds=n_folds, fold_len=a.fold_len, early_stop_len=a.early_stop_len,
        min_train_sessions=a.min_train_sessions, test_end=a.test_end, earnings=a.earnings,
        market=a.market, macro=a.macro, session_shape=a.session_shape,
        after_hours=a.after_hours, premarket=a.premarket, pm_groups=a.pm_groups, seed=a.seed,
    ), max_iter=a.max_iter, max_train_rows=a.max_train_rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
