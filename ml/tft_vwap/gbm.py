"""R7: plain gradient-boosted quantile baseline, no neural net (does the TFT earn its cost?).

    uv run tft-vwap-gbm --panel PANEL --n-folds 0 --min-train-sessions 252 --fold-len 63 \
        --early-stop-len 21 --market --macro --after-hours --session-shape [--folds 1,3,5]
        [--premarket on|optional --pm-groups px,full] [--open] [--vol] [--tail-iter 600]

Same panel, input-group flags and fold boundaries as `tft-vwap` (make_folds), same run contract.
One sample per (symbol, forecast day d): the features are the last session's inputs (row d-1:
the enabled observed groups, incl. earnings) plus lags 0, 1, 2, 5, 20 of vwap_pct_prev_day, and
the known-future inputs of d (calendar and, when enabled, the pre-market groups with their flags
from row d). The target is the move vs the last VWAP. One HistGradientBoostingRegressor(
loss="quantile") per quantile, fitted on forecast days before the fold's early-stop block
(no early stopping: a fixed `--max-iter`; `--tail-iter` for the outer levels, P15 and below and
P85 and above, whose loss keeps falling longer), outputs sorted so none cross.

`--open` adds day d's open vs the last VWAP for every stock (known from 09:30, also for stocks
with no pre-market trades; the panel's gap_open_vs_prev_vwap_pct of row d). `--vol` adds
past-only volatility: the std of the move over 5, 20 and 60 sessions, the mean |move| and the
mean intraday spread over 20. `--premarket optional` (px,full) trains one model for 09:30 and
09:45: every training row with pre-market data appears twice, as it is and with the full group
blank (values 0, flag 0, as on a 09:30 morning), and the test rows are scored in both states:
full in the run dir, px in the sibling <run>~px (as an `optional` TFT run). Writes
runs/<stamp>-gbm-<n>sym/{config.json, summary.json, fold_k/{test_predictions,metrics}} with the
TFT runs' columns, so compare, the dashboard and overlay read it like a TFT run, plus the
MODEL.md card (`--label`, `--about`, `--purpose`; model_card).
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

import model_card

from . import cards
from .config import (
    EARNINGS_MODES,
    KNOWN_REALS,
    PM_FULL_REALS,
    PM_GROUP_CHOICES,
    PREMARKET_MODES,
    STATE_SUFFIX,
    TrainConfig,
)
from .dataset import Fold, load_panel, make_folds
from .evaluate import qcol
from .runout import write_fold, write_summary

LAGS = (0, 1, 2, 5, 20)  # of vwap_pct_prev_day, relative to the last session
TAIL = 0.15  # levels <= TAIL or >= 1 - TAIL use --tail-iter
OPEN_CLIP = 0.3
FULL_COLS = [f"f_{c}" for c in PM_FULL_REALS]


def vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only volatility per (symbol, day_idx), known at that session's close."""
    g = df.groupby("symbol")
    ret, spread = g["vwap_pct_prev_day"], g["intraday_spread_pct"]
    out = df[["symbol", "day_idx"]].copy()
    for n, lo in ((5, 3), (20, 10), (60, 20)):
        out[f"f_vol{n}"] = ret.transform(lambda s, n=n, lo=lo: s.rolling(n, min_periods=lo).std())
    out["f_absmove20"] = ret.transform(lambda s: s.abs().rolling(20, min_periods=10).mean())
    out["f_spread20"] = spread.transform(lambda s: s.rolling(20, min_periods=10).mean())
    return out


def build_features(df: pd.DataFrame, cfg: TrainConfig, open_all: bool = False,
                   vol: bool = False) -> pd.DataFrame:
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
    if vol:
        out = out.merge(vol_features(df), on=["symbol", "day_idx"], how="left",
                        validate="one_to_one")
    nxt = df[["symbol", "day_idx", *known]].copy()
    if open_all:  # the open is known at 09:30 for every stock, pre-market trades or not
        nxt["open_all"] = (df["gap_open_vs_prev_vwap_pct"].fillna(0.0)
                           .clip(-OPEN_CLIP, OPEN_CLIP).to_numpy())
        known = [*known, "open_all"]
    nxt["day_idx"] -= 1  # row d's known inputs attach to the session before it
    out = out.merge(nxt.rename(columns={c: f"f_{c}" for c in known}), on=["symbol", "day_idx"],
                    how="inner", validate="one_to_one")
    out["day_idx"] += 1
    out = out.rename(columns={"today_vwap": "prev_vwap", "target_tomorrow_vwap": "actual"})
    return out.dropna(subset=["actual"]).reset_index(drop=True)


def _xcols(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns if c.startswith("f_")]


def blank_full(frame: pd.DataFrame) -> pd.DataFrame:
    """The 09:30 view of rows: the full pre-market group blank (values 0, flag 0)."""
    out = frame.copy()
    out[[c for c in FULL_COLS if c in out.columns]] = 0.0
    return out


def fit_predict(train: pd.DataFrame, target: pd.DataFrame | list[pd.DataFrame],
                levels: list[float], seed: int = 0, max_iter: int = 200,
                tail_iter: int | None = None, **hgb) -> np.ndarray | list[np.ndarray]:
    """Sorted quantile moves (rows x levels) for `target` (or for each frame of a list),
    fitted on `train` rows only."""
    cols = _xcols(train)
    y = train["actual"].to_numpy() / train["prev_vwap"].to_numpy() - 1
    targets = target if isinstance(target, list) else [target]
    preds = [[] for _ in targets]
    for q in levels:
        it = tail_iter if tail_iter and (q <= TAIL + 1e-9 or q >= 1 - TAIL - 1e-9) else max_iter
        kw = {"max_iter": it, "learning_rate": 0.05, "max_leaf_nodes": 31,
              "min_samples_leaf": 200, "random_state": seed, **hgb}
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw).fit(train[cols], y)
        for p, t in zip(preds, targets):
            p.append(m.predict(t[cols]))
    out = [np.sort(np.column_stack(p), axis=1) for p in preds]  # no crossing
    return out if isinstance(target, list) else out[0]


def run_fold(feat: pd.DataFrame, df: pd.DataFrame, cfg: TrainConfig, fold: Fold,
             max_iter: int, max_train_rows: int | None, tail_iter: int | None = None
             ) -> dict[str, pd.DataFrame]:
    """Test forecasts per scoring state: {"full": ...} (or the run's own state), plus "px"
    for an optional run."""
    train = feat[feat["day_idx"] < fold.es_start]
    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=cfg.seed)
    test = feat[feat["day_idx"].between(fold.test_start, fold.test_end)].reset_index(drop=True)
    states = {cards.state_of(cfg): test}
    if cfg.premarket == "optional":
        has_full = train["f_pm_full_known"] == 1
        train = pd.concat([train, blank_full(train[has_full])], ignore_index=True)
        states["px"] = blank_full(test)
    moves = fit_predict(train, list(states.values()), cfg.quantiles, cfg.seed, max_iter,
                        tail_iter)
    keys = [c for c in ("date", "industry_code") if c in df.columns]
    out = {}
    for state, m in zip(states, moves):
        res = test[["symbol", "day_idx"]].copy()
        for i, level in enumerate(cfg.quantiles):
            res[qcol(level)] = test["prev_vwap"].to_numpy() * (1 + m[:, i])
        res["prev_vwap"], res["actual"] = test["prev_vwap"], test["actual"]
        out[state] = res.merge(df[["symbol", "day_idx", *keys]], on=["symbol", "day_idx"],
                               how="left", validate="one_to_one")
    return out


def run(cfg: TrainConfig, *, max_iter: int = 200, max_train_rows: int | None = None,
        card: dict | None = None, tail_iter: int | None = None, open_all: bool = False,
        vol: bool = False) -> Path:
    if cfg.premarket == "optional" and cfg.pm_group_names != ["px", "full"]:
        raise ValueError("gbm --premarket optional needs --pm-groups px,full")
    df, meta = load_panel(cfg)
    folds = make_folds(df, cfg)
    cfg.test_end, cfg.n_folds = folds[-1].test_end, len(folds)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    out = cfg.out_dir / f"{stamp}-gbm-{df['symbol'].nunique()}sym"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"model": "gbm", "config": cfg.to_dict(), "panel_meta": meta,
         "folds": [asdict(f) for f in folds], "max_iter": max_iter, "tail_iter": tail_iter,
         "open_all": open_all, "vol": vol, "lags": list(LAGS)}, indent=2, default=str))
    main_state = cards.state_of(cfg)
    dirs = {main_state: out}
    if cfg.premarket == "optional":
        dirs["px"] = out.with_name(out.name + STATE_SUFFIX["px"])
    card = dict(card or {})
    label = card.pop("label", None) or model_card.default_label("GBM", out)
    extras = [f"{max_iter} iterations"
              + (f" ({tail_iter} for P{round(TAIL * 100)} and below / above "
                 f"P{round(100 - TAIL * 100)})" if tail_iter else "")]
    if open_all:
        extras.append("the open for every stock")
    if vol:
        extras.append("past volatility (5/20/60-session std, mean |move|, spread)")
    if max_train_rows:
        extras.append(f"up to {max_train_rows:,} training rows")
    if cfg.premarket == "optional":
        extras.append("pre-market rows also trained with the full group blank (one model for "
                      "09:30 and 09:45)")
    inputs = (cards.inputs_text(cfg, "HGB per quantile (no TFT)", masks=False)
              + f" Last-day inputs + lags {list(LAGS[1:])} of the move; " + "; ".join(extras) + ".")

    def write_cards(status: str) -> None:
        for state, path in dirs.items():
            suffix = STATE_SUFFIX.get(state, "") if state != main_state else ""
            forecast = cards.FORECAST[state]
            model_card.write(
                path, kind="GBM", forecast=forecast, inputs=inputs, status=status,
                folds=cards.folds_text(cfg, len(folds)) + (
                    f" Scored in the {state} state ({forecast}): the same model as "
                    f"`{out.name}`." if suffix else ""),
                label=label + suffix,
                result=model_card.contract_result(path) if status == "done" else None, **card)

    write_cards("running")
    feat = build_features(df, cfg, open_all=open_all, vol=vol)
    print(f"run {out.name}: {len(feat):,} samples, {len(_xcols(feat))} features")
    done = {s: {} for s in dirs}
    for k, fold in enumerate(folds):
        if cfg.only_folds is not None and k not in cfg.only_folds:
            continue
        for state, res in run_fold(feat, df, cfg, fold, max_iter, max_train_rows,
                                   tail_iter).items():
            m = write_fold(dirs[state], k, res, {"fold": asdict(fold), "state": state})
            done[state][k] = res
            print(f"fold {k} [{state}]: {len(res):,} test rows, pinball "
                  f"{m['test']['quantile_loss']:.5f} (naive {m['test']['naive_quantile_loss']:.5f})")
    for state, path in dirs.items():
        if path != out:
            (path / "config.json").write_text((out / "config.json").read_text())
        if done[state]:
            write_summary(path, done[state])
        (path / "state.json").write_text(json.dumps(
            {"status": "done", "path": os.path.realpath(path)}, indent=2))
    write_cards("done")
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
    p.add_argument("--premarket", choices=PREMARKET_MODES, default="off",
                   help="on = pre-market groups are features (trees see 0/flag 0 when missing); "
                        "optional = also trained with the full group blank, scored as full "
                        "and as px (<run>~px)")
    p.add_argument("--pm-groups", choices=PM_GROUP_CHOICES, default=d.pm_groups)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tail-iter", type=int, default=None,
                   help=f"iterations for levels <= {TAIL} and >= {1 - TAIL:g} (default --max-iter)")
    p.add_argument("--open", action="store_true", help="the open vs the last VWAP, every stock")
    p.add_argument("--vol", action="store_true", help="past-only volatility features")
    p.add_argument("--max-train-rows", type=int, default=None, help="subsample the training rows")
    p.add_argument("--seed", type=int, default=d.seed)
    model_card.add_args(p)
    a = p.parse_args(argv)
    n_folds = a.n_folds if a.n_folds is not None else (0 if a.min_train_sessions else d.n_folds)
    out = run(TrainConfig(
        panel=a.panel, out_dir=a.out, target=a.target, symbols=a.symbols,
        only_folds=[int(x) for x in a.folds.split(",")] if a.folds else None,
        n_folds=n_folds, fold_len=a.fold_len, early_stop_len=a.early_stop_len,
        min_train_sessions=a.min_train_sessions, test_end=a.test_end, earnings=a.earnings,
        market=a.market, macro=a.macro, session_shape=a.session_shape,
        after_hours=a.after_hours, premarket=a.premarket, pm_groups=a.pm_groups, seed=a.seed,
    ), max_iter=a.max_iter, max_train_rows=a.max_train_rows,
        card=model_card.from_args(a, "tft-vwap-gbm", argv), tail_iter=a.tail_iter,
        open_all=a.open, vol=a.vol)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
