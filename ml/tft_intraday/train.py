"""Walk-forward intraday TFT training CLI.

    uv run tft-intraday --symbols 50 --max-epochs 1 --limit-train-batches 20   # smoke run
    uv run tft-intraday --baselines-only                                        # no model
    uv run tft-intraday                                                         # the default blocks

One model forecasts all six targets (config.TARGETS) from each 15 s decision point. Each run
writes runs/intraday/<run_id>/{config.json, summary.json, report/} plus, per fold,
fold_<k>/{best.ckpt, metrics.json, baselines.json, intraday_<target>.parquet, logs/}.
The prediction files are the dashboard's intraday contract (forecast_dash/intraday_contract.py):
one row per (symbol, date, t_s) test decision, every value a log move from the last price at t.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from torch import nn

from . import baselines, daily_join, report
from .config import EMBEDDING_SIZES, MOMENTUM_REF, TARGETS, IntradayConfig
from .dataset import (
    KEYS,
    Fold,
    build_train,
    evaluation_set,
    features,
    load_days,
    make_folds,
    model_frame,
    pick_symbols,
    read_meta,
)
from .losses import MaskedMultiLoss

QCOLS = report.QCOLS
MOMCOLS = report.MODELS["momentum"]
CLIMCOLS = report.MODELS["climatology"]
DAILYCOLS = report.MODELS["daily"]
# decision-row columns every prediction file carries (the panel's reference columns)
CARRY = ["t_s", "tod_bucket", "sector_code", "rvol", "y", "last", "prev_vwap", "ref_session"]


def _embedding_sizes(train_ds) -> dict[str, tuple[int, int]]:
    sizes = {}
    for name, dim in EMBEDDING_SIZES.items():
        if name in train_ds._categorical_encoders:
            n = len(train_ds._categorical_encoders[name].classes_)
            sizes[name] = (n, max(1, min(dim, n)))
    return sizes


def load(cfg: IntradayConfig, dates: list[str], symbols, daily) -> pd.DataFrame:
    return daily_join.attach(load_days(cfg.panel, dates, symbols), daily)


def forecast_frame(index: pd.DataFrame, q: np.ndarray, raw: pd.DataFrame, target: str,
                   tables: dict) -> pd.DataFrame:
    """Model quantiles for decisions `index` (symbol, date, step) -> one contract frame."""
    out = index.reset_index(drop=True).copy()
    out[QCOLS] = np.sort(q, axis=1)
    ref = MOMENTUM_REF[target]
    cols = [*KEYS, *CARRY, target, "tmask", *([ref] if ref not in CARRY else [])]
    if target == "r_close":
        cols += ["has_daily", *daily_join.DQCOLS]
    out = out.merge(raw[cols], on=KEYS, how="inner", validate="one_to_one")
    ok = baselines.valid(out, target)
    out["mask"] = ok.astype("int8")
    out["actual"] = out[target].where(ok)
    out["ref_trail"] = out[ref]
    out[MOMCOLS] = baselines.momentum_quantiles(out, target, tables["momentum"])
    out[CLIMCOLS] = baselines.climatology_quantiles(out, target, tables["climatology"])
    if target == "r_close":  # the morning forecast, re-based to a move from the last price
        dq = out[daily_join.DQCOLS].to_numpy(float) - out["y"].to_numpy(float)[:, None]
        out[DAILYCOLS] = dq
        out["has_daily"] = out["has_daily"].astype("int8")
        out = out.drop(columns=daily_join.DQCOLS)
    out = out.drop(columns=[target, "tmask", "step"] + ([ref] if ref not in CARRY else []))
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    return out


def decisions(raw: pd.DataFrame, stride_steps: int, first_step: int) -> pd.DataFrame:
    """The test decision points the model is graded on (baselines-only runs): the stride grid
    from the first step with a minimum-length encoder."""
    d = raw.loc[(raw["step"] % stride_steps == 0) & (raw["step"] >= first_step), KEYS]
    return d.reset_index(drop=True)


def write_frames(frames: dict[str, pd.DataFrame], fold_dir: Path) -> dict:
    scores = {}
    for t, f in frames.items():
        f.to_parquet(fold_dir / f"intraday_{t}.parquet", index=False)
        s = report.score_target(f[f["mask"] == 1].reset_index(drop=True), t)["by_target"]
        scores[t] = s.drop(columns="target").set_index("model").to_dict("index") if not s.empty else {}
    return scores


def train_fold(cfg: IntradayConfig, meta: dict, fold: Fold, fold_dir: Path, symbols, daily
               ) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    fold_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    step_s = int(meta["step_s"])
    feats = features(meta, cfg)

    raw_train = load(cfg, fold.train_dates, symbols, daily)
    tables = {"momentum": baselines.fit_momentum(raw_train),
              "climatology": baselines.fit_climatology(raw_train)}
    (fold_dir / "baselines.json").write_text(json.dumps(tables))
    raw_test = load(cfg, fold.test_dates, symbols, daily)
    stride = max(1, cfg.test_stride_s // step_s)
    result = {"fold": asdict(fold), "train_rows": len(raw_train), "test_rows": len(raw_test)}

    if cfg.baselines_only:
        del raw_train
        idx = decisions(raw_test, stride, cfg.min_encoder_steps - 1)
        rows = idx.merge(raw_test, on=KEYS)
        frames = {t: forecast_frame(idx, baselines.momentum_quantiles(rows, t, tables["momentum"]),
                                    raw_test, t, tables) for t in TARGETS}
    else:
        train_frame = model_frame(raw_train, feats, step_s)
        del raw_train
        gc.collect()
        train_ds = build_train(train_frame, feats, cfg)
        del train_frame
        gc.collect()
        es_frame = model_frame(load(cfg, fold.es_dates, symbols, daily), feats, step_s)
        es_ds = evaluation_set(train_ds, es_frame, max(1, cfg.es_stride_s // step_s))
        test_ds = evaluation_set(train_ds, model_frame(raw_test, feats, step_s), stride)
        print(f"  datasets ready in {time.time() - t0:.0f}s: train {len(train_ds):,} samples, "
              f"early stop {len(es_ds):,}, test {len(test_ds):,}")
        train_dl = train_ds.to_dataloader(train=True, batch_size=cfg.batch_size,
                                          num_workers=cfg.num_workers)
        es_dl = es_ds.to_dataloader(train=False, batch_size=cfg.batch_size * 2,
                                    num_workers=cfg.num_workers)
        model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            attention_head_size=cfg.attention_head_size,
            hidden_continuous_size=cfg.hidden_continuous_size,
            dropout=cfg.dropout,
            loss=MaskedMultiLoss(cfg.quantiles, len(TARGETS)),
            # the default logging metrics would be weighted by the raw bitmask
            logging_metrics=nn.ModuleList([]),
            embedding_sizes=_embedding_sizes(train_ds),
            log_interval=-1,
            reduce_on_plateau_patience=max(1, cfg.patience - 1),
        )
        ckpt = ModelCheckpoint(dirpath=fold_dir, filename="best", monitor="val_loss",
                               save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator=cfg.accelerator,
            devices=1,
            gradient_clip_val=cfg.gradient_clip_val,
            limit_train_batches=cfg.limit_train_batches,
            callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.patience), ckpt],
            logger=CSVLogger(fold_dir, name="logs", version=""),
            enable_model_summary=False,
        )
        t1 = time.time()
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=es_dl)
        result.update(train_samples=len(train_ds), es_samples=len(es_ds),
                      epochs=trainer.current_epoch + 1, train_seconds=round(time.time() - t1, 1),
                      best_val_loss=(float(ckpt.best_model_score)
                                     if ckpt.best_model_score is not None else None))
        del train_ds, train_dl, es_ds, es_dl, es_frame
        gc.collect()

        best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
        t2 = time.time()
        pred = best.predict(
            test_ds.to_dataloader(train=False, batch_size=cfg.batch_size * 4,
                                  num_workers=cfg.num_workers),
            mode="quantiles", return_index=True,
            trainer_kwargs={"accelerator": cfg.accelerator, "devices": 1, "logger": False,
                            "enable_progress_bar": False})
        idx = pred.index[["symbol", "date", "step"]].astype({"symbol": str, "date": str})
        idx["step"] = idx["step"].astype("int64") - 1  # the sample predicts its decision row's targets
        frames = {t: forecast_frame(idx, pred.output[i][:, 0, :].cpu().numpy(), raw_test, t, tables)
                  for i, t in enumerate(TARGETS)}
        result["predict_seconds"] = round(time.time() - t2, 1)

    result["test"] = write_frames(frames, fold_dir)
    result["seconds"] = round(time.time() - t0, 1)
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run(cfg: IntradayConfig) -> Path:
    meta = read_meta(cfg.panel)
    folds = make_folds(meta, cfg)
    symbols = pick_symbols(cfg.panel, meta["dates"], cfg.symbols, cfg.seed)
    n_sym = len(symbols) if symbols else max(meta["days"][d]["symbols"] for d in meta["dates"])
    daily = daily_join.load_daily(cfg.daily_run, cfg.daily_fold) if cfg.daily_run else None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    run_id = f"{stamp}-intraday-{n_sym}sym" + ("-baselines" if cfg.baselines_only else "")
    run_dir = cfg.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "model": "tft_intraday", "targets": list(TARGETS), "horizon_s": meta["horizon_s"],
        "step_s": meta["step_s"], "eval_stride_s": cfg.test_stride_s,
        "daily_run": None if cfg.daily_run is None else cfg.daily_run.name,
        "folds": [{"fold": k, **asdict(f)} for k, f in enumerate(folds)],
        "config": cfg.to_dict(), "symbols": symbols,
        "panel_meta": {k: meta[k] for k in ("schema_version", "step_s", "dates", "rows",
                                            "features", "tmask_bits", "target_unit")},
    }, indent=2, default=str))
    print(f"run {run_id}: {len(folds)} folds, {n_sym} symbols")

    results = {}
    for k, fold in enumerate(folds):
        if cfg.only_folds is not None and k not in cfg.only_folds:
            continue
        print(f"fold {k}: train {fold.train_dates[0]}..{fold.train_dates[-1]} "
              f"({len(fold.train_dates)} days), early stop {fold.es_dates}, test {fold.test_dates}")
        results[k] = train_fold(cfg, meta, fold, run_dir / f"fold_{k}", symbols, daily)
        gc.collect()

    tables = report.build_report(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(
        {"folds": results,
         "by_target": tables["by_target"].to_dict("records") if "by_target" in tables else []},
        indent=2, default=str))
    if "by_target" in tables:
        print(report.format_summary(tables["by_target"]))
    print(f"results: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    d = IntradayConfig()
    p = argparse.ArgumentParser(prog="tft-intraday", description=__doc__.split("\n")[0])
    p.add_argument("--panel", type=Path, default=d.panel)
    p.add_argument("--out", type=Path, default=d.out_dir)
    p.add_argument("--daily-run", default=str(d.daily_run),
                   help="tft_vwap run whose morning forecast is joined in ('none' to skip)")
    p.add_argument("--daily-fold", type=int, default=d.daily_fold)
    p.add_argument("--test-blocks", default=None,
                   help="test dates per fold: 'D1,D2;D3' (default: "
                        + ";".join(",".join(b) for b in d.test_blocks) + ")")
    p.add_argument("--daily-folds", action="store_true",
                   help="one fold per test date of the default blocks")
    p.add_argument("--folds", default=None, help="comma list of 0-based folds to run")
    p.add_argument("--symbols", type=int, default=None, help="random subset of N symbols")
    p.add_argument("--no-group", action="store_true", help="drop the market/sector inputs")
    p.add_argument("--encoder-steps", type=int, default=d.encoder_steps)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--limit-train-batches", type=int, default=d.limit_train_batches)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--hidden-size", type=int, default=d.hidden_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--accelerator", default=d.accelerator)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args(argv)
    blocks = d.test_blocks
    if a.test_blocks:
        blocks = [b.split(",") for b in a.test_blocks.split(";")]
    if a.daily_folds:
        blocks = [[x] for b in blocks for x in b]
    torch.set_float32_matmul_precision("medium")
    run(IntradayConfig(
        panel=a.panel, out_dir=a.out,
        daily_run=None if a.daily_run.lower() == "none" else Path(a.daily_run),
        daily_fold=a.daily_fold, test_blocks=blocks,
        only_folds=[int(x) for x in a.folds.split(",")] if a.folds else None,
        symbols=a.symbols, use_group=not a.no_group, encoder_steps=a.encoder_steps,
        max_epochs=a.max_epochs, limit_train_batches=a.limit_train_batches,
        batch_size=a.batch_size, hidden_size=a.hidden_size, learning_rate=a.learning_rate,
        accelerator=a.accelerator, num_workers=a.num_workers, baselines_only=a.baselines_only,
        seed=a.seed,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
