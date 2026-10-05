"""Walk-forward TFT training CLI.

    uv run tft-vwap --symbols 200 --folds 3 --max-epochs 3      # quick smoke run
    uv run tft-vwap                                              # all symbols, all folds

Each run writes runs/<run_id>/{config.json, summary.json} plus, per fold,
fold_<k>/{best.ckpt, metrics.json, test_predictions.parquet, logs/} (CSV loss curves).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from .config import TrainConfig
from .dataset import Fold, build_datasets, load_panel, make_folds
from .evaluate import calibration, forecast_frame, metrics
from .report import build_report


def _embedding_sizes(train_ds, cfg: TrainConfig) -> dict[str, tuple[int, int]]:
    sizes = {}
    for name, dim in cfg.embedding_sizes.items():
        n = len(train_ds._categorical_encoders[name].classes_)
        sizes[name] = (n, max(1, min(dim, n)))
    return sizes


def train_fold(df: pd.DataFrame, cfg: TrainConfig, fold: Fold, fold_dir: Path) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    t0 = time.time()
    train_ds, es_ds, test_ds = build_datasets(df, cfg, fold)
    train_dl = train_ds.to_dataloader(
        train=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    es_dl = es_ds.to_dataloader(
        train=False, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers
    )
    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        hidden_continuous_size=cfg.hidden_continuous_size,
        dropout=cfg.dropout,
        loss=QuantileLoss(quantiles=cfg.quantiles),
        embedding_sizes=_embedding_sizes(train_ds, cfg),
        log_interval=-1,
        reduce_on_plateau_patience=max(1, cfg.patience - 1),
    )
    ckpt = ModelCheckpoint(dirpath=fold_dir, filename="best", monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=1,
        gradient_clip_val=cfg.gradient_clip_val,
        limit_train_batches=cfg.limit_train_batches,
        callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.patience), ckpt],
        logger=CSVLogger(fold_dir, name="logs", version=""),
        enable_model_summary=False,
        deterministic=False,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=es_dl)
    train_secs = time.time() - t0

    best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
    pk = {"accelerator": cfg.accelerator, "devices": 1, "logger": False,
          "enable_progress_bar": False}
    es_frame = forecast_frame(best, es_ds, df, cfg, pk)
    test_frame = forecast_frame(best, test_ds, df, cfg, pk)
    # row keys travel with the forecasts, so a run can be graded without the panel
    keys = [c for c in ("date", "industry_code") if c in df.columns]
    test_frame.merge(df[["symbol", "day_idx", *keys]], on=["symbol", "day_idx"], how="left",
                     validate="one_to_one").to_parquet(fold_dir / "test_predictions.parquet",
                                                       index=False)

    result = {
        "fold": asdict(fold),
        "train_samples": len(train_ds),
        "epochs": trainer.current_epoch + 1,
        "best_val_loss": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
        "train_seconds": round(train_secs, 1),
        "early_stop": metrics(es_frame, cfg.quantiles),
        "test": metrics(test_frame, cfg.quantiles),
        "test_calibration": calibration(test_frame, cfg),
    }
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


_SHOW = ("rows", "mape", "naive_mape", "mape_skill", "median_ape", "naive_median_ape",
         "directional_accuracy", "up_rate", "interval_coverage", "quantile_loss",
         "naive_quantile_loss")


def summarize(results: list[dict]) -> pd.DataFrame:
    rows = [{"fold": i, "test_days": f"{r['fold']['test_start']}-{r['fold']['test_end']}",
             "epochs": r["epochs"], **{k: r["test"].get(k) for k in _SHOW}}
            for i, r in results]
    table = pd.DataFrame(rows).set_index("fold")
    if len(table) > 1:
        table.loc["mean"] = table.mean(numeric_only=True)
    return table


def run(cfg: TrainConfig) -> Path:
    df, meta = load_panel(cfg)
    folds = make_folds(df, cfg)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    run_id = f"{stamp}-{cfg.target}-{df['symbol'].nunique()}sym"
    run_dir = cfg.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"config": cfg.to_dict(), "panel_meta": meta, "folds": [asdict(f) for f in folds]},
        indent=2, default=str))
    print(f"run {run_id}: {len(df):,} rows, {df['symbol'].nunique():,} symbols, "
          f"day_idx {df['day_idx'].min()}-{df['day_idx'].max()}")

    results = []
    for k, fold in enumerate(folds):
        if cfg.only_folds is not None and k not in cfg.only_folds:
            continue
        print(f"fold {k}: train <{fold.es_start}, early-stop {fold.es_start}-{fold.test_start - 1}, "
              f"test {fold.test_start}-{fold.test_end}")
        results.append((k, train_fold(df, cfg, fold, run_dir / f"fold_{k}")))

    table = summarize(results)
    # pooled over every test fold: the calibration report the strategy would actually face
    pooled = pd.concat(
        [pd.read_parquet(run_dir / f"fold_{k}" / "test_predictions.parquet") for k, _ in results],
        ignore_index=True,
    )
    pooled_cal = calibration(pooled, cfg)
    for name, rows in pooled_cal.items():
        pd.DataFrame(rows).to_csv(run_dir / f"calibration_{name}.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(
        {"folds": {k: r for k, r in results}, "test_table": table.reset_index().to_dict("records"),
         "pooled_test": metrics(pooled, cfg.quantiles), "pooled_calibration": pooled_cal},
        indent=2, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.max_rows", 200, "display.float_format", "{:.4f}".format):
        print(table)
        print(f"\npooled test set: {len(pooled):,} forecasts")
        print(_quantile_report(pooled_cal["quantiles"]))
        print(_signal_report(pooled_cal["signals"]))
    build_report(run_dir)  # per-stock / per-day calibration grid, bias and ladder hits
    print(f"results: {run_dir} (calibration report: {run_dir / 'report'})")
    return run_dir


def _quantile_report(rows: list[dict]) -> str:
    t = pd.DataFrame(rows)
    t = t[t["quantile"].round(2).isin([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])]
    t = t.assign(forecast=t["quantile"].map("P{:.0%}".format).str.replace("%", ""))
    t = t.set_index("forecast")[["observed"]].T.rename(index={"observed": "share of actuals below"})
    return "\nquantile calibration (ideal: share = level)\n" + t.to_string()


def _signal_report(rows: list[dict]) -> str:
    t = pd.DataFrame(rows)
    t["event"] = t["move"].map(lambda m: f"{'>=' if m > 0 else '<='} {m:+.0%}")
    t["call"] = t["confidence"].map(lambda c: f">= {c:.0%}")
    t = t[["event", "call", "signals", "signal_share", "mean_prob", "hit_rate", "base_rate"]]
    return ("\nprobability calls: when P(next-day vwap move) >= call, how often it happened\n"
            + t.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    d = TrainConfig()
    p = argparse.ArgumentParser(prog="tft-vwap", description=__doc__.split("\n")[0])
    p.add_argument("--panel", type=Path, default=d.panel)
    p.add_argument("--out", type=Path, default=d.out_dir)
    p.add_argument("--target", choices=["vwap", "return"], default=d.target)
    p.add_argument("--normalizer", choices=["encoder", "group"], default=d.normalizer)
    p.add_argument("--symbols", type=int, default=None, help="random subset of N symbols")
    p.add_argument("--folds", default=None, help="comma list of 0-based folds to run")
    p.add_argument("--n-folds", type=int, default=d.n_folds)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--limit-train-batches", type=int, default=d.limit_train_batches)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--hidden-size", type=int, default=d.hidden_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--accelerator", default=d.accelerator)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--use-sentiment", action="store_true")
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args(argv)
    torch.set_float32_matmul_precision("medium")
    run(TrainConfig(
        panel=a.panel, out_dir=a.out, target=a.target, normalizer=a.normalizer,
        symbols=a.symbols,
        only_folds=[int(x) for x in a.folds.split(",")] if a.folds else None,
        n_folds=a.n_folds, max_epochs=a.max_epochs, limit_train_batches=a.limit_train_batches,
        batch_size=a.batch_size, hidden_size=a.hidden_size, learning_rate=a.learning_rate,
        accelerator=a.accelerator, num_workers=a.num_workers, use_sentiment=a.use_sentiment,
        seed=a.seed,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
