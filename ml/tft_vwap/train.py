"""Walk-forward TFT training CLI.

    uv run tft-vwap --symbols 200 --folds 3 --max-epochs 3      # quick smoke run
    uv run tft-vwap                                              # all symbols, all folds

Each run writes runs/<run_id>/{config.json, state.json, summary.json} plus, per fold,
fold_<k>/{best.ckpt, resume.ckpt, metrics.json, test_predictions.parquet, logs/} (CSV loss curves).

Long runs go in segments: `--max-hours 2.5` stops at an epoch end (also on a runs/<id>/STOP
file, Ctrl-C or SIGTERM), writes state.json {"status": "paused"} and exits without scoring the
half-trained fold; `--resume runs/<id> [--max-hours H]` continues it where it stopped.
Each epoch prints a `[progress]` line with the time left, also kept in runs/<id>/progress.json
and shown by `tft-vwap-status` (progress.py).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from .config import (
    EARNINGS_MODES,
    PM_GROUP_CHOICES,
    PREMARKET_MODES,
    STATE_SUFFIX,
    TrainConfig,
    from_run,
)
from .dataset import Fold, PmMaskCollate, build_datasets, load_panel, make_folds
from .evaluate import calibration, forecast_frame, metrics
from .importance import by_group, pool, variable_importance
from .progress import Progress
from .report import build_report


def _embedding_sizes(train_ds, cfg: TrainConfig) -> dict[str, tuple[int, int]]:
    sizes = {}
    for name, dim in cfg.embedding_sizes.items():
        n = len(train_ds._categorical_encoders[name].classes_)
        sizes[name] = (n, max(1, min(dim, n)))
    return sizes


class TimeBudget(Callback):
    """Pauses training at an epoch end once the next epoch would overrun the budget, or when
    `stop_file` exists. Stateless: the clock starts when the run segment starts (not carried
    across resumes) and the epoch time is a running average. A pause is told apart from an
    early stop by `paused`: EarlyStopping decides in on_validation_end, before this hook, so a
    stop that is already set is the model finishing, not a pause. On a pause the whole state
    is saved to resume.ckpt in on_train_end, after the LR scheduler stepped."""

    def __init__(self, max_hours: float | None, stop_file: Path | None, resume_ckpt: Path):
        self.budget = None if max_hours is None else max_hours * 3600
        self.stop_file, self.resume_ckpt = stop_file, resume_ckpt
        self.start = time.monotonic()
        self.epoch_secs: float | None = None
        self.paused = False
        self._t = self.start

    def stop_requested(self) -> bool:
        return bool(self.stop_file and self.stop_file.exists())

    def exhausted(self) -> bool:
        if self.stop_requested():
            return True
        used = time.monotonic() - self.start + (self.epoch_secs or 0)
        return self.budget is not None and used > self.budget

    def on_train_epoch_start(self, trainer, pl_module):
        self._t = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        dt = time.monotonic() - self._t
        self.epoch_secs = dt if self.epoch_secs is None else 0.7 * self.epoch_secs + 0.3 * dt
        if trainer.should_stop or trainer.current_epoch + 1 >= trainer.max_epochs:
            return
        if self.exhausted():
            self.paused = trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        if self.paused:
            trainer.save_checkpoint(self.resume_ckpt)


def _epoch_of(ckpt: Path) -> int:
    return int(torch.load(ckpt, map_location="cpu", weights_only=False)["epoch"])


def train_fold(df: pd.DataFrame, cfg: TrainConfig, fold: Fold, fold_dir: Path,
               budget: TimeBudget | None = None, progress: Progress | None = None) -> dict | None:
    """Train, then score one fold. None when the budget paused it (nothing is scored)."""
    resume_ckpt = fold_dir / "resume.ckpt"
    resuming = resume_ckpt.exists()
    start_epoch = _epoch_of(resume_ckpt) if resuming else 0
    pl.seed_everything(cfg.seed + start_epoch, workers=True)
    t0 = time.time()
    train_ds, es_ds, test_ds = build_datasets(df, cfg, fold)
    masked = cfg.premarket == "optional"
    train_dl = train_ds.to_dataloader(
        train=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        **({"collate_fn": PmMaskCollate.for_dataset(
            train_ds, {"px": cfg.pm_mask_px, "full": cfg.pm_mask_full})} if masked else {}))
    es_dl = es_ds.to_dataloader(
        train=False, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers,
        # the same samples blank every epoch (the training rates, by a hash of symbol and day)
        **({"collate_fn": PmMaskCollate.for_dataset(
            es_ds, {"px": cfg.pm_mask_px, "full": cfg.pm_mask_full}, deterministic=True)}
           if masked else {}))
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
    # no version counter: a second best-v1.ckpt would be a stale file for predict to find
    ckpt = ModelCheckpoint(dirpath=fold_dir, filename="best", monitor="val_loss", save_top_k=1,
                           enable_version_counter=False)
    # every epoch, improving or not (save_last only saves on improvement): the resume point
    last = ModelCheckpoint(dirpath=fold_dir, filename="resume", monitor=None, save_top_k=1,
                           enable_version_counter=False)
    budget = budget or TimeBudget(None, None, resume_ckpt)
    budget.resume_ckpt, budget.paused = resume_ckpt, False
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=1,
        gradient_clip_val=cfg.gradient_clip_val,
        limit_train_batches=cfg.limit_train_batches,
        callbacks=[ckpt, last, EarlyStopping(monitor="val_loss", patience=cfg.patience), budget,
                   *([progress] if progress else [])],
        logger=CSVLogger(fold_dir, name="logs", version=""),
        enable_model_summary=False,
        deterministic=False,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=es_dl,
                ckpt_path=str(resume_ckpt) if resuming else None, weights_only=False)
    if budget.paused:
        return None
    resume_ckpt.unlink(missing_ok=True)
    train_secs = time.time() - t0

    best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
    pk = {"accelerator": cfg.accelerator, "devices": 1, "logger": False,
          "enable_progress_bar": False}
    # row keys travel with the forecasts, so a run can be graded without the panel
    keys = [c for c in ("date", "industry_code") if c in df.columns]

    def save(frame, directory):
        directory.mkdir(parents=True, exist_ok=True)
        frame.merge(df[["symbol", "day_idx", *keys]], on=["symbol", "day_idx"], how="left",
                    validate="one_to_one").to_parquet(directory / "test_predictions.parquet",
                                                      index=False)

    es_frame = forecast_frame(best, es_ds, df, cfg, pk)
    test_frame = forecast_frame(best, test_ds, df, cfg, pk)
    save(test_frame, fold_dir)
    if masked:  # the same checkpoint per scoring state; the main run is the last (data's own)
        for state in cfg.scoring_states[:-1]:
            blank = forecast_frame(best, test_ds, df, cfg, pk,
                                   collate_fn=PmMaskCollate.for_state(test_ds, state))
            sib = fold_dir.parent.with_name(fold_dir.parent.name + STATE_SUFFIX[state]) / fold_dir.name
            save(blank, sib)
            (sib / "metrics.json").write_text(json.dumps(
                {"fold": asdict(fold), "state": state, "test": metrics(blank, cfg.quantiles)},
                indent=2))
    imp = variable_importance(best, test_ds.to_dataloader(
        train=False, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers))
    (fold_dir / "importance.json").write_text(json.dumps(
        {"variables": imp, "groups": by_group(imp)}, indent=2))

    result = {
        "fold": asdict(fold),
        "train_samples": len(train_ds),
        "epochs": trainer.current_epoch,  # after fit: epochs completed (resumed ones included)
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


def _set_state(run_dir: Path, status: str, **extra) -> None:
    path = run_dir / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(status=status, updated=datetime.now(UTC).isoformat(timespec="seconds"), **extra)
    path.write_text(json.dumps(state, indent=2))


def run(cfg: TrainConfig, *, max_hours: float | None = None, resume: Path | None = None) -> Path:
    """Train every selected fold. With `resume` (a run dir) continue that run: cfg must come
    from from_run, finished folds are skipped and the paused fold continues from resume.ckpt.
    Returns the run dir; its state.json says paused or done."""
    df, meta = load_panel(cfg)
    if resume is not None:
        run_dir = Path(resume)
        saved = json.loads((run_dir / "config.json").read_text())
        if saved["panel_meta"] != meta:
            raise ValueError(f"{cfg.panel} changed since the run started (meta differs); "
                             "a resumed run needs the same panel")
        # ModelCheckpoint restores its best score only for the same dirpath
        started_at = json.loads((run_dir / "state.json").read_text())["path"]
        if os.path.realpath(run_dir) != started_at:
            raise ValueError(f"{run_dir} is not where the run started ({started_at}); "
                             "move it back to resume")
        folds = make_folds(df, cfg)
        if [asdict(f) for f in folds] != saved["folds"]:
            raise ValueError("fold boundaries differ from the saved run")
    else:
        folds = make_folds(df, cfg)
        # pin the folds: make_folds otherwise anchors to the panel's end
        cfg.test_end, cfg.n_folds = folds[-1].test_end, len(folds)
        if cfg.premarket != "off":
            cfg.premarket_cutoff = meta.get("premarket_cutoff")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
        run_dir = cfg.out_dir / f"{stamp}-{cfg.target}-{df['symbol'].nunique()}sym"
        run_dir.mkdir(parents=True, exist_ok=True)
        info = {"config": cfg.to_dict(), "panel_meta": meta, "folds": [asdict(f) for f in folds]}
        (run_dir / "config.json").write_text(json.dumps(info, indent=2, default=str))
        if cfg.premarket == "optional":
            for state in cfg.scoring_states[:-1]:
                sib = run_dir.with_name(run_dir.name + STATE_SUFFIX[state])
                sib.mkdir(exist_ok=True)
                (sib / "config.json").write_text(json.dumps(
                    {"view_of": run_dir.name, "state": state, **info}, indent=2, default=str))
    print(f"run {run_dir.name}: {len(df):,} rows, {df['symbol'].nunique():,} symbols, "
          f"day_idx {df['day_idx'].min()}-{df['day_idx'].max()}")

    budget = TimeBudget(max_hours, run_dir / "STOP", run_dir / "resume.ckpt")
    _set_state(run_dir, "running", path=os.path.realpath(run_dir))
    selected = [k for k in range(len(folds)) if cfg.only_folds is None or k in cfg.only_folds]
    results, trained = [], False  # trained: a fold ran in this segment
    for k, fold in enumerate(folds):
        if k not in selected:
            continue
        done = run_dir / f"fold_{k}" / "metrics.json"
        if done.exists():
            results.append((k, json.loads(done.read_text())))
            continue
        result = None
        # a segment always trains at least one epoch; the budget is checked between folds
        if not (budget.stop_requested() or (trained and budget.exhausted())):
            print(f"fold {k}: train <{fold.es_start}, early-stop {fold.es_start}-"
                  f"{fold.test_start - 1}, test {fold.test_start}-{fold.test_end}")
            try:
                progress = Progress(run_dir / "progress.json", fold=k,
                                    pos=selected.index(k) + 1, n_folds=len(selected),
                                    prior=[r for _, r in results], patience=cfg.patience)
                result = train_fold(df, cfg, fold, run_dir / f"fold_{k}", budget, progress)
            except SystemExit:  # Ctrl-C (Lightning exits) or SIGTERM: a clean pause, no scoring
                _set_state(run_dir, "paused", fold=k)
                print(f"interrupted in fold {k}; continue with: tft-vwap --resume {run_dir}")
                raise
        if result is None:
            _set_state(run_dir, "paused", fold=k)
            print(f"paused in fold {k}; continue with: tft-vwap --resume {run_dir}")
            return run_dir
        results.append((k, result))
        trained = True

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
         "pooled_test": metrics(pooled, cfg.quantiles), "pooled_calibration": pooled_cal,
         "importance_by_group": _pooled_importance(run_dir, results)},
        indent=2, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.max_rows", 200, "display.float_format", "{:.4f}".format):
        print(table)
        print(f"\npooled test set: {len(pooled):,} forecasts")
        print(_quantile_report(pooled_cal["quantiles"]))
        print(_signal_report(pooled_cal["signals"]))
    build_report(run_dir)  # per-stock / per-day calibration grid, bias and ladder hits
    (run_dir / "progress.json").unlink(missing_ok=True)
    _set_state(run_dir, "done")
    print(f"results: {run_dir} (calibration report: {run_dir / 'report'})")
    return run_dir


def _pooled_importance(run_dir: Path, results: list) -> dict:
    """Variable-selection weight per input group, per fold and averaged over folds."""
    per_fold = {}
    for k, _ in results:
        f = run_dir / f"fold_{k}" / "importance.json"
        if f.exists():
            per_fold[k] = json.loads(f.read_text())["groups"]
    return {"per_fold": per_fold, "mean": pool(list(per_fold.values())) if per_fold else {}}


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
    p.add_argument("--n-folds", type=int, default=None,
                   help=f"test blocks (default {d.n_folds}; 0 = as many as fit, see "
                        "--min-train-sessions)")
    p.add_argument("--fold-len", type=int, default=d.fold_len, help="sessions per test block")
    p.add_argument("--early-stop-len", type=int, default=d.early_stop_len,
                   help="early-stop sessions in front of each test block")
    p.add_argument("--min-train-sessions", type=int, default=None,
                   help="at least this many training sessions in the first fold; given without "
                        "--n-folds it fits as many folds as the panel allows (252 is the "
                        "usual choice)")
    p.add_argument("--test-end", type=int, default=None,
                   help="day_idx of the last test session (default: the panel's last); "
                        "pass an earlier run's to repeat its folds")
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--limit-train-batches", type=int, default=d.limit_train_batches)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--hidden-size", type=int, default=d.hidden_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--accelerator", default=d.accelerator)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--use-sentiment", action="store_true")
    p.add_argument("--earnings", choices=EARNINGS_MODES, default=d.earnings,
                   help="earnings-calendar inputs: exact release dates, point in time, or none "
                        "(FEATURES.md)")
    p.add_argument("--market", action="store_true", help="market and sector returns (schema 4)")
    p.add_argument("--macro", action="store_true", help="yields and inflation (schema 4)")
    p.add_argument("--session-shape", action="store_true",
                   help="high/low position and gap vs VWAP (schema 4)")
    p.add_argument("--after-hours", action="store_true", help="after-hours of day t (schema 4)")
    p.add_argument("--premarket", choices=PREMARKET_MODES, default=d.premarket,
                   help="today's pre-market as a known-future input; optional = also trained "
                        "with groups masked and scored per state into <run>~night / <run>~px")
    p.add_argument("--pm-groups", choices=PM_GROUP_CHOICES, default=d.pm_groups,
                   help="pre-market groups that are inputs: px (prices, 09:30) or px,full "
                        "(+ VWAP/volume/range, 09:45)")
    p.add_argument("--pm-mask-full", type=float, default=d.pm_mask_full,
                   help="share of training samples with the full group blank (optional mode)")
    p.add_argument("--pm-mask-px", type=float, default=d.pm_mask_px,
                   help="share with the prices group blank; full is blank too (optional mode)")
    p.add_argument("--dedupe-train", action=argparse.BooleanOptionalAction,
                   default=d.dedupe_train,
                   help="one training sample per (symbol, day) instead of per encoder length")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--max-hours", type=float, default=None,
                   help="pause at an epoch end after this long (continue with --resume)")
    p.add_argument("--resume", type=Path, default=None,
                   help="continue a paused run dir; only --max-hours may be given with it")
    a = p.parse_args(argv)
    torch.set_float32_matmul_precision("medium")
    if a.resume:
        given = {k for k, v in vars(a).items() if v != p.get_default(k)} - {"resume", "max_hours"}
        if given:
            p.error("--resume reuses the run's saved settings; only --max-hours may be given "
                    f"with it (got {sorted(given)})")
        run(from_run(a.resume), max_hours=a.max_hours, resume=a.resume)
        return 0
    n_folds = a.n_folds if a.n_folds is not None else (0 if a.min_train_sessions else d.n_folds)
    run(TrainConfig(
        panel=a.panel, out_dir=a.out, target=a.target, normalizer=a.normalizer,
        symbols=a.symbols,
        only_folds=[int(x) for x in a.folds.split(",")] if a.folds else None,
        n_folds=n_folds, fold_len=a.fold_len, early_stop_len=a.early_stop_len,
        min_train_sessions=a.min_train_sessions, test_end=a.test_end, max_epochs=a.max_epochs,
        limit_train_batches=a.limit_train_batches,
        batch_size=a.batch_size, hidden_size=a.hidden_size, learning_rate=a.learning_rate,
        accelerator=a.accelerator, num_workers=a.num_workers, use_sentiment=a.use_sentiment,
        earnings=a.earnings, market=a.market, macro=a.macro, session_shape=a.session_shape,
        after_hours=a.after_hours, premarket=a.premarket, pm_groups=a.pm_groups,
        pm_mask_full=a.pm_mask_full, pm_mask_px=a.pm_mask_px,
        dedupe_train=a.dedupe_train, seed=a.seed,
    ), max_hours=a.max_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
