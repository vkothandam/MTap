"""Training progress and time left (`tft-vwap-status [runs/<id> | runs]`).

While a run trains, `Progress` keeps runs/<id>/progress.json current (every 30 s and at each
epoch end) and prints one line per epoch. The time left is a range, because a fold ends either
by early stopping or at max_epochs:

    earliest  the fold early-stops as soon as it can (patience - wait more epochs without a
              better val_loss); every later fold stops after patience + 1 epochs
    estimate  the completed folds' mean epochs and training time (mid-range before any fold
              has finished)
    latest    every fold runs to max_epochs

Scoring a fold (a minute or two on the full panel) is not included. Pausing loses at most the
current epoch: `touch runs/<id>/STOP` stops at the next epoch end, `tft-vwap --resume` continues.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from lightning.pytorch.callbacks import Callback

WRITE_EVERY = 30.0  # seconds between progress.json updates inside an epoch


def estimate_left(*, epoch_secs: float, completed: int, frac_left: float, max_epochs: int,
                  wait: int, patience: int, folds_after: int, prior_epochs: list[int],
                  prior_secs: list[float]) -> dict:
    """Seconds left as {"fold": [earliest, estimate, latest], "run": [...]}. `completed`
    epochs of this fold are done; inside an epoch `frac_left` (0..1] of it remains, at an
    epoch end it is 0. `wait` is the early-stop count after the last validation."""
    partial = 1 - frac_left if frac_left > 0 else 0.0  # share of the current epoch already done
    hi = max_epochs - completed - partial
    lo = max(min(patience - wait, max_epochs - completed) - partial, frac_left)
    mid = (lo + hi) / 2
    if prior_epochs:
        mid = min(max(sum(prior_epochs) / len(prior_epochs) - completed - partial, lo), hi)
    fold = [x * epoch_secs for x in (lo, mid, hi)]
    per_fold_lo = min(patience + 1, max_epochs) * epoch_secs
    per_fold_hi = max_epochs * epoch_secs
    per_fold_mid = (sum(prior_secs) / len(prior_secs) if prior_secs
                    else (per_fold_lo + per_fold_hi) / 2)
    run = [f + folds_after * p for f, p in zip(fold, (per_fold_lo, per_fold_mid, per_fold_hi))]
    return {"fold": [round(x, 1) for x in fold], "run": [round(x, 1) for x in run]}


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class Progress(Callback):
    """Writes progress.json for fold `fold` (position `pos` of `n_folds` selected folds).
    `prior` are the metrics of folds already finished in this run, any segment."""

    def __init__(self, path: Path, *, fold: int, pos: int, n_folds: int, prior: list[dict],
                 patience: int, every: float = WRITE_EVERY):
        self.path, self.fold, self.pos, self.n_folds = path, fold, pos, n_folds
        self.prior_epochs = [r["epochs"] for r in prior]
        self.prior_secs = [r["train_seconds"] for r in prior]
        self.patience, self.every = patience, every
        self.epoch_secs: float | None = None
        self._epoch_t0 = self._last_write = time.monotonic()

    def _wait(self, trainer) -> int:
        es = trainer.early_stopping_callback
        return int(es.wait_count) if es is not None else 0

    def _snapshot(self, trainer, batch: int | None, completed: int) -> dict:
        """`completed`: epochs finished in this fold, resumed ones included."""
        nb = trainer.num_training_batches
        nb = int(nb) if isinstance(nb, (int, float)) and nb != float("inf") else None
        frac_left, epoch_secs = 0.0, self.epoch_secs
        if batch is not None and nb:
            frac_left = max(1 - batch / nb, 1e-6)
            if epoch_secs is None and batch:  # first epoch: extrapolate from its batches so far
                epoch_secs = (time.monotonic() - self._epoch_t0) * nb / batch
        data = {
            "fold": self.fold, "fold_pos": self.pos, "folds": self.n_folds,
            "epoch": completed + (1 if batch is not None else 0), "max_epochs": trainer.max_epochs,
            "batch": batch, "batches": nb, "early_stop_wait": self._wait(trainer),
            "patience": self.patience,
            "epoch_seconds": round(epoch_secs, 1) if epoch_secs else None,
            "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if epoch_secs:
            left = estimate_left(epoch_secs=epoch_secs, completed=completed, frac_left=frac_left,
                                 max_epochs=trainer.max_epochs, wait=data["early_stop_wait"],
                                 patience=self.patience, folds_after=self.n_folds - self.pos,
                                 prior_epochs=self.prior_epochs, prior_secs=self.prior_secs)
            if batch is None and self._wait(trainer) >= self.patience:  # early-stopped
                after = [r - f for r, f in zip(left["run"], left["fold"])]
                left = {"fold": [0, 0, 0], "run": after}
            data |= {"fold_left_seconds": left["fold"], "run_left_seconds": left["run"],
                     "eta": (datetime.now().astimezone()
                             + timedelta(seconds=left["run"][1])).isoformat(timespec="minutes")}
        return data

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if time.monotonic() - self._last_write >= self.every:
            self._last_write = time.monotonic()
            _write(self.path, self._snapshot(trainer, batch_idx + 1, trainer.current_epoch))

    def on_train_epoch_end(self, trainer, pl_module):
        # validation has run by now, so the time covers both and wait_count is current
        dt = time.monotonic() - self._epoch_t0
        self.epoch_secs = dt if self.epoch_secs is None else 0.7 * self.epoch_secs + 0.3 * dt
        # current_epoch moves on only after this hook
        data = self._snapshot(trainer, None, trainer.current_epoch + 1)
        self._last_write = time.monotonic()
        _write(self.path, data)
        print(f"[progress] {line(data)}", flush=True)


def _dur(secs: float) -> str:
    return f"{secs / 60:.0f} min" if secs < 5400 else f"{secs / 3600:.1f} h"


def _span(left: list[float]) -> str:
    lo, mid, hi = left
    return f"{_dur(lo)}–{_dur(hi)} (est {_dur(mid)})"


def line(p: dict) -> str:
    s = (f"fold {p['fold']} ({p['fold_pos']}/{p['folds']}) epoch {p['epoch']}/{p['max_epochs']}"
         + (f" batch {p['batch']}/{p['batches']}" if p.get("batch") else "")
         + f", early-stop wait {p['early_stop_wait']}/{p['patience']}")
    if p.get("epoch_seconds"):
        s += f", {_dur(p['epoch_seconds'])}/epoch"
    if p.get("fold_left_seconds"):
        s += (f" | fold left {_span(p['fold_left_seconds'])}"
              f" | run left {_span(p['run_left_seconds'])}, ETA {p['eta'][11:16]}")
    return s


def status(run_dir: Path, now: datetime | None = None) -> str:
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").exists() else {}
    head = f"{run_dir.name}: {state.get('status', 'unknown')}"
    pfile = run_dir / "progress.json"
    if state.get("status") == "done" or not pfile.exists():
        return head
    p = json.loads(pfile.read_text())
    now = now or datetime.now().astimezone()
    age = (now - datetime.fromisoformat(p["updated"])).total_seconds()
    out = [head, f"  {line(p)}", f"  updated {_dur(age) if age >= 60 else f'{age:.0f} s'} ago"]
    if state.get("status") == "running" and age > max(3 * WRITE_EVERY, 2 * (p.get("epoch_seconds") or 0)):
        out.append("  no update for a while: the process may have stopped")
    if state.get("status") == "running":
        out.append(f"  pause at the next epoch end: touch {run_dir / 'STOP'}  "
                   f"(continue: tft-vwap --resume {run_dir})")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-status", description=__doc__.split("\n")[0])
    p.add_argument("path", type=Path, nargs="?", default=Path("runs"),
                   help="a run dir, or a runs dir (lists its running and paused runs)")
    a = p.parse_args(argv)
    if (a.path / "state.json").exists():
        print(status(a.path))
        return 0
    found = False
    for d in sorted(a.path.glob("*/state.json")):
        if json.loads(d.read_text()).get("status") in ("running", "paused"):
            print(status(d.parent))
            found = True
    if not found:
        print(f"no running or paused runs in {a.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
