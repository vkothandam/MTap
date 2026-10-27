"""screener-train: fit the opening-range screener on a base TFT run's out-of-fold forecasts.

    uv run screener-train --panel ../out/tft/panel.parquet --base-run runs/<id> [--folds 0,1,2]

Fold k is evaluated by a model fitted only on forecasts of folds before it (their test blocks
lie wholly before fold k's test_start), so the test fold is never seen; fold 0 has nothing
earlier and is skipped. The shipped model is fitted on all selected folds.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .evaluate import summarise
from .features import STATES, tft_features, training_frame
from .model import fit, save
from .rules import Rules, load_rules, rule_pass


def load_base(base_run: Path, folds: list[int] | None) -> tuple[list[dict], pd.DataFrame]:
    """(fold boundaries from config.json, out-of-fold forecasts tagged with `fold`)."""
    info = json.loads((base_run / "config.json").read_text())
    bounds = info["folds"]
    parts = []
    for k, _ in enumerate(bounds):
        path = base_run / f"fold_{k}" / "test_predictions.parquet"
        if (folds is None or k in folds) and path.exists():
            parts.append(pd.read_parquet(path).assign(fold=k))
    if not parts:
        raise FileNotFoundError(f"no fold_*/test_predictions.parquet under {base_run}")
    return bounds, pd.concat(parts, ignore_index=True)


def walk_forward(frame: pd.DataFrame, bounds: list[dict], state: str) -> pd.DataFrame:
    """Out-of-fold screener scores for every fold that has an earlier fold to train on."""
    parts = []
    for k in sorted(frame["fold"].unique()):
        start = bounds[k]["test_start"]
        past = frame[frame["day_idx"] < start]  # earlier folds only: never fold k itself
        test = frame[frame["fold"] == k]
        if past.empty or test.empty:
            continue
        m = fit(past, state)
        parts.append(pd.concat([test, m.predict(test)], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def train(panel: Path, base_run: Path, out_dir: Path, folds: list[int] | None = None,
          rules: Rules | None = None, states: tuple[str, ...] = STATES) -> Path:
    rules = rules or Rules()
    bounds, oof = load_base(base_run, folds)
    frame = training_frame(pd.read_parquet(panel), tft_features(oof, prices=True)
                           .assign(fold=oof["fold"].to_numpy()))
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    model_dir = out_dir / f"{stamp}-{base_run.name}"
    summary = {"base_run": str(base_run), "panel": str(panel.resolve()), "rules": rules.to_dict(),
               "folds": sorted(int(k) for k in frame["fold"].unique()), "rows": len(frame),
               "states": {}}
    models = {}
    for state in states:
        models[state] = fit(frame, state)
        scored = walk_forward(frame, bounds, state)
        summary["states"][state] = (summarise(scored, rules) if len(scored)
                                    else {"days": 0, "note": "needs at least two folds"})
        print(f"{state}: {json.dumps(summary['states'][state])}")
    save(models, model_dir)
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"screener model: {model_dir}")
    return model_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--base-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/screener"))
    ap.add_argument("--folds", help="comma-separated fold numbers (default: all)")
    ap.add_argument("--settings", type=Path, help="settings.toml with a [screener] section")
    a = ap.parse_args(argv)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    train(a.panel, a.base_run, a.out_dir, folds, load_rules(a.settings))


if __name__ == "__main__":
    main()
