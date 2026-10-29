"""screener-train: fit the opening-range screener on a base TFT run's out-of-fold forecasts.

    uv run screener-train --panel ../out/tft/panel.parquet --base-run runs/<id> [--folds 0,1,2]

Fold k is evaluated by a model fitted only on forecasts of folds before it (their test blocks
lie wholly before fold k's test_start), so the test fold is never seen; fold 0 has nothing
earlier and is skipped. The shipped model is fitted on all selected folds. The model dir gets a
MODEL.md card (`--label`, `--about`, `--purpose`; model_card) with the walk-forward result.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import model_card

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
          rules: Rules | None = None, states: tuple[str, ...] = STATES,
          card: dict | None = None) -> Path:
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
    model_card.write(
        model_dir, kind="Screener", forecast="09:30" if states == ("px",) else "09:30/09:45",
        inputs=f"HGB on log range from the open, states {', '.join(states)}, on "
               f"{model_card.label_of(base_run)}'s out-of-fold forecasts (`{base_run.name}`) "
               f"+ panel `{panel}`; top {rules.top_n} per day.",
        folds=f"Base folds {summary['folds']}; each scored by a model fitted on earlier folds "
              f"only ({len(frame):,} rows).",
        status="done", result=_card_result(summary["states"]), **(card or {}))
    print(f"screener model: {model_dir}")
    return model_dir


def _card_result(states: dict) -> str:
    parts = []
    for state, s in states.items():
        if not s.get("days"):
            parts.append(f"{state}: not scored ({s.get('note', 'no days')})")
            continue
        r, atr = s["mean_range"], s.get("screener_minus_naive_atr") or {}
        text = (f"{state}: top-{s['top_n']} mean range {r['screener']:.2%} vs universe median "
                f"{r.get('universe_median', float('nan')):.2%} over {s['days']} days")
        if atr.get("mean") is not None:
            text += (f"; vs the ATR ranking {atr['mean']:+.2%} (95% CI {atr['lo']:+.2%} to "
                     f"{atr['hi']:+.2%})")
        parts.append(text)
    return "; ".join(parts) + ". Full table: summary.json."


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--base-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/screener"))
    ap.add_argument("--folds", help="comma-separated fold numbers (default: all)")
    ap.add_argument("--settings", type=Path, help="settings.toml with a [screener] section")
    model_card.add_args(ap)
    a = ap.parse_args(argv)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    train(a.panel, a.base_run, a.out_dir, folds, load_rules(a.settings),
          card=model_card.from_args(a, "screener-train", argv))


if __name__ == "__main__":
    main()
