"""Walk-forward recalibration of a run's quantiles: each quantile moves by the residual quantile
learned on the run's own earlier test folds, so the share of actuals below P_tau comes out near
tau (the dashboard's q err, grade and coverage).

    uv run tft-vwap-recal runs/<RUN> [--window 4] [--by pm,width] [--folds 5,7,9]

For target fold k, the calibration rows are the run's test forecasts of the `window` folds
before k (never k itself). Per group, the shift at level tau is the tau-quantile of
(actual move − forecast P_tau move) over those rows; it is added to fold k's P_tau, then the
quantiles are sorted so none cross. A shift (not a scale) keeps the pinball loss level while it
fixes the levels: a quantile's own residual quantile is exactly the correction that puts tau of
the rows below it.

Groups (`--by`):
- `pm`: the row has full pre-market data (the panel's pm_full_known), so stocks with and
  without pre-market get their own shifts;
- `width`: the forecast P5–P95 width in terciles, with the cut points taken from the
  calibration rows (forecast-time information only).

A group with fewer than MIN_ROWS calibration rows uses the shift of all rows. Folds with no
earlier fold are skipped; folds with fewer than `window` earlier ones are flagged `thin`.
Live, the same shifts come from the latest `window` folds (or graded live forecasts).
Writes runs/<RUN>-cal/ in the run contract with a MODEL.md card (`--label`, default the base
label + "-cal").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import model_card

from .compare import _panel_of, qlevels
from .overlay import load_folds
from .runout import write_fold, write_summary

MIN_ROWS = 2000
BY_CHOICES = ("pm", "width")


def shifts(resid: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Per level tau: the tau-quantile of its residual column (rows x levels)."""
    return np.array([np.quantile(resid[:, i], t) for i, t in enumerate(levels)])


def groups(r_q: np.ndarray, pm: np.ndarray | None, cuts: np.ndarray | None) -> np.ndarray:
    """Group id per row: pm flag x width tercile (0 when a part is off)."""
    g = np.zeros(len(r_q), dtype=int)
    if cuts is not None:
        g = np.digitize(r_q[:, -1] - r_q[:, 0], cuts)
    if pm is not None:
        g = g + 3 * pm.astype(int)
    return g


def recalibrate(by_fold: dict[int, pd.DataFrame], levels: list[float], cols: list[str],
                window: int = 4, by: tuple[str, ...] = ("pm",),
                pm: dict[int, np.ndarray] | None = None, folds: list[int] | None = None
                ) -> tuple[dict[int, pd.DataFrame], dict[int, list[int]]]:
    """(recalibrated fold frames, the folds each was calibrated on). `pm` maps a fold to its
    rows' 0/1 pre-market flag (needed for by=pm)."""
    lv = np.asarray(levels)

    def moves(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prev = f["prev_vwap"].to_numpy()
        return (np.sort(f[cols].to_numpy() / prev[:, None] - 1, axis=1),
                f["actual"].to_numpy() / prev - 1)

    out, used = {}, {}
    for k in sorted(by_fold):
        if folds is not None and k not in folds:
            continue
        earlier = [j for j in sorted(by_fold) if j < k][-window:]
        if not earlier:
            continue
        cq, cy = zip(*(moves(by_fold[j]) for j in earlier))
        cq, cy = np.concatenate(cq), np.concatenate(cy)
        tq, _ = moves(by_fold[k])
        cuts = (np.quantile(cq[:, -1] - cq[:, 0], [1 / 3, 2 / 3]) if "width" in by else None)
        cpm = np.concatenate([pm[j] for j in earlier]) if "pm" in by else None
        gc = groups(cq, cpm, cuts)
        gt = groups(tq, pm[k] if "pm" in by else None, cuts)
        resid = cy[:, None] - cq
        every = shifts(resid, lv)
        adj = np.empty_like(tq)
        for g in np.unique(gt):
            sel = gc == g
            adj[gt == g] = shifts(resid[sel], lv) if sel.sum() >= MIN_ROWS else every
        new = np.sort(tq + adj, axis=1)
        res = by_fold[k].copy()
        res[cols] = res["prev_vwap"].to_numpy()[:, None] * (1 + new)
        out[k], used[k] = res, earlier
    return out, used


def pm_flags(by_fold: dict[int, pd.DataFrame], panel: Path) -> dict[int, np.ndarray]:
    p = pd.read_parquet(panel, columns=["symbol", "day_idx", "pm_full_known"])
    flags = {}
    for k, f in by_fold.items():
        keys = pd.DataFrame({"symbol": f["symbol"].str.split("~").str[0].to_numpy(),
                             "day_idx": f["day_idx"].to_numpy()})
        flags[k] = (keys.merge(p, on=["symbol", "day_idx"], how="left")["pm_full_known"]
                    .fillna(0).to_numpy() == 1)
    return flags


def run(base: Path, window: int = 4, by: tuple[str, ...] = ("pm",), folds: list[int] | None = None,
        panel: Path | None = None, out: Path | None = None, card: dict | None = None) -> Path:
    base = Path(base)
    by_fold = load_folds(base)
    if not by_fold:
        raise FileNotFoundError(f"no fold_*/test_predictions.parquet under {base}")
    cols_by_level = qlevels(next(iter(by_fold.values())))
    levels, cols = list(cols_by_level), list(cols_by_level.values())
    pm = None
    if "pm" in by:
        panel = panel or _panel_of(base) or _panel_of(base.with_name(base.name.split("~")[0]))
        if not panel or not Path(panel).exists():
            raise FileNotFoundError("--by pm needs the panel (config.json or --panel)")
        pm = pm_flags(by_fold, Path(panel))
    done, used = recalibrate(by_fold, levels, cols, window, by, pm, folds)
    if not done:
        raise ValueError("no fold has an earlier fold to calibrate on")
    out = Path(out) if out else base.with_name(f"{base.name}-cal")
    thin = [k for k, e in used.items() if len(e) < window]
    for k, res in done.items():
        write_fold(out, k, res, {"calibrated_on_folds": used[k], "thin": k in thin})
        print(f"fold {k}: calibrated on folds {used[k]}{' (thin)' if k in thin else ''}")
    saved = json.loads((base / "config.json").read_text()) if (base / "config.json").exists() else {}
    (out / "config.json").write_text(json.dumps(
        {**saved, "model": "recal", "base": str(base), "window": window, "by": list(by),
         "folds_done": sorted(done), "thin_folds": thin}, indent=2, default=str))
    write_summary(out, done, {"thin_folds": thin})
    old = model_card.read(base)
    card = dict(card or {})
    card["label"] = card.get("label") or f"{model_card.label_of(base)}-cal"
    card["summary"] = card.get("summary") or (
        f"{old.get('summary') or base.name}, recalibrated on its earlier folds")
    model_card.write(
        out, kind="Recal", forecast=old.get("forecast", "night"),
        inputs=f"{model_card.label_of(base)} (`{base.name}`) with each quantile shifted by its "
               f"residual quantile over the {window} earlier folds"
               + (f", per {' x '.join(by)} group" if by else "") + ".",
        folds=f"Folds {sorted(done)}" + (f"; thin (fewer than {window} earlier): {thin}."
                                         if thin else "."),
        status="done", result=model_card.contract_result(out), **card)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-recal", description=__doc__.split("\n")[0])
    p.add_argument("base_run", type=Path)
    p.add_argument("--window", type=int, default=4, help="earlier folds to calibrate on")
    p.add_argument("--by", default="pm", help=f"comma list of {BY_CHOICES}, or '' for none")
    p.add_argument("--folds", help="comma list of target folds (default: all with an earlier one)")
    p.add_argument("--panel", type=Path, help="panel for --by pm (default: the run's config)")
    p.add_argument("--out", type=Path, help="output run dir (default: <base>-cal)")
    model_card.add_args(p)
    a = p.parse_args(argv)
    by = tuple(x for x in a.by.split(",") if x)
    if set(by) - set(BY_CHOICES):
        p.error(f"--by takes {BY_CHOICES}")
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    out = run(a.base_run, a.window, by, folds, a.panel, a.out,
              model_card.from_args(a, "tft-vwap-recal", argv))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
