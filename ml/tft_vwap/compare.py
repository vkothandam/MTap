"""Paired comparison of two runs on the same forecasts.

    uv run tft-vwap-compare runs/<A> runs/<B> [--folds 1,3,5] [--drop-split]

Joins the runs' fold_*/test_predictions.parquet on (symbol, day_idx), scores each run's pinball
loss (mean over the quantiles, on returns vs prev_vwap) per day, and tests the paired
difference A - B over days with a bootstrap 95% interval (negative: A is better). Days, not
rows, are resampled: stocks on one day share the market's move. Reports the difference on
earnings-window days (Earnings_Eve / Earnings_Day / Post_Earnings_Reaction in the panel's
results_window, if the panel is there) vs the other days, the P10-P90 and P5-P95 coverage of
each run, and a per-fold table. Prints a text report and writes compare.json (in A's run dir
unless --out). --drop-split drops series-break segments (symbols containing "~").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EARNINGS_WINDOW_DAYS = ("Earnings_Eve", "Earnings_Day", "Post_Earnings_Reaction")
BOOTSTRAP = 2000


def load(run_dir: Path, folds: list[int] | None) -> pd.DataFrame:
    paths = sorted(Path(run_dir).glob("fold_*/test_predictions.parquet"))
    frames = [pd.read_parquet(p).assign(fold=int(p.parent.name.split("_")[1])) for p in paths]
    f = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if folds is not None:
        f = f[f["fold"].isin(folds)]
    if f.empty:
        raise FileNotFoundError(f"no test forecasts under {run_dir} for folds {folds}")
    return f


def qlevels(frame: pd.DataFrame) -> dict[float, str]:
    cols = [c for c in frame.columns if len(c) == 3 and c[0] == "q" and c[1:].isdigit()]
    return {int(c[1:]) / 100: c for c in sorted(cols)}


def pinball(frame: pd.DataFrame, levels: dict[float, str]) -> np.ndarray:
    """Per-row pinball loss on returns, averaged over the quantiles."""
    prev = frame["prev_vwap"].to_numpy()
    r_act = frame["actual"].to_numpy() / prev - 1
    r_q = frame[list(levels.values())].to_numpy() / prev[:, None] - 1
    err = r_act[:, None] - r_q
    lv = np.asarray(list(levels))[None, :]
    return np.maximum(lv * err, (lv - 1) * err).mean(axis=1)


def coverage(frame: pd.DataFrame, lo: str, hi: str) -> float:
    return float(((frame[lo] <= frame["actual"]) & (frame["actual"] <= frame[hi])).mean())


def bootstrap_ci(x: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(d: pd.Series) -> dict:
    lo, hi = bootstrap_ci(d.to_numpy())
    return {"days": int(len(d)), "diff": float(d.mean()) if len(d) else None,
            "ci_low": lo, "ci_high": hi}


def earnings_days(panel: Path, keys: pd.DataFrame) -> pd.Series | None:
    """Per (symbol, day_idx) row of `keys`: is it in an earnings window? None without panel."""
    if not Path(panel).exists():
        return None
    try:
        p = pd.read_parquet(panel, columns=["symbol", "day_idx", "results_window"])
    except (KeyError, ValueError, OSError):
        return None
    p["earn"] = p["results_window"].isin(EARNINGS_WINDOW_DAYS)
    base = keys["symbol"].str.split("~").str[0]  # series-break segments keep their ticker
    m = pd.DataFrame({"symbol": base, "day_idx": keys["day_idx"]}).merge(
        p[["symbol", "day_idx", "earn"]], on=["symbol", "day_idx"], how="left")
    return m["earn"].fillna(False).astype(bool)


def compare(run_a: Path, run_b: Path, folds: list[int] | None = None,
            drop_split: bool = False) -> dict:
    a, b = load(run_a, folds), load(run_b, folds)
    key = ["symbol", "day_idx"]
    cols_a, cols_b = qlevels(a), qlevels(b)
    if list(cols_a) != list(cols_b):
        raise ValueError("the runs forecast different quantile levels")
    a = a[[*key, "fold", "prev_vwap", "actual", *cols_a.values()]]
    b = b[[*key, "prev_vwap", "actual", *cols_b.values()]]
    j = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    if drop_split:
        j = j[~j["symbol"].str.contains("~", regex=False)]
    if j.empty:
        raise ValueError("the runs have no forecasts in common")
    fa = j[[*key, "fold", "prev_vwap_a", "actual_a", *(f"{c}_a" for c in cols_a.values())]]
    fa.columns = [*key, "fold", "prev_vwap", "actual", *cols_a.values()]
    fb = j[[*key, "prev_vwap_b", "actual_b", *(f"{c}_b" for c in cols_b.values())]]
    fb.columns = [*key, "prev_vwap", "actual", *cols_b.values()]
    j = j[key + ["fold"]].assign(loss_a=pinball(fa, cols_a), loss_b=pinball(fb, cols_b))

    lo, hi = min(cols_a), max(cols_a)
    p10, p90 = (cols_a.get(0.1), cols_a.get(0.9))

    def cover(f):
        out = {"p5_p95": coverage(f, cols_a[lo], cols_a[hi])}
        if p10 and p90:
            out["p10_p90"] = coverage(f, p10, p90)
        return out

    day = j.groupby("day_idx")[["loss_a", "loss_b"]].mean()
    diff = day["loss_a"] - day["loss_b"]
    result = {"a": str(run_a), "b": str(run_b), "rows": int(len(j)), "folds": folds,
              "loss_a": float(j["loss_a"].mean()), "loss_b": float(j["loss_b"].mean()),
              "paired": paired_stats(diff),
              "coverage": {"a": cover(fa), "b": cover(fb)}}

    panel = _panel_of(run_a)
    earn = earnings_days(panel, j) if panel else None
    if earn is not None:
        sub = {}
        for name, sel in (("earnings", earn.to_numpy()), ("other", ~earn.to_numpy())):
            d = j[sel].groupby("day_idx")[["loss_a", "loss_b"]].mean()
            sub[name] = {**paired_stats(d["loss_a"] - d["loss_b"]), "rows": int(sel.sum())}
        result["by_earnings"] = sub

    per_fold = []
    for k, g in j.groupby("fold"):
        d = g.groupby("day_idx")[["loss_a", "loss_b"]].mean()
        sel = fa["fold"] == k
        per_fold.append({"fold": int(k), "rows": int(len(g)), "loss_a": float(g["loss_a"].mean()),
                         "loss_b": float(g["loss_b"].mean()),
                         **paired_stats(d["loss_a"] - d["loss_b"]),
                         "cov_p10_p90_a": coverage(fa[sel], p10, p90) if p10 and p90 else None,
                         "cov_p10_p90_b": coverage(fb[sel.to_numpy()], p10, p90)
                         if p10 and p90 else None})
    result["per_fold"] = per_fold
    return result


def _panel_of(run_dir: Path) -> Path | None:
    f = Path(run_dir) / "config.json"
    if not f.exists():
        return None
    panel = json.loads(f.read_text()).get("config", {}).get("panel")
    return Path(panel) if panel else None


def _verdict(p: dict) -> str:
    if p["diff"] is None or np.isnan(p["ci_low"]):
        return "n/a"
    if p["ci_high"] < 0:
        return "A better"
    return "B better" if p["ci_low"] > 0 else "no clear difference"


def format_report(r: dict) -> str:
    p = r["paired"]
    lines = [f"A = {r['a']}", f"B = {r['b']}", f"{r['rows']:,} common forecasts, {p['days']} days",
             f"pinball loss: A {r['loss_a']:.6f}  B {r['loss_b']:.6f}",
             f"A - B per day: {p['diff']:+.6f}  95% CI [{p['ci_low']:+.6f}, {p['ci_high']:+.6f}]"
             f"  -> {_verdict(p)}  (negative: A better)"]
    for name, s in r.get("by_earnings", {}).items():
        lines.append(f"  {name:9s} days: {s['diff']:+.6f}  [{s['ci_low']:+.6f}, {s['ci_high']:+.6f}]"
                     f"  ({s['days']} days, {s['rows']:,} rows)")
    for side in ("a", "b"):
        c = r["coverage"][side]
        lines.append(f"coverage {side.upper()}: P10-P90 {c.get('p10_p90', float('nan')):.1%}"
                     f" (nominal 80%), P5-P95 {c['p5_p95']:.1%} (nominal 90%)")
    t = pd.DataFrame(r["per_fold"])
    t["verdict"] = [_verdict(x) for x in r["per_fold"]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.6f}".format):
        lines += ["", "per fold", t[["fold", "rows", "loss_a", "loss_b", "diff", "ci_low",
                                     "ci_high", "verdict"]].to_string(index=False)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-compare", description=__doc__.split("\n")[0])
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--folds", help="comma list of fold numbers (default: all)")
    p.add_argument("--drop-split", action="store_true",
                   help="drop series-break segments (symbols containing '~')")
    p.add_argument("--out", type=Path, help="compare.json path (default: in run A)")
    a = p.parse_args(argv)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    r = compare(a.run_a, a.run_b, folds, a.drop_split)
    print(format_report(r))
    out = a.out or a.run_a / "compare.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
