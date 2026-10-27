"""Paired comparison of two runs on the same forecasts.

    uv run tft-vwap-compare runs/<A> runs/<B> [--folds 1,3,5] [--drop-split]
        [--state night|px|full] [--state-b ...] [--only-pm [px|full]]

Joins the runs' fold_*/test_predictions.parquet on (symbol, date) when both runs carry dates
(so a 2-year and a 5-year panel line up), else on (symbol, day_idx); scores each run's pinball
loss (mean over the quantiles, on returns vs prev_vwap) per day, and tests the paired
difference A - B over days with a bootstrap 95% interval (negative: A is better). Days, not
rows, are resampled: stocks on one day share the market's move. Reports the difference on
earnings-window days (Earnings_Eve / Earnings_Day / Post_Earnings_Reaction in the panel's
results_window, if the panel is there) vs the other days, the P10-P90 and P5-P95 coverage of
each run, and a per-fold table. Prints a text report and writes compare.json (in A's run dir
unless --out). --drop-split drops series-break segments (symbols containing "~").
--state picks run A's scoring state: the sibling dir <A>~night / <A>~px of an `optional` run
(full = A itself); --state-b the same for B. --only-pm [px|full] keeps only rows where the
pre-market data exists (the panel's pm_px_known / pm_full_known flag is 1; default full), the
bake-off's common test rows.
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


def state_dir(run: Path, state: str | None) -> Path:
    """The run dir that holds a pre-market state's forecasts (full = the run itself)."""
    run = Path(run)
    if state in (None, "full"):
        return run
    d = run.with_name(run.name + {"night": "~night", "px": "~px"}[state])
    if not d.is_dir():
        raise FileNotFoundError(f"{d} missing: {run.name} has no {state!r} state (train it with "
                                "--premarket optional)")
    return d


def pm_rows(frame: pd.DataFrame, panel: Path, group: str) -> pd.Series:
    """Per row of `frame`: does the panel hold the pre-market `group` data for that day?"""
    flag = {"px": "pm_px_known", "full": "pm_full_known"}[group]
    p = pd.read_parquet(panel, columns=["symbol", "day_idx", flag])
    base = frame["symbol"].str.split("~").str[0]
    m = pd.DataFrame({"symbol": base, "day_idx": frame["day_idx"]}).merge(
        p, on=["symbol", "day_idx"], how="left")
    return (m[flag].fillna(0) == 1).set_axis(frame.index)


def match_key(a: pd.DataFrame, b: pd.DataFrame) -> str:
    """Rows are matched on date when both runs have a date for every row, else day_idx."""
    ok = all("date" in f.columns and f["date"].notna().all() for f in (a, b))
    return "date" if ok else "day_idx"


def compare(run_a: Path, run_b: Path, folds: list[int] | None = None,
            drop_split: bool = False, state_a: str | None = None, state_b: str | None = None,
            only_pm: str | None = None) -> dict:
    panel_a = _panel_of(run_a)
    run_a, run_b = state_dir(run_a, state_a), state_dir(run_b, state_b)
    a, b = load(run_a, folds), load(run_b, folds)
    if only_pm:
        panel = panel_a or _panel_of(run_b)
        if not panel or not Path(panel).exists():
            raise FileNotFoundError("--only-pm needs the panel named in the run's config.json")
        a = a[pm_rows(a, panel, only_pm)]
        if a.empty:
            raise ValueError(f"no test rows with pre-market {only_pm} data")
    mk = match_key(a, b)
    key = ["symbol", mk]
    if mk == "date":  # the paired days are dates; day_idx would differ between panels
        a, b = a.assign(date=pd.to_datetime(a["date"])), b.assign(date=pd.to_datetime(b["date"]))
    cols_a, cols_b = qlevels(a), qlevels(b)
    if list(cols_a) != list(cols_b):
        raise ValueError("the runs forecast different quantile levels")
    a_idx = a["day_idx"] if mk == "date" else None  # run A's own day_idx, for the panel lookups
    a = a[[*key, "fold", "prev_vwap", "actual", *cols_a.values()]]
    if a_idx is not None:
        a = a.assign(day_idx=a_idx)
    b = b[[*key, "prev_vwap", "actual", *cols_b.values()]]
    j = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    if a_idx is not None:
        j = j.rename(columns={"day_idx": "day_idx_a"})
    if drop_split:
        j = j[~j["symbol"].str.contains("~", regex=False)]
    if j.empty:
        raise ValueError("the runs have no forecasts in common")
    day_col = mk
    a_day = j["day_idx_a"] if a_idx is not None else j["day_idx"]
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

    day = j.groupby(day_col)[["loss_a", "loss_b"]].mean()
    diff = day["loss_a"] - day["loss_b"]
    result = {"a": str(run_a), "b": str(run_b), "rows": int(len(j)), "folds": folds,
              "matched_on": f"(symbol, {mk})", "only_pm": only_pm,
              "loss_a": float(j["loss_a"].mean()), "loss_b": float(j["loss_b"].mean()),
              "paired": paired_stats(diff),
              "coverage": {"a": cover(fa), "b": cover(fb)}}

    panel = _panel_of(run_a)
    earn = earnings_days(panel, j[["symbol"]].assign(day_idx=a_day)) if panel else None
    if earn is not None:
        sub = {}
        for name, sel in (("earnings", earn.to_numpy()), ("other", ~earn.to_numpy())):
            d = j[sel].groupby(day_col)[["loss_a", "loss_b"]].mean()
            sub[name] = {**paired_stats(d["loss_a"] - d["loss_b"]), "rows": int(sel.sum())}
        result["by_earnings"] = sub

    per_fold = []
    for k, g in j.groupby("fold"):
        d = g.groupby(day_col)[["loss_a", "loss_b"]].mean()
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
    lines = [f"A = {r['a']}", f"B = {r['b']}", f"{r['rows']:,} common forecasts matched on {r['matched_on']}, {p['days']} days"
             + (f", only rows with pre-market {r['only_pm']}" if r.get("only_pm") else ""),
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
    p.add_argument("--state", choices=("night", "px", "full"), help="run A's pre-market state")
    p.add_argument("--state-b", choices=("night", "px", "full"), help="run B's pre-market state")
    p.add_argument("--only-pm", nargs="?", const="full", choices=("px", "full"),
                   help="only rows where the panel has this pre-market group (default full)")
    p.add_argument("--out", type=Path, help="compare.json path (default: in run A)")
    a = p.parse_args(argv)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    r = compare(a.run_a, a.run_b, folds, a.drop_split, a.state, a.state_b, a.only_pm)
    print(format_report(r))
    out = a.out or a.run_a / "compare.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
