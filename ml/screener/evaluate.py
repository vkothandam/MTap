"""R8: do the screener's top picks travel further from the open than simple rankings?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import Rules, rule_pass

BASELINES = ("random", "naive_atr", "tft_width", "pm_volume")
_RANKERS = {"naive_atr": "atr14_pct", "tft_width": "tft_width90", "pm_volume": "pm_vol_pct_10d"}


def daily_picks(df: pd.DataFrame, top_n: int, seed: int = 0) -> pd.DataFrame:
    """Per day (day_idx): mean realised range_from_open and mean hl range of the top_n picks
    of each ranking, among the rows in `df` (already the liquid / rule-passing universe).
    Needs `score`. Days with fewer than top_n candidates are skipped."""
    rng = np.random.default_rng(seed)
    rows = []
    for day, g in df.groupby("day_idx"):
        if len(g) < top_n:
            continue
        row = {"day_idx": day, "n": len(g), "universe_median": g["range_from_open"].median(),
               "universe_mean": g["range_from_open"].mean()}
        picks = {"screener": g.nlargest(top_n, "score"),
                 "random": g.iloc[rng.choice(len(g), top_n, replace=False)]}
        for name, col in _RANKERS.items():
            if col in g and g[col].notna().sum() >= top_n:
                picks[name] = g.nlargest(top_n, col)
        for name, p in picks.items():
            row[name] = p["range_from_open"].mean()
        row["screener_hit"] = (g.nlargest(top_n, "score")["range_from_open"]
                               > g["range_from_open"].median()).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean and 95% CI of per-day differences, resampling days."""
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return {"mean": None, "lo": None, "hi": None, "days": 0}
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return {"mean": float(diff.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "days": int(len(diff))}


def summarise(oof: pd.DataFrame, rules: Rules) -> dict:
    """`oof`: out-of-fold scored rows (score, features, range_from_open, day_idx)."""
    liquid = oof[rule_pass(oof, rules)]
    days = daily_picks(liquid, rules.top_n)
    if days.empty:
        return {"days": 0}
    out = {"days": int(len(days)), "top_n": rules.top_n,
           "mean_range": {c: float(days[c].mean()) for c in
                          ("screener", "universe_median", *BASELINES) if c in days}}
    out["hit_rate_vs_universe_median"] = float(days["screener_hit"].mean())
    for base in BASELINES:
        if base in days:
            out[f"screener_minus_{base}"] = paired_bootstrap(days["screener"] - days[base])
    return out
