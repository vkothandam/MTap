"""Screener files the Watchlist page reads (written by ml/screener; no ml imports here).

    <dir>/screener-<date>.parquet   whole universe for that morning: symbol, state, score,
                                    pred_range, rule_pass, rank, lean, open, last, prev_close,
                                    dollar_volume_20d, gap_prev_close, earn_near, split_day,
                                    pm_vol_pct_10d (09:45 lists only), ... model features
    <dir>/outcomes.parquet          optional; what happened: date, symbol, open, high, low,
                                    range_from_open, hl_range_open (screener-outcomes)
    settings.toml [screener]        optional; slider defaults (min_price, min_dollar_volume, ...)
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

LIST_RE = re.compile(r"^screener-(\d{4}-\d{2}-\d{2})\.parquet$")
OUTCOMES = "outcomes.parquet"
EARNINGS = ("any", "only", "exclude")


@dataclass
class Rules:
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    min_pm_vol_pct: float = 0.0
    min_abs_gap_pct: float = 0.0
    exclude_split_day: bool = True
    earnings: str = "any"
    top_n: int = 20


def load_rules(settings: Path | None) -> Rules:
    if settings is None or not Path(settings).exists():
        return Rules()
    section = tomllib.loads(Path(settings).read_text()).get("screener", {})
    return Rules(**{k: v for k, v in section.items() if k in {f.name for f in fields(Rules)}})


def list_dates(root: Path) -> list[str]:
    """Dates that have a screener list, newest first."""
    return sorted((m.group(1) for p in Path(root).glob("screener-*.parquet")
                   if (m := LIST_RE.match(p.name))), reverse=True)


def load_list(root: Path, date: str) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / f"screener-{date}.parquet")


def load_outcomes(root: Path) -> pd.DataFrame | None:
    p = Path(root) / OUTCOMES
    return pd.read_parquet(p) if p.exists() else None


def apply_rules(df: pd.DataFrame, r: Rules) -> pd.DataFrame:
    """Re-apply the rule filters to a saved list and re-rank (best score = rank 1). Rows without
    a score (no open yet) never pass."""
    ok = df["score"].notna()
    if r.min_price > 0:
        ok &= df["open"].fillna(df.get("last")) >= r.min_price
    if r.min_dollar_volume > 0:
        ok &= df["dollar_volume_20d"] >= r.min_dollar_volume
    if r.min_pm_vol_pct > 0 and "pm_vol_pct_10d" in df:
        ok &= df["pm_vol_pct_10d"] >= r.min_pm_vol_pct
    if r.min_abs_gap_pct > 0:
        ok &= df["gap_prev_close"].abs() >= r.min_abs_gap_pct
    if r.exclude_split_day and "split_day" in df:
        ok &= ~df["split_day"].fillna(False).astype(bool)
    if r.earnings == "only":
        ok &= df["earn_near"] == 1
    elif r.earnings == "exclude":
        ok &= df["earn_near"] != 1
    out = df.assign(rule_pass=ok.fillna(False))
    out["rank"] = out["score"].where(out["rule_pass"]).rank(ascending=False, method="first")
    return out.sort_values("rank", na_position="last").reset_index(drop=True)


def grade_lists(root: Path, outcomes: pd.DataFrame, rules: Rules | None = None) -> pd.DataFrame:
    """One row per past list that has outcomes: realised range_from_open of the top-N picks
    (mean) vs the median of every stock in that list's universe, and the share of picks above
    that median. With `rules` the saved list is re-filtered, else its own rule_pass/rank stand."""
    rows = []
    for date in sorted(list_dates(root)):
        o = outcomes[outcomes["date"] == date]
        if o.empty:
            continue
        lst = load_list(root, date)
        n = (rules or Rules()).top_n
        if rules is not None:
            lst = apply_rules(lst, rules)
        g = lst.merge(o[["symbol", "range_from_open", "hl_range_open"]], on="symbol")
        if g.empty:
            continue
        med = g["range_from_open"].median()
        top = g[g["rule_pass"] & (g["rank"] <= n)]
        if top.empty:
            continue
        rows.append({"date": date, "state": lst["state"].iloc[0] if "state" in lst else "",
                     "picks": len(top), "universe": len(g),
                     "top_range": top["range_from_open"].mean(), "universe_median": med,
                     "ratio": top["range_from_open"].mean() / med if med else np.nan,
                     "hit_rate": float((top["range_from_open"] > med).mean()),
                     "predicted": top["pred_range"].mean()})
    return pd.DataFrame(rows)
