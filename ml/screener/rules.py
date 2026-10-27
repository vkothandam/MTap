"""Rule filters: the [screener] section of settings.toml, applied to the scored universe."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

EARNINGS_CHOICES = ("any", "only", "exclude")


@dataclass
class Rules:
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0  # 20-day mean of volume x VWAP
    min_pm_vol_pct: float = 0.0  # pm_vol_pct_10d (09:45 only; the 09:30 state has no volume)
    min_abs_gap_pct: float = 0.0  # |open vs previous close|, fraction (0.01 = 1%)
    exclude_split_day: bool = True
    earnings: str = "any"  # EARNINGS_CHOICES
    top_n: int = 20

    def __post_init__(self):
        if self.earnings not in EARNINGS_CHOICES:
            raise ValueError(f"earnings must be one of {EARNINGS_CHOICES}, got {self.earnings!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def load_rules(path: Path | str | None = None) -> Rules:
    """The [screener] section of a settings toml; defaults when no file or no section."""
    if path is None or not Path(path).exists():
        return Rules()
    section = tomllib.loads(Path(path).read_text()).get("screener", {})
    known = {f.name for f in fields(Rules)}
    return Rules(**{k: v for k, v in section.items() if k in known})


def rule_pass(df: pd.DataFrame, rules: Rules) -> pd.Series:
    """True per row that passes every rule. Missing inputs fail a rule only when it is active
    (a NaN dollar volume fails min_dollar_volume > 0)."""
    ok = pd.Series(True, index=df.index)
    price = df["open"].fillna(df.get("last", np.nan)) if "open" in df else df["last"]
    if rules.min_price > 0:
        ok &= price >= rules.min_price
    if rules.min_dollar_volume > 0:
        ok &= df["dollar_volume_20d"] >= rules.min_dollar_volume
    if rules.min_pm_vol_pct > 0 and "pm_vol_pct_10d" in df:
        ok &= df["pm_vol_pct_10d"] >= rules.min_pm_vol_pct
    if rules.min_abs_gap_pct > 0:
        ok &= df["gap_prev_close"].abs() >= rules.min_abs_gap_pct
    if rules.exclude_split_day and "split_day" in df:
        ok &= ~df["split_day"].fillna(False).astype(bool)
    if rules.earnings == "only":
        ok &= df["earn_near"] == 1
    elif rules.earnings == "exclude":
        ok &= df["earn_near"] != 1
    return ok.fillna(False)


def rank_passing(score: pd.Series, passed: pd.Series) -> pd.Series:
    """1 = best score among the rows that pass; <NA> for the rest."""
    r = score.where(passed).rank(ascending=False, method="first")
    return r.astype("Int64")
