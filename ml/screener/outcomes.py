"""screener-outcomes: what each stock did on past days, for the dashboard to grade old lists.

Writes <out-dir>/outcomes.parquet: date, symbol, open, high, low, range_from_open,
hl_range_open for every panel row in the last --days sessions. The dashboard reads this file
only (not the panel)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import bars, targets


def outcomes(panel: Path, days: int = 60) -> pd.DataFrame:
    df = pd.read_parquet(panel)
    df = df[df["day_idx"] > df["day_idx"].max() - days]
    b = bars(df)
    out = b[["symbol", "day_idx", "open", "high", "low"]].merge(
        targets(b), on=["symbol", "day_idx"])
    out.insert(0, "date", pd.to_datetime(df["date"]).dt.date.astype(str).to_numpy())
    return out.dropna(subset=["range_from_open"]).drop(columns="day_idx").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out-dir", type=Path, default=Path("../out/screener"))
    a = ap.parse_args(argv)
    out = outcomes(a.panel, a.days)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out_dir / "outcomes.parquet", index=False)
    print(f"{len(out):,} rows, {out['date'].nunique()} days -> {a.out_dir / 'outcomes.parquet'}")


if __name__ == "__main__":
    main()
