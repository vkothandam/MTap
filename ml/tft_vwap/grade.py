"""Grade a run's live forecasts once their session is in the panel.

    uv run tft-vwap-grade runs/<run_id> [--panel PATH]

For every runs/<run_id>/forecast_<date>.parquet whose <date> now has bars in the panel, writes
one row per symbol to runs/<run_id>/actuals.parquet (rebuilt on each run):

    date          the forecast session
    symbol
    actual_move   vwap(date) / vwap(asof) - 1, both from the same panel (the model's target)
    actual_vwap   the forecast's today_vwap * (1 + actual_move): the actual on the forecast's
                  own price basis, so a split between asof and date can't distort the grade

The move is taken from one export, so the split adjustment is consistent across both sessions.
Forecasts without an `asof` column (older files) use the panel session just before <date>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import from_run

COLUMNS = ["date", "symbol", "actual_move", "actual_vwap"]


def grade(run_dir: Path, panel: Path | None = None) -> pd.DataFrame:
    run_dir = Path(run_dir)
    panel = panel or from_run(run_dir).panel
    p = pd.read_parquet(panel, columns=["symbol", "date", "today_vwap"])
    p["date"] = pd.to_datetime(p["date"])
    sessions = pd.Index(sorted(p["date"].unique()))
    vwap = p.set_index(["date", "symbol"])["today_vwap"]
    parts = []
    for f in sorted(run_dir.glob("forecast_*.parquet")):
        fc = pd.read_parquet(f)
        day = pd.Timestamp(fc["date"].iloc[0])
        if day not in sessions:
            continue
        if "asof" in fc:
            asof = pd.Timestamp(fc["asof"].iloc[0])
        else:
            i = sessions.get_loc(day)
            if i == 0:
                continue
            asof = sessions[i - 1]
        now = vwap.reindex(pd.MultiIndex.from_product([[day], fc["symbol"]])).to_numpy()
        then = vwap.reindex(pd.MultiIndex.from_product([[asof], fc["symbol"]])).to_numpy()
        move = now / then - 1
        g = pd.DataFrame({"date": day.date().isoformat(), "symbol": fc["symbol"],
                          "actual_move": move,
                          "actual_vwap": fc["today_vwap"].to_numpy() * (1 + move)})
        parts.append(g[g["actual_move"].notna()])
    out = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame({c: pd.Series(dtype=float if c.startswith("actual") else str)
                              for c in COLUMNS}))
    out.to_parquet(run_dir / "actuals.parquet", index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-grade", description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--panel", type=Path, help="panel holding the actuals (default: the run's)")
    a = p.parse_args(argv)
    out = grade(a.run_dir, a.panel)
    for day, g in out.groupby("date"):
        print(f"{day}: {len(g):,} symbols graded, median actual move {g['actual_move'].median():+.2%}")
    print(f"wrote {a.run_dir / 'actuals.parquet'} ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
