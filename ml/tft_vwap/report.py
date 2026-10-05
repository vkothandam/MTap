"""Calibration report over a finished run's saved test forecasts. Nothing is retrained.

    uv run tft-vwap-report runs/<run_id>                    # overall grid, ladder, bias leaders
    uv run tft-vwap-report runs/<run_id> --symbol AAPL      # one stock's grid / ladder / bias
    uv run tft-vwap-report runs/<run_id> --moves 0.01,0.02,-0.01

Writes runs/<run_id>/report/*.csv (see calibration.py for the definitions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .calibration import bias, grid, ladder_hits
from .config import TrainConfig, from_run

DAY_KEY = "date"


def load_test_forecasts(run_dir: Path, cfg: TrainConfig) -> pd.DataFrame:
    """Every fold's test forecasts, with date + industry joined from the run's panel."""
    paths = sorted(Path(run_dir).glob("fold_*/test_predictions.parquet"))
    if not paths:
        raise FileNotFoundError(f"no fold_*/test_predictions.parquet under {run_dir}")
    frame = pd.concat(
        [pd.read_parquet(p).assign(fold=int(p.parent.name.split("_")[1])) for p in paths],
        ignore_index=True,
    )
    need = [c for c in (DAY_KEY, "industry_code") if c not in frame.columns]  # older runs
    if need and cfg.panel.exists():
        keys = pd.read_parquet(cfg.panel, columns=["symbol", "day_idx", *need])
        frame = frame.merge(keys, on=["symbol", "day_idx"], how="left")
    if DAY_KEY not in frame.columns:  # panel moved: fall back to day_idx as the day key
        frame[DAY_KEY] = frame["day_idx"]
    if "industry_code" not in frame.columns:
        frame["industry_code"] = "unknown"
    frame["industry_code"] = frame["industry_code"].fillna("unknown").astype(str)
    return frame


def build_report(run_dir: Path, *, moves: list[float] | None = None,
                 prob_buckets: list[float] | None = None) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    cfg = from_run(run_dir, moves=moves, prob_buckets=prob_buckets)
    f = load_test_forecasts(run_dir, cfg)
    tables = {
        "grid_overall": grid(f, cfg),
        "grid_by_symbol": grid(f, cfg, ["symbol"]),
        "grid_by_industry": grid(f, cfg, ["industry_code"]),
        "grid_by_day": grid(f, cfg, [DAY_KEY]),
        "ladder_overall": ladder_hits(f, cfg),
        "ladder_by_symbol": ladder_hits(f, cfg, ["symbol"]),
        "bias_by_symbol": bias(f, cfg, ["symbol", "industry_code"]),
        "bias_by_day": bias(f, cfg, [DAY_KEY]),
    }
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    (out / "report_config.json").write_text(json.dumps({
        "forecasts": len(f), "symbols": int(f["symbol"].nunique()), "folds": sorted(f["fold"].unique().tolist()),
        "moves": cfg.moves, "prob_buckets": cfg.prob_buckets, "confidences": cfg.confidences,
    }, indent=2))
    return tables


def event_label(move: float) -> str:
    return f"{'>=' if move > 0 else '<='} {move:+.1%}"


def grid_pivot(g: pd.DataFrame) -> pd.DataFrame:
    """Rows = event, cols = forecast-probability bucket, cell = 'came true / said (n)'."""
    if g.empty:
        return pd.DataFrame()
    g = g.sort_values("move", key=lambda m: m.where(m < 0, 1 + m))  # downside first, then up
    cell = g.apply(lambda r: f"{r.hit_rate:.0%} / {r.mean_prob:.0%} ({r.n})", axis=1)
    t = g.assign(event=g["move"].map(event_label), cell=cell)
    order = list(dict.fromkeys(t["event"]))
    piv = t.pivot(index="event", columns="bucket", values="cell").reindex(order).fillna("")
    return piv[sorted(piv.columns, key=lambda b: float(b.split("%")[0]))]  # "5%-10%" first


def ladder_table(lh: pd.DataFrame) -> pd.DataFrame:
    t = lh.assign(call=lh["side"] + " " + lh["confidence"].map("{:.0%}".format))
    t = t.set_index("call")[["mean_move", "hit_rate", "deviation", "z", "n"]]
    return t.rename(columns={"mean_move": "avg_ladder_move"})


def _print_tables(tables: dict[str, pd.DataFrame], symbol: str | None, min_n: int, top: int) -> None:
    fmt = {"display.width": 220, "display.max_columns": 30, "display.max_rows": 200,
           "display.float_format": "{:.4f}".format}
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print("calibration grid, all stocks: cell = came true / model said (n)")
        print(grid_pivot(tables["grid_overall"]).to_string())
        print("\nladder: the move the model gives at each confidence, and how often it was cleared")
        print(ladder_table(tables["ladder_overall"]).to_string())
        b = tables["bias_by_symbol"]
        b = b[b["n"] >= min_n]
        cols = ["symbol", "industry_code", "n", "mean_pit", "bias", "bias_z", "up_deviation",
                "down_deviation", "label"]
        print(f"\nmost optimistic stocks (forecasts above what happened; n >= {min_n})")
        print(b.nlargest(top, "bias")[cols].to_string(index=False))
        print(f"\nmost pessimistic stocks (forecasts below what happened; n >= {min_n})")
        print(b.nsmallest(top, "bias")[cols].to_string(index=False))
        print("\nbias label counts:", b["label"].value_counts().to_dict())
        if symbol:
            lh, gs = tables["ladder_by_symbol"], tables["grid_by_symbol"]
            if not (lh["symbol"] == symbol).any():
                print(f"\n{symbol}: no test forecasts in this run")
                return
            print(f"\n{symbol}: calibration grid (came true / model said (n))")
            if (gs["symbol"] == symbol).any():
                print(grid_pivot(gs[gs["symbol"] == symbol]).to_string())
            else:
                print("no forecast reached the lowest probability bucket for any move")
            print(f"\n{symbol}: ladder")
            print(ladder_table(lh[lh["symbol"] == symbol]).to_string())
            bs = tables["bias_by_symbol"]
            print(f"\n{symbol}: bias")
            print(bs[bs["symbol"] == symbol][cols].to_string(index=False))


def _floats(s: str | None) -> list[float] | None:
    return [float(x) for x in s.split(",")] if s else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-report", description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--symbol")
    p.add_argument("--moves", help="comma list of fractional moves, e.g. 0.01,0.02,-0.01")
    p.add_argument("--prob-buckets", help="comma list of bucket edges, e.g. 0.5,0.6,0.7,0.8,0.9,0.95")
    p.add_argument("--min-n", type=int, default=60, help="min forecasts for the bias leaderboards")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args(argv)
    tables = build_report(a.run_dir, moves=_floats(a.moves), prob_buckets=_floats(a.prob_buckets))
    _print_tables(tables, a.symbol, a.min_n, a.top)
    print(f"\nreport: {a.run_dir / 'report'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
