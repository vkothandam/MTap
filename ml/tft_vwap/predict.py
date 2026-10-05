"""Tomorrow's next-day VWAP move distribution from a frozen, already-trained checkpoint.

    uv run tft-vwap-predict runs/<run_id>                                  # counts per move / call
    uv run tft-vwap-predict runs/<run_id> --symbol AAPL                    # one stock's spread
    uv run tft-vwap-predict runs/<run_id> --side up --move 0.01 --confidence 0.7   # screener

The checkpoint is only read: no training, no weight updates. Its stored dataset parameters (the
fitted encoders/normalisers) are applied to the newest panel rows, and one future row per symbol
at the next session (calendar from panel.meta.json["next_session"]) is the decoder step.
Each probability sits next to how often that stock's test forecasts in the same probability
bucket came true (report/grid_by_symbol.csv), and the all-stock figure (grid_overall.csv).

Writes runs/<run_id>/forecast_<date>.{parquet,csv} (one row per symbol: quantiles, ladder) and
forecast_<date>_moves.csv (one row per symbol and move: probability + historical hit rates).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from .calibration import SIDE_DOWN, SIDE_UP, bucket_labels, bucket_of, ladder
from .config import KNOWN_REALS, TrainConfig, from_run
from .dataset import prepare, read_meta
from .evaluate import prob_below, qcol
from .report import build_report, event_label


def latest_fold(run_dir: Path) -> int:
    folds = sorted(int(p.parent.name.split("_")[1]) for p in run_dir.glob("fold_*/best.ckpt"))
    if not folds:
        raise FileNotFoundError(f"no fold_*/best.ckpt under {run_dir}")
    return folds[-1]


def future_rows(df: pd.DataFrame, next_session: dict) -> pd.DataFrame:
    """One row per symbol trading on the panel's last session, moved to the next session.
    Known inputs come from the calendar; the rest are copies the decoder never reads."""
    last = int(df["day_idx"].max())
    if next_session["day_idx"] != last + 1:
        raise ValueError(f"meta next_session day_idx {next_session['day_idx']} != last panel "
                         f"day_idx {last} + 1; re-export the panel")
    fut = df[df["day_idx"] == last].copy()
    fut["day_idx"] = last + 1
    for c in KNOWN_REALS:
        fut[c] = float(next_session[c])
    return fut


def forecast(run_dir: Path, *, fold: int | None = None, cfg: TrainConfig | None = None,
             panel: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """(per-symbol forecast frame with return quantiles + ladder, next_session)."""
    run_dir = Path(run_dir)
    cfg = cfg or from_run(run_dir, panel=panel)
    fold = latest_fold(run_dir) if fold is None else fold
    meta = read_meta(cfg.panel)
    nxt = meta.get("next_session")
    if not nxt:
        raise ValueError(f"{cfg.panel}: meta has no next_session; re-export with "
                         "`sourcing-py etrade export-tft --include-inference`")

    df = prepare(pd.read_parquet(cfg.panel), symbols=cfg.symbols, seed=cfg.seed)
    last = int(df["day_idx"].max())
    active = df.loc[df["day_idx"] == last, "symbol"]
    hist = df[df["symbol"].isin(active) & (df["day_idx"] > last - cfg.encoder_length)]
    data = pd.concat([hist, future_rows(df, nxt)], ignore_index=True)

    model = TemporalFusionTransformer.load_from_checkpoint(
        run_dir / f"fold_{fold}" / "best.ckpt", map_location="cpu")
    model.eval()
    ds = TimeSeriesDataSet.from_parameters(model.dataset_parameters, data, predict=True)
    pred = model.predict(
        ds.to_dataloader(train=False, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers),
        mode="quantiles", return_index=True,
        trainer_kwargs={"accelerator": cfg.accelerator, "devices": 1, "logger": False,
                        "enable_progress_bar": False},
    )
    q = np.sort(pred.output[:, 0, :].cpu().numpy(), axis=1)

    out = pred.index[["symbol"]].reset_index(drop=True)
    today = df[df["day_idx"] == last].set_index("symbol")
    out["industry_code"] = out["symbol"].map(today["industry_code"])
    out["today_vwap"] = out["symbol"].map(today["today_vwap"])
    r_q = q if cfg.target == "return" else q / out["today_vwap"].to_numpy()[:, None] - 1
    out.insert(0, "date", nxt["date"])
    out.insert(1, "fold", fold)
    for i, level in enumerate(cfg.quantiles):
        out[qcol(level)] = r_q[:, i]  # fractional move vs today's vwap
    for name, move in ladder(r_q, cfg.quantiles, cfg.confidences).items():
        out[name] = move
    return out.sort_values("symbol").reset_index(drop=True), nxt


def move_table(fc: pd.DataFrame, cfg: TrainConfig, grid_sym: pd.DataFrame | None,
               grid_all: pd.DataFrame | None, moves: list[float] | None = None) -> pd.DataFrame:
    """One row per (symbol, move): model probability, its bucket, and the historical hit rate
    of that bucket for the stock (hist_*) and for all stocks (all_*)."""
    r_q = fc[[qcol(level) for level in cfg.quantiles]].to_numpy()
    labels = np.asarray(bucket_labels(cfg.prob_buckets))
    parts = []
    for move in moves or cfg.moves:
        below = prob_below(r_q, cfg.quantiles, move)
        prob = 1 - below if move > 0 else below
        b = bucket_of(prob, cfg.prob_buckets)
        parts.append(pd.DataFrame({
            "symbol": fc["symbol"], "side": SIDE_UP if move > 0 else SIDE_DOWN, "move": move,
            "prob": prob, "bucket": np.where(b >= 0, labels[np.clip(b, 0, None)], "below grid"),
        }))
    t = pd.concat(parts, ignore_index=True)
    cols = ["n", "hit_rate", "deviation"]
    for prefix, g, keys in (("hist", grid_sym, ["symbol", "move", "bucket"]),
                            ("all", grid_all, ["move", "bucket"])):
        if g is None or g.empty:
            for c in cols:
                t[f"{prefix}_{c}"] = np.nan
            continue
        g = g.assign(move=g["move"].round(6))[[*keys, *cols]]
        t = t.assign(move=t["move"].round(6)).merge(
            g.rename(columns={c: f"{prefix}_{c}" for c in cols}), on=keys, how="left")
    return t


def _load_grids(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep = run_dir / "report"
    if not (rep / "grid_by_symbol.csv").exists():
        print("report/ missing: building it from the run's test forecasts first")
        build_report(run_dir)
    read = lambda name: pd.read_csv(rep / f"{name}.csv", dtype={"symbol": str})
    return read("grid_by_symbol"), read("grid_overall")


def _pct(x) -> str:
    return "" if pd.isna(x) else f"{x:.0%}"


def _symbol_view(sym: str, fc: pd.DataFrame, mt: pd.DataFrame, cfg: TrainConfig) -> str:
    row = fc[fc["symbol"] == sym]
    if row.empty:
        return f"{sym}: no forecast (not trading on the panel's last session, or too little history)"
    r = row.iloc[0]
    lines = [f"{sym} ({r['industry_code']}), today vwap {r['today_vwap']:.2f}, forecast for {r['date']}"]
    lad = pd.DataFrame({
        "confidence": [f"{c:.0%}" for c in cfg.confidences],
        "upside (move >= x)": [f"{r[f'up_{round(c * 100)}']:+.2%}" for c in cfg.confidences],
        "downside (move <= x)": [f"{r[f'down_{round(c * 100)}']:+.2%}" for c in cfg.confidences],
    })
    lines += ["\nladder: the move the model gives at each confidence", lad.to_string(index=False)]
    m = mt[mt["symbol"] == sym].sort_values("move", key=lambda s: s.where(s < 0, 1 + s))
    view = pd.DataFrame({
        "event": m["move"].map(event_label), "model": m["prob"].map(_pct), "bucket": m["bucket"],
        f"{sym} came true": m["hist_hit_rate"].map(_pct), "n": m["hist_n"].astype("Int64"),
        "all stocks came true": m["all_hit_rate"].map(_pct), "all n": m["all_n"].astype("Int64"),
    })
    lines += ["\nprobability per move vs how often the same bucket came true in testing",
              view.to_string(index=False)]
    return "\n".join(lines)


def _screen(mt: pd.DataFrame, move: float, confidence: float) -> pd.DataFrame:
    m = mt[np.isclose(mt["move"], move) & (mt["prob"] >= confidence - 1e-9)]
    return m.sort_values("prob", ascending=False)


def _counts(mt: pd.DataFrame, confidences: list[float]) -> pd.DataFrame:
    t = pd.DataFrame({f">= {c:.0%}": mt.assign(ok=mt["prob"] >= c - 1e-9)
                      .groupby("move", sort=True)["ok"].sum() for c in confidences})
    t.index = t.index.map(event_label)
    return t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-predict", description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--fold", type=int, help="checkpoint to use (default: the latest fold)")
    p.add_argument("--panel", type=Path, help="panel to forecast from (default: the run's)")
    p.add_argument("--symbol")
    p.add_argument("--side", choices=[SIDE_UP, SIDE_DOWN])
    p.add_argument("--move", type=float, help="screener move size, e.g. 0.01 for 1%%")
    p.add_argument("--confidence", type=float, default=0.7, help="screener minimum probability")
    p.add_argument("--top", type=int, default=50)
    a = p.parse_args(argv)
    if (a.side is None) != (a.move is None):
        p.error("--side and --move go together")

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # device banner, tips
    warnings.filterwarnings("ignore", module=r"lightning\..*")
    run_dir = a.run_dir
    cfg = from_run(run_dir, panel=a.panel)
    fc, nxt = forecast(run_dir, fold=a.fold, cfg=cfg)
    screen_move = None if a.move is None else abs(a.move) * (1 if a.side == SIDE_UP else -1)
    moves = sorted({*cfg.moves, *([screen_move] if screen_move is not None else [])})
    grid_sym, grid_all = _load_grids(run_dir)
    mt = move_table(fc, cfg, grid_sym, grid_all, moves)

    stem = run_dir / f"forecast_{nxt['date']}"
    fc.to_parquet(f"{stem}.parquet", index=False)
    fc.to_csv(f"{stem}.csv", index=False)
    mt.to_csv(f"{stem}_moves.csv", index=False)
    print(f"forecast for {nxt['date']} from fold_{int(fc['fold'].iloc[0])}: {len(fc):,} symbols")

    with pd.option_context("display.width", 220, "display.max_rows", 500):
        if a.symbol:
            print(_symbol_view(a.symbol, fc, mt, cfg))
        elif screen_move is not None:
            s = _screen(mt, screen_move, a.confidence)
            print(f"\n{len(s):,} stocks with P({event_label(screen_move)}) >= {a.confidence:.0%}"
                  " (hist = that stock's test hit rate in the same bucket; all = every stock)")
            show = s.head(a.top).assign(prob=s["prob"].map(_pct),
                                        hist_hit_rate=s["hist_hit_rate"].map(_pct),
                                        all_hit_rate=s["all_hit_rate"].map(_pct),
                                        hist_n=s["hist_n"].astype("Int64"),
                                        all_n=s["all_n"].astype("Int64"))
            print(show[["symbol", "prob", "bucket", "hist_hit_rate", "hist_n", "all_hit_rate",
                        "all_n"]].to_string(index=False))
        else:
            print("\nstocks per event and call level (P(event) >= level)")
            print(_counts(mt, cfg.confidences).to_string())
    print(f"\nwrote {stem}.parquet/.csv and {stem.name}_moves.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
