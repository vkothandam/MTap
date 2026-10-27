"""Tomorrow's next-day VWAP move distribution from a frozen, already-trained checkpoint.

    uv run tft-vwap-predict runs/<run_id>                                  # counts per move / call
    uv run tft-vwap-predict runs/<run_id> --symbol AAPL                    # one stock's spread
    uv run tft-vwap-predict runs/<run_id> --side up --move 0.01 --confidence 0.7   # screener
    uv run tft-vwap-predict runs/<run_id> --asof 2026-10-02     # forecast a past session, to grade

The checkpoint is only read: no training, no weight updates. Its stored dataset parameters (the
fitted encoders/normalisers) are applied to the newest panel rows, and one future row per symbol
at the next session (calendar from panel.meta.json["next_session"]) is the decoder step.
Each probability sits next to how often that stock's test forecasts in the same probability
bucket came true (report/grid_by_symbol.csv), and the all-stock figure (grid_overall.csv).

Writes runs/<run_id>/forecast_<date>.{parquet,csv} (one row per symbol: quantiles, ladder) and
forecast_<date>_moves.csv (one row per symbol and move: probability + historical hit rates).
A model with pre-market inputs reads them from panel.next.parquet (next to the panel, filled by
`etrade export-next`; same export_id as the meta). --state sets which pre-market the forecast
sees: night (both groups blank), px (prices only: full blanked) or full (the panel.next flags
as they are; prices blank forces full blank too, as in training). Each state writes into its
own run dir: full -> runs/<id>/ (the main run), night -> runs/<id>~night/, px -> runs/<id>~px/
(the sibling dirs `optional` training creates; made here for other runs, with the main run's
report grids), so forecast_<date>.parquet never collides between states and grade.py's glob
forecast_????-??-??.parquet grades each dir. Without --state an `optional` run writes all its
states; any other run writes its own (full, or px for --pm-groups px).
--symbols-file F (one symbol per line) restricts the forecast to those symbols: the rows equal
the full run's rows for them. It writes forecast_<date>_subset.* (not graded, never replaces
the full file), e.g. the 09:30 shortlist.
<date> is the forecast session; `asof` is the session whose bars it was made from. --asof D
cuts the panel at D, so the session after D (already in the panel) can be graded against what
happened. Every run then refreshes actuals.parquet (see grade.py).
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
from .config import (
    KNOWN_REALS,
    PM_FULL_REALS,
    PM_GROUPS,
    PREMARKET_REALS,
    STATE_SUFFIX,
    STATES,
    TrainConfig,
    from_run,
)
from .dataset import prepare, read_meta
from .evaluate import prob_below, qcol
from .grade import grade
from .report import build_report, event_label


def latest_fold(run_dir: Path) -> int:
    folds = sorted(int(p.parent.name.split("_")[1]) for p in run_dir.glob("fold_*/best.ckpt"))
    if not folds:
        raise FileNotFoundError(f"no fold_*/best.ckpt under {run_dir}")
    return folds[-1]


def force_state(next_df: pd.DataFrame, state: str, per_symbol: list[str]) -> pd.DataFrame:
    """Set the pre-market inputs of the next-session rows to a scoring state: night blanks every
    group (value 0, flag 0), px blanks the full group, full keeps the data. Prices blank
    always blanks full too (the model never saw full without prices)."""
    nxt = next_df.copy()
    for c in per_symbol:
        if c not in nxt.columns:
            nxt[c] = 0.0
    if state == "night":
        nxt[per_symbol] = 0.0
        return nxt
    flag = PM_GROUPS["px"][-1]
    if flag in nxt.columns:
        off = nxt[flag].fillna(0) == 0
        for c in PM_FULL_REALS:
            if c in nxt.columns:
                nxt.loc[off, c] = 0.0
    if state == "px":
        for c in PM_FULL_REALS:
            if c in nxt.columns:
                nxt[c] = 0.0
    return nxt


def future_rows(df: pd.DataFrame, next_session: dict, known_reals: list[str] | None = None,
                next_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per symbol trading on the panel's last session, moved to the next session.
    Every known-future input of the model is filled: calendar ones from `next_session`, the
    per-symbol ones (pre-market) from `next_df` (symbol + those columns). A symbol missing from
    `next_df` gets blank pre-market (flags pm_px_known / pm_full_known 0, values 0). Anything else unfilled raises; the
    remaining columns are copies the decoder never reads."""
    known_reals = list(KNOWN_REALS if known_reals is None else known_reals)
    last = int(df["day_idx"].max())
    if next_session["day_idx"] != last + 1:
        raise ValueError(f"meta next_session day_idx {next_session['day_idx']} != last panel "
                         f"day_idx {last} + 1; re-export the panel")
    fut = df[df["day_idx"] == last].copy()
    fut["day_idx"] = last + 1
    per_symbol = [c for c in known_reals if c not in KNOWN_REALS]
    for c in known_reals:
        if c in KNOWN_REALS:
            fut[c] = float(next_session[c])
    if per_symbol:
        if next_df is None:
            raise ValueError(f"model needs per-symbol known inputs {per_symbol} but no "
                             "next-session rows were given (panel.next.parquet)")
        lacking = [c for c in per_symbol if c not in next_df.columns]
        if lacking:
            raise ValueError(f"next-session rows lack known inputs {lacking}")
        nxt = next_df.drop_duplicates("symbol").set_index("symbol")
        missing = ~fut["symbol"].isin(nxt.index)
        for c in per_symbol:
            fut[c] = fut["symbol"].map(nxt[c])
            if c in PREMARKET_REALS:  # no pre-market row: blank, as in training
                fut.loc[missing, c] = 0.0
            elif missing.any():
                raise ValueError(f"known input {c} is unfilled for {int(missing.sum())} symbols")
            fut[c] = fut[c].astype("float64")
    unfilled = [c for c in known_reals if fut[c].isna().any()]
    if unfilled:
        raise ValueError(f"known inputs unfilled for the next session: {unfilled}")
    return fut


def read_next(cfg: TrainConfig) -> pd.DataFrame:
    """panel.next.parquet beside the panel (symbol, day_idx, per-symbol known inputs and an
    export_id column), checked against the meta: same export, and the pre-market cutoff the
    model was trained with."""
    path = cfg.panel.with_name(cfg.panel.stem + ".next.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: the model uses pre-market inputs; run "
                                "`sourcing-py etrade export-next`")
    meta = read_meta(cfg.panel)
    nxt = pd.read_parquet(path)
    ids = set(nxt["export_id"].astype(str)) if "export_id" in nxt else set()
    if meta.get("export_id") is None or ids != {str(meta["export_id"])}:
        raise ValueError(f"{path.name} export_id {sorted(ids)} != panel meta "
                         f"{meta.get('export_id')!r}: they come from different exports")
    if cfg.premarket_cutoff and meta.get("premarket_cutoff") != cfg.premarket_cutoff:
        raise ValueError(f"panel pre-market cutoff {meta.get('premarket_cutoff')!r} != the "
                         f"model's {cfg.premarket_cutoff!r}")
    return nxt


def session_after(df: pd.DataFrame, day_idx: int) -> dict:
    """The calendar of the session after `day_idx`, read off the panel's own rows there (same
    keys as meta next_session)."""
    rows = df[df["day_idx"] == day_idx + 1]
    if rows.empty:
        raise ValueError(f"no panel rows on day_idx {day_idx + 1}: nothing to forecast after it")
    r = rows.iloc[0]
    return {"date": pd.Timestamp(r["date"]).date().isoformat(), "day_idx": day_idx + 1,
            **{c: float(r[c]) for c in KNOWN_REALS}}


def forecast(run_dir: Path, *, fold: int | None = None, cfg: TrainConfig | None = None,
             panel: Path | None = None, asof: str | None = None,
             state: str = "full", symbols: list[str] | None = None
             ) -> tuple[pd.DataFrame, dict]:
    """(per-symbol forecast frame with return quantiles + ladder, next_session). With `asof`,
    the panel is cut at that session (the latest one on or before it) and the forecast is for
    the session after it. `state` (STATES) sets the pre-market the forecast sees; `symbols`
    restricts it to those symbols (same rows as the full forecast for them)."""
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    run_dir = Path(run_dir)
    cfg = cfg or from_run(run_dir, panel=panel)
    fold = latest_fold(run_dir) if fold is None else fold
    df = prepare(pd.read_parquet(cfg.panel), symbols=cfg.symbols, seed=cfg.seed,
                 series_breaks=cfg.series_breaks, extra=cfg.extra_reals)
    model = TemporalFusionTransformer.load_from_checkpoint(
        run_dir / f"fold_{fold}" / "best.ckpt", map_location="cpu")
    model.eval()
    known = list(model.dataset_parameters["time_varying_known_reals"])
    per_symbol = [c for c in known if c not in KNOWN_REALS]
    next_df = None
    if asof is None:
        if per_symbol and state != "night":
            next_df = read_next(cfg)
        nxt = read_meta(cfg.panel).get("next_session")
        if not nxt:
            raise ValueError(f"{cfg.panel}: meta has no next_session; re-export with "
                             "`sourcing-py etrade export-tft --include-inference`")
    else:
        on_or_before = df.loc[pd.to_datetime(df["date"]) <= pd.Timestamp(asof), "day_idx"]
        if on_or_before.empty:
            raise ValueError(f"--asof {asof} is before the panel starts")
        nxt = session_after(df, int(on_or_before.max()))
        # the real next-day rows, before the cut: only the known pre-market columns are copied
        next_df = df.loc[df["day_idx"] == nxt["day_idx"], ["symbol", *per_symbol]]
        df = df[df["day_idx"] < nxt["day_idx"]]
    if per_symbol:
        if next_df is None:  # night: nothing to read
            next_df = pd.DataFrame({c: [] for c in ["symbol", *per_symbol]})
        next_df = force_state(next_df, state, per_symbol)
    last = int(df["day_idx"].max())
    active = df.loc[df["day_idx"] == last, "symbol"]
    if symbols is not None:
        active = active[active.isin(symbols)]
    hist = df[df["symbol"].isin(active) & (df["day_idx"] > last - cfg.encoder_length)]
    fut = future_rows(df, nxt, known, next_df)
    data = pd.concat([hist, fut[fut["symbol"].isin(active)]], ignore_index=True)
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
    out.insert(1, "asof", pd.Timestamp(today["date"].iloc[0]).date().isoformat())
    out.insert(2, "fold", fold)
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


def read_symbols(path: Path) -> list[str]:
    """One symbol per line, or a CSV with a `symbol` column (the morning's shortlist-D.csv)."""
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if lines and lines[0].split(",")[0].strip().lower() == "symbol":
        return [x.split(",")[0].strip() for x in lines[1:]]
    return lines


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
    p.add_argument("--asof", help="forecast the session after this date (YYYY-MM-DD) from the "
                                  "panel as it stood then (default: after the panel's last day)")
    p.add_argument("--state", choices=STATES, help="pre-market the forecast sees (see above; "
                   "default: all states of an optional run, else the run's own)")
    p.add_argument("--symbols-file", type=Path, help="only these symbols (one per line); "
                   "written as forecast_<date>_subset.*")
    a = p.parse_args(argv)
    if (a.side is None) != (a.move is None):
        p.error("--side and --move go together")

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # device banner, tips
    warnings.filterwarnings("ignore", module=r"lightning\..*")
    run_dir = a.run_dir
    cfg = from_run(run_dir, panel=a.panel)
    symbols = read_symbols(a.symbols_file) if a.symbols_file else None
    main_state = "full" if "full" in cfg.pm_group_names else "px"
    if a.state:
        states = [a.state]
    elif cfg.premarket == "optional":
        states = [*cfg.scoring_states[:-1], main_state]
    else:
        states = [main_state]
    screen_move = None if a.move is None else abs(a.move) * (1 if a.side == SIDE_UP else -1)
    moves = sorted({*cfg.moves, *([screen_move] if screen_move is not None else [])})
    suffix = "_subset" if symbols is not None else ""

    def state_dir(state: str) -> Path:
        # the main run's own state lives in run_dir; the others in sibling dirs
        if state == main_state or cfg.premarket == "off":
            return run_dir
        return run_dir.with_name(run_dir.name + STATE_SUFFIX[state])

    def write(directory: Path, frame: pd.DataFrame, date: str) -> pd.DataFrame:
        # a sibling without its own test forecasts (not trained as `optional`) reuses the
        # main run's grids
        own = any(directory.glob("fold_*/test_predictions.parquet"))
        grid_sym, grid_all = _load_grids(directory if own else run_dir)
        table = move_table(frame, cfg, grid_sym, grid_all, moves)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"forecast_{date}{suffix}"
        frame.to_parquet(f"{out}.parquet", index=False)
        frame.to_csv(f"{out}.csv", index=False)
        table.to_csv(f"{out}_moves.csv", index=False)
        return table

    written, shown = [], None
    for state in states:
        fc, nxt = forecast(run_dir, fold=a.fold, cfg=cfg, asof=a.asof, state=state,
                           symbols=symbols)
        d = state_dir(state)
        mt = write(d, fc, nxt["date"])
        written.append(d)
        print(f"{state}: forecast for {nxt['date']} from fold_{int(fc['fold'].iloc[0])}: "
              f"{len(fc):,} symbols -> {d / ('forecast_' + nxt['date'] + suffix)}.parquet")
        if state == states[-1]:
            shown = (fc, mt)
    fc, mt = shown
    stem = state_dir(states[-1]) / f"forecast_{nxt['date']}{suffix}"

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
    if not suffix:
        for d in dict.fromkeys(written):
            act = grade(d, cfg.panel)
            print(f"{d.name}/actuals.parquet: {act['date'].nunique()} forecast day(s) graded, "
                  f"{len(act):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
