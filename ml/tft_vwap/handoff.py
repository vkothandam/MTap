"""The daily forecast handed to the intraday model: one row per (symbol, date), out of fold only.

    uv run tft-vwap-handoff runs/<run_id> [--live] [--state night|px|full] [--out PATH]

Reads the run's fold_*/test_predictions.parquet: each walk-forward test fold forecasts its
sessions with a model trained only on earlier ones, so the folds together give one honest
forecast per (symbol, date). With --live it adds the run's forecast_<date>.parquet files (the
production model's live forecasts, graded from actuals.parquet), for dates after the last fold.
Never re-predicts a day the model trained on. Writes runs/<run_id>/handoff.parquet (or --out)
and handoff.meta.json next to it.

`date` is the forecast session D. Every column below is known before D opens. Moves are logs
vs `prev_vwap`, the VWAP of the stock's previous session:

    dq05 … dq95       the 19 quantiles, sorted
    d_med             P50
    d_width, d_width50, d_width90   P90 - P10, P75 - P25, P95 - P5
    d_skew            (P90 + P10 - 2 P50) / (P90 - P10)
    d_tail            (P95 - P5) / (P75 - P25)
    d_p_up            share of the 19 quantiles above 0
    d_width_rel       d_width / the stock's mean d_width over its previous HIST sessions
    hist_mae, hist_cov80, hist_bias   the forecast's |error| of P50, P10-P90 coverage and mean
                      signed error (actual - P50) over the stock's previous HIST sessions; the
                      actual of D-1 is VWAP(D-1), known at D's open
    hist_n            how many previous sessions those use (0: no history, the hist_* are NaN)
    sec_d_med         mean d_med of the other stocks in the sector (industry_code[:2]) that day;
                      NaN alone in the sector or with no numeric industry_code ("unknown")
    mkt_d_med         mean d_med of all stocks that day
    d_width_rank      d_width's percentile among that day's stocks (0 … 1)
    d_state           0 night, 1 09:30 prices, 2 09:45 full: which daily forecast this is
    fold, source      the daily test fold (-1 live) and 'oof' or 'live'

Symbols keep their series-break suffix removed (A~1 is the earlier security that traded as A).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import qlevels

HIST = 20
STATES = {"night": 0, "px": 1, "full": 2}


def _oof(run_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(run_dir.glob("fold_*/test_predictions.parquet")):
        f = pd.read_parquet(p)
        cols = list(qlevels(f).values())
        with np.errstate(invalid="ignore", divide="ignore"):  # a quantile <= 0: dropped in build
            lm = np.log(f[cols].to_numpy(float) / f["prev_vwap"].to_numpy(float)[:, None])
            act = np.log(f["actual"].to_numpy(float) / f["prev_vwap"].to_numpy(float))
        parts.append(pd.DataFrame({"symbol": f["symbol"].to_numpy(), "date": f["date"].to_numpy(),
                                   "industry_code": f["industry_code"].to_numpy(),
                                   "prev_vwap": f["prev_vwap"].to_numpy(), "_act": act,
                                   "fold": int(p.parent.name.split("_")[1]), "source": "oof"})
                     .join(pd.DataFrame(lm, columns=[f"dq{c[1:]}" for c in cols])))
    if not parts:
        raise FileNotFoundError(f"no fold_*/test_predictions.parquet under {run_dir}")
    return pd.concat(parts, ignore_index=True)


def _live(run_dir: Path) -> pd.DataFrame:
    """forecast_<date>.parquet (return quantiles vs today_vwap = VWAP(asof)), graded where
    actuals.parquet has the day."""
    files = sorted(run_dir.glob("forecast_????-??-??.parquet"))
    if not files:
        return pd.DataFrame()
    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    cols = list(qlevels(f).values())
    with np.errstate(invalid="ignore", divide="ignore"):
        lm = np.log1p(f[cols].to_numpy(float))
    out = pd.DataFrame({"symbol": f["symbol"].to_numpy(),
                        "date": pd.to_datetime(f["date"]).to_numpy(),
                        "industry_code": f["industry_code"].to_numpy(),
                        "prev_vwap": f["today_vwap"].to_numpy(), "_act": np.nan,
                        "fold": -1, "source": "live"}).join(
        pd.DataFrame(lm, columns=[f"dq{c[1:]}" for c in cols]))
    a = run_dir / "actuals.parquet"
    if a.exists():
        g = pd.read_parquet(a, columns=["date", "symbol", "actual_move"])
        g["date"] = pd.to_datetime(g["date"])
        out = out.drop(columns="_act").merge(g, on=["date", "symbol"], how="left")
        out["_act"] = np.log1p(out.pop("actual_move"))
    return out


def build(run_dir: Path, live: bool = False, state: str = "night") -> pd.DataFrame:
    run_dir = Path(run_dir)
    d = _oof(run_dir)
    d["date"] = pd.to_datetime(d["date"])
    if live:
        lv = _live(run_dir)
        if not lv.empty:
            d = pd.concat([d, lv[lv["date"] > d["date"].max()]], ignore_index=True)
    dq = [c for c in d.columns if c.startswith("dq")]
    # a series break (A~1, the earlier security under ticker A): the current one wins a shared date
    d["_older"] = d["symbol"].str.contains("~", regex=False)
    d["symbol"] = d["symbol"].str.split("~").str[0]
    ok = (d["prev_vwap"] > 0) & np.isfinite(d[dq]).all(axis=1)
    d = (d[ok].sort_values(["symbol", "date", "_older"])
         .drop_duplicates(["symbol", "date"], keep="first").drop(columns="_older"))
    d[dq] = np.sort(d[dq].to_numpy(float), axis=1)

    q = {c: d[c].to_numpy() for c in dq}
    p05, p10, p25, p50, p75, p90, p95 = (q[f"dq{k}"] for k in ("05", "10", "25", "50", "75",
                                                               "90", "95"))
    d["d_med"] = p50
    d["d_width"], d["d_width50"], d["d_width90"] = p90 - p10, p75 - p25, p95 - p05
    with np.errstate(invalid="ignore", divide="ignore"):
        d["d_skew"] = np.where(p90 > p10, (p90 + p10 - 2 * p50) / (p90 - p10), 0.0)
        d["d_tail"] = np.where(p75 > p25, (p95 - p05) / (p75 - p25), np.nan)
    d["d_p_up"] = (d[dq].to_numpy() > 0).mean(axis=1)

    # the stock's own record, previous sessions only (shift 1)
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    d["_err"] = d["_act"] - d["d_med"]
    d["_in"] = ((d["_act"] >= d["dq10"]) & (d["_act"] <= d["dq90"])).astype(float)
    d.loc[d["_act"].isna(), "_in"] = np.nan
    g = d.groupby("symbol", sort=False)

    def past(col: str, how: str = "mean") -> pd.Series:
        return g[col].transform(lambda s: getattr(s.shift(1).rolling(HIST, min_periods=1), how)())

    d["_abs"] = d["_err"].abs()
    d["hist_mae"], d["hist_cov80"], d["hist_bias"] = past("_abs"), past("_in"), past("_err")
    d["hist_n"] = past("_err", "count").fillna(0).astype("int16")
    with np.errstate(invalid="ignore", divide="ignore"):
        d["d_width_rel"] = d["d_width"] / past("d_width")

    # the day's context
    # a sector is a numeric industry_code's first 2 digits; "unknown" stocks have none
    sec = d["industry_code"].astype(str).str[:2]
    d["_sec"] = sec.where(sec.str.fullmatch(r"\d\d"), "")
    by_day = d.groupby("date")
    d["mkt_d_med"] = by_day["d_med"].transform("mean")
    d["d_width_rank"] = by_day["d_width"].rank(pct=True)
    s = d.groupby(["date", "_sec"])["d_med"]
    n = s.transform("count") - 1
    d["sec_d_med"] = np.where((n > 0) & (d["_sec"] != ""),
                              (s.transform("sum") - d["d_med"]) / n.clip(lower=1), np.nan)

    d["d_state"] = np.int8(STATES[state])
    cols = ["symbol", "date", "industry_code", "prev_vwap", *dq, "d_med", "d_width",
            "d_width50", "d_width90", "d_skew", "d_tail", "d_p_up", "d_width_rel", "hist_mae",
            "hist_cov80", "hist_bias", "hist_n", "sec_d_med", "mkt_d_med", "d_width_rank",
            "d_state", "fold", "source"]
    out = d[cols].copy()
    f32 = [c for c in cols[4:] if c not in ("hist_n", "d_state", "fold", "source")]
    out[f32] = out[f32].astype("float32")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tft-vwap-handoff", description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--live", action="store_true",
                   help="add the run's live forecast_<date>.parquet after the last fold")
    p.add_argument("--state", choices=list(STATES), default="night",
                   help="which daily forecast the run's files are (recorded as d_state)")
    p.add_argument("--out", type=Path, help="default: <run_dir>/handoff.parquet")
    a = p.parse_args(argv)
    out = build(a.run_dir, live=a.live, state=a.state)
    path = a.out or a.run_dir / "handoff.parquet"
    out.to_parquet(path, index=False)
    meta = {"daily_run": str(a.run_dir), "state": a.state, "hist_sessions": HIST,
            "rows": len(out), "symbols": int(out["symbol"].nunique()),
            "first_date": str(out["date"].min().date()), "last_date": str(out["date"].max().date()),
            "sources": out["source"].value_counts().to_dict(),
            "folds": {int(k): [str(v.min().date()), str(v.max().date())]
                      for k, v in out.groupby("fold")["date"]}}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}: {len(out):,} rows, {meta['symbols']:,} symbols, "
          f"{meta['first_date']} … {meta['last_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
