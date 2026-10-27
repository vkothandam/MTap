"""screener-score: today's ranked list from live opens, pre-market and the TFT forecast.

    uv run screener-score --date D --model runs/screener/<id> --prices prices-D.jsonl \
        --pm premarket-D.jsonl --forecast forecast_D.parquet --state px|full \
        [--symbols-file shortlist-D.csv] [--panel panel.parquet]

Writes out/screener/screener-D.parquet (the whole universe) and screener-D-top20.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    PM_CLIP,
    advance_earnings,
    feature_names,
    finish,
    history_features,
    tft_features,
)
from .model import ScreenerModel, load
from .rules import Rules, load_rules, rule_pass, rank_passing

HISTORY_SESSIONS = 40  # enough for ATR(14) and the 20-day dollar volume


def read_jsonl(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["symbol"])
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol"])


def latest_history(panel_path: Path) -> pd.DataFrame:
    """Last known session's history features per symbol (the previous session of the target)."""
    import pyarrow.parquet as pq

    last = pq.read_table(panel_path, columns=["day_idx"]).column("day_idx").to_numpy().max()
    panel = pd.read_parquet(panel_path, filters=[("day_idx", ">=", int(last) - HISTORY_SESSIONS)])
    h = history_features(panel)
    h = h[h["day_idx"] == h.groupby("symbol")["day_idx"].transform("max")]
    return advance_earnings(h).reset_index(drop=True)


def live_frame(hist: pd.DataFrame, fc: pd.DataFrame, prices: pd.DataFrame, pm: pd.DataFrame,
               date: str, state: str) -> pd.DataFrame:
    """Join the pieces for the symbols in the TFT forecast (the universe)."""
    d = tft_features(fc, prices=False).merge(hist.drop(columns=["day_idx"]), on="symbol",
                                             how="left")
    keep = [c for c in ("symbol", "open", "last") if c in prices.columns]
    d = d.merge(prices[keep].drop_duplicates("symbol"), on="symbol", how="left")
    if "last" not in d:
        d["last"] = np.nan
    if "open" not in d:
        d["open"] = np.nan
    d["last"] = d["last"].astype(float)
    d["open"] = d["open"].astype(float)
    pv = d["prev_vwap"]
    d["pm_last_pct_prev_vwap"] = (d["last"] / pv - 1).clip(-PM_CLIP, PM_CLIP)
    pmc = pm.drop_duplicates("symbol").set_index("symbol") if len(pm) else pd.DataFrame()
    ok = pd.Series(False, index=d.index)
    for c in ("pm_vwap", "pm_volume", "pm_high", "pm_low"):
        d[c] = d["symbol"].map(pmc[c]) if c in pmc.columns else np.nan
    if "complete" in pmc.columns:
        ok = d["symbol"].map(pmc["complete"]).fillna(False).astype(bool)
    d["pm_vwap_pct_prev_vwap"] = ((d["pm_vwap"] / pv - 1).clip(-PM_CLIP, PM_CLIP)).where(ok)
    vol = 100 * d["pm_volume"] / d.get("vol_10d_mean", np.nan)
    d["pm_vol_pct_10d"] = np.log1p(vol).clip(0, 6).where(ok)
    d["pm_range_pct"] = ((d["pm_high"] - d["pm_low"]) / pv).where(ok)
    d["date"] = date
    d["day_of_week"] = pd.Timestamp(date).dayofweek + 1
    d["split_day"] = False
    d["dollar_volume_20d"] = d["dollar_volume_20d"]
    return finish(d)


def score_frame(d: pd.DataFrame, model: ScreenerModel, rules: Rules) -> pd.DataFrame:
    """Score, apply rules, rank among the passing. Rows without an open score NaN and fail."""
    out = d.copy()
    pred = model.predict(out)
    has_open = out["open"].notna()
    out["pred_range"] = pred["pred_range"].where(has_open)
    out["score"] = pred["score"].where(has_open)
    out["rule_pass"] = rule_pass(out, rules) & has_open
    out["rank"] = rank_passing(out["score"], out["rule_pass"])
    return out.sort_values(["rank", "score"], ascending=[True, False], na_position="last") \
        .reset_index(drop=True)


OUT_COLUMNS = ["symbol", "date", "state", "score", "pred_range", "rule_pass", "rank", "lean",
               "open", "last", "prev_close", "prev_vwap", "dollar_volume_20d", "split_day"]


def score(date: str, model_dir: Path, prices: Path, pm: Path | None, forecast: Path, state: str,
          panel: Path | None = None, symbols_file: Path | None = None, rules: Rules | None = None,
          out_dir: Path = Path("../out/screener")) -> pd.DataFrame:
    model_dir = Path(model_dir)
    summary = json.loads((model_dir / "summary.json").read_text())
    rules = rules or Rules(**summary.get("rules", {}))
    fc = pd.read_parquet(forecast)
    if symbols_file is not None:
        keep = set(pd.read_csv(symbols_file)["symbol"])
        fc = fc[fc["symbol"].isin(keep)]
    hist = latest_history(panel or Path(summary["panel"]))
    d = live_frame(hist, fc, read_jsonl(prices), read_jsonl(pm), date, state)
    out = score_frame(d, load(model_dir, state), rules)
    out["state"] = state
    feats = [f for f in feature_names(state) if f not in OUT_COLUMNS]
    out = out[[*OUT_COLUMNS, *feats]]
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"screener-{date}.parquet", index=False)
    top = out[out["rule_pass"]].head(rules.top_n)
    top.to_csv(out_dir / f"screener-{date}-top{rules.top_n}.csv", index=False)
    cols = ["rank", "symbol", "score", "pred_range", "lean", "open", "prev_close"]
    print(top[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--pm", type=Path)
    ap.add_argument("--forecast", type=Path, required=True)
    ap.add_argument("--state", choices=("px", "full"), required=True)
    ap.add_argument("--panel", type=Path, help="default: the panel the model was trained on")
    ap.add_argument("--symbols-file", type=Path)
    ap.add_argument("--settings", type=Path, help="settings.toml with a [screener] section")
    ap.add_argument("--out-dir", type=Path, default=Path("../out/screener"))
    a = ap.parse_args(argv)
    score(a.date, a.model, a.prices, a.pm, a.forecast, a.state, a.panel, a.symbols_file,
          load_rules(a.settings) if a.settings else None, a.out_dir)


if __name__ == "__main__":
    main()
