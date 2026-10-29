"""Direction and move-size metrics per model on common rows."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/vkothandaraman/development/pprojects/MTap/ml")
from tft_vwap.compare import load, qlevels, pm_rows
from tft_vwap.calibration import prob_below
R = Path("/Users/vkothandaraman/development/pprojects/MTap/ml/runs")
PANEL = Path("/Users/vkothandaraman/development/pprojects/MTap/out/tft/exp5y-r1/panel.parquet")

def rows(run, folds, only_pm=None):
    f = load(R / run, folds); f["date"] = pd.to_datetime(f["date"])
    if only_pm: f = f[pm_rows(f, PANEL, only_pm)]
    lv = qlevels(f); levels = list(lv); prev = f["prev_vwap"].to_numpy()
    r = f["actual"].to_numpy() / prev - 1
    rq = np.sort(f[list(lv.values())].to_numpy() / prev[:, None] - 1, axis=1)
    return pd.DataFrame({"symbol": f["symbol"].to_numpy(), "date": f["date"].to_numpy(), "r": r,
                         "m": rq[:, 9], "p_up": 1 - prob_below(rq, levels, np.zeros_like(r))}
                        ).set_index(["symbol", "date"])

def metrics(d):
    nz = (d["r"] != 0) & (d["m"] != 0)
    hit = np.sign(d["m"]) == np.sign(d["r"])
    big = nz & (d["r"].abs() > 0.02)
    conf = (d["p_up"] >= 0.65) | (d["p_up"] <= 0.35)
    conf_hit = ((d["p_up"] >= 0.65) & (d["r"] > 0)) | ((d["p_up"] <= 0.35) & (d["r"] < 0))
    ic = d.reset_index().groupby("date").apply(lambda g: g["m"].corr(g["r"], method="spearman"))
    return {"direction_%": 100 * hit[nz].mean(),
            "dir_big_moves_%": 100 * hit[big].mean(),
            "confident_calls_%rows": 100 * conf.mean(), "confident_hit_%": 100 * conf_hit[conf].mean(),
            "median_|P50 err|_%": 100 * (d["r"] - d["m"]).abs().median(),
            "rank_IC": ic.mean(), "IC_days_>0_%": 100 * (ic > 0).mean()}

def table(runs, folds, only_pm=None):
    data = {n: rows(run, folds, only_pm) for n, run in runs.items()}
    common = None
    for d in data.values(): common = d.index if common is None else common.intersection(d.index)
    t = pd.DataFrame([{"model": n, **metrics(d.loc[common])} for n, d in data.items()])
    return t.round(2), len(common)

pd.set_option("display.width", 250)
night = {"R0": "20261028-030847Z-return-4438sym", "R1": "20261028-172535Z-return-4438sym",
         "R1c": "20261028-233305Z-return-4438sym", "R6c": "20261028-030847Z-return-4438sym-event",
         "R7": "20261029-153450Z-gbm-4438sym"}
t, n = table(night, list(range(1, 15))); print(f"NIGHT folds 1-14, {n} rows"); print(t.to_string(index=False))
bake = {"R0": night["R0"], "R1c": night["R1c"], "R4~night": "20261029-040858Z-return-4438sym~night",
        "R4~px 09:30": "20261029-040858Z-return-4438sym~px", "R4p 09:30": "20261029-090243Z-return-4438sym",
        "R6p 09:30": "20261028-233305Z-return-4438sym-overlay-px", "R4 09:45": "20261029-040858Z-return-4438sym", "R4f 09:45": "20261029-122046Z-return-4438sym",
        "R6f 09:45": "20261028-233305Z-return-4438sym-overlay-full",
        "R7": "20261029-153450Z-gbm-4438sym", "R7f 09:45": "20261029-170229Z-gbm-4438sym"}
t, n = table(bake, [1, 3, 5, 7, 9, 11, 13], "full"); print(f"\nBAKE-OFF odd folds, full pre-market rows, {n} rows"); print(t.to_string(index=False))
