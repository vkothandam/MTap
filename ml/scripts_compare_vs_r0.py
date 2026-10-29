"""All runs vs R0 on common rows: pinball, skill vs naive, paired diff vs R0 (bootstrap over days),
coverage, direction, PIT bias/z/label, mean |quantile calibration error|."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/vkothandaraman/development/pprojects/MTap/ml")
from tft_vwap.compare import load, qlevels, pm_rows, bootstrap_ci
from tft_vwap.calibration import prob_below

R = Path("/Users/vkothandaraman/development/pprojects/MTap/ml/runs")
PANEL = Path("/Users/vkothandaraman/development/pprojects/MTap/out/tft/exp5y-r1/panel.parquet")

def rows(run, folds, keys=None, only_pm=None):
    f = load(R / run, folds)
    f["date"] = pd.to_datetime(f["date"])
    if only_pm:
        f = f[pm_rows(f, PANEL, only_pm)]
    lv = qlevels(f); levels = np.array(list(lv)); cols = list(lv.values())
    prev = f["prev_vwap"].to_numpy(); r = f["actual"].to_numpy() / prev - 1
    rq = np.sort(f[cols].to_numpy() / prev[:, None] - 1, axis=1)
    err = r[:, None] - rq
    loss = np.maximum(levels * err, (levels - 1) * err).mean(1)
    naive = np.maximum(levels * r[:, None], (levels - 1) * r[:, None]).mean(1)
    out = pd.DataFrame({"symbol": f["symbol"].to_numpy(), "date": f["date"].to_numpy(),
                        "loss": loss, "naive": naive,
                        "c90": (rq[:, 0] <= r) & (r <= rq[:, -1]),
                        "c80": (rq[:, 1] <= r) & (r <= rq[:, -2]),
                        "dir_ok": np.sign(rq[:, 9]) == np.sign(r), "dir_n": (rq[:, 9] != 0) & (r != 0),
                        "pit": prob_below(rq, list(levels), r)})
    for i, l in enumerate(levels):
        out[f"b{i}"] = r <= rq[:, i]
    return out, levels

def table(runs, folds, only_pm=None, ref="R0"):
    data = {}
    for name, (run, state_folds) in runs.items():
        d, levels = rows(run, state_folds or folds, only_pm=only_pm)
        data[name] = d.set_index(["symbol", "date"])
    common = None
    for d in data.values():
        common = d.index if common is None else common.intersection(d.index)
    res = []
    ref_day = data[ref].loc[common].groupby(level="date")["loss"].mean()
    for name, d in data.items():
        d = d.loc[common]
        loss, naive = d["loss"].mean(), d["naive"].mean()
        day = d.groupby(level="date")["loss"].mean()
        diff = (day - ref_day).to_numpy()
        lo, hi = bootstrap_ci(diff)
        n = len(d); bias = 0.5 - d["pit"].mean(); z = bias / np.sqrt(1 / 12 / n)
        obs = np.array([d[f"b{i}"].mean() for i in range(len(levels))])
        res.append({"model": name, "rows": n, "loss": loss, "skill_%": 100 * (1 - loss / naive),
                    "vs_R0_%": 100 * (loss / data[ref].loc[common]["loss"].mean() - 1),
                    "ci_lo_%": 100 * lo / ref_day.mean(), "ci_hi_%": 100 * hi / ref_day.mean(),
                    "P5-P95_%": 100 * d["c90"].mean(), "P10-P90_%": 100 * d["c80"].mean(),
                    "dir_%": 100 * d.loc[d["dir_n"], "dir_ok"].mean(),
                    "bias_pts": 100 * bias, "bias_z": z,
                    "label": "calibrated" if abs(z) < 2 else ("optimistic" if bias > 0 else "pessimistic"),
                    "mean_|q_err|_pts": 100 * np.abs(obs - levels).mean(),
                    "max_|q_err|_pts": 100 * np.abs(obs - levels).max()})
    return pd.DataFrame(res), len(common), common.get_level_values("date").nunique()

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
fmt = lambda t: t.round({"loss": 6, "skill_%": 2, "vs_R0_%": 2, "ci_lo_%": 2, "ci_hi_%": 2, "P5-P95_%": 1,
                         "P10-P90_%": 1, "dir_%": 1, "bias_pts": 2, "bias_z": 1, "mean_|q_err|_pts": 2, "max_|q_err|_pts": 2})
which = sys.argv[1]
if which == "night":
    F = list(range(1, 15))
    runs = {"R0": ("20261028-030847Z-return-4438sym", None), "R1": ("20261028-172535Z-return-4438sym", None),
            "R1c": ("20261028-233305Z-return-4438sym", None), "R6c": ("20261028-030847Z-return-4438sym-event", None),
            "R7": ("20261029-153450Z-gbm-4438sym", None)}
    t, n, days = table(runs, F)
    print(f"NIGHT, folds 1-14, {n} common rows, {days} days"); print(fmt(t).to_string(index=False))
    t, n, days = table({"R0": runs["R0"], "R1a": ("20261028-214331Z-return-4438sym", None)}, list(range(1, 7)))
    print(f"\nR1a, folds 1-6, {n} rows, {days} days"); print(fmt(t).to_string(index=False))
    pr = pd.read_parquet("/Users/vkothandaraman/development/pprojects/MTap/ml/runs/20261005-025909Z-return-3712sym/fold_3/test_predictions.parquet", columns=["date"]) if False else None
elif which == "bake":
    F = [1, 3, 5, 7, 9, 11, 13]
    runs = {"R0": ("20261028-030847Z-return-4438sym", None), "R1c": ("20261028-233305Z-return-4438sym", None),
            "R4~night": ("20261029-040858Z-return-4438sym~night", None),
            "R4~px (09:30)": ("20261029-040858Z-return-4438sym~px", None),
            "R4p (09:30)": ("20261029-090243Z-return-4438sym", None),
            "R6p (09:30)": ("20261028-233305Z-return-4438sym-overlay-px", None),
            "R4 full (09:45)": ("20261029-040858Z-return-4438sym", None),
            "R4f (09:45)": ("20261029-122046Z-return-4438sym", None),
            "R6f (09:45)": ("20261028-233305Z-return-4438sym-overlay-full", None),
            "R7": ("20261029-153450Z-gbm-4438sym", None),
            "R7f (09:45)": ("20261029-170229Z-gbm-4438sym", None)}
    t, n, days = table(runs, F, only_pm="full")
    print(f"BAKE-OFF, odd folds, rows with full pre-market, {n} common rows, {days} days"); print(fmt(t).to_string(index=False))
