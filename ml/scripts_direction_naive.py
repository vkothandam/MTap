import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/vkothandaraman/development/pprojects/MTap/ml")
from tft_vwap.compare import load, pm_rows
R = Path("/Users/vkothandaraman/development/pprojects/MTap/ml/runs")
PANEL = "/Users/vkothandaraman/development/pprojects/MTap/out/tft/exp5y-r1/panel.parquet"
f = load(R / "20261028-233305Z-return-4438sym-overlay-full", [1, 3, 5, 7, 9, 11, 13])
f = f[pm_rows(f, Path(PANEL), "full")]
r = f["actual"] / f["prev_vwap"] - 1
p = pd.read_parquet(PANEL, columns=["symbol", "day_idx", "open_pct_prev_vwap", "pm_last_pct_prev_vwap", "pm_vwap_pct_prev_vwap"])
p_prev = pd.read_parquet(PANEL, columns=["symbol", "day_idx", "close_pct_rth_vwap", "ah_vwap_pct_rth_vwap"]).assign(day_idx=lambda d: d["day_idx"] + 1)
m = f[["symbol", "day_idx"]].assign(r=r.to_numpy()).merge(p, on=["symbol", "day_idx"], how="left").merge(p_prev, on=["symbol", "day_idx"], how="left")
for c in ["open_pct_prev_vwap", "pm_last_pct_prev_vwap", "pm_vwap_pct_prev_vwap", "close_pct_rth_vwap", "ah_vwap_pct_rth_vwap"]:
    nz = (m["r"] != 0) & (m[c].fillna(0) != 0)
    big = nz & (m["r"].abs() > 0.02)
    hit = np.sign(m[c]) == np.sign(m["r"])
    ic = m[nz].groupby("day_idx").apply(lambda g: g[c].corr(g["r"], method="spearman")).mean()
    print(f"{c:24s} direction {100*hit[nz].mean():.1f}%  big moves {100*hit[big].mean():.1f}%  rank IC {ic:.2f}  rows {nz.mean()*100:.0f}%")
