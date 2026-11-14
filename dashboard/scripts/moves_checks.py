"""Read-only checks on the Moves export (LEARNINGS "Intraday v2 Moves export check", Q43).

Rebuilds the dashboard's Moves table (pockets_page.get_moves) without Streamlit, from the stored run files:
+X% / -X%, "by the window's end" (b1..bK combined), calibrated on `reached` (rolling 20 sessions).

    cd dashboard && uv run python scripts/moves_checks.py [--x 0.01] [--b 5] [--min-p 0.55] [--caps large,mega|all]

Outputs (tables, the per-arm frames as a cache) go to <repo>/temp/analysis/moves_checks/ (git-ignored).

1  per-P-bucket rates with day-clustered 95% CIs, with and without 08-06 / 09-02 / 08-12
2  IA1 vs IA1-noS vs IA1A on the same settings
3  does P(up) - P(down) predict the direction (all warm rows)
4  the first-touch race (+X before -0.7X, from the per-window highs / lows: bounds where both cross in
   the same window)
"""
from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_dash import pockets_calc as P
from forecast_dash import pockets_contract as C

RUNS = Path(__file__).resolve().parents[2] / "ml" / "runs" / "intraday_ft"
ARMS = {"IA1": ("20261107-141854-ft-B", "B"), "IA1-noS": ("20261107-181646-ft-B-noS", "B"),
        "IA1A": ("20261107-162439-ft-A", "A")}
OUT = Path(__file__).resolve().parents[2] / "temp" / "analysis" / "moves_checks"   # git-ignored
BIG_DAYS = ["2026-08-06", "2026-09-02", "2026-08-12"]
rng = np.random.default_rng(0)


def load(run: Path, fold: str, x: float, b: int) -> pd.DataFrame:
    """One row per decision: P (raw, cal), warm and the outcomes for both sides, as the Moves tab has them."""
    cols = [*C.KEYS, "real_p0", *[f"real_cm_{s}_b{k}" for s in ("up", "dn") for k in C.BINS],
            *[f"real_mass_up_b{k}" for k in C.BINS]]
    f = C.load_pred(run, fold, cols)
    f["date"] = f["date"].astype(str)
    e = C.load_extremes(run, fold)
    ext = e.reindex(f["dec_row"].to_numpy()).reset_index(drop=True)
    targets = C.targets_of(run, fold)
    Q = np.load(run / f"fold_{fold}" / "quantiles.npy", mmap_mode="r")
    dec = f["dec_row"].to_numpy()

    def q(t):
        return P.dust(np.asarray(Q[:, targets.index(t)], np.float64)[dec], t)

    out = {"date": f["date"].to_numpy(), "symbol": f["symbol"].to_numpy(), "minute": f["minute"].to_numpy(),
           "sigma": f["sigma"].to_numpy(), "live_set": f["live_set"].to_numpy(),
           "cap_band": f["cap_band"].astype(str).to_numpy()}
    for side in ("up", "dn"):
        raws, outs = [], []
        for k in range(1, b + 1):
            q_m = q(f"mass_up_b{k}") if f"mass_up_b{k}" in targets else None
            raws.append(P.move_prob(q(f"cm_{side}_b{k}"), q_m, f["sigma"].to_numpy(), x, side)["p"])
            outs.append(P.move_outcome(f, side, k, x, ext))
        raw = P.move_prob_cum(raws)
        oc = P.move_outcome_cum(outs, side)
        cal, warm = P.move_recal(raw, oc["reached"].to_numpy(), f["date"].to_numpy())
        out[f"raw_{side}"], out[f"p_{side}"], out["warm"] = raw, cal, warm
        for c in ("reached", "touched", "end_pct", "high_pct", "low_pct"):
            out[f"{c}_{side}"] = oc[c].to_numpy()
    d = pd.DataFrame(out)
    # per-window extremes (log vs P0) for the race
    for k in range(1, b + 1):
        d[f"hi_b{k}"], d[f"lo_b{k}"] = ext[f"hi_b{k}"].to_numpy(np.float64), ext[f"lo_b{k}"].to_numpy(np.float64)
    return d


def boot(day, v, reps=2000):
    """Mean of v and its 95% CI, resampling days (rows of a day move together)."""
    ok = np.isfinite(v)
    day, v = np.asarray(day)[ok], np.asarray(v, np.float64)[ok]
    if not len(v):
        return np.nan, np.nan, np.nan
    u, di = np.unique(day, return_inverse=True)
    s, n = np.bincount(di, v, len(u)), np.bincount(di, minlength=len(u)).astype(float)
    idx = rng.integers(0, len(u), (reps, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


def ci(day, v):
    m, lo, hi = boot(day, v)
    return f"{m:.0%} [{lo:.0%}, {hi:.0%}]" if np.isfinite(m) else "–"


def picks(d: pd.DataFrame, side: str, min_p: float) -> pd.DataFrame:
    """The Moves list's rows for a side: calibrated P ≥ min_p, outcome known."""
    m = (d[f"p_{side}"] >= min_p) & d[f"reached_{side}"].notna() & d["warm"]   # cold days: P is raw
    return d[m]


def side_stats(g: pd.DataFrame, side: str) -> dict:
    oth = "dn" if side == "up" else "up"
    sign = 1 if side == "up" else -1
    return {"rows": len(g), "days": g["date"].nunique(), "said": g[f"p_{side}"].mean(),
            "reached": ci(g["date"], g[f"reached_{side}"]),
            "touched same": ci(g["date"], g[f"touched_{side}"]),
            "touched other": ci(g["date"], g[f"touched_{oth}"]),
            "ended on side": ci(g["date"], (sign * g[f"end_pct_{side}"] > 0).astype(float)
                                .where(g[f"end_pct_{side}"].notna())),
            "median end": f"{np.nanmedian(g[f'end_pct_{side}']):+.2%}"}


def check1(d, min_p):
    rows = []
    edges = [min_p, 0.60, 0.65, 0.70, 0.80, 1.01]
    for drop in (False, True):
        dd = d[~d["date"].isin(BIG_DAYS)] if drop else d
        for side in ("up", "dn"):
            g = picks(dd, side, min_p)
            for lo, hi in pairwise(edges):
                h = g[(g[f"p_{side}"] >= lo) & (g[f"p_{side}"] < hi)]
                if len(h):
                    rows.append({"without 3 days": drop, "side": side, "P": f"{lo:.2f}–{min(hi, 1):.2f}",
                                 **side_stats(h, side)})
            rows.append({"without 3 days": drop, "side": side, "P": "all", **side_stats(g, side)})
    return pd.DataFrame(rows)


def auc(score, y):
    """P(score of a positive > score of a negative), ties half."""
    ok = np.isfinite(score) & np.isfinite(y)
    s, y = score[ok], y[ok].astype(bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def check3(d):
    w = d[d["warm"]].copy()
    rows = []
    for kind in ("p", "raw"):
        lean = (w[f"{kind}_up"] - w[f"{kind}_dn"]).to_numpy()
        end = w["end_pct_up"].to_numpy()      # the same last price either side
        end = np.where(np.isfinite(end) & (end != 0), end, np.nan)   # unknown or flat close: no direction
        ended_up = np.where(np.isfinite(end), end > 0, np.nan)
        up_only = (w["touched_up"] == 1) & (w["touched_dn"] == 0)
        dn_only = (w["touched_up"] == 0) & (w["touched_dn"] == 1)
        one = (up_only | dn_only).to_numpy()
        ic = []
        for _, g in w.assign(lean=lean).groupby("date"):
            if len(g) >= 50:
                ic.append(g["lean"].rank().corr(g["end_pct_up"].rank()))
        ic = np.array(ic)
        icb = np.percentile([rng.choice(ic, len(ic)).mean() for _ in range(2000)], [2.5, 97.5])
        dec = pd.qcut(pd.Series(lean).rank(method="first"), 10, labels=False).to_numpy()
        rows.append({
            "P": kind, "rows": len(w), "days": w["date"].nunique(), "close known": f"{np.isfinite(end).mean():.0%}",
            "AUC lean → ended up": f"{auc(lean, ended_up):.3f}",
            "AUC lean → only +X touched (of one-sided)": f"{auc(lean[one], up_only.to_numpy()[one].astype(float)):.3f}",
            "one-sided share": f"{one.mean():.0%}",
            "daily rank IC lean vs end": f"{ic.mean():+.3f} [{icb[0]:+.3f}, {icb[1]:+.3f}]",
            "top-decile lean: ended up": f"{np.nanmean(ended_up[dec == 9]):.0%}",
            "bottom-decile lean: ended up": f"{np.nanmean(ended_up[dec == 0]):.0%}",
            "base: ended up": f"{np.nanmean(ended_up):.0%}",
            "top-decile: touched +X / −X": f"{np.nanmean(w['touched_up'].to_numpy()[dec == 9]):.0%} / "
                                           f"{np.nanmean(w['touched_dn'].to_numpy()[dec == 9]):.0%}",
            "bottom-decile: touched +X / −X": f"{np.nanmean(w['touched_up'].to_numpy()[dec == 0]):.0%} / "
                                              f"{np.nanmean(w['touched_dn'].to_numpy()[dec == 0]):.0%}",
        })
    return pd.DataFrame(rows)


def race(g: pd.DataFrame, side: str, x: float, stop: float, b: int) -> pd.DataFrame:
    """Per row: win (target first), loss (stop first), neither by the close, or both in the same first
    window (order unknown at window resolution). Windows: 0-5, 5-15, 15-30, 30-60 min, 60 min-close."""
    tgt, stp = (np.log1p(x), np.log1p(-stop)) if side == "up" else (np.log1p(-x), np.log1p(stop))
    n = len(g)
    res = np.full(n, "neither", object)
    done = np.zeros(n, bool)
    known = np.ones(n, bool)
    for k in range(1, b + 1):
        hi, lo = g[f"hi_b{k}"].to_numpy(), g[f"lo_b{k}"].to_numpy()
        known &= ~(np.isnan(hi) & np.isnan(lo)) | done   # a window without a price leaves it unknown
        if side == "up":
            t, s = hi >= tgt - 1e-12, lo <= stp + 1e-12
        else:
            t, s = lo <= tgt + 1e-12, hi >= stp - 1e-12
        t, s = np.nan_to_num(t, nan=0).astype(bool), np.nan_to_num(s, nan=0).astype(bool)
        new = ~done & (t | s)
        res[new & t & ~s], res[new & s & ~t], res[new & t & s] = "win", "loss", "both"
        if k == b:
            res[new & t & s] = "both (last window)"
        done |= new
    res[~done & ~known] = "unknown"
    return pd.Series(res, index=g.index)


def check4(d, x, min_p, b):
    stop = 0.7 * x
    rows = []
    for side in ("up", "dn"):
        sign = 1 if side == "up" else -1
        for name, g in (("picks", picks(d, side, min_p)),
                        ("all warm rows", d[d["warm"] & d[f"reached_{side}"].notna()])):
            r = race(g, side, x, stop, b)
            vc = r.value_counts(normalize=True)
            win, loss = vc.get("win", 0), vc.get("loss", 0)
            amb = vc.get("both", 0) + vc.get("both (last window)", 0)
            nei = r == "neither"
            end = sign * g["end_pct_up"].to_numpy()
            # gross return per trade: +x / -stop / the close; the ambiguous ones bounded both ways
            base = win * x - loss * stop + np.nansum(end[nei.to_numpy()]) / len(g)
            rows.append({"side": side, "rows": name, "n": len(g),
                         "target first": f"{win:.0%}", "stop first": f"{loss:.0%}",
                         "both in the first window that crossed": f"{amb:.0%}",
                         "neither by the close": f"{nei.mean():.0%}",
                         "win share (bounds)": f"{win / (win + loss + amb):.0%} – {(win + amb) / (win + loss + amb):.0%}",
                         "gross per trade (bounds)": f"{base - amb * stop:+.2%} … {base + amb * x:+.2%}"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.01)
    ap.add_argument("--b", type=int, default=5)
    ap.add_argument("--min-p", type=float, default=0.55)
    ap.add_argument("--from", dest="d0", default="2026-08-05")
    ap.add_argument("--to", dest="d1", default="2026-10-20")
    ap.add_argument("--caps", default="large,mega", help="the export's sidebar filter; \"all\" for none")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    res = {}
    for arm, (run, fold) in ARMS.items():
        cache = OUT / f"{arm}_x{a.x}_b{a.b}.parquet"
        if cache.is_file():
            d = pd.read_parquet(cache)
        else:
            print(f"loading {arm} …", flush=True)
            d = load(RUNS / run, fold, a.x, a.b)
            d.to_parquet(cache)
        # as the dashboard: calibrated on every row, the sidebar filter applied after
        if a.caps != "all":
            d = d[d["cap_band"].isin(a.caps.split(","))].reset_index(drop=True)
        res[arm] = d
    print(f"caps: {a.caps}")
    noS = res["IA1-noS"]
    win = noS[(noS["date"] >= a.d0) & (noS["date"] <= a.d1)]
    print(f"\n== reproduce the export: IA1-noS {a.d0} → {a.d1}, +{a.x:.0%}, b1..b{a.b}, cal P ≥ {a.min_p}")
    for s in ("up", "dn"):
        g = picks(win, s, a.min_p)
        print(f"  {s}: {len(g)} rows, {g['date'].nunique()} days, said {g[f'p_{s}'].mean():.2f}, "
              f"reached {g[f'reached_{s}'].mean():.0%}, touched {g[f'touched_{s}'].mean():.0%}")
    print("\n== 1. per P bucket, IA1-noS, export window (95% CIs by day)")
    t1 = check1(win, a.min_p)
    print(t1.to_string(index=False))
    print("\n== 2. arms on the same settings")
    rows = []
    for arm, d in res.items():
        for span, dd in (("export window", d[(d["date"] >= a.d0) & (d["date"] <= a.d1)]), ("whole test", d)):
            if not len(dd):
                continue
            for s in ("up", "dn"):
                rows.append({"arm": arm, "span": f"{span} ({dd['date'].min()} → {dd['date'].max()})", "side": s,
                             **side_stats(picks(dd, s, a.min_p), s)})
    t2 = pd.DataFrame(rows)
    print(t2.to_string(index=False))
    print("\n== 3. P(up) − P(down) as a direction call (all warm rows)")
    t3 = pd.concat([check3(d).assign(arm=arm) for arm, d in res.items()])
    print(t3.T.to_string())
    print(f"\n== 4. race: +{a.x:.0%} before −{0.7 * a.x:.1%} (down: the mirror), IA1-noS export window + arms whole test")
    t4 = pd.concat([check4(win, a.x, a.min_p, a.b).assign(arm="IA1-noS", span="export window")]
                   + [check4(d, a.x, a.min_p, a.b).assign(arm=arm, span="whole test") for arm, d in res.items()])
    print(t4.to_string(index=False))
    for n, t in (("check1", t1), ("check2", t2), ("check3", t3), ("check4", t4)):
        t.to_csv(OUT / f"{n}_{a.caps.replace(",", "+")}.csv", index=False)
    (OUT / "args.json").write_text(json.dumps(vars(a)))


if __name__ == "__main__":
    main()
