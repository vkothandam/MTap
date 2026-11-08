"""Tables for Intraday → Pockets, from an intraday-ft run (pockets_contract.py). Pure functions.

Every score is against the realised targets stored in the run file, on the rows the filters keep:
- pinball skill = 1 - model loss / climatology loss (climatology = the training span's quantiles per
  5-minute bucket x sigma tercile x liquidity tercile, from fold_<k>/clim.parquet);
- hits = the share of outcomes at or below each quantile (should be 0.10 / 0.25 / 0.50 / 0.75 / 0.90);
- coverage = the share inside P10 .. P90 (should be 80%);
- rank IC = Spearman correlation of P50 with the outcome across the stocks at the same minute, the
  mean over minutes (it ranks the stocks; a level shift doesn't change it).
CIs = day bootstrap (whole test days resampled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calib
from .insights import BAD, GOOD, INFO, WARN, Insight
from .pockets_contract import BINS, OVERLAYS, TAUS

N_BOOT = 200
MIN_RANK = 10              # stocks needed at a minute for its rank IC
SHAPE = ("cm_up", "cm_dn", "bulge", "sag", "rvol")
FAMILIES = {"cm_up": "centre of mass above P0 (σ)", "cm_dn": "centre of mass below P0 (σ)",
            "bulge": "volume above P0 × distance (σ)", "sag": "volume below P0 × distance (σ)",
            "rvol": "bin volume vs normal (log)", "mass_up": "share of the volume above P0",
            "net": "bulge − sag (σ)", "occ_up": "share of the time above P0",
            "occ_dn": "share of the time below P0"}
ZERO_MASS = ("bulge", "sag")
DUST = 0.01                # σ; a bulge / sag quantile below it reads as 0 (the run's report does the same)
EVENTS = {"p_mass_up60": ("mass_up60", 0.5, "most of the next hour's volume above P0"),
          "p_bulge60": ("bulge60", 0.5, "next hour's up pocket ≥ 0.5σ"),
          "p_sag60": ("sag60", 0.5, "next hour's down pocket ≥ 0.5σ")}
MIN_BUCKETS = ((0, 15), (15, 60), (60, 120), (120, 240), (240, 330), (330, 390))
GROUPS = {"cap_band": "cap band", "sector": "sector", "industry": "industry", "minute_bucket": "time of day",
          "symbol": "stock", "live_set": "live set"}


def hhmm(minute) -> str:
    m = int(minute) + 570
    return f"{m // 60:02d}:{m % 60:02d}"


def minute_bucket(minute: pd.Series) -> pd.Series:
    edges = [b[0] for b in MIN_BUCKETS] + [390]
    labels = [f"{hhmm(a)}-{hhmm(b)}" for a, b in MIN_BUCKETS]
    return pd.cut(minute, edges, right=False, labels=labels).astype(str)


def dust(q: np.ndarray, target: str) -> np.ndarray:
    return np.where(q < DUST, 0.0, q) if target.rsplit("_b", 1)[0].removesuffix("60") in ZERO_MASS else q


# ---- climatology -------------------------------------------------------------------------------

def cells(minute, sigma, log_dvol, cl: pd.DataFrame) -> np.ndarray:
    """The climatology cell of each row (the trainer's formula: 5-min bucket x sigma x dvol tercile)."""
    r = cl.iloc[0]
    return ((np.asarray(minute) // 5) * 9 + np.digitize(sigma, r.sigma_cuts) * 3
            + np.digitize(log_dvol, r.dvol_cuts))


def clim_quantiles(cl: pd.DataFrame, target: str, cell: np.ndarray) -> np.ndarray | None:
    """(rows, 5) climatology quantiles of `target` per row (the overall ones for an empty cell)."""
    c = cl[cl.target == target]
    if c.empty:
        return None
    cols = [f"q{round(t * 100)}" for t in TAUS]
    allq = c[c.cell == -1][cols].to_numpy(np.float64)[0]
    q = c[c.cell >= 0].set_index("cell")[cols].reindex(cell).to_numpy(np.float64)
    return np.where(np.isfinite(q), q, allq[None, :])


def clim_rate(cr: pd.DataFrame, event: str, cell: np.ndarray) -> np.ndarray:
    c = cr[cr.event == event]
    overall = float(c[c.cell == -1].rate.iloc[0])
    return c[c.cell >= 0].set_index("cell").rate.reindex(cell).fillna(overall).to_numpy(np.float64)


# ---- scores ------------------------------------------------------------------------------------

def pinball(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    e = y[:, None] - q
    tau = np.asarray(TAUS)[None, :]
    return np.maximum(tau * e, (tau - 1) * e).mean(1)


def boot_skill(day: np.ndarray, num: np.ndarray, den: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """1 - sum(num) / sum(den) and its day-bootstrap 95% CI."""
    ok = np.isfinite(num) & np.isfinite(den)
    g = pd.DataFrame({"d": day[ok], "a": num[ok], "b": den[ok]}).groupby("d").sum()
    if g.empty or g.b.sum() <= 0:
        return np.nan, np.nan, np.nan
    A, B = g.a.to_numpy(), g.b.to_numpy()
    k = np.random.default_rng(seed).integers(0, len(g), (N_BOOT, len(g)))
    s = 1 - A[k].sum(1) / B[k].sum(1)
    return float(1 - A.sum() / B.sum()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def rank_ic(date, minute, p, y, min_n: int = MIN_RANK) -> float:
    """Spearman correlation of p with y across the stocks at each (date, minute) with at least
    min_n of them; the mean of the per-day means (the run report's rule)."""
    d = pd.DataFrame({"d": date, "m": minute, "p": p, "y": y}).dropna()
    d = d[d.groupby(["d", "m"]).p.transform("size") >= min_n]
    if d.empty:
        return np.nan
    g = d.groupby(["d", "m"])
    d["rp"], d["ry"] = g.p.rank(), g.y.rank()
    g = d.groupby(["d", "m"])
    a, b = d.rp - g.rp.transform("mean"), d.ry - g.ry.transform("mean")
    k = [d.d, d.m]
    num = (a * b).groupby(k).sum()
    den = np.sqrt((a * a).groupby(k).sum() * (b * b).groupby(k).sum())
    ic = num / den.where(den > 0)
    return float(ic.groupby(level=0).mean().mean())


def score(f: pd.DataFrame, q: np.ndarray, y: np.ndarray, qc: np.ndarray | None, seed: int = 0,
          ic: bool = True) -> dict:
    """The shape scores of one target on rows f (date, minute): n, hits, coverage, skill vs clim."""
    ok = np.isfinite(y) & np.isfinite(q).all(1)
    out = {"n": int(ok.sum())}
    if not ok.any():
        return out | {"cover80": np.nan, "p50_hit": np.nan, "skill_clim": np.nan, "skill_lo": np.nan,
                      "skill_hi": np.nan, "rank_ic": np.nan, "hits": [np.nan] * len(TAUS)}
    yy, qq = y[ok], q[ok]
    out["hits"] = [float((yy <= qq[:, k]).mean()) for k in range(len(TAUS))]
    out["p50_hit"] = out["hits"][2]
    out["cover80"] = float(((yy >= qq[:, 0]) & (yy <= qq[:, -1])).mean())
    if qc is not None:
        s = boot_skill(f.date.to_numpy()[ok], pinball(qq, yy), pinball(qc[ok], yy), seed)
    else:
        s = (np.nan,) * 3
    out["skill_clim"], out["skill_lo"], out["skill_hi"] = s
    out["rank_ic"] = rank_ic(f.date.to_numpy()[ok], f.minute.to_numpy()[ok], qq[:, 2], yy) if ic else np.nan
    return out


def by_group(f: pd.DataFrame, q: np.ndarray, y: np.ndarray, qc: np.ndarray | None, key: str,
             min_n: int = 1) -> pd.DataFrame:
    """score() per value of f[key]; rank IC only for keys that leave several stocks per minute."""
    rows = []
    many = key not in ("symbol",)
    for val, idx in f.groupby(key, observed=True).indices.items():
        if len(idx) < min_n:
            continue
        s = score(f.iloc[idx], q[idx], y[idx], None if qc is None else qc[idx], ic=many)
        rows.append({key: str(val), **{k: v for k, v in s.items() if k != "hits"}})
    cols = [key, "n", "cover80", "p50_hit", "skill_clim", "skill_lo", "skill_hi", "rank_ic"]
    return pd.DataFrame(rows, columns=cols).sort_values("n", ascending=False, ignore_index=True)


def hits_table(hits: list[float]) -> pd.DataFrame:
    """quantile_chart's table: level, observed, gap."""
    t = pd.DataFrame({"level": TAUS, "observed": hits})
    t["gap"] = t.observed - t.level
    return t


# ---- the report's go-rule table ---------------------------------------------------------------

def report_shape(rep: dict | None, fold: str) -> pd.DataFrame:
    """The run report's pocket-shape rows of one fold: raw and (if scored) recalibrated."""
    if not rep:
        return pd.DataFrame()
    fo = next((x for x in rep.get("folds", []) if str(x.get("fold")) == str(fold)), None)
    if fo is None:
        return pd.DataFrame()
    rows = []
    for p in fo["pinball"]:
        if "hits" not in p:
            continue
        fam, b = p["target"].rsplit("_b", 1) if "_b" in p["target"] else (p["target"], "")
        r = {"target": p["target"], "family": fam, "bin": int(b) if b else 0,
             "skill_clim": p["clim"]["skill"], "skill_lo": p["clim"]["lo"], "skill_hi": p["clim"]["hi"],
             "p50_hit": p["hits"][2][1], "cover80": p["cover"][1],
             "rank_ic": p["ic"]["mean"], "clim_rank_ic": p["clim_ic"]["mean"],
             "p50_lo": p["hits"][2][0], "cover_lo": p["cover"][0]}
        if "cal" in p:
            c = p["cal"]
            r |= {"cal_skill_clim": c["clim"]["skill"], "cal_p50_hit": c["hits"][2][1],
                  "cal_cover80": c["cover"][1], "cal_p50_lo": c["hits"][2][0],
                  "cal_cover_lo": c["cover"][0]}
        rows.append(r)
    return pd.DataFrame(rows)


def go_rule(t: pd.DataFrame) -> pd.DataFrame:
    """Per shape target: does it pass Q41? skill CI above 0, rank IC above climatology's, and
    calibrated (P50 hit 0.45-0.55, coverage 75-85%) raw or after the recalibration. Outcomes exactly 0
    (bulge / sag with no volume on a side) tie with a quantile at 0, so a hit rate is a span from
    "below" to "at or below" (the report's a-b); calibrated = the span reaches the ideal band."""
    if t.empty:
        return t
    t = t.copy()

    def calibrated(p50_lo, p50, cov_lo, cov):
        return ((p50_lo <= 0.55) & (p50 >= 0.45)) & ((cov_lo <= 0.85) & (cov >= 0.75))

    t["calibrated"] = calibrated(t.get("p50_lo", t.p50_hit), t.p50_hit,
                                 t.get("cover_lo", t.cover80), t.cover80)
    if "cal_p50_hit" in t:
        t["calibrated_after_recal"] = calibrated(t.get("cal_p50_lo", t.cal_p50_hit), t.cal_p50_hit,
                                                 t.get("cal_cover_lo", t.cal_cover80),
                                                 t.cal_cover80).fillna(False)
    else:
        t["calibrated_after_recal"] = False
    t["passes"] = ((t.skill_lo > 0) & (t.rank_ic > t.clim_rank_ic)
                   & (t.calibrated | t.calibrated_after_recal))
    return t


def headline(rep: dict | None) -> dict:
    """Per fold: the numbers the runs are compared on."""
    out = {}
    for fo in (rep or {}).get("folds", []):
        sh = {p["target"]: p for p in fo["pinball"] if "hits" in p}
        r = {"rows": fo["rows"], "days": fo["days"],
             "net60_rank_ic": fo["rank_ic"].get("net60__all", {}).get("mean", np.nan),
             "mass_up60_auc": fo["events"]["p_mass_up60"]["all"]["auc"]}
        for t in ("cm_up_b1", "cm_up_b4", "cm_dn_b4", "bulge_b4", "sag_b4", "rvol_b1", "rvol_b4"):
            if t in sh:
                r[f"{t}_skill"] = sh[t]["clim"]["skill"]
                r[f"{t}_ic"] = sh[t]["ic"]["mean"]
        out[str(fo["fold"])] = r
    return out


# ---- events and overlays -----------------------------------------------------------------------

def cdf_at(q: np.ndarray, v: float) -> np.ndarray:
    """P(X <= v) from the 5 quantiles: linear between them, exponential tails outside (the trainer's
    readout rule, so a rare event can read below 5%)."""
    q = np.asarray(q, np.float64)
    tau = np.asarray(TAUS)
    s_lo = np.maximum(q[..., 1] - q[..., 0], 1e-6) / np.log(tau[1] / tau[0])
    s_hi = np.maximum(q[..., -1] - q[..., -2], 1e-6) / np.log((1 - tau[-2]) / (1 - tau[-1]))
    with np.errstate(over="ignore"):
        lo_tail = tau[0] * np.exp(-np.maximum(q[..., 0] - v, 0) / s_lo)
        hi_tail = 1 - (1 - tau[-1]) * np.exp(-np.maximum(v - q[..., -1], 0) / s_hi)
    out = np.where(v < q[..., 0], lo_tail, np.where(v >= q[..., -1], hi_tail, np.nan))
    for k in range(len(TAUS) - 1):
        lo, hi = q[..., k], q[..., k + 1]
        inside = (v >= lo) & (v < hi)
        out = np.where(inside, tau[k] + (tau[k + 1] - tau[k]) * (v - lo) / np.maximum(hi - lo, 1e-12), out)
    return out


def event_hit(f: pd.DataFrame, event: str) -> np.ndarray:
    t, thr, _ = EVENTS[event]
    y = f[f"real_{t}"]
    return (y > thr).astype(float).where(y.notna()).to_numpy()


def event_grid(probs: dict[str, np.ndarray], hit: np.ndarray, width: float = 0.1) -> pd.DataFrame:
    """Reliability cells per overlay variant (column `overlay`), for charts.reliability(group=)."""
    edges = [0.0, *[round(e, 4) for e in np.arange(width, 1, width)], 1.0]
    ok = np.isfinite(hit)
    parts = []
    for v, p in probs.items():
        m = ok & np.isfinite(p)
        g = calib.cells(p[m], hit[m], edges)
        g.insert(0, "overlay", OVERLAYS.get(v, v))
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def overlay_scores(f: pd.DataFrame, probs: dict[str, np.ndarray], hit: np.ndarray, rate: np.ndarray,
                   key: str | None = None) -> pd.DataFrame:
    """Brier skill vs climatology of one event per overlay variant, overall or per value of f[key]:
    n, came_true, then per variant said_<v> (mean probability) and skill_<v> (with lo / hi for the
    first and the last variant: the base and the full overlay)."""
    groups = {"all rows": np.arange(len(f))} if key is None else f.groupby(key, observed=True).indices
    day = f.date.to_numpy()
    first, last = next(iter(probs)), list(probs)[-1]
    rows = []
    for val, idx in groups.items():
        y = hit[idx]
        if not np.isfinite(y).any():
            continue
        base = (rate[idx] - y) ** 2
        r = {"group": str(val), "n": int(np.isfinite(y).sum()), "came_true": float(np.nanmean(y))}
        for v, p in probs.items():
            s, lo, hi = boot_skill(day[idx], (p[idx] - y) ** 2, base)
            r[f"said_{v}"] = float(np.nanmean(p[idx]))
            r[f"skill_{v}"] = s
            if v in (first, last):
                r[f"skill_{v}_lo"], r[f"skill_{v}_hi"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)


# ---- one stock, one day ------------------------------------------------------------------------

def replay(f: pd.DataFrame, q: dict[str, np.ndarray], b: int) -> pd.DataFrame:
    """One stock-day's decisions: time, P0, and per side the bin-b centre of mass band in σ and $,
    with what happened. q: {target: (rows, 5)} for cm_up_b<b>, cm_dn_b<b>."""
    up, dn = q[f"cm_up_b{b}"], q[f"cm_dn_b{b}"]
    p0, sig = f.real_p0.to_numpy(np.float64), f.sigma.to_numpy(np.float64)

    def px(d):
        return p0[:, None] * np.exp(sig[:, None] * d) if d.ndim == 2 else p0 * np.exp(sig * d)

    r = pd.DataFrame({"minute": f.minute.to_numpy(), "time": [hhmm(m) for m in f.minute], "p0": p0})
    for k, lab in enumerate(("p10", "p50", "p90")):
        j = (0, 2, 4)[k]
        r[f"up_{lab}"] = px(up[:, j])
        r[f"dn_{lab}"] = px(-dn[:, j])
    r["up_real"] = px(f[f"real_cm_up_b{b}"].to_numpy(np.float64))
    r["dn_real"] = px(-f[f"real_cm_dn_b{b}"].to_numpy(np.float64))
    return r.sort_values("minute", ignore_index=True)


def pocket_fan(row: pd.Series, q: dict[str, np.ndarray], i: int, bins=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """One decision across the bins: per bin the cm_up / cm_dn quantiles (σ and $) and the outcome."""
    p0, sig = float(row.real_p0), float(row.sigma)
    rows = []
    for b in bins:
        up, dn = q[f"cm_up_b{b}"][i], q[f"cm_dn_b{b}"][i]
        rows.append({"bin": b, **{f"up_{n}": up[k] for k, n in enumerate(("p10", "p25", "p50", "p75", "p90"))},
                     **{f"dn_{n}": -dn[k] for k, n in enumerate(("p10", "p25", "p50", "p75", "p90"))},
                     "up_real": float(row.get(f"real_cm_up_b{b}", np.nan)),
                     "dn_real": -float(row.get(f"real_cm_dn_b{b}", np.nan)),
                     "rvol_p50": q[f"rvol_b{b}"][i][2] if f"rvol_b{b}" in q else np.nan,
                     "rvol_real": float(row.get(f"real_rvol_b{b}", np.nan))})
    t = pd.DataFrame(rows)
    for c in [c for c in t.columns if c.startswith(("up_", "dn_"))]:
        t[f"{c}_usd"] = p0 * np.exp(sig * t[c])
    t.insert(1, "window", t["bin"].map(BINS))
    return t


# ---- In short ----------------------------------------------------------------------------------

def _pct(x) -> str:
    return f"{x:+.1%}"


def go_insights(t: pd.DataFrame) -> list[Insight]:
    if t.empty:
        return [Insight(INFO, "This run's report has no pocket-shape rows (re-score it with "
                              "`intraday-ft-report`).")]
    out = []
    for fam in SHAPE:
        d = t[t.family == fam]
        if d.empty:
            continue
        n, k = len(d), int(d.passes.sum())
        lvl = GOOD if k == n else WARN if k else BAD
        cal = "after the recalibration" if (d.calibrated_after_recal & ~d.calibrated).any() else "raw"
        out.append(Insight(lvl, f"`{fam}` ({FAMILIES[fam]}): passes in {k} of {n} bins; skill vs "
                                f"climatology {_pct(d.skill_clim.min())} … {_pct(d.skill_clim.max())}, "
                                f"rank IC {d.rank_ic.min():+.2f} … {d.rank_ic.max():+.2f} (climatology "
                                f"{d.clim_rank_ic.min():+.2f} … {d.clim_rank_ic.max():+.2f}); calibrated {cal}"
                                if k else
                                f"`{fam}` ({FAMILIES[fam]}): fails in every bin"))
    return out


def shape_insights(s: dict, target: str, recal: bool, warm_days: int) -> list[Insight]:
    if not s.get("n"):
        return [Insight(INFO, "No rows with an outcome under these filters.")]
    out = []
    p50, cov = s["p50_hit"], s["cover80"]
    ok = 0.45 <= p50 <= 0.55 and 0.75 <= cov <= 0.85
    out.append(Insight(GOOD if ok else WARN,
                       f"`{target}` {'recalibrated' if recal else 'raw'}: {p50:.0%} of outcomes at or "
                       f"below P50 (ideal 50%), {cov:.0%} inside P10–P90 (ideal 80%), n = {s['n']:,}"))
    if np.isfinite(s["skill_clim"]):
        lvl = GOOD if s["skill_lo"] > 0 else BAD if s["skill_hi"] < 0 else WARN
        out.append(Insight(lvl, f"Pinball skill vs climatology {_pct(s['skill_clim'])} "
                                f"[{_pct(s['skill_lo'])}, {_pct(s['skill_hi'])}]"))
    if np.isfinite(s.get("rank_ic", np.nan)):
        out.append(Insight(GOOD if s["rank_ic"] > 0.05 else INFO,
                           f"P50 ranks the stocks at a minute with rank IC {s['rank_ic']:+.3f}"))
    if recal:
        out.append(Insight(INFO, f"Recalibrated = each day's quantiles shifted by the errors of the 20 "
                                 f"sessions before it; scored on the {warm_days} days that have them"))
    return out


def group_insights(t: pd.DataFrame, key: str, min_n: int) -> list[Insight]:
    d = t[(t.n >= min_n) & t.skill_clim.notna()]
    if d.empty:
        return [Insight(INFO, f"No {GROUPS.get(key, key)} has {min_n:,} rows or more.")]
    lo, hi = d.loc[d.skill_clim.idxmin()], d.loc[d.skill_clim.idxmax()]
    neg = d[d.skill_hi < 0]
    out = [Insight(INFO, f"Best {GROUPS.get(key, key)}: {hi[key]} {_pct(hi.skill_clim)} (n = {hi.n:,}); "
                         f"worst: {lo[key]} {_pct(lo.skill_clim)} (n = {lo.n:,})")]
    if len(neg):
        out.append(Insight(BAD, f"Worse than climatology beyond noise in {len(neg)} of {len(d)}: "
                                + ", ".join(map(str, neg[key].head(8)))))
    else:
        out.append(Insight(GOOD, f"No {GROUPS.get(key, key)} is worse than climatology beyond noise "
                                 f"({len(d)} with n ≥ {min_n:,})"))
    return out


def overlay_insights(t: pd.DataFrame, event: str, variants) -> list[Insight]:
    if t.empty:
        return []
    r = t.iloc[0]
    a, b = variants[0], variants[-1]
    out = [Insight(INFO, f"{EVENTS[event][2]}: came true {r.came_true:.1%}; said {r[f'said_{a}']:.1%} "
                         f"({OVERLAYS[a]}) / {r[f'said_{b}']:.1%} ({OVERLAYS[b]})")]
    for v in (a, b):
        s, lo, hi = r[f"skill_{v}"], r[f"skill_{v}_lo"], r[f"skill_{v}_hi"]
        lvl = GOOD if lo > 0 else BAD if hi < 0 else WARN
        out.append(Insight(lvl, f"Brier skill vs climatology, {OVERLAYS[v]}: {_pct(s)} [{_pct(lo)}, {_pct(hi)}]"))
    d = r[f"skill_{b}"] - r[f"skill_{a}"]
    out.append(Insight(INFO, f"The overlay ({OVERLAYS[b]}) changes the skill by {d * 100:+.2f} pts"))
    return out
