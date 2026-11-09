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


# ---- moves: the ±X% filter ---------------------------------------------------------------------
# P(the volume on one side of P0 centres at least X away) in a bin, from the stored quantiles:
#   p_forms  = P(that side holds >= 5% of the bin's volume) (below it the run has no centre: CM_MIN)
#   p_beyond = P(centre >= the target | the side formed) = 1 - cdf_at(cm quantiles, distance / σ)
#   p        = p_forms × p_beyond, optionally recalibrated on the previous 20 sessions' outcomes.

CM_MIN = 0.05              # the trainer's rule: no centre of mass on a side with < 5% of the volume
SIDES = {"up": "up (+X%)", "dn": "down (−X%)"}
MOVE_WINDOW = 20           # sessions before the day the calibration learns from
MOVE_MIN_BIN = 150         # decisions per calibration bin (the top bin too)


def move_level(x: float, side: str) -> float:
    """The target's log distance from P0 (positive): ln(1 + X) up, −ln(1 − X) down."""
    return float(np.log1p(x) if side == "up" else -np.log1p(-x))


def move_prob(q_cm: np.ndarray, q_mass: np.ndarray | None, sigma, x: float, side: str) -> dict:
    """p_forms, p_beyond and p per row. q_cm: cm_<side>_bK quantiles (σ, a positive distance);
    q_mass: mass_up_bK quantiles (None: the side is taken as formed)."""
    v = move_level(x, side) / np.asarray(sigma, np.float64)
    beyond = 1 - cdf_at(q_cm, v)
    if q_mass is None:
        forms = np.ones(len(beyond))
    else:
        forms = 1 - cdf_at(q_mass, CM_MIN) if side == "up" else cdf_at(q_mass, 1 - CM_MIN)
    return {"p_forms": forms, "p_beyond": beyond, "p": forms * beyond}


AFTER = ("high_pct", "high_min", "low_pct", "low_min", "vwap_pct", "twap_pct", "end_pct", "cvwap_pct",
         "ctwap_pct")


def move_outcome(f: pd.DataFrame, side: str, b: int, x: float, ext: pd.DataFrame | None) -> pd.DataFrame:
    """What happened in bin b after each decision of f (rows aligned with ext, the extremes):
    reached = the side's centre of mass at or beyond the target (NaN when the bin had no trade),
    formed, centre_pct (the centre's move from P0), and from the extremes touched, best_pct /
    best_min (the high up, the low down, and its minute after the decision), worst_pct (the
    opposite extreme), and the prices after the entry (AFTER, % vs P0): the high / low and their
    minute, the VWAP / TWAP and the last price of the bin, the VWAP / TWAP from the entry to its end."""
    sig = f["sigma"].to_numpy(np.float64)
    cm = f[f"real_cm_{side}_b{b}"].to_numpy(np.float64)
    other = f[f"real_cm_{'dn' if side == 'up' else 'up'}_b{b}"].to_numpy(np.float64)
    mass = f"real_mass_up_b{b}"
    known = np.isfinite(f[mass].to_numpy(np.float64)) if mass in f else np.isfinite(cm) | np.isfinite(other)
    sign = 1.0 if side == "up" else -1.0
    reach = np.nan_to_num(cm, nan=-np.inf) * sig >= move_level(x, side)
    out = {"formed": np.where(known, np.isfinite(cm), np.nan),
           "reached": np.where(known, reach, np.nan),
           "centre_pct": np.expm1(sign * sig * cm)}
    nan = np.full(len(f), np.nan)

    def col(c):
        return ext[c].to_numpy(np.float64) if ext is not None and c in ext else nan

    for name, c in (("high", "hi"), ("low", "lo")):
        out[f"{name}_pct"] = np.expm1(col(f"{c}_b{b}"))
        t = col(f"t_{c}_b{b}")
        out[f"{name}_min"] = np.where(t >= 0, t, np.nan)
    for name, c in (("vwap_pct", "vw"), ("twap_pct", "tw"), ("end_pct", "cl"), ("cvwap_pct", "cvw"),
                    ("ctwap_pct", "ctw")):
        out[name] = np.expm1(col(f"{c}_b{b}"))
    best, worst = ("high", "low") if side == "up" else ("low", "high")
    out["touched"] = np.where(np.isfinite(out[f"{best}_pct"]),
                              sign * np.log1p(out[f"{best}_pct"]) >= move_level(x, side) - 1e-12, np.nan)
    out["best_pct"], out["worst_pct"] = out[f"{best}_pct"], out[f"{worst}_pct"]
    out["best_min"] = out[f"{best}_min"]
    return pd.DataFrame(out, index=f.index)


CUM_BINS = {1: "0-5 min", 2: "0-15 min", 3: "0-30 min", 4: "0-60 min", 5: "the entry to the close"}


def move_window(b: int, cum: bool) -> str:
    """The window's label: bin b alone, or every bin from the entry to b's end."""
    return CUM_BINS[b] if cum else BINS[b]


def move_prob_cum(ps: list[np.ndarray]) -> np.ndarray:
    """P that it happens in at least one of the windows, from each window's P (as if independent;
    the rolling calibration then corrects what that gets wrong)."""
    return 1 - np.prod(1 - np.column_stack(ps), axis=1)


def move_outcome_cum(outs: list[pd.DataFrame], side: str) -> pd.DataFrame:
    """move_outcome over windows b1 .. bK together: happened / formed / touched in any of them (NaN
    only when no window has an outcome); the furthest real centre, the high / low (with its minute)
    over all of them, best / worst as the side reads them; the VWAP / TWAP from the entry to bK's end
    and the last price at its end."""
    def stack(c):
        return np.column_stack([o[c].to_numpy(np.float64) for o in outs])

    def pick(val, t, up):
        v = stack(val)
        j = np.nanargmax(np.nan_to_num(v if up else -v, nan=-np.inf), axis=1)
        e = (np.fmax if up else np.fmin).reduce(v, axis=1)
        return e, np.where(np.isfinite(e), stack(t)[np.arange(len(v)), j], np.nan)

    out = {c: np.fmax.reduce(stack(c), axis=1) for c in ("formed", "reached", "touched")}
    out["centre_pct"] = (np.fmax if side == "up" else np.fmin).reduce(stack("centre_pct"), axis=1)
    out["high_pct"], out["high_min"] = pick("high_pct", "high_min", True)
    out["low_pct"], out["low_min"] = pick("low_pct", "low_min", False)
    last = outs[-1]
    out["vwap_pct"], out["twap_pct"] = last["cvwap_pct"].to_numpy(), last["ctwap_pct"].to_numpy()
    out["end_pct"] = last["end_pct"].to_numpy()
    out["cvwap_pct"], out["ctwap_pct"] = out["vwap_pct"], out["twap_pct"]
    best, worst = ("high", "low") if side == "up" else ("low", "high")
    out["best_pct"], out["best_min"] = out[f"{best}_pct"], out[f"{best}_min"]
    out["worst_pct"] = out[f"{worst}_pct"]
    return pd.DataFrame(out, index=outs[0].index)


def move_centre_cum(ps: list[np.ndarray], p50s: list[np.ndarray], p90s: list[np.ndarray],
                    centres: list[np.ndarray]) -> dict:
    """Each window has its own centre: with the windows combined, the one the model gave the highest
    P (its window number, the predicted P50 / P90 there, σ) and where that window's centre really was."""
    P = np.nan_to_num(np.column_stack(ps), nan=-1.0)
    j = np.argmax(P, axis=1)
    r = np.arange(len(P))
    return {"win": j + 1, "p50": np.column_stack(p50s)[r, j], "p90": np.column_stack(p90s)[r, j],
            "centre_pct": np.column_stack(centres)[r, j]}


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) fit by pool-adjacent-violators."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(float(wi))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            wts[-2], cnt[-2] = wsum, cnt[-2] + cnt[-1]
            del vals[-1], wts[-1], cnt[-1]
    return np.repeat(vals, cnt)


def move_recal(p: np.ndarray, hit: np.ndarray, day, window: int = MOVE_WINDOW,
               min_bin: int = MOVE_MIN_BIN) -> tuple[np.ndarray, np.ndarray]:
    """Each day's P mapped through what happened at each P on the previous `window` sessions (only
    past outcomes, as live): those sessions' decisions sorted by P into bins of at least min_bin each,
    the hit rates made non-decreasing (isotonic), then read off by linear interpolation. Equal-count
    bins keep the top bin as solid as the rest, so a high P has to have happened that often before.
    Returns (p_cal, warm): days without a full window (or < 10 bins of rows) keep the raw P, warm False."""
    p, hit = np.asarray(p, np.float64), np.asarray(hit, np.float64)
    days, di = np.unique(np.asarray(day), return_inverse=True)
    ok = np.isfinite(p) & np.isfinite(hit)
    order = np.argsort(di, kind="stable")
    cut = np.searchsorted(di[order], np.arange(len(days) + 1))
    out, warm = p.copy(), np.zeros(len(days), bool)
    for d in range(window, len(days)):
        r = order[cut[d - window]:cut[d]]
        r = r[ok[r]]
        if len(r) < 10 * min_bin:
            continue
        r = r[np.argsort(p[r], kind="stable")]
        edges = np.linspace(0, len(r), len(r) // min_bin + 1).astype(int)
        n = np.diff(edges)
        x = np.add.reduceat(p[r], edges[:-1]) / n
        y = _pav(np.add.reduceat(hit[r], edges[:-1]) / n, n)
        t = order[cut[d]:cut[d + 1]]
        out[t] = np.interp(p[t], x, y)
        warm[d] = True
    return out, warm[di]


def move_runs(date, symbol, minute, match) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive matching decisions of a stock-day form one run: (start of a run, its length at
    the start row, 0 elsewhere). Rows may come in any order."""
    d = pd.DataFrame({"date": np.asarray(date), "symbol": np.asarray(symbol),
                      "minute": np.asarray(minute), "m": np.asarray(match, bool)})
    o = np.lexsort((d.minute.to_numpy(), d.symbol.to_numpy(), d.date.to_numpy()))
    s = d.iloc[o]
    same = (s.date.to_numpy()[1:] == s.date.to_numpy()[:-1]) & (s.symbol.to_numpy()[1:] == s.symbol.to_numpy()[:-1])
    m = s.m.to_numpy()
    prev = np.r_[False, same & m[:-1]]
    start = m & ~prev
    rid = np.cumsum(start)
    n = np.bincount(rid[m], minlength=rid.max() + 1 if len(rid) else 1)
    length = np.where(start, n[rid], 0)
    st_, ln = np.empty(len(d), bool), np.empty(len(d), np.int64)
    st_[o], ln[o] = start, length
    return st_, ln


_MV_STATS = {"said": "p", "reached": "reached", "touched": "touched", "not_formed": "not_formed",
             "centre_pct": "centre_move", "best_pct": "best_move", "captured": "captured"}


def _mv_agg(g: pd.DataFrame) -> dict:
    return {"predictions": len(g), "stocks": g["symbol"].nunique(), "said": g["p"].mean(),
            "reached": g["reached"].mean(), "touched": g["touched"].mean(),
            "not_formed": 1 - g["formed"].mean(), "median_centre_pct": g["centre_pct"].abs().median(),
            "median_best_pct": g["best_pct"].abs().median(), "median_vwap_move": g["vwap_move"].median(),
            "median_captured": g["captured"].median()}


def _boot_gap(day, d, seed: int = 0) -> tuple[float, float]:
    """95% day-bootstrap CI of mean(d) (d = happened − said per row)."""
    ok = np.isfinite(d)
    if ok.sum() < 2:
        return np.nan, np.nan
    days, inv = np.unique(np.asarray(day)[ok], return_inverse=True)
    s, n = np.bincount(inv, weights=d[ok]), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(days), (N_BOOT, len(days)))
    g = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975))


def move_summary(mt: pd.DataFrame, scored: dict[str, int]) -> pd.DataFrame:
    """The period: one row per side (and both together when there are two): predictions, stocks,
    days with any, decisions scored, said vs reached (gap with its day-bootstrap CI), touched,
    never formed, the median centre / best move and momentum captured."""
    rows = []
    groups = [(s, mt[mt.side == s]) for s in scored]
    if len(scored) > 1:
        groups.append(("both", mt))
    for s, g in groups:
        n_scored = sum(scored.values()) if s == "both" else scored[s]
        r = {"side": s, **_mv_agg(g), "days": g["date"].nunique(), "decisions": n_scored,
             "share_flagged": len(g) / n_scored if n_scored else np.nan}
        r["gap"] = r["reached"] - r["said"]
        r["gap_lo"], r["gap_hi"] = _boot_gap(g["date"], (g["reached"] - g["p"]).to_numpy(np.float64))
        rows.append(r)
    cols = ["side", "predictions", "stocks", "days", "decisions", "share_flagged", "said", "reached",
            "gap", "gap_lo", "gap_hi", "touched", "not_formed", "median_centre_pct", "median_best_pct",
            "median_vwap_move", "median_captured"]
    return pd.DataFrame(rows, columns=cols)


def move_by_day(mt: pd.DataFrame, scored_days: pd.DataFrame) -> pd.DataFrame:
    """One row per day and side: predictions, happened (centre reached), didn't, said (the mean P) and
    max_said, the rates, the median best high / low and VWAP move.
    scored_days: date, side, decisions (every scored decision, so days without a match show 0)."""
    g = mt.groupby(["date", "side"])
    t = pd.DataFrame({"predictions": g.size(), "happened": g["reached"].apply(lambda s: int((s == 1).sum())),
                      "did_not": g["reached"].apply(lambda s: int((s == 0).sum())),
                      "stocks": g["symbol"].nunique(), "said": g["p"].mean(), "max_said": g["p"].max(),
                      "reached": g["reached"].mean(), "touched": g["touched"].mean(),
                      "median_best_pct": g["best_pct"].apply(lambda s: s.abs().median()),
                      "median_vwap_move": g["vwap_move"].median(), "calibrated": g["warm"].all()})
    t = scored_days.set_index(["date", "side"]).join(t, how="left")
    for c in ("predictions", "happened", "did_not", "stocks"):
        t[c] = t[c].fillna(0).astype(int)
    t["calibrated"] = t["calibrated"].astype("boolean")
    return t.reset_index().sort_values(["date", "side"], ascending=[False, True], ignore_index=True)


def move_times(mt: pd.DataFrame, width: int = 15) -> pd.DataFrame:
    """Predictions per `width`-minute bucket of the day and side: happened, didn't, unknown, said."""
    if mt.empty:
        return pd.DataFrame(columns=["time", "side", "predictions", "happened", "did_not", "unknown",
                                     "said", "reached"])
    b = (mt["minute"] // width) * width
    g = mt.assign(bucket=b).groupby(["bucket", "side"])
    t = pd.DataFrame({"predictions": g.size(), "happened": g["reached"].apply(lambda s: int((s == 1).sum())),
                      "did_not": g["reached"].apply(lambda s: int((s == 0).sum())),
                      "said": g["p"].mean(), "reached": g["reached"].mean()}).reset_index()
    t["unknown"] = t["predictions"] - t["happened"] - t["did_not"]
    t.insert(0, "time", [f"{hhmm(m)}-{hhmm(m + width)}" for m in t["bucket"]])
    return t.drop(columns="bucket")[["time", "side", "predictions", "happened", "did_not", "unknown",
                                      "said", "reached"]]


def move_lean(mt: pd.DataFrame) -> pd.DataFrame:
    """Of the predictions, which way the model leaned (P up vs P down at the same decision) and which
    way the bin's volume went (bulge − sag that happened): per lean, how often the leaned side's centre
    reached the target, and how often the other side's did."""
    if mt.empty:
        return pd.DataFrame()
    up = mt["p_up"] >= mt["p_dn"]
    rows = []
    for lean, m in (("up (P up ≥ P down)", up), ("down (P down > P up)", ~up)):
        g = mt[m]
        if g.empty:
            continue
        lean_hit = np.where(up[m], g["reached_up"], g["reached_dn"])
        other_hit = np.where(up[m], g["reached_dn"], g["reached_up"])
        rows.append({"lean": lean, "predictions": len(g), "share": len(g) / len(mt),
                     "mean_p_up": g["p_up"].mean(), "mean_p_dn": g["p_dn"].mean(),
                     "net_p50": g["net_p50"].mean(), "went_up": (g["real_net"] > 0).where(
                         g["real_net"].notna()).mean(),
                     "lean_reached": np.nanmean(lean_hit) if np.isfinite(lean_hit).any() else np.nan,
                     "other_reached": np.nanmean(other_hit) if np.isfinite(other_hit).any() else np.nan})
    return pd.DataFrame(rows)


def move_insights(s: pd.DataFrame, lean: pd.DataFrame, x: float, window: str, min_p: float, calibrated: bool,
                  both: int, cold_days: int, has_ext: bool) -> list[Insight]:
    if s.empty or not s["predictions"].sum():
        return [Insight(INFO, f"No decision with P ≥ {min_p:.0%} for ±{x:.1%} in {window} under these "
                              "filters: lower the minimum P or X.")]
    out = []
    for _, r in s[s.side != "both"].iterrows():
        if not r.predictions:
            continue
        sign = "+" if r.side == "up" else "−"
        txt = (f"{sign}{x:.1%} in {window}, P ≥ {min_p:.0%} ({'calibrated' if calibrated else 'raw'}): "
               f"{r.predictions:,} predictions on {r.stocks:,} stocks over {r.days} days "
               f"({r.share_flagged:.1%} of {r.decisions:,} decisions); said {r.said:.0%}, the volume "
               f"centred at {sign}{x:.1%} or beyond in {r.reached:.0%}")
        if has_ext and np.isfinite(r.touched):
            txt += (f"; the price touched {sign}{x:.1%} in {r.touched:.0%}, median best "
                    f"{'high' if r.side == 'up' else 'low'} {sign}{r.median_best_pct:.2%}")
        ok = np.isfinite(r.gap_lo) and r.gap_lo <= 0 <= r.gap_hi
        out.append(Insight(GOOD if ok else WARN, txt))
        if np.isfinite(r.gap_lo):
            out.append(Insight(GOOD if ok else WARN,
                               f"{r.side}: happened − said = {r.gap * 100:+.1f} pts [{r.gap_lo * 100:+.1f}, "
                               f"{r.gap_hi * 100:+.1f}] (days resampled)"
                               + ("" if ok else ": P is off beyond noise here")))
    if not lean.empty:
        u = lean[lean.lean.str.startswith("up")]
        share = float(u.share.iloc[0]) if len(u) else 0.0
        hit = (lean.lean_reached * lean.predictions).sum() / lean.predictions.sum()
        oth = (lean.other_reached * lean.predictions).sum() / lean.predictions.sum()
        out.append(Insight(INFO, f"Lean: {share:.0%} of the predictions had P up ≥ P down. The leaned "
                                 f"side reached in {hit:.0%}, the other side in {oth:.0%}"
                                 + (": the lean picks no direction" if abs(hit - oth) < 0.03 else "")))
    if both:
        out.append(Insight(INFO, f"{both:,} decisions matched up and down at once: the model sizes "
                                 "the move, it doesn't call the direction"))
    if calibrated and cold_days:
        out.append(Insight(INFO, f"{cold_days} day(s) in the range have no 20 earlier sessions: raw P there"))
    if not has_ext:
        out.append(Insight(WARN, "No extremes.parquet in this fold: touched / best high / low are blank. "
                                 "Run `intraday-ft-extremes` on the run."))
    return out
