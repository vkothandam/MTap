"""Executive summaries: plain-English findings from the calib tables. Pure functions.

Each `*_insights` function returns a list of Insight(level, text), most important first. The level
only picks the icon. The thresholds below are judgement calls, kept in one place so they can be
tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import calib
from .calib import SIDE_DOWN, SIDE_UP

GOOD, WARN, BAD, INFO = "good", "warn", "bad", "info"

COVERAGE_TOL = 0.02                    # |coverage - nominal| within this: bands sized right
BIAS_SLIGHT, BIAS_STRONG = 0.01, 0.03  # |0.5 - mean PIT|
DEV_OK, DEV_BAD = 0.03, 0.10           # |came true - said| of a set of calls
LIFT_OK = 1.2                          # came true / base rate: the call ranks well
FOLD_SPREAD = 0.10                     # per-fold call deviation range that counts as unstable
NOISE_SHARE = 0.0455                   # P(|z| >= 2) for a calibrated forecast
MIN_CALLS = 100                        # calls needed before a call level is named best / worst
MIN_COMPARE_ROWS = 20_000              # fewer forecasts than this: run-vs-run gaps may be noise


@dataclass(frozen=True)
class Insight:
    level: str
    text: str


def _pts(x: float) -> str:
    return f"{x * 100:+.0f} pts"


def _call_label(r) -> str:
    return f"{calib.event_label(r['move'])} at P ≥ {r['confidence']:.0%}"


def _grade(dev: float) -> str:
    return GOOD if abs(dev) < DEV_OK else WARN if abs(dev) < DEV_BAD else BAD


def _lean(bias: float) -> str:
    return ("optimistic: actual moves land below the forecast" if bias > 0
            else "pessimistic: actual moves land above the forecast")


# ---- tables ----------------------------------------------------------------------------------

def call_table(events: calib.Events, confidences) -> pd.DataFrame:
    """One row per (event, call level): the calls with P >= c, how often they came true, the
    base rate (how often the event happened at all) and lift = came true / base rate."""
    parts = []
    for move, (prob, hit) in sorted(events.items()):
        t = calib.threshold_curve(prob, hit, sorted(confidences))
        t.insert(0, "event", calib.event_label(move))
        t.insert(0, "side", calib.side_of(move))
        t.insert(0, "move", move)
        t["share"] = t["n"] / max(1, len(prob))
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["move", "side", "event", "confidence", "n", "mean_prob",
                                     "hit_rate", "ci_lo", "ci_hi", "base_rate", "share",
                                     "deviation", "lift"])
    out = pd.concat(parts, ignore_index=True)
    out["deviation"] = out["hit_rate"] - out["mean_prob"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["lift"] = out["hit_rate"] / out["base_rate"]
    return out


def side_rollup(calls: pd.DataFrame) -> pd.DataFrame:
    """Per (side, call level), averaged over the selected moves, weighted by calls. A row can
    count under several moves (P(>= +1%) >= 70% implies P(>= +0.5%) >= 70%), so `n` is the
    number of (call, move) pairs, not distinct forecasts."""
    d = calls[calls["n"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["side", "confidence", "n", "mean_prob", "hit_rate",
                                     "base_rate", "deviation", "lift"])
    w = d.assign(**{c: d[c] * d["n"] for c in ("mean_prob", "hit_rate", "base_rate")})
    g = w.groupby(["side", "confidence"])[["n", "mean_prob", "hit_rate", "base_rate"]].sum()
    for c in ("mean_prob", "hit_rate", "base_rate"):
        g[c] = g[c] / g["n"]
    g["deviation"] = g["hit_rate"] - g["mean_prob"]
    g["lift"] = g["hit_rate"] / g["base_rate"]
    return g.reset_index()


def quantile_table(r_q: np.ndarray, r_act: np.ndarray, levels) -> pd.DataFrame:
    """Per forecast quantile: the share of actual moves that landed below it (ideal = level)."""
    observed = (r_act[:, None] < r_q).mean(axis=0) if len(r_act) else np.full(len(levels), np.nan)
    t = pd.DataFrame({"level": levels, "observed": observed})
    t["gap"] = t["observed"] - t["level"]
    return t


def fold_table(keys: pd.DataFrame, r_q: np.ndarray, r_act: np.ndarray, levels,
               events: calib.Events, confidence: float) -> pd.DataFrame:
    """Per test fold: span, skill, coverage, bias and each side's calls at `confidence`."""
    fold = keys["fold"].to_numpy()
    days = keys["date"].to_numpy()
    rows = []
    for k in np.unique(fold):
        i = fold == k
        s = calib.summary(r_q[i], r_act[i], levels)
        row = {"fold": int(k), "first_day": days[i].min(), "last_day": days[i].max(),
               "rows": s["rows"], "pinball_skill": s["pinball_skill"], "coverage": s["coverage"],
               "bias": s["bias"]}
        roll = side_rollup(call_table({m: (p[i], h[i]) for m, (p, h) in events.items()},
                                      [confidence]))
        for side in (SIDE_UP, SIDE_DOWN):
            r = roll[roll["side"] == side]
            row[f"{side}_calls"] = int(r["n"].iloc[0]) if len(r) else 0
            row[f"{side}_said"] = float(r["mean_prob"].iloc[0]) if len(r) else np.nan
            row[f"{side}_came_true"] = float(r["hit_rate"].iloc[0]) if len(r) else np.nan
            row[f"{side}_deviation"] = float(r["deviation"].iloc[0]) if len(r) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _day(x) -> str:
    return str(pd.Timestamp(x).date()) if isinstance(x, np.datetime64 | pd.Timestamp) else str(x)


# ---- whole run -------------------------------------------------------------------------------

def run_insights(summ: dict, calls: pd.DataFrame, folds: pd.DataFrame | None,
                 confidence: float) -> list[Insight]:
    """Headline findings: skill, band width, overall lean, each side's calls at `confidence`,
    the best and worst call levels and fold-to-fold stability."""
    out: list[Insight] = []
    if not summ.get("rows"):
        return out
    s = summ["pinball_skill"]
    if s > 0.05:
        out.append(Insight(GOOD, f"The forecast distribution beats a naive 'no move' forecast: "
                                 f"quantile loss is {s:.0%} lower."))
    elif s > 0:
        out.append(Insight(WARN, f"The forecast distribution only just beats a naive 'no move' "
                                 f"forecast (quantile loss {s:.1%} lower)."))
    else:
        out.append(Insight(BAD, f"The forecast distribution does not beat a naive 'no move' "
                                f"forecast (quantile loss {-s:.1%} higher)."))

    cov, nom = summ["coverage"], summ["coverage_nominal"]
    if abs(cov - nom) <= COVERAGE_TOL:
        out.append(Insight(GOOD, f"The bands are about the right width: P5–P95 held {cov:.1%} "
                                 f"of actual moves (target {nom:.0%})."))
    else:
        how = ("too narrow: big moves happen more often than the model expects" if cov < nom
               else "too wide: the model expects more big moves than happen")
        out.append(Insight(WARN, f"The bands are {how}. P5–P95 held {cov:.1%} of actual moves "
                                 f"(target {nom:.0%})."))

    b = summ["bias"]
    if abs(b) < BIAS_SLIGHT:
        out.append(Insight(GOOD, f"No material overall lean (bias {b:+.3f}; 0 = none)."))
    else:
        level = WARN if abs(b) < BIAS_STRONG else BAD
        size = "slightly" if abs(b) < BIAS_STRONG else "clearly"
        out.append(Insight(level, f"Overall the model is {size} {_lean(b)} (bias {b:+.3f})."))

    out += _side_calls(calls, confidence)
    out += _best_worst(calls)
    if folds is not None and len(folds) > 1:
        out += _fold_stability(folds, confidence)
    return out


def _side_calls(calls: pd.DataFrame, confidence: float) -> list[Insight]:
    roll = side_rollup(calls[np.isclose(calls["confidence"], confidence)])
    out, devs = [], {}
    for _, r in roll.iterrows():
        dev = r["deviation"]
        devs[r["side"]] = dev
        what = ("about right" if abs(dev) < DEV_OK
                else "overconfident" if dev < 0 else "too cautious")
        out.append(Insight(_grade(dev), (
            f"{r['side'].capitalize()}side calls at P ≥ {confidence:.0%} are {what}: the model "
            f"said {r['mean_prob']:.0%} on average and they came true {r['hit_rate']:.0%} "
            f"({_pts(dev)}), averaged over the selected moves.")))
        if r["lift"] >= LIFT_OK and dev < -DEV_OK:
            out.append(Insight(INFO, (
                f"The {r['side']}side probability still ranks well: those calls came true "
                f"{r['lift']:.1f}× as often as the base rate ({r['base_rate']:.0%}). A high "
                f"number points the right way even though it overstates the odds.")))
        elif r["lift"] < LIFT_OK:
            out.append(Insight(BAD, (
                f"{r['side'].capitalize()}side calls barely beat the base rate (came true "
                f"{r['hit_rate']:.0%} vs {r['base_rate']:.0%} for any forecast).")))
    if len(devs) == 2 and abs(devs[SIDE_UP] - devs[SIDE_DOWN]) > DEV_OK:
        worse = min(devs, key=devs.get)
        better = SIDE_UP if worse == SIDE_DOWN else SIDE_DOWN
        out.append(Insight(INFO, (
            f"{worse.capitalize()}side calls are less reliable than {better}side calls "
            f"({_pts(devs[worse])} vs {_pts(devs[better])}).")))
    return out


def _best_worst(calls: pd.DataFrame) -> list[Insight]:
    d = calls[calls["n"] >= MIN_CALLS].dropna(subset=["deviation"])
    if d.empty:
        return []
    best = d.loc[d["deviation"].abs().idxmin()]
    worst = d.loc[d["deviation"].idxmin()]
    out = [Insight(GOOD if abs(best["deviation"]) < DEV_OK else INFO, (
        f"Most trustworthy call: {_call_label(best)}. The model said {best['mean_prob']:.0%} "
        f"and it came true {best['hit_rate']:.0%} ({int(best['n']):,} calls)."))]
    if worst["deviation"] < -DEV_OK and worst.name != best.name:
        out.append(Insight(BAD if worst["deviation"] < -DEV_BAD else WARN, (
            f"Least trustworthy call: {_call_label(worst)}. The model said "
            f"{worst['mean_prob']:.0%} but it came true only {worst['hit_rate']:.0%} "
            f"({int(worst['n']):,} calls).")))
    return out


def _fold_stability(folds: pd.DataFrame, confidence: float) -> list[Insight]:
    out = []
    for side in (SIDE_UP, SIDE_DOWN):
        f = folds.dropna(subset=[f"{side}_deviation"])
        f = f[f[f"{side}_calls"] >= MIN_CALLS]
        if len(f) < 2:
            continue
        dev = f[f"{side}_deviation"]
        lo, hi = f.loc[dev.idxmin()], f.loc[dev.idxmax()]
        if hi[f"{side}_deviation"] - lo[f"{side}_deviation"] > FOLD_SPREAD:
            out.append(Insight(WARN, (
                f"{side.capitalize()}side calls at P ≥ {confidence:.0%} are not stable across "
                f"time. In fold {lo['fold']} ({_day(lo['first_day'])} to "
                f"{_day(lo['last_day'])}, {int(lo[f'{side}_calls']):,} calls) they were off by "
                f"{_pts(lo[f'{side}_deviation'])}. In fold {hi['fold']} "
                f"({int(hi[f'{side}_calls']):,} calls) they were off by "
                f"{_pts(hi[f'{side}_deviation'])}. The pooled numbers mix regimes, so check "
                f"fold by fold.")))
        else:
            out.append(Insight(GOOD, (
                f"{side.capitalize()}side calls at P ≥ {confidence:.0%} are consistent across "
                f"folds ({_pts(dev.min())} to {_pts(dev.max())}).")))
    losing = folds[folds["pinball_skill"] <= 0]
    if len(losing):
        out.append(Insight(BAD, "Fold(s) " + ", ".join(str(k) for k in losing["fold"])
                           + " do not beat the naive forecast."))
    return out


# ---- per page --------------------------------------------------------------------------------

def grid_insights(g: pd.DataFrame, min_n: int) -> list[Insight]:
    """Calibration tab: how many grid cells are consistent with calibration, plus the worst
    overconfident and most cautious cells."""
    d = g[g["n"] >= min_n]
    if d.empty:
        return [Insight(INFO, f"No grid cell has {min_n}+ forecasts; lower 'Min n per grid "
                              "point'.")]
    ece = float((d["deviation"].abs() * d["n"]).sum() / d["n"].sum())
    close = float((d["deviation"].abs() < DEV_OK).mean())
    noise = float(((d["ci_lo"] <= d["mean_prob"]) & (d["mean_prob"] <= d["ci_hi"])).mean())
    out = [Insight(_grade(ece) if close >= 0.7 else WARN, (
        f"On average a probability is off by {ece * 100:.1f} pts, weighted by forecasts. Most "
        f"forecasts sit in the low buckets, which pulls this down. Unweighted, {close:.0%} of "
        f"the {len(d)} cells are within ±{DEV_OK * 100:.0f} pts of what was said, and "
        f"{noise:.0%} are within noise (95% interval). With large n even small gaps are "
        f"significant."))]
    lo, hi = d.loc[d["deviation"].idxmin()], d.loc[d["deviation"].idxmax()]
    if lo["deviation"] < -DEV_OK:
        out.append(Insight(_grade(lo["deviation"]), (
            f"Most overconfident: {lo['event']} when the model said {lo['bucket']}. It said "
            f"{lo['mean_prob']:.0%} and it came true {lo['hit_rate']:.0%} (n={int(lo['n']):,}).")))
    if hi["deviation"] > DEV_OK:
        out.append(Insight(_grade(hi["deviation"]), (
            f"Most cautious: {hi['event']} when the model said {hi['bucket']}. It said "
            f"{hi['mean_prob']:.0%} and it came true {hi['hit_rate']:.0%} (n={int(hi['n']):,}).")))
    return out


def ladder_insights(lh: pd.DataFrame) -> list[Insight]:
    """Ladder: does the actual clear the model's own c-confidence move c of the time?"""
    out = []
    for side, d in lh.groupby("side", sort=False):
        worst = d.loc[d["deviation"].abs().idxmax()]
        out.append(Insight(_grade(worst["deviation"]), (
            f"Ladder, {side}side: at {worst['confidence']:.0%} confidence the actual cleared the "
            f"model's own move {worst['hit_rate']:.0%} of the time (largest gap of the selected "
            f"levels: {_pts(worst['deviation'])}).")))
    return out


def threshold_insights(t: pd.DataFrame, event: str, confidence: float) -> list[Insight]:
    """Threshold tab: the chosen call level, the level where the hit rate reaches each target,
    and whether higher confidence buys accuracy at all."""
    t = t[t["n"] > 0]
    if t.empty:
        return [Insight(INFO, f"No forecast gives {event} any probability in the 5–95% range.")]
    out = []
    row = t[np.isclose(t["confidence"], confidence)]
    if len(row) and row["n"].iloc[0] > 0:
        r = row.iloc[0]
        lift = r["hit_rate"] / r["base_rate"] if r["base_rate"] else np.nan
        out.append(Insight(_grade(r["hit_rate"] - r["mean_prob"]), (
            f"{event} at P ≥ {confidence:.0%}: {int(r['n']):,} calls came true "
            f"{r['hit_rate']:.0%} (the model said {r['mean_prob']:.0%}). Base rate "
            f"{r['base_rate']:.0%}, so {lift:.1f}× better than not filtering.")))
    else:
        out.append(Insight(INFO, f"No calls at P ≥ {confidence:.0%} for {event}."))
    solid = t[t["n"] >= MIN_CALLS]
    reach = []
    for target in (0.5, 0.6, 0.7, 0.8):
        hit = solid[solid["hit_rate"] >= target]
        if len(hit):
            first = hit.iloc[0]
            reach.append(f"{target:.0%} from P ≥ {first['confidence']:.0%} "
                         f"({int(first['n']):,} calls)")
    if reach:
        out.append(Insight(INFO, "The hit rate reaches " + "; ".join(reach) + "."))
    elif len(solid):
        out.append(Insight(WARN, (f"The hit rate never reaches 50% at any call level with "
                                  f"{MIN_CALLS}+ calls (best {solid['hit_rate'].max():.0%}).")))
    if len(solid) >= 2:
        peak = solid.loc[solid["hit_rate"].idxmax()]
        top = solid.iloc[-1]
        if top["hit_rate"] < peak["hit_rate"] - DEV_OK:
            out.append(Insight(WARN, (
                f"Higher confidence stops helping. Calls at P ≥ {top['confidence']:.0%} came "
                f"true {top['hit_rate']:.0%}, below the peak of {peak['hit_rate']:.0%} at P ≥ "
                f"{peak['confidence']:.0%}. The model's most extreme numbers are its least "
                f"reliable.")))
    return out


def stock_insights(b: pd.DataFrame, industries: pd.DataFrame | None = None) -> list[Insight]:
    """Across stocks: is per-stock lean real or noise, which way, and which stocks / industries
    lean most."""
    if b.empty:
        return []
    counts = b["label"].value_counts()
    opt, pes = int(counts.get("optimistic", 0)), int(counts.get("pessimistic", 0))
    expected = NOISE_SHARE * len(b)
    flagged = opt + pes
    if flagged <= 1.5 * expected:
        out = [Insight(GOOD, (
            f"Per-stock lean is mostly noise: {flagged:,} of {len(b):,} stocks are flagged and "
            f"~{expected:,.0f} would be flagged by chance alone."))]
    else:
        side = ("mostly pessimistic" if pes > 2 * opt else "mostly optimistic" if opt > 2 * pes
                else "split both ways")
        out = [Insight(WARN, (
            f"Some stocks really lean: {flagged:,} of {len(b):,} are flagged ({opt:,} "
            f"optimistic, {pes:,} pessimistic; {side}) and only ~{expected:,.0f} would be by "
            f"chance. Look at those stocks before trusting their calls."))]
    top = b.reindex(b["bias_z"].abs().sort_values(ascending=False).index).head(3)
    top = top[top["bias_z"].abs() >= 2]
    if len(top):
        out.append(Insight(INFO, "Strongest lean: " + ", ".join(
            f"{r.symbol} ({r.label}, z {r.bias_z:+.1f}, n={r.n})" for r in top.itertuples())
            + "."))
    if industries is not None and len(industries):
        real = industries[industries["bias_z"].abs() >= 2]
        if len(real):
            worst = real.reindex(real["bias_z"].abs().sort_values(ascending=False).index).head(3)
            out.append(Insight(INFO, (
                f"{len(real)} of {len(industries)} industries have |z| ≥ 2, strongest: "
                + ", ".join(f"{r.industry_code} ({r.label}, z {r.bias_z:+.1f})"
                            for r in worst.itertuples())
                + ". z treats every forecast as independent, but stocks in one industry move "
                  "together on the same day, so these z values are overstated.")))
        else:
            out.append(Insight(GOOD, "No industry leans beyond noise."))
    return out


def day_insights(d: pd.DataFrame) -> list[Insight]:
    """By day: are misses market-wide (regime days) and which days were worst."""
    if d.empty:
        return []
    regime = d[d["bias_z"].abs() >= 2]
    expected = NOISE_SHARE * len(d)
    if len(regime) <= 1.5 * expected:
        out = [Insight(GOOD, (
            f"{len(regime)} of {len(d)} days are regime days (~{expected:.0f} expected by "
            f"chance), so misses look stock-specific, not market-wide."))]
    else:
        out = [Insight(WARN, (
            f"{len(regime)} of {len(d)} days are regime days, when the whole market missed the "
            f"same way (~{expected:.0f} expected by chance). Errors come in market-wide waves "
            f"that a per-stock model cannot see coming."))]
    lo, hi = d.loc[d["bias"].idxmin()], d.loc[d["bias"].idxmax()]
    out.append(Insight(INFO, (
        f"Worst optimistic day {_day(hi['day'])} (bias {hi['bias']:+.2f}, z {hi['bias_z']:+.0f}); "
        f"worst pessimistic day {_day(lo['day'])} (bias {lo['bias']:+.2f}, "
        f"z {lo['bias_z']:+.0f}).")))
    if "deviation" in d.columns:
        dev = d["deviation"].dropna()
        if len(dev):
            out.append(Insight(INFO, (
                f"Overlay event: came true − said ranged {_pts(dev.min())} to "
                f"{_pts(dev.max())} across days (median {_pts(dev.median())}).")))
    return out


def stock_detail_insights(symbol: str, b: pd.Series, r_q: np.ndarray, r_act: np.ndarray,
                          u: np.ndarray, nominal: float) -> list[Insight]:
    """One stock: lean, band coverage and the PIT shape."""
    out = []
    if b["label"] == "calibrated":
        out.append(Insight(GOOD, f"{symbol} has no lean beyond noise (bias {b['bias']:+.3f}, "
                                 f"z {b['bias_z']:+.1f}, n={int(b['n'])})."))
    else:
        out.append(Insight(WARN, f"For {symbol} the model is {_lean(b['bias'])} (bias "
                                 f"{b['bias']:+.3f}, z {b['bias_z']:+.1f}, n={int(b['n'])})."))
    cov = float(((r_q[:, 0] <= r_act) & (r_act <= r_q[:, -1])).mean())
    se = np.sqrt(nominal * (1 - nominal) / max(1, len(r_act)))
    if abs(cov - nominal) <= max(COVERAGE_TOL, 2 * se):
        out.append(Insight(GOOD, f"P5–P95 held {cov:.0%} of its moves (target {nominal:.0%})."))
    else:
        how = "too narrow" if cov < nominal else "too wide"
        out.append(Insight(WARN, f"Its bands are {how}: P5–P95 held {cov:.0%} of its moves "
                                 f"(target {nominal:.0%})."))
    tails = float(((u <= 0.1) | (u >= 0.9)).mean())
    if tails > 0.3:
        out.append(Insight(WARN, (f"{tails:.0%} of actual moves landed in the outer 10% "
                                  f"tails (20% expected): a U-shaped PIT, so the bands are "
                                  f"too narrow for this stock.")))
    elif tails < 0.1:
        out.append(Insight(INFO, (f"Only {tails:.0%} of actual moves landed in the outer 10% "
                                  f"tails (20% expected): the bands are wider than needed.")))
    return out


def tomorrow_insights(s: pd.DataFrame, event: str, confidence: float,
                      min_hist: int = 20, day: str | None = None,
                      band: tuple[float, float] | None = None) -> list[Insight]:
    """Forecast: how many stocks qualify and how many have test history backing the number.
    On a graded day (`s` has `inside`), also how many shown stocks' actuals fell in `band`."""
    if s.empty:
        when = f"on {day}" if day else "for this day"
        return [Insight(INFO, f"No stock has P({event}) ≥ {confidence:.0%} {when}.")]
    backed = s[s["hist_n"].fillna(0) >= min_hist]
    held = backed[backed["hist_hit_rate"] >= confidence - 1e-9]
    out = [Insight(INFO, (
        f"{len(s):,} stocks have P({event}) ≥ {confidence:.0%}. {len(backed):,} of them have "
        f"{min_hist}+ test forecasts in the same probability bucket, and in {len(held):,} of "
        f"those the event came true at least {confidence:.0%} of the time."))]
    allc = s.dropna(subset=["all_hit_rate"]).groupby("bucket")[["all_hit_rate", "all_n"]].first()
    if len(allc):
        out.append(Insight(INFO, "Across all stocks, past calls in these buckets came true: "
                           + ", ".join(f"{b} → {r.all_hit_rate:.0%} (n={int(r.all_n):,})"
                                       for b, r in allc.iterrows()) + "."))
    if "inside" in s and band is not None and s["inside"].notna().any():
        got = s["inside"].dropna().astype(bool)
        nom = band[1] - band[0]
        lvl = GOOD if abs(got.mean() - nom) <= COVERAGE_TOL else WARN
        out.append(Insight(lvl, f"{day} has happened: the actual VWAP landed inside the "
                                f"{_band(*band)} range for {got.mean():.0%} of these "
                                f"{len(got):,} stocks (target {nom:.0%}). The full grade is in "
                                f"Forecast vs actual."))
    return out


def _band(lo: float, hi: float) -> str:
    return f"P{lo * 100:.0f}–P{hi * 100:.0f}"


def live_insights(g: dict, lo: float, hi: float, by_day: pd.DataFrame | None = None,
                  day: str | None = None) -> list[Insight]:
    """Forecast vs actual: did the band hold, which way the misses went, direction, skill, and
    (with several graded days) how steady that is."""
    if not g.get("rows"):
        return [Insight(INFO, "No graded forecasts yet.")]
    band, cov, nom = _band(lo, hi), g["coverage"], g["coverage_nominal"]
    when = f" on {day}" if day else ""
    out = []
    if abs(cov - nom) <= COVERAGE_TOL:
        out.append(Insight(GOOD, f"The {band} band held {cov:.0%} of {g['rows']:,} stocks' "
                                 f"actual VWAP{when} (target {nom:.0%}): about the right width."))
    else:
        how = "too narrow" if cov < nom else "wider than needed"
        out.append(Insight(WARN if cov < nom else INFO,
                           f"The {band} band held {cov:.0%} of {g['rows']:,} stocks' actual "
                           f"VWAP{when} (target {nom:.0%}): {how} for this day."))
    ab, be = g["above"], g["below"]
    out.append(Insight(INFO, f"Misses: {ab:.0%} landed above the band, {be:.0%} below "
                             f"(about {(1 - nom) / 2:.0%} each expected)."))
    b = g["bias"]
    if abs(b) >= BIAS_SLIGHT:
        lean = ("optimistic: actuals landed below the forecast" if b > 0
                else "pessimistic: actuals landed above the forecast")
        out.append(Insight(WARN if abs(b) >= BIAS_STRONG else INFO,
                           f"The forecasts leaned {lean} (bias {b:+.3f}). Median actual move "
                           f"{g['median_actual']:+.2%} vs median forecast "
                           f"{g['median_forecast']:+.2%}."))
    else:
        out.append(Insight(GOOD, f"No overall lean (bias {b:+.3f}): actuals were spread "
                                 f"around the forecast median."))
    out.append(Insight(INFO, f"The forecast median had the direction right for "
                             f"{g['direction_hit']:.0%} of stocks."))
    sk = g["pinball_skill"]
    out.append(Insight(GOOD if sk > 0 else BAD,
                       f"Quantile loss is {abs(sk):.0%} {'lower' if sk > 0 else 'higher'} than "
                       f"a naive 'no move' forecast."))
    if by_day is not None and len(by_day) > 1:
        out.append(Insight(INFO, (
            f"Over {len(by_day)} graded days the {band} band held {by_day['coverage'].mean():.0%}"
            f" on average (range {by_day['coverage'].min():.0%}–{by_day['coverage'].max():.0%})."
        )))
    else:
        out.append(Insight(INFO, "This is one day. Stocks move together, so a single day's "
                                 "coverage swings with the whole market; judge after several "
                                 "graded days."))
    return out


def live_headline(by_day: pd.DataFrame, lo: float, hi: float) -> list[Insight]:
    """Summary tab: one line on the live (out-of-sample) forecasts, if any were graded."""
    if by_day is None or by_day.empty:
        return []
    last = by_day.iloc[-1]
    nom = last["coverage_nominal"]
    lvl = GOOD if abs(last["coverage"] - nom) <= COVERAGE_TOL else WARN
    more = (f"; {len(by_day)} days graded, {by_day['coverage'].mean():.0%} on average"
            if len(by_day) > 1 else "")
    return [Insight(lvl, f"Live forecast for {last['day']}: the {_band(lo, hi)} band held "
                         f"{last['coverage']:.0%} of actual VWAPs (target {nom:.0%}){more}. "
                         f"See Forecast → Forecast vs actual.")]


def compare_insights(summaries: dict[str, dict], calls: dict[str, pd.DataFrame],
                     confidence: float, same_rows: bool) -> list[Insight]:
    """Compare runs: the winner on each metric, and whose calls at `confidence` held best."""
    runs = {n: s for n, s in summaries.items() if s.get("rows")}
    if len(runs) < 2:
        return [Insight(INFO, "Pick two or more runs to get a comparison.")]
    out = []
    if not same_rows:
        out.append(Insight(WARN, "These runs are scored on different rows (stocks or days), so "
                                 "a gap can come from the data rather than the model."))
    # (label, shown value, format, score: higher is better)
    metrics = [
        ("quantile skill vs naive", "pinball_skill", "{:.1%}", lambda s: s["pinball_skill"]),
        ("median direction right", "direction", "{:.1%}", lambda s: s.get("direction", np.nan)),
        ("band coverage closest to target", "coverage", "{:.1%}",
         lambda s: -abs(s["coverage"] - s["coverage_nominal"])),
        ("smallest overall lean", "bias", "{:+.3f}", lambda s: -abs(s["bias"])),
        ("lowest probability error (ECE)", "ece", "{:.3f}", lambda s: -s.get("ece", np.nan)),
    ]
    wins: dict[str, int] = dict.fromkeys(runs, 0)
    lines = []
    for label, shown, fmt, score in metrics:
        scores = {n: score(s) for n, s in runs.items() if not np.isnan(score(s))}
        if not scores:
            continue
        best = max(scores, key=scores.get)
        wins[best] += 1
        lines.append(f"{label}: **{best}** (" + ", ".join(
            f"{n} {fmt.format(runs[n][shown])}" for n in scores) + ")")
    most = max(wins.values())
    top = [n for n, w in wins.items() if w == most]
    if len(top) == 1:
        out.append(Insight(GOOD, f"**{top[0]}** wins {most} of {len(lines)} metrics."))
    else:
        out.append(Insight(INFO, f"No clear winner: {', '.join(f'**{n}**' for n in top)} win "
                                 f"{most} of {len(lines)} metrics each."))
    fewest = min(s["rows"] for s in runs.values())
    if fewest < MIN_COMPARE_ROWS:
        out.append(Insight(WARN, f"Only {fewest:,} forecasts in the smallest run. Gaps between "
                                 f"runs on so few rows can be noise."))
    out += [Insight(INFO, line) for line in lines]
    devs = {}
    for n, c in calls.items():
        roll = side_rollup(c[np.isclose(c["confidence"], confidence)])
        if len(roll):
            devs[n] = roll.set_index("side")["deviation"]
    if len(devs) >= 2:
        score = {n: float(d.abs().mean()) for n, d in devs.items()}
        best = min(score, key=score.get)
        out.append(Insight(INFO, (
            f"Calls at P ≥ {confidence:.0%} held best in **{best}** ("
            + "; ".join(f"{n}: " + ", ".join(f"{side} {_pts(v)}" for side, v in d.items())
                        for n, d in devs.items()) + ").")))
    return out
