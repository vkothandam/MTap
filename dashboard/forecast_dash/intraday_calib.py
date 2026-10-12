"""Scoring for the intraday model. Pure numpy/pandas, built on calib.py.

All forecasts are log moves from the last price at the decision time t. To grade against another
reference (trailing VWAP, session VWAP) subtract that reference's own move from the quantiles and
the actual: r_q = q - ref, r_act = actual - ref. Then "no move" (0) is the naive forecast "the
VWAP ahead equals the reference", and calib.py's maths applies unchanged.

Per decision row we compute small metrics once (`row_metrics`); every table is a groupby of them.

The baseline that matters is climatology: the training days' spread of each move from the last
price, per 15-minute slot. The VWAP ahead stays near the last price while the trailing VWAP lags,
so momentum ("the VWAP ahead equals the trailing VWAP") and the direction call vs the trailing
VWAP flatter any forecast centred near the last price. Runs written before climatology existed
fall back to momentum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calib
from .insights import BAD, GOOD, INFO, WARN, Insight
from .intraday_contract import CLIMCOLS, DAILYCOLS, LEVELS, MOMCOLS, QCOLS

TARGET_LABELS = {"r_1m": "Next 1 min VWAP", "r_5m": "Next 5 min", "r_15m": "Next 15 min",
                 "r_60m": "Next 60 min", "r_rod": "Rest of day", "r_close": "Whole-day VWAP"}
TARGET_SHORT = {"r_1m": "1 min", "r_5m": "5 min", "r_15m": "15 min", "r_60m": "60 min",
                "r_rod": "rest of day", "r_close": "whole day"}
REF_LABELS = {"trail": "Trailing VWAP", "last": "Last price", "session": "Session VWAP"}
REF_COLUMN = {"trail": "ref_trail", "session": "ref_session", "last": None}
COVERAGE_TOL = 0.03       # |coverage80 - 0.80| within this: bands sized right
MIN_ROWS = 200            # fewer decision points than this: a rate is noise
Q10, Q50, Q90 = QCOLS[1], QCOLS[9], QCOLS[17]
HEADLINE = ["target", "n", "coverage80", "skill_clim", "skill_mom", "dir_last"]
BASELINE_LABELS = {"skill_clim": "climatology", "skill_mom": "momentum"}


def target_label(t: str) -> str:
    return TARGET_LABELS.get(t, t)


def hhmm(t_s) -> np.ndarray:
    """Clock time (ET) of seconds since the 09:30 open."""
    s = np.asarray(t_s, dtype=int) + 9 * 3600 + 1800
    return np.array([f"{x // 3600:02d}:{x % 3600 // 60:02d}" for x in s])


# ---- shifting to a reference ---------------------------------------------------------------------

def ref_move(df: pd.DataFrame, ref: str) -> np.ndarray:
    """The reference's log move from the last price (0 for the last price itself)."""
    col = REF_COLUMN[ref]
    return np.zeros(len(df)) if col is None else df[col].to_numpy(dtype=float)


def shift(df: pd.DataFrame, ref: str = "trail", cols=QCOLS) -> np.ndarray:
    """Quantiles `cols` as moves from the reference, sorted per row."""
    q = df[cols].to_numpy(dtype=float) - ref_move(df, ref)[:, None]
    return np.sort(q, axis=1)


def shift_actual(df: pd.DataFrame, ref: str = "trail") -> np.ndarray:
    return df["actual"].to_numpy(dtype=float) - ref_move(df, ref)


def _pinball_rows(r_q: np.ndarray, r_act: np.ndarray, levels=LEVELS) -> np.ndarray:
    lv = np.asarray(levels, dtype=float)[None, :]
    err = r_act[:, None] - r_q
    return np.maximum(lv * err, (lv - 1) * err).mean(axis=1)


def _direction(df: pd.DataFrame, ref: str) -> np.ndarray:
    """1/0 = the median called the side of the reference right/wrong; NaN = it did not move."""
    m = ref_move(df, ref)
    pred, act = df[Q50].to_numpy(dtype=float) - m, df["actual"].to_numpy(dtype=float) - m
    return np.where(act == 0, np.nan, (np.sign(pred) == np.sign(act)).astype(float))


# ---- per-row metrics and the tables built from them ------------------------------------------

def row_metrics(df: pd.DataFrame, ref: str = "trail") -> pd.DataFrame:
    """One row per decision point: inside80, pinball (model / random walk / climatology /
    momentum), PIT u, band width and direction hits against each reference."""
    r_q, r_act = shift(df, ref), shift_actual(df, ref)
    has_mom = all(c in df for c in MOMCOLS)
    has_clim = all(c in df for c in CLIMCOLS)
    out = pd.DataFrame({
        "inside80": ((r_q[:, 1] <= r_act) & (r_act <= r_q[:, 17])).astype(float),
        "pin": _pinball_rows(r_q, r_act),
        "pin_rw": _pinball_rows(np.zeros_like(r_q), r_act),
        "pin_mom": _pinball_rows(shift(df, ref, MOMCOLS), r_act) if has_mom else np.nan,
        "pin_clim": _pinball_rows(shift(df, ref, CLIMCOLS), r_act) if has_clim else np.nan,
        "u": calib.pit(r_q, r_act, LEVELS),
        "width80": r_q[:, 17] - r_q[:, 1],
        **{f"dir_{r}": _direction(df, r) for r in REF_COLUMN},
    }, index=df.index)
    return out


def _agg(m: pd.DataFrame, by) -> pd.DataFrame:
    g = m.groupby(by, observed=True)
    out = g.agg(n=("pin", "size"), coverage80=("inside80", "mean"), pinball=("pin", "mean"),
                _rw=("pin_rw", "sum"), _mom=("pin_mom", "sum"), _clim=("pin_clim", "sum"),
                _p=("pin", "sum"),
                pit_bias=("u", lambda u: 0.5 - u.mean()), width80_med=("width80", "median"),
                dir_last=("dir_last", "mean"), dir_trail=("dir_trail", "mean"),
                dir_session=("dir_session", "mean"))
    out["skill_rw"] = 1 - out["_p"] / out["_rw"].where(out["_rw"] > 0)
    out["skill_mom"] = 1 - out["_p"] / out["_mom"].where(out["_mom"] > 0)
    out["skill_clim"] = 1 - out["_p"] / out["_clim"].where(out["_clim"] > 0)
    cols = ["n", "coverage80", "pinball", "skill_clim", "skill_rw", "skill_mom", "dir_last",
            "dir_trail",
            "dir_session", "pit_bias", "width80_med"]
    return out[cols].reset_index()


def score(df: pd.DataFrame, ref: str = "trail") -> dict:
    """The run-level scores of one target as a dict ({} when there are no rows)."""
    if df.empty:
        return {}
    t = _agg(row_metrics(df, ref).assign(_all=0), "_all")
    return t.drop(columns="_all").iloc[0].to_dict()


def summary(frames: dict[str, pd.DataFrame], ref: str = "trail") -> pd.DataFrame:
    """One row per target (in the order given) with `score`."""
    rows = [{"target": t, **score(d, ref)} for t, d in frames.items() if not d.empty]
    return pd.DataFrame(rows)


def by_group(df: pd.DataFrame, key: str | list[str], ref: str = "trail") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = [key] if isinstance(key, str) else list(key)
    m = row_metrics(df, ref)
    for k in keys:
        m[k] = df[k].to_numpy()
    return _agg(m.dropna(subset=keys), keys)


def by_tod(df: pd.DataFrame, ref: str = "trail") -> pd.DataFrame:
    """Scores per 15-minute slot; adds `time` (the slot's start, ET)."""
    t = by_group(df, "tod_bucket", ref)
    if not t.empty:
        t.insert(1, "time", hhmm(t["tod_bucket"].to_numpy() * 900))
    return t


def by_symbol(df: pd.DataFrame, ref: str = "trail", min_n: int = 1) -> pd.DataFrame:
    t = by_group(df, "symbol", ref)
    if t.empty:
        return t
    sector = df.groupby("symbol")["sector_code"].agg(lambda s: s.dropna().iloc[0]
                                                      if s.notna().any() else "unknown")
    t.insert(1, "sector_code", t["symbol"].map(sector))
    return t[t["n"] >= min_n].sort_values(skill_col(t), ascending=False).reset_index(drop=True)


def by_sector(df: pd.DataFrame, ref: str = "trail") -> pd.DataFrame:
    return by_group(df.assign(sector_code=df["sector_code"].fillna("unknown")),
                    "sector_code", ref)


def by_day(df: pd.DataFrame, ref: str = "trail") -> pd.DataFrame:
    t = by_group(df, "date", ref)
    return t.sort_values("date").reset_index(drop=True) if not t.empty else t


# ---- calibration -----------------------------------------------------------------------------

def quantile_calibration(df: pd.DataFrame, ref: str = "trail") -> pd.DataFrame:
    """level, observed (share of actuals below that quantile), gap."""
    r_q, r_act = shift(df, ref), shift_actual(df, ref)
    obs = (r_act[:, None] < r_q).mean(axis=0) if len(r_act) else np.full(len(LEVELS), np.nan)
    t = pd.DataFrame({"level": LEVELS, "observed": obs})
    t["gap"] = t["observed"] - t["level"]
    return t


def prob_grid(df: pd.DataFrame, ref: str = "trail", width: float = 0.1) -> pd.DataFrame:
    """P(VWAP ahead above the reference): said vs came true per probability bucket
    (calib.cells, plus an `event` column for the chart)."""
    r_q, r_act = shift(df, ref), shift_actual(df, ref)
    prob = 1 - calib.prob_below(r_q, LEVELS, 0.0)
    g = calib.cells(prob, r_act > 0, calib.bucket_edges(width))
    g.insert(0, "event", f"above {REF_LABELS[ref].lower()}")
    return g


# ---- intraday vs the morning daily forecast (r_close) -----------------------------------------

def vs_daily(df: pd.DataFrame, ref: str = "last") -> pd.DataFrame:
    """r_close rows that have a daily forecast: per time-of-day slot, the intraday model's
    pinball vs the daily model's, on the same actuals. skill = 1 - intraday / daily."""
    if df.empty or "has_daily" not in df or not all(c in df for c in DAILYCOLS):
        return pd.DataFrame()
    d = df[df["has_daily"] == 1]
    if d.empty:
        return pd.DataFrame()
    r_act = shift_actual(d, ref)
    m = pd.DataFrame({"tod_bucket": d["tod_bucket"].to_numpy(),
                      "intraday_pinball": _pinball_rows(shift(d, ref), r_act),
                      "daily_pinball": _pinball_rows(shift(d, ref, DAILYCOLS), r_act)})
    t = m.groupby("tod_bucket").agg(n=("daily_pinball", "size"),
                                    intraday_pinball=("intraday_pinball", "mean"),
                                    daily_pinball=("daily_pinball", "mean")).reset_index()
    t["skill"] = 1 - t["intraday_pinball"] / t["daily_pinball"].where(t["daily_pinball"] > 0)
    t.insert(1, "time", hhmm(t["tod_bucket"].to_numpy() * 900))
    return t


def vs_daily_total(vd: pd.DataFrame) -> float:
    """n-weighted skill over all slots."""
    if vd.empty:
        return np.nan
    i, d = (vd[c] * vd["n"] for c in ("intraday_pinball", "daily_pinball"))
    return float(1 - i.sum() / d.sum()) if d.sum() > 0 else np.nan


# ---- session replay --------------------------------------------------------------------------

def replay_choices(df: pd.DataFrame) -> dict[str, list[str]]:
    """date -> symbols with decision points that day."""
    g = df.groupby("date")["symbol"].unique()
    return {d: sorted(s) for d, s in sorted(g.items())}


def replay(df: pd.DataFrame, date: str, symbol: str) -> pd.DataFrame:
    """Per decision time in $: last, session VWAP so far, the VWAP that followed (actual), the
    predicted P10 / P50 / P90 of it, and for r_close the morning daily band."""
    d = df[(df["date"] == date) & (df["symbol"] == symbol)].sort_values("t_s")
    last = d["last"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "t_s": d["t_s"].to_numpy(), "time": hhmm(d["t_s"].to_numpy()), "last": last,
        "session_vwap": last * np.exp(d["ref_session"].to_numpy(dtype=float)),
        "actual_vwap": last * np.exp(d["actual"].to_numpy(dtype=float)),
        **{p: last * np.exp(d[c].to_numpy(dtype=float))
           for p, c in (("p10", Q10), ("p50", Q50), ("p90", Q90))}})
    if "has_daily" in d and all(c in d for c in DAILYCOLS):
        ok = (d["has_daily"] == 1).to_numpy()
        for p, c in (("daily_p10", DAILYCOLS[1]), ("daily_p50", DAILYCOLS[9]),
                     ("daily_p90", DAILYCOLS[17])):
            out[p] = np.where(ok, last * np.exp(d[c].to_numpy(dtype=float)), np.nan)
    return out.reset_index(drop=True)


# ---- the report (precomputed) or the same numbers computed from predictions ------------------

def headline(report: dict[str, pd.DataFrame] | None, frames: dict[str, pd.DataFrame]
             ) -> pd.DataFrame:
    """Per target: HEADLINE (n, coverage80, skill vs climatology and momentum, direction vs the
    last price). From report/by_target.parquet (the "tft" rows) when present, else computed from
    `frames` (target -> predictions)."""
    cols = HEADLINE
    bt = (report or {}).get("by_target")
    if bt is not None and {"target", "model"} <= set(bt.columns):
        t = bt[bt["model"] == "tft"]
        if not t.empty:
            return t.reindex(columns=cols).reset_index(drop=True)
    s = summary(frames, "trail")
    return s.reindex(columns=cols) if not s.empty else pd.DataFrame(columns=cols)


# ---- insight lines (In short) ----------------------------------------------------------------

def skill_col(t: pd.DataFrame) -> str:
    """The headline skill column: vs climatology, or vs momentum for runs without it."""
    return "skill_clim" if "skill_clim" in t and t["skill_clim"].notna().any() else "skill_mom"


def _lvl(ok: bool, warn: bool = False) -> str:
    return GOOD if ok else WARN if warn else BAD


def summary_insights(s: pd.DataFrame, ref: str) -> list[Insight]:
    if s.empty:
        return []
    out = []
    for _, r in s.iterrows():
        lab = target_label(r["target"])
        gap = r["coverage80"] - 0.8
        if abs(gap) > COVERAGE_TOL:
            out.append(Insight(WARN, f"{lab}: the P10–P90 band held {r['coverage80']:.0%} of "
                               f"outcomes (target 80%): "
                               f"{'too narrow' if gap < 0 else 'too wide'}."))
    k = skill_col(s)
    base = BASELINE_LABELS[k]
    best = s.loc[s[k].idxmax()]
    worst = s.loc[s[k].idxmin()]
    beats = int((s[k] > 0).sum())
    out.insert(0, Insight(_lvl(beats == len(s), beats > 0),
                          f"The model beats the {base} baseline on {beats} of {len(s)} "
                          f"targets (pinball loss)."))
    if len(s) > 1:
        out.append(Insight(INFO, f"Best vs {base}: {target_label(best['target'])} "
                           f"({best[k]:+.1%}); weakest: "
                           f"{target_label(worst['target'])} ({worst[k]:+.1%})."))
    if k == "skill_clim" and s["skill_mom"].notna().any():
        out.append(Insight(INFO, f"Vs momentum it is {s['skill_mom'].min():+.0%} … "
                           f"{s['skill_mom'].max():+.0%}, but that bar is low: the VWAP ahead "
                           "stays near the last price while the trailing VWAP lags."))
    rw_neg = s[s["skill_rw"] <= 0]
    if len(rw_neg):
        out.append(Insight(BAD, "No better than \"the VWAP ahead equals "
                           f"{REF_LABELS[ref].lower()}\" for: "
                           + ", ".join(target_label(t) for t in rw_neg["target"]) + "."))
    d = s["dir_last"].mean()
    out.append(Insight(_lvl(d > 0.55, d > 0.5), f"The median called which side of the last "
                       f"price the VWAP ahead lands on {d:.1%} of the time (50% = coin flip)."))
    dt = s["dir_trail"].mean()
    if dt > d + 0.05:
        out.append(Insight(INFO, f"Vs the trailing VWAP it is {dt:.0%}, which any forecast "
                           "centred on the last price gets: the trailing VWAP lags the price."))
    return out


def calibration_insights(qc: pd.DataFrame, g: pd.DataFrame, n: int) -> list[Insight]:
    if qc.empty or n == 0:
        return []
    out = []
    worst = qc.loc[qc["gap"].abs().idxmax()]
    big = abs(worst["gap"]) > 0.03
    out.append(Insight(_lvl(not big, abs(worst["gap"]) <= 0.06),
                       f"Largest quantile gap: P{worst['level'] * 100:.0f} had "
                       f"{worst['observed']:.1%} of outcomes below it ({worst['gap'] * 100:+.1f} pts)."))
    lo, hi = qc.iloc[1], qc.iloc[17]
    cov = hi["observed"] - lo["observed"]
    out.append(Insight(_lvl(abs(cov - 0.8) <= COVERAGE_TOL), f"P10–P90 band held {cov:.0%} of "
                       f"outcomes (target 80%)."))
    g = g[g["n"] >= MIN_ROWS]
    if len(g):
        dev = float((g["deviation"].abs() * g["n"]).sum() / g["n"].sum())
        out.append(Insight(_lvl(dev < 0.03, dev < 0.08), f"P(up) said vs came true differs by "
                           f"{dev:.1%} on average across probability buckets."))
    return out


def tod_insights(t: pd.DataFrame) -> list[Insight]:
    if t.empty:
        return []
    t = t[t["n"] >= MIN_ROWS] if (t["n"] >= MIN_ROWS).any() else t
    off = t.loc[(t["coverage80"] - 0.8).abs().idxmax()]
    k = skill_col(t)
    best, worst = t.loc[t[k].idxmax()], t.loc[t[k].idxmin()]
    return [
        Insight(_lvl(abs(off["coverage80"] - 0.8) <= COVERAGE_TOL, abs(off["coverage80"] - 0.8) <= 0.08),
                f"Band coverage is furthest from 80% at {off['time']}: {off['coverage80']:.0%}."),
        Insight(INFO, f"Skill vs {BASELINE_LABELS[k]} is best at {best['time']} "
                      f"({best[k]:+.1%}) and weakest at {worst['time']} ({worst[k]:+.1%})."),
        Insight(INFO, f"Median P10–P90 width goes from {t['width80_med'].iloc[0]:.2%} "
                      f"({t['time'].iloc[0]}) to {t['width80_med'].iloc[-1]:.2%} "
                      f"({t['time'].iloc[-1]})."),
    ]


def replay_insights(r: pd.DataFrame, date: str, symbol: str) -> list[Insight]:
    if r.empty:
        return []
    inside = float(((r["actual_vwap"] >= r["p10"]) & (r["actual_vwap"] <= r["p90"])).mean())
    up = (np.sign(r["p50"] - r["session_vwap"]) == np.sign(r["actual_vwap"] - r["session_vwap"]))
    out = [Insight(_lvl(inside >= 0.7, inside >= 0.5),
                   f"{symbol} on {date}: the outcome landed inside the P10–P90 band at "
                   f"{inside:.0%} of {len(r)} decision times (target 80%)."),
           Insight(INFO, f"The median had the right side of the session VWAP at {up.mean():.0%} "
                         f"of decision times.")]
    return out


def stocks_insights(t: pd.DataFrame) -> list[Insight]:
    if t.empty:
        return []
    t = t[t["n"] >= MIN_ROWS] if (t["n"] >= MIN_ROWS).any() else t
    k = skill_col(t)
    beats = int((t[k] > 0).sum())
    out = [Insight(_lvl(beats >= 0.8 * len(t), beats >= 0.5 * len(t)),
                   f"The model beats {BASELINE_LABELS[k]} on {beats} of {len(t)} stocks.")]
    off = int(((t["coverage80"] - 0.8).abs() > 0.08).sum())
    out.append(Insight(_lvl(off == 0, off <= 0.2 * len(t)),
                       f"{off} stocks have band coverage more than 8 pts from 80%."))
    b, w = t.loc[t[k].idxmax()], t.loc[t[k].idxmin()]
    out.append(Insight(INFO, f"Best: {b['symbol']} ({b[k]:+.1%}); weakest: "
                             f"{w['symbol']} ({w[k]:+.1%})."))
    return out


def vs_daily_insights(vd: pd.DataFrame) -> list[Insight]:
    if vd.empty:
        return [Insight(INFO, "No decision points have a morning daily forecast in this run.")]
    tot = vs_daily_total(vd)
    out = [Insight(_lvl(tot > 0, tot > -0.02),
                   f"Overall the intraday model's whole-day VWAP forecast is {abs(tot):.1%} "
                   f"{'better' if tot > 0 else 'worse'} than the morning daily forecast "
                   f"(pinball loss).")]
    early, late = vd.iloc[0], vd.iloc[-1]
    out.append(Insight(INFO, f"At {early['time']} the skill is {early['skill']:+.1%}; at "
                             f"{late['time']} it is {late['skill']:+.1%}. It should grow as the "
                             f"day reveals more."))
    return out
