"""intraday-ft-report: scores of an intraday-ft run against the realised targets (docs/intraday-features.md §7).

Everything is scored on the test months only, joined to the targets the build computed from the bars
after each entry. Each score comes with two baselines scored on the same rows:
- climatology = the training span's quantiles / event rates per 5-minute bucket x sigma tercile x
  liquidity tercile (terciles cut on the training span);
- persistence = the realised pocket of the 15 minutes up to the decision, repeated (a point forecast:
  every quantile = it; for events, the training rate given the sign of the past-15-min net).
Skill = 1 - model loss / baseline loss. CIs = day bootstrap (whole test days resampled, 500 draws).

    intraday-ft-report runs/intraday_ft/<run>

Writes report.json and report.md in the run directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import recal as Rc
from . import targets as Tg
from . import windows as W

TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
N_BOOT = 500
PERS = ("bulge", "sag", "mass_up", "net")   # pers_* columns that have a target twin in every bin
EVENTS = {"p_mass_up60": ("mass_up60", 0.5), "p_bulge60": ("bulge60", 0.5), "p_sag60": ("sag60", 0.5)}
SHAPE = ("cm_up", "cm_dn", "bulge", "sag", "rvol")   # the pocket-shape targets the go rule reads (Q41)
ZERO_MASS = ("bulge", "sag")   # exactly 0 in 22-31% of bins (no volume on that side of the entry)
DUST = 0.01   # sigma; a ZERO_MASS quantile below it is read as 0 everywhere in the report (the quantile
              # loss can't sit exactly on the point mass; picked on fold B, checked on fold A)
SLICES = ("minute_bucket", "cap_band", "live_set", "delisted_later", "poll_tier")
MIN_BUCKETS = ((0, 15), (15, 60), (60, 120), (120, 240), (240, 330), (330, 390))
TOP = 20


# ---------------------------------------------------------------- baselines

def _cells(minute, sigma, log_dvol, cuts):
    return (np.asarray(minute) // 5) * 9 + np.digitize(sigma, cuts["sigma"]) * 3 + np.digitize(log_dvol, cuts["dvol"])


def climatology(months, root, targets, n_max=2_000_000, seed=0):
    """Training-span quantiles (TAUS) and means per cell, for each target column of dec.parquet; the
    tercile cuts; the event rates given the sign of the past-15-min net (the persistence baseline)."""
    parts = []
    for m in months:
        mo = W.open_month(m, root)
        r = mo.rows
        d = mo.dec[["row", "i", *[t for t in targets if t in mo.dec.columns], "pers_net"]].copy()
        d["sigma"], d["log_dvol"] = r.sig.to_numpy()[d.row], r.log_dvol.to_numpy()[d.row]
        parts.append(d.drop(columns="row"))
    d = pd.concat(parts, ignore_index=True)
    if len(d) > n_max:
        d = d.sample(n_max, random_state=seed)
    cuts = {"sigma": np.nanquantile(d.sigma, [1 / 3, 2 / 3]), "dvol": np.nanquantile(d.log_dvol, [1 / 3, 2 / 3])}
    d["cell"] = _cells(d.i, d.sigma, d.log_dvol, cuts)
    cols = [t for t in targets if t in d.columns]
    q = d.groupby("cell")[cols].quantile(list(TAUS)).unstack(level=-1)   # cell x (target, tau)
    allq = d[cols].quantile(list(TAUS))
    rates = {}
    for ev, (t, thr) in EVENTS.items():
        y = (d[t] > thr).astype(np.float64).where(d[t].notna())   # float: a bool column with NaN is object
        rates[ev] = {"cell": y.groupby(d.cell).mean(), "all": float(y.mean()),
                     "pers": y.groupby(np.sign(d.pers_net.fillna(0))).mean()}
    return {"cuts": cuts, "q": q, "allq": allq, "rates": rates}


def write_clim(fd, clim):
    """fold_<k>/clim.parquet: the climatology quantiles per cell (cell -1 = every cell) and target,
    with the tercile cuts; clim_rates.parquet: the event rates per cell. So a reader (the dashboard)
    can score skill vs climatology on any subset of the rows; cell = _cells()."""
    q = clim["q"].stack(level=0, future_stack=True).reset_index()
    q.columns = ["cell", "target", *[f"q{round(t * 100)}" for t in TAUS]]
    a = clim["allq"].T.reset_index()
    a.columns = ["target", *[f"q{round(t * 100)}" for t in TAUS]]
    out = pd.concat([q, a.assign(cell=-1)], ignore_index=True)
    out["sigma_cuts"] = [list(map(float, clim["cuts"]["sigma"]))] * len(out)
    out["dvol_cuts"] = [list(map(float, clim["cuts"]["dvol"]))] * len(out)
    out.to_parquet(fd / "clim.parquet", index=False)
    r = [pd.DataFrame({"cell": v["cell"].index.astype(int), "event": ev, "rate": v["cell"].to_numpy(float)})
         for ev, v in clim["rates"].items()]
    r += [pd.DataFrame({"cell": [-1], "event": [ev], "rate": [v["all"]]}) for ev, v in clim["rates"].items()]
    pd.concat(r, ignore_index=True).to_parquet(fd / "clim_rates.parquet", index=False)


def clim_quantiles(clim, cell, t):
    """(rows x 5) climatology quantiles of target t for each row's cell (the overall ones if the cell
    is empty)."""
    q = clim["q"][t].reindex(cell).to_numpy(np.float64) if t in clim["q"].columns.get_level_values(0) else None
    fall = clim["allq"][t].to_numpy(np.float64)
    if q is None:
        return np.broadcast_to(fall, (len(cell), len(TAUS)))
    return np.where(np.isfinite(q), q, fall[None, :])


# ---------------------------------------------------------------- scores

def pinball(q, y):
    """Mean pinball loss over the 5 quantiles, per row (NaN where y is)."""
    e = y[:, None] - q
    tau = np.asarray(TAUS)[None, :]
    return np.maximum(tau * e, (tau - 1) * e).mean(1)


def boot_ratio(day, num, den, seed=0):
    """1 - sum(num) / sum(den) with a day-bootstrap 95% CI. num / den: per-row losses."""
    ok = np.isfinite(num) & np.isfinite(den)
    g = pd.DataFrame({"d": day[ok], "a": num[ok], "b": den[ok]}).groupby("d").sum()
    if g.empty or g.b.sum() <= 0:
        return {"skill": np.nan, "lo": np.nan, "hi": np.nan, "n": int(ok.sum()), "days": len(g)}
    rng = np.random.default_rng(seed)
    A, B = g.a.to_numpy(), g.b.to_numpy()
    k = rng.integers(0, len(g), (N_BOOT, len(g)))
    s = 1 - A[k].sum(1) / B[k].sum(1)
    return {"skill": float(1 - A.sum() / B.sum()), "lo": float(np.quantile(s, 0.025)),
            "hi": float(np.quantile(s, 0.975)), "n": int(ok.sum()), "days": len(g)}


def boot_mean(day, x, seed=0):
    """The mean of per-day means of x, with a day-bootstrap 95% CI."""
    ok = np.isfinite(x)
    g = pd.Series(x[ok]).groupby(day[ok]).mean()
    if g.empty:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "days": 0}
    rng = np.random.default_rng(seed)
    v = g.to_numpy()
    b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
            "days": len(v)}


def rank_ic(f, pred, real, min_n=10):
    """Spearman correlation across the stocks at the same (date, minute), one value per group; the
    groups' day means -> boot_mean."""
    d = pd.DataFrame({"date": f.date, "minute": f.minute, "p": pred, "y": real}).dropna()
    g = d.groupby(["date", "minute"])
    d["rp"], d["ry"] = g.p.rank(), g.y.rank()
    d["n"] = g.p.transform("size")
    d = d[d.n >= min_n]
    g = d.groupby(["date", "minute"])
    a, b = d.rp - g.rp.transform("mean"), d.ry - g.ry.transform("mean")
    num = (a * b).groupby([d.date, d.minute]).sum()
    den = np.sqrt((a * a).groupby([d.date, d.minute]).sum() * (b * b).groupby([d.date, d.minute]).sum())
    ic = (num / den.where(den > 0)).reset_index(name="ic")
    return boot_mean(ic.date.to_numpy(), ic.ic.to_numpy())


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(p) & np.isfinite(y)
    yy = y[ok] > 0.5
    return float(roc_auc_score(yy, p[ok])) if 0 < yy.sum() < len(yy) else np.nan


# ---------------------------------------------------------------- the run

def load_fold(fd):
    f = pd.read_parquet(fd / "intraday_ft_pred.parquet")
    Q = np.load(fd / "quantiles.npy", mmap_mode="r")
    qj = json.loads((fd / "quantiles.json").read_text())
    return f, Q, qj["targets"]


def score_fold(fd, fold, root, seed=0):
    f, Q, qnames = load_fold(fd)
    tr, _, _ = W.split(fold)
    clim = climatology(tr, root, qnames, seed=seed)
    Qr = np.asarray(Q[f.dec_row.to_numpy()], np.float64)
    for j, t in enumerate(qnames):
        if t.rsplit("_b", 1)[0].removesuffix("60") in ZERO_MASS:
            Qr[:, j] = np.where(Qr[:, j] < DUST, 0.0, Qr[:, j])
    day = f.date.to_numpy()
    cell = _cells(f.minute, f.sigma, f.log_dvol, clim["cuts"])
    # the rolling recalibration (recal.py): offsets from the previous WINDOW test sessions' outcomes,
    # written for the dashboard and scored below on the sessions that have a full window
    rc = Rc.fit_fold(Qr, qnames, f, zero_mass=ZERO_MASS, dust=DUST)
    rc.to_parquet(fd / "recal.parquet", index=False)
    write_clim(fd, clim)
    warm = pd.Index(f.date.astype(str)).isin(rc.date[rc.n_days >= Rc.WINDOW].unique())
    out = {"fold": fold, "rows": len(f), "days": int(f.date.nunique()), "pinball": [], "coverage": {},
           "recal": {"window": Rc.WINDOW, "rows": int(warm.sum()), "days": int(f.date[warm].nunique())}}
    # pinball skill per quantile target vs climatology and (bin targets with a twin) persistence
    for j, t in enumerate(qnames):
        y = f.get(f"real_{t}")
        if y is None:
            continue
        y = y.to_numpy(np.float64)
        lm = pinball(Qr[:, j], y)
        lc = pinball(clim_quantiles(clim, cell, t), y)
        row = {"target": t, "clim": boot_ratio(day, lm, lc, seed)}
        base = t.rsplit("_b", 1)[0]
        if base in PERS and f"real_pers_{base}" in f:
            p = f[f"real_pers_{base}"].to_numpy(np.float64)
            row["pers"] = boot_ratio(day, lm, pinball(np.repeat(p[:, None], len(TAUS), 1), y), seed)
        ok = np.isfinite(y)
        out["coverage"][t] = float(((y >= Qr[:, j, 0]) & (y <= Qr[:, j, -1]))[ok].mean()) if ok.any() else np.nan
        if base in SHAPE and ok.any():
            # calibration: the share of outcomes below / at or below each quantile (TAUS should lie between),
            # P10-P90 coverage the same way, and how well each P50 ranks the stocks at the same minute. The
            # two shares differ only on ties: outcomes exactly 0 against a quantile read as 0 (DUST).
            c = clim_quantiles(clim, cell, t)
            q, yy = Qr[ok, j], y[ok]
            row["hits"] = [[float((yy < q[:, k]).mean()), float((yy <= q[:, k]).mean())] for k in range(len(TAUS))]
            row["cover"] = [float(((yy > q[:, 0]) & (yy <= q[:, -1])).mean()),
                            float(((yy >= q[:, 0]) & (yy <= q[:, -1])).mean())]
            row["clim_coverage"] = float(((y >= c[:, 0]) & (y <= c[:, -1]))[ok].mean())
            row["ic"] = rank_ic(f, Qr[:, j, 2], y)
            row["clim_ic"] = rank_ic(f, c[:, 2], y)
            # the same after the recalibration, on the warmed-up sessions (raw skill on those rows too)
            qc = Rc.apply(Qr[:, j], Rc.offsets_for(rc, t, f.date), *Rc.bounds(t))
            w = warm & ok
            q, yy = qc[w], y[w]
            row["cal"] = {"hits": [[float((yy < q[:, k]).mean()), float((yy <= q[:, k]).mean())] for k in range(len(TAUS))],
                          "cover": [float(((yy > q[:, 0]) & (yy <= q[:, -1])).mean()),
                                    float(((yy >= q[:, 0]) & (yy <= q[:, -1])).mean())],
                          "clim": boot_ratio(day[w], pinball(q, yy), lc[w], seed),
                          "raw": boot_ratio(day[w], lm[w], lc[w], seed)}
        out["pinball"].append(row)
    # events: Brier skill, AUC, per overlay variant. The `all` variant is re-read from its stored
    # quantiles with the current cdf_at, so runs scored before a readout fix compare like for like.
    from .tft import cdf_at
    out["events"] = {}
    for ev, (t, thr) in EVENTS.items():
        if t in qnames:
            f[f"{ev}__all"] = 1 - cdf_at(Qr[:, qnames.index(t)], thr)
        y = (f[f"real_{t}"] > thr).astype(float).where(f[f"real_{t}"].notna()).to_numpy()
        rc = clim["rates"][ev]["cell"].reindex(cell).fillna(clim["rates"][ev]["all"]).to_numpy(np.float64)
        rp = (clim["rates"][ev]["pers"].reindex(np.sign(f.real_pers_net.fillna(0)).to_numpy())
              .fillna(clim["rates"][ev]["all"]).to_numpy(np.float64))
        res = {}
        for v in ("none", "sec", "ind", "all"):
            p = f[f"{ev}__{v}"].to_numpy(np.float64)
            res[v] = {"brier_clim": boot_ratio(day, (p - y) ** 2, (rc - y) ** 2, seed),
                      "brier_pers": boot_ratio(day, (p - y) ** 2, (rp - y) ** 2, seed), "auc": auc(p, y)}
        res["auc_clim"], res["auc_pers"] = auc(rc, y), auc(rp, y)
        out["events"][ev] = res
    # direction: rank IC of net60 P50 (each overlay) and of the persistence net; per bin from Q
    real = f.real_net60.to_numpy(np.float64)
    out["rank_ic"] = {f"net60__{v}": rank_ic(f, f[f"net60_p50__{v}"].to_numpy(), real) for v in ("none", "all")}
    out["rank_ic"]["net60__pers"] = rank_ic(f, f.real_pers_net.to_numpy(), real)
    for k in range(1, Tg.NB + 1):
        t = f"net_b{k}"
        if t in qnames:
            out["rank_ic"][t] = rank_ic(f, Qr[:, qnames.index(t), 2], f[f"real_{t}"].to_numpy(np.float64))
    # the race (P1's sanity anchor: AUC ~0.51) and t50 rank IC
    race = f.real_race.to_numpy()
    out["race_auc_up"] = auc(f.p_race_up__all.to_numpy(np.float64), (race == 0).astype(float))
    side = np.isin(race, (0, 1))   # the side alone: up vs down on the rows where one level is hit first
    pu, pd_ = f.p_race_up__all.to_numpy(np.float64)[side], f.p_race_dn__all.to_numpy(np.float64)[side]
    out["race_auc_side"] = auc(pu / np.maximum(pu + pd_, 1e-12), (race[side] == 0).astype(float))
    out["t50_up_ic"] = rank_ic(f, f.t50_up_p50__all.to_numpy(), f.real_t50_up.to_numpy(np.float64))
    burst = (f.real_t50_up <= 10).astype(float).where(f.real_t50_up.notna()).to_numpy()
    out["burst_auc_up"] = auc(f.p_burst_up__all.to_numpy(np.float64), burst)
    # top / bottom 20 by predicted net60 at each (date, minute): realised net60 spread
    fd_ = f.assign(p=f.net60_p50__all).dropna(subset=["p", "real_net60"])   # rank on the kept rows only
    g = fd_.groupby(["date", "minute"])
    rk = g.p.rank(ascending=False, method="first")
    n = g.p.transform("size")
    top = fd_.real_net60[rk <= TOP].groupby(fd_.date[rk <= TOP]).mean()
    bot = fd_.real_net60[(n - rk) < TOP].groupby(fd_.date[(n - rk) < TOP]).mean()
    spread = (top - bot).dropna()
    out["top_bottom_net60"] = boot_mean(spread.index.to_numpy(), spread.to_numpy())
    # slices of the headline numbers
    f["minute_bucket"] = pd.cut(f.minute, [b[0] for b in MIN_BUCKETS] + [390], right=False,
                                labels=[f"{a}-{b}" for a, b in MIN_BUCKETS]).astype(str)
    yv = (f.real_mass_up60 > 0.5).astype(float).where(f.real_mass_up60.notna()).to_numpy()
    rc = clim["rates"]["p_mass_up60"]["cell"].reindex(cell).fillna(clim["rates"]["p_mass_up60"]["all"]).to_numpy(np.float64)
    pm = f.p_mass_up60__all.to_numpy(np.float64)
    out["slices"] = {}
    for s in SLICES:
        rows = []
        for val, idx in f.groupby(s).indices.items():
            rows.append({"value": str(val), "n": len(idx),
                         "brier_clim": boot_ratio(day[idx], (pm[idx] - yv[idx]) ** 2, (rc[idx] - yv[idx]) ** 2, seed),
                         "net60_ic": rank_ic(f.iloc[idx], f.net60_p50__all.to_numpy()[idx], real[idx], min_n=5)})
        out["slices"][s] = rows
    return out


def build(run_dir, root=D.CACHE, seed=0):
    run_dir = Path(run_dir)
    rep = {"run": run_dir.name, "folds": []}
    for fd in sorted(run_dir.glob("fold_*")):
        if (fd / "intraday_ft_pred.parquet").is_file():
            rep["folds"].append(score_fold(fd, fd.name.split("_", 1)[1], root, seed))
    (run_dir / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    (run_dir / "report.md").write_text(to_markdown(rep))
    return rep


def _ci(d, k="skill"):
    return f"{d[k] * 100:+.1f}% [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]" if np.isfinite(d[k]) else "n/a"


def _ic(d):
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if np.isfinite(d["mean"]) else "n/a"


def _span(ab, scale=1):
    """A below .. at-or-below pair as one number when they agree, else a-b."""
    a, b = (x * scale for x in ab)
    fmt = ".0f" if scale == 100 else ".2f"
    return f"{a:{fmt}}" if f"{a:{fmt}}" == f"{b:{fmt}}" else f"{a:{fmt}}-{b:{fmt}}"


def to_markdown(rep):
    L = [f"# intraday-ft report: {rep['run']}", ""]
    for fo in rep["folds"]:
        L += [f"## Fold {fo['fold']}: {fo['rows']:,} test decisions over {fo['days']} days", "",
              "**Direction** (go rule: CI clear of 0 vs climatology and vs persistence)", "",
              "| measure | value |", "| --- | --- |"]
        for k, v in fo["rank_ic"].items():
            L.append(f"| rank IC {k} | {_ic(v)} |")
        e = fo["events"]["p_mass_up60"]
        L += [f"| P(mass_up60 > 50%) Brier skill vs clim (all) | {_ci(e['all']['brier_clim'])} |",
              f"| P(mass_up60 > 50%) Brier skill vs pers (all) | {_ci(e['all']['brier_pers'])} |",
              (f"| P(mass_up60 > 50%) AUC model / clim / pers | {e['all']['auc']:.3f} / {e['auc_clim']:.3f} / "
               f"{e['auc_pers']:.3f} |"),
              f"| race AUC (up first vs not; mostly activity) | {fo['race_auc_up']:.3f} |",
              f"| race AUC (up vs down, rows where one is hit) | {fo.get('race_auc_side', np.nan):.3f} |",
              f"| top-{TOP} minus bottom-{TOP} realised net60 (sigma) | {_ic(fo['top_bottom_net60'])} |", "",
              ("**Overlays** (Brier skill vs climatology; `all` re-read from the stored quantiles with the "
               "current readout, the others as written at scoring)"), "", "| event | none | sec | ind | all |",
              "| --- | --- | --- | --- | --- |"]
        for ev, r in fo["events"].items():
            L.append(f"| {ev} | " + " | ".join(_ci(r[v]["brier_clim"]) for v in ("none", "sec", "ind", "all")) + " |")
        L += ["", ("**Pocket shape** (go rule, Q41): share of outcomes below each quantile (should be "
                   "0.10 / 0.25 / 0.50 / 0.75 / 0.90), P10-P90 coverage (should be 80%), rank IC of P50 across "
                   f"the stocks at a minute. a-b = below .. at or below: outcomes exactly 0 tie with a quantile "
                   f"under {DUST} sigma (read as 0), and the target lies anywhere in between"),
              "", ("| target | below P10 / P25 / P50 / P75 / P90 | coverage model / clim | rank IC model | "
                   "rank IC clim |"), "| --- | --- | --- | --- | --- |"]
        for p in fo["pinball"]:
            if "hits" in p:
                L.append(f"| {p['target']} | {' / '.join(_span(h) for h in p['hits'])} | "
                         f"{_span(p['cover'], 100)}% / {p['clim_coverage'] * 100:.0f}% | "
                         f"{_ic(p['ic'])} | {_ic(p['clim_ic'])} |")
        rcl = fo.get("recal")
        if rcl:
            L += ["", (f"**Recalibrated** (each session's quantiles + the offsets that the previous {rcl['window']} "
                       f"sessions' outcomes call for; {rcl['days']} sessions with a full window, {rcl['rows']:,} rows)"),
                  "", "| target | below P10 / P25 / P50 / P75 / P90 | coverage | pinball skill vs clim | raw, same rows |",
                  "| --- | --- | --- | --- | --- |"]
            for p in fo["pinball"]:
                if "cal" in p:
                    c = p["cal"]
                    L.append(f"| {p['target']} | {' / '.join(_span(h) for h in c['hits'])} | {_span(c['cover'], 100)}% | "
                             f"{_ci(c['clim'])} | {_ci(c['raw'])} |")
        L += ["", "**Pinball skill** per target", "", "| target | vs climatology | vs persistence | P10-P90 coverage |",
              "| --- | --- | --- | --- |"]
        for p in fo["pinball"]:
            L.append(f"| {p['target']} | {_ci(p['clim'])} | {_ci(p['pers']) if 'pers' in p else ''} | "
                     f"{fo['coverage'].get(p['target'], np.nan) * 100:.0f}% |")
        for s, rows in fo["slices"].items():
            L += ["", f"**Slice: {s}**", "", "| value | n | Brier skill mass_up60 vs clim | net60 rank IC |",
                  "| --- | --- | --- | --- |"]
            L += [f"| {r['value']} | {r['n']:,} | {_ci(r['brier_clim'])} | {_ic(r['net60_ic'])} |" for r in rows]
        L.append("")
    return "\n".join(L)


def card_result(rep):
    """One line for MODEL.md: per fold, the direction numbers the go rule reads."""
    parts = []
    for fo in rep["folds"]:
        e = fo["events"]["p_mass_up60"]["all"]
        sh = {p["target"]: p for p in fo["pinball"] if "hits" in p}
        shape = "; ".join(f"{t} pinball skill {_ci(sh[t]['clim'])}, rank IC {_ic(sh[t]['ic'])} (clim "
                          f"{_ic(sh[t]['clim_ic'])})" for t in ("cm_up_b4", "rvol_b4") if t in sh)
        parts.append(f"fold {fo['fold']}: shape: {shape or 'n/a'}. Direction: net60 rank IC {_ic(fo['rank_ic']['net60__all'])} (persistence "
                     f"{_ic(fo['rank_ic']['net60__pers'])}); P(mass_up60>50%) Brier skill vs clim "
                     f"{_ci(e['brier_clim'])}, vs pers {_ci(e['brier_pers'])}, AUC {e['auc']:.3f}")
    return "; ".join(parts) or "No fold scored."


def main(argv=None):
    p = argparse.ArgumentParser(prog="intraday-ft-report", description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--cache", type=Path, default=D.CACHE)
    a = p.parse_args(argv)
    rep = build(a.run_dir, a.cache)
    print(card_result(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
