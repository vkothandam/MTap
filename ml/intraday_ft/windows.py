"""Day batches over the month frames: folds, standardisation, overlay vocabularies (docs/intraday-features.md §6).

The model runs once per stock-day over the whole session, causally (an LSTM plus attention masked to
earlier minutes), and reads its state at each decision minute i. That equals an encoder over minutes
0 .. i for every decision, without re-reading the day once per decision. A batch = a set of
stock-days with all their kept decisions:
- x (days x 390 x n_seq), standardised, clipped to +-10, NaN -> 0; plus two per-minute columns
  pm_vwap / pm_rvol = the 09:15 cut for minutes 0-14 and the 09:30 cut from minute 15 (Q-L: the
  09:30 cut is not known live before 09:45, so it can't be a day-level static);
- static (days x n_static), cat (days x 4: cap band, sector, industry, symbol; 0 = unknown);
- per decision: its day in the batch, the minute i, the known decoder inputs per bin (bins x 4), the
  bin targets (bins x n_bin, with a mask), the row-level targets and the class targets.
Standardisation = per column median / IQR from the training months only. The loss divides each target
by its training IQR (y_*_scale), so every target weighs about the same.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import targets as Tg

FOLDS = {"B": {"train": ("2023-01", "2025-12"), "test": ("2026-01", "2026-10")},
         "A": {"train": ("2023-01", "2024-12"), "test": ("2025-01", "2025-12")}}
VALID_MONTHS = 2   # early stopping: the last 2 months of the training span
PM_CUT = ("pm_vwap", "pm_rvol")   # per-minute: the 555 cut before minute 15, the 570 cut from 15
PM_SWITCH = 15
STATIC_ALL = ("y_close", "y_post", "gap", "pm_open", "pm_vwap555", "pm_vwap570", "pm_rvol555", "pm_rvol570",
              "spy_gap", "spx_y", "spx_gap", "sigma", "log_dvol", "log_mcap", "beta60", "results", "poll_tier")
STATIC_REALS = tuple(c for c in STATIC_ALL if not c.startswith(PM_CUT))
CAPS = ("unknown", "micro", "small", "mid", "large", "mega")
CATS = ("cap_band", "sector", "industry", "symbol")
KNOWN_DEC = ("bin_minutes", "bin_normal_share", "bin_to_close", "bin_cut")
BIN_REAL = ("bulge", "sag", "mass_up", "cm_up", "cm_dn", "net", "rvol", "occ_up", "occ_dn", "pi_up", "pi_dn")
ROW_REAL = ("t50_up", "t50_dn", "vol_to_up1", "vol_to_dn1", "mass_up60", "bulge60", "sag60", "net60")
ROW_CLASS = {"race": 3, "t_peak_up": Tg.NB, "t_peak_dn": Tg.NB}
MIN_SYMBOL_SESSIONS = 60  # the symbol id is "unknown" below this many training sessions
CLIP = 10.0


def split(fold):
    """(train months, valid months, test months) of a fold."""
    f = FOLDS[fold]
    tr = D.months(*f["train"])
    return tr[:-VALID_MONTHS], tr[-VALID_MONTHS:], D.months(*f["test"])


@dataclass
class Month:
    month: str
    rows: pd.DataFrame
    seq: np.ndarray        # memmap, stock-days x 390 x n_seq, float16
    dec: pd.DataFrame
    seq_cols: list


def open_month(month, root=D.CACHE) -> Month:
    d = Path(root) / "frames" / f"month={month}"
    meta = json.loads((d / "meta.json").read_text())
    return Month(month, pd.read_parquet(d / "rows.parquet"), np.load(d / "seq.npy", mmap_mode="r"),
                 pd.read_parquet(d / "dec.parquet"), meta["seq_cols"])


def _robust(X):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(X, [25, 50, 75], axis=0)
    iqr = q[2] - q[0]
    return (np.nan_to_num(q[1]).astype(np.float32),
            np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0).astype(np.float32))


def _bin_targets(d):
    return np.stack([d[[f"{t}_b{k}" for t in BIN_REAL]].to_numpy(np.float32) for k in range(1, Tg.NB + 1)], 1)


def fit_scaler(months, root=D.CACHE, days=3000, seed=0):
    """Training-span statistics from a sample of stock-days: (median, IQR) per sequence column and per
    static real, and each target's IQR for the loss weights."""
    rng = np.random.default_rng(seed)
    seqs, stats, yb, yr = [], [], [], []
    per = max(1, days // max(len(months), 1))
    seq_cols = None
    for m in months:
        mo = open_month(m, root)
        seq_cols = seq_cols or mo.seq_cols
        take = np.sort(rng.choice(len(mo.rows), min(per, len(mo.rows)), replace=False))
        pos = [mo.seq_cols.index(c) for c in seq_cols]
        seqs.append(np.asarray(mo.seq[take], np.float32)[..., pos].reshape(-1, len(seq_cols)))
        stats.append(mo.rows[list(STATIC_ALL)].to_numpy(np.float32))
        d = mo.dec.sample(min(len(mo.dec), per * 20), random_state=seed)
        yb.append(_bin_targets(d))
        yr.append(d[list(ROW_REAL)].to_numpy(np.float32))
    _, ib = _robust(np.concatenate(yb).reshape(-1, len(BIN_REAL)))
    _, ir = _robust(np.concatenate(yr))
    return {"seq_cols": seq_cols, "seq": _robust(np.concatenate(seqs)), "static": _robust(np.concatenate(stats)),
            "y_bin_scale": ib, "y_row_scale": ir}


def fit_vocab(months, root=D.CACHE):
    """Index maps for the overlay categoricals (0 = unknown), from the training months only; a symbol
    (stock id) needs >= MIN_SYMBOL_SESSIONS training sessions."""
    sec, ind, sym = set(), set(), {}
    for m in months:
        r = open_month(m, root).rows
        sec |= set(r.sector.astype(str))
        ind |= set(r.industry.astype(str))
        for s, c in r.groupby("sid").size().items():
            sym[s] = sym.get(s, 0) + int(c)
    return {"sector": {s: i + 1 for i, s in enumerate(sorted(sec - {"", "None", "nan"}))},
            "industry": {s: i + 1 for i, s in enumerate(sorted(ind - {"", "None", "nan"}))},
            "symbol": {s: i + 1 for i, s in enumerate(sorted(s for s, c in sym.items() if c >= MIN_SYMBOL_SESSIONS))}}


def save_prep(path, sc, voc):
    """The scaler and vocab of a fold, as JSON (needed to score with a saved model)."""
    j = {k: [a.tolist() for a in v] if isinstance(v, tuple) else (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in sc.items()}
    Path(path).write_text(json.dumps({"scaler": j, "vocab": voc}))


def load_prep(path):
    j = json.loads(Path(path).read_text())
    s = j["scaler"]
    sc = {"seq_cols": s["seq_cols"], "seq": tuple(np.asarray(a, np.float32) for a in s["seq"]),
          "static": tuple(np.asarray(a, np.float32) for a in s["static"]),
          "y_bin_scale": np.asarray(s["y_bin_scale"], np.float32), "y_row_scale": np.asarray(s["y_row_scale"], np.float32)}
    return sc, j["vocab"]


def decoder_known(i, share):
    """The known decoder inputs per bin for decisions at minutes i: (decisions x 5 x 4) = bin minutes / 60,
    the bin's share of the stock's normal day volume, minutes to the close at the bin's start / 390, and
    the cut-at-close flag. share: (decisions x 390) normal volume share per minute."""
    i = np.asarray(i)
    out = np.zeros((len(i), Tg.NB, len(KNOWN_DEC)), np.float32)
    c = np.concatenate([np.zeros((len(i), 1)), np.nancumsum(share, 1)], 1)
    rr = np.arange(len(i))
    for k, (lo, hi) in enumerate(Tg.BINS):
        a = np.minimum(i + 1 + lo, 390)
        b = np.minimum(i + 1 + min(hi, 390), 390)
        out[:, k, 0] = (b - a) / 60.0
        out[:, k, 1] = c[rr, b] - c[rr, a]
        out[:, k, 2] = (390 - a) / 390.0
        out[:, k, 3] = (i + 1 + hi > 390) if hi < 390 else 0
    return out


def overlay_codes(rows, voc):
    """(days x 4) int codes: cap band, sector, industry, symbol; 0 = unknown."""
    return np.stack([np.array([CAPS.index(c) if c in CAPS else 0 for c in rows.cap_band]),
                     rows.sector.astype(str).map(voc["sector"]).fillna(0).to_numpy(),
                     rows.industry.astype(str).map(voc["industry"]).fillna(0).to_numpy(),
                     rows.sid.map(voc["symbol"]).fillna(0).to_numpy()], 1).astype(np.int64)


class Days:
    """The stock-days of a list of months, ready for day batches. Targets (float16: ~4 GB for the
    ~55M training decisions of fold B) and keys are held in memory; the sequences stay memory-mapped
    and are read per batch.

    every: keep every n-th sampled decision of a stock-day (the build samples every 5 minutes, so
      every=3 = every 15 minutes); day_frac: keep this share of the stock-days (seeded);
    keep_frame: also keep the decisions' rows (every target column) for the run file (test sets)."""

    def __init__(self, months, sc, voc, root=D.CACHE, every=1, day_frac=1.0, seed=0, keep_frame=False):
        rng = np.random.default_rng(seed)
        self.months = [open_month(m, root) for m in months]
        self.sc, self.voc, self.seq_cols = sc, voc, list(sc["seq_cols"])
        self.col_pos = [np.array([mo.seq_cols.index(c) for c in self.seq_cols]) for mo in self.months]
        self.nshare = self.seq_cols.index("normal_rvol") if "normal_rvol" in self.seq_cols else None
        sm, si = sc["static"]
        units, parts, frames = [], [], []
        self.stat, self.cat, self.pm = [], [], []
        for k, mo in enumerate(self.months):
            r = mo.rows
            keep = rng.random(len(r)) < day_frac
            st = np.nan_to_num(np.clip((r[list(STATIC_ALL)].to_numpy(np.float32) - sm) / si, -CLIP, CLIP))
            ix = {c: j for j, c in enumerate(STATIC_ALL)}
            self.stat.append(st[:, [ix[c] for c in STATIC_REALS]].astype(np.float32))
            self.pm.append(np.stack([st[:, [ix[f"{c}555"] for c in PM_CUT]], st[:, [ix[f"{c}570"] for c in PM_CUT]]],
                                    1).astype(np.float32))   # days x 2 cuts x 2 columns
            self.cat.append(overlay_codes(r, voc))
            d = mo.dec[keep[mo.dec.row.to_numpy()]]
            if every > 1:
                first = d.groupby("row").i.transform("min").to_numpy()
                d = d[((d.i.to_numpy() - first) // 5) % every == 0]
            d = d.sort_values(["row", "i"], kind="stable")
            units += [(k, j) for j in np.flatnonzero(keep)]
            parts.append({"month": np.full(len(d), k, np.int16), "row": d.row.to_numpy(np.int64),
                          "i": d.i.to_numpy(np.int16), "y_bin": _bin_targets(d).astype(np.float16),
                          "y_row": d[list(ROW_REAL)].to_numpy(np.float16),
                          "y_cls": d[list(ROW_CLASS)].fillna(-1).to_numpy(np.int64)})
            if keep_frame:
                frames.append(d.assign(month=k).reset_index(drop=True))
        self.units = np.array(units, np.int64).reshape(-1, 2)
        cat = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
        self.d_month, self.d_row, self.d_i = cat["month"], cat["row"], cat["i"]
        self.y_bin, self.y_row, self.y_cls = cat["y_bin"], cat["y_row"], cat["y_cls"]
        self.frame = pd.concat(frames, ignore_index=True) if keep_frame else None
        key = self.d_month.astype(np.int64) * 10_000_000 + self.d_row
        ukey = self.units[:, 0] * 10_000_000 + self.units[:, 1]
        self.first = np.searchsorted(key, ukey)
        self.last = np.searchsorted(key, ukey, side="right")

    def __len__(self):
        return len(self.units)

    @property
    def n_decisions(self):
        return len(self.d_i)

    def batch(self, ids):
        """A dict of arrays for stock-days `ids` (indices into self.units) and all their decisions."""
        ids = np.asarray(ids)
        med, iqr = self.sc["seq"]
        n = len(self.seq_cols)
        x = np.zeros((len(ids), 390, n + len(PM_CUT)), np.float32)
        stat = np.zeros((len(ids), len(STATIC_REALS)), np.float32)
        cat = np.zeros((len(ids), len(CATS)), np.int64)
        share = np.full((len(ids), 390), 1 / 390, np.float32)
        d_unit, d_rows = [], []
        early = (np.arange(390) < PM_SWITCH)[:, None]
        for b, u in enumerate(ids):
            k, j = self.units[u]
            mo = self.months[k]
            s = np.asarray(mo.seq[j], np.float32)[:, self.col_pos[k]]
            if self.nshare is not None:
                share[b] = s[:, self.nshare]
            x[b, :, :n] = (s - med) / iqr
            x[b, :, n:] = np.where(early, self.pm[k][j, 0][None, :], self.pm[k][j, 1][None, :])
            stat[b], cat[b] = self.stat[k][j], self.cat[k][j]
            rr = np.arange(self.first[u], self.last[u])
            d_rows.append(rr)
            d_unit.append(np.full(len(rr), b))
        d_rows, d_unit = np.concatenate(d_rows), np.concatenate(d_unit)
        i = self.d_i[d_rows].astype(np.int64)
        yb, yr = self.y_bin[d_rows].astype(np.float32), self.y_row[d_rows].astype(np.float32)
        mb, mr = np.isfinite(yb), np.isfinite(yr)
        # Masked targets are 0, not nan_to_num's float max: a huge masked value overflows in the loss
        # (inf x mask 0 = NaN). Non-finite = undefined, e.g. pi_* where the bin's normal volume is 0.
        return {"x": np.nan_to_num(np.clip(x, -CLIP, CLIP)), "static": stat, "cat": cat, "d_unit": d_unit,
                "d_i": i, "x_dec": decoder_known(i, share[d_unit]), "y_bin": np.where(mb, yb, 0.0),
                "m_bin": mb, "y_row": np.where(mr, yr, 0.0), "m_row": mr,
                "y_cls": self.y_cls[d_rows], "d_rows": d_rows}

    def meta(self, d_rows):
        """Keys, slices and the realised targets (real_*) of decisions `d_rows`, for the run file
        (needs keep_frame=True)."""
        d = self.frame.iloc[np.asarray(d_rows)]
        r = pd.concat([self.months[k].rows.iloc[g.row.to_numpy()].set_index(g.index)
                       for k, g in d.groupby("month", sort=False)]).loc[d.index]
        f = pd.DataFrame({"date": [D.date_of(x) for x in r.day], "minute": d.i.to_numpy(), "stock_id": r.sid.to_numpy(),
                          "symbol": r.tk.to_numpy(), "sector": r.sector.to_numpy(), "industry": r.industry.to_numpy(),
                          "cap_band": r.cap_band.to_numpy(), "live_set": r.live.to_numpy(),
                          "delisted_later": r.delisted_later.to_numpy(), "poll_tier": r.poll_tier.to_numpy(),
                          "sigma": r.sig.to_numpy(), "log_dvol": r.log_dvol.to_numpy()})
        real = d.drop(columns=["row", "i", "month"]).add_prefix("real_").reset_index(drop=True)
        return pd.concat([f.reset_index(drop=True), real], axis=1)


def seq_names(sc):
    """The model's sequence columns: the frame columns plus the per-minute pre-market cut."""
    return [*sc["seq_cols"], *PM_CUT]
