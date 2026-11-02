"""Minute bars in the NAS layout: `<root>/month=YYYY-MM/*.parquet`.

Each month folder holds that month's bars only, whatever its file name says (the file is named
after its quarter). Columns: ticker, t (UTC epoch ms, the bar's start), o, h, l, c, v, vw.
Only the regular session is kept: 09:30-16:00 New York time, minute 0 = 09:30.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

OPEN_ET = 570          # 09:30 in minutes after midnight, New York time
NMIN = 390             # minutes in a regular session
COLS = ["ticker", "t", "o", "h", "l", "c", "v", "vw"]
DAY_KEY = 100_000      # stock-day key = ticker code * DAY_KEY + epoch day


def month_dirs(root: str | Path) -> list[Path]:
    """The month folders, oldest first."""
    return sorted(p for p in Path(root).glob("month=*") if p.is_dir())


def et_minutes(t_ms: np.ndarray) -> np.ndarray:
    """Epoch minutes on the New York clock, DST included (one UTC offset per UTC date: the
    regular session never straddles a change, which happens at 02:00 local on a Sunday)."""
    m = np.asarray(t_ms, dtype=np.int64) // 60_000
    day = m // 1440
    u = np.unique(day)
    noon = pd.DatetimeIndex(pd.to_datetime(u * 1440 + 720, unit="m", utc=True))
    local = noon.tz_convert("America/New_York").tz_localize(None)
    off = (local - noon.tz_localize(None)) // pd.Timedelta(minutes=1)
    return m + np.asarray(off, dtype=np.int64)[np.searchsorted(u, day)]


def _f32(tb, name: str) -> np.ndarray:
    col = tb[name].combine_chunks().fill_null(np.nan)
    return col.to_numpy(zero_copy_only=False).astype(np.float32)


def read_month(folder: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """One month's regular-session bars.

    Returns
      daily: one row per stock-day with a bar: tk, day (epoch days, New York date), dv (volume),
             dpv (dollar volume), open (the first bar's open).
      bars:  per bar: sd (row in `daily`), mi (minute 0-389), h, l, c, v, vw (float32).
    """
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet file in {folder}")
    tb = pq.read_table(files, columns=COLS)
    tk = pc.dictionary_encode(tb["ticker"].combine_chunks())
    et = et_minutes(tb["t"].to_numpy())
    mod = et % 1440
    reg = (mod >= OPEN_ET) & (mod < OPEN_ET + NMIN)
    code = tk.indices.to_numpy(zero_copy_only=False)[reg].astype(np.int64)
    key = code * DAY_KEY + et[reg] // 1440
    u, inv = np.unique(key, return_inverse=True)
    mi = (mod[reg] - OPEN_ET).astype(np.int16)
    o, h, lo, c, v, vw = (_f32(tb, k)[reg] for k in ("o", "h", "l", "c", "v", "vw"))
    del tb
    v = np.nan_to_num(v, nan=0.0)
    px = np.where(np.isfinite(vw), vw, c)
    first = np.lexsort((mi, inv))
    first = first[np.r_[0, np.flatnonzero(np.diff(inv[first])) + 1]]
    names = np.asarray(tk.dictionary.to_pylist(), dtype=object)
    daily = pd.DataFrame({
        "tk": names[u // DAY_KEY], "day": (u % DAY_KEY).astype(np.int32),
        "dv": np.bincount(inv, weights=v, minlength=len(u)),
        "dpv": np.bincount(inv, weights=np.where(v > 0, v * px, 0), minlength=len(u)),
        "open": np.where(np.isfinite(o[first]), o[first], c[first]),
    })
    bars = {"sd": inv.astype(np.int32), "mi": mi, "h": h, "l": lo, "c": c, "v": v, "vw": vw}
    return daily, bars


def dense(parts: list[tuple[dict[str, np.ndarray], np.ndarray]], n: int) -> dict[str, np.ndarray]:
    """Dense (n, 390) minute arrays for the chosen stock-days.

    `parts`: per month, its bars and a map from the month's stock-day row to the output row
    (-1 = not wanted). Minutes with no bar get zero volume and the last price carried forward
    (NaN before the first trade); a missing VWAP / high / low falls back to that price.
    """
    V = np.zeros((n, NMIN), np.float32)
    C, VW, H, L = (np.full((n, NMIN), np.nan, np.float32) for _ in range(4))
    for bars, rows in parts:
        r = rows[bars["sd"]]
        k = r >= 0
        r, mi = r[k], bars["mi"][k]
        V[r, mi] = bars["v"][k]
        for D, name in ((C, "c"), (VW, "vw"), (H, "h"), (L, "l")):
            D[r, mi] = bars[name][k]
    idx = np.where(np.isnan(C), 0, np.arange(NMIN))
    np.maximum.accumulate(idx, axis=1, out=idx)
    Cf = np.take_along_axis(C, idx, 1)
    del C, idx
    for D in (VW, H, L):
        np.copyto(D, Cf, where=np.isnan(D))
    return {"V": V, "VW": VW, "C": Cf, "H": H, "L": L}
