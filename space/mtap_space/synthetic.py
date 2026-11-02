"""Synthetic minute bars in the NAS layout, for tests (here and in the dashboard)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def sessions(start="2025-09-15", end="2025-11-26"):
    return [d for d in pd.bdate_range(start, end)]


def utc_ms(day: pd.Timestamp, et_minute: int) -> int:
    """Epoch ms of `et_minute` minutes after midnight New York time on `day`."""
    local = pd.Timestamp(day.date()).tz_localize("America/New_York") \
        + pd.Timedelta(minutes=et_minute)
    return int(local.tz_convert("UTC").value // 1_000_000)


def synthetic_bars(days, tickers=TICKERS, seed=0, drop=0.05):
    """Minute bars in the NAS columns. Prices random-walk; each stock trades 1M–3M shares a
    day; ~5% of regular minutes have no bar; a few pre/after-hours bars are added."""
    rng = np.random.default_rng(seed)
    out = []
    for k, tk in enumerate(tickers):
        px = 20.0 + 10 * k
        for day in days:
            m = np.r_[np.arange(570, 960), [480, 540, 960, 1000]]  # + 08:00, 09:00, 16:00, 16:40
            keep = (rng.random(len(m)) > drop) | (m == 570) | (m >= 960)
            m = m[keep]
            ret = rng.normal(0, 0.0012, len(m))
            c = px * np.exp(np.cumsum(ret))
            o = np.r_[px, c[:-1]]
            h = np.maximum(o, c) * (1 + rng.uniform(0, 0.001, len(m)))
            lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.001, len(m)))
            v = rng.gamma(2.0, (1e6 + 5e5 * k) / 780, len(m))
            out.append(pd.DataFrame({
                "ticker": tk, "t": [utc_ms(day, int(x)) for x in m], "o": o, "h": h, "l": lo,
                "c": c, "v": v, "vw": (o + c + h + lo) / 4, "n": 10}))
            px = c[-1] * np.exp(rng.normal(0, 0.01))
    return pd.concat(out, ignore_index=True)


def write_bars(root, bars):
    """month=YYYY-MM/ folders, one month each, with a quarter-style file name like the NAS."""
    t = pd.to_datetime(bars.t, unit="ms", utc=True).dt.tz_convert("America/New_York")
    for month, g in bars.groupby(t.dt.strftime("%Y-%m")):
        d = root / f"month={month}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g.reset_index(drop=True), preserve_index=False),
                       d / "bars-2025-07-01_2025-09-30.parquet")
    return root
