"""Phase 2a — export the TFT training panel (panel.parquet + panel.meta.json).

One row per (symbol, day_idx) over a liquid common-stock/ADR universe, with split-adjusted
(daily_bars.split_factor) but otherwise raw, un-normalized values: TFT's GroupNormalizer does the per-symbol scaling at training time.
The Parquet file plus its meta sidecar is the ONLY contract with the downstream ML code.

Leakage rule: every input column is known at the close of day t. The forward VWAPs other
than the target (vwap_nx_10d / vwap_end_week / vwap_nx_qtr) and results_window (whose
pre-earnings buckets come from an upcoming filing date) are deliberately not exported.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pyarrow as pa

from ..common import config
from ..news import db as news_db
from ..utils import trading_calendar as tc
from . import db

SCHEMA_VERSION = 1

_DEFAULTS = {
    "issue_types": ("CS", "DR"),
    "min_bars": 120,
    "min_median_vwap": 1.0,
    "min_median_volume": 50_000.0,
}


def default_out_path() -> Path:
    return config.out_root() / "tft" / "panel.parquet"


def _calendar_features(end: date, calendar: str) -> pa.Table:
    """Known-future calendar features per session, keyed by the global day_idx."""
    # Extend past the last bar so the newest session still has a "next session".
    rows = tc.trading_calendar(end=end + timedelta(days=30), calendar=calendar)
    cols: dict[str, list] = {k: [] for k in (
        "date", "day_idx", "day_of_week", "month", "is_month_end", "is_quarter_end",
        "sessions_gap_next",
    )}
    for (d, idx), nxt in pairwise(rows):
        nd = nxt[0]
        cols["date"].append(d)
        cols["day_idx"].append(idx)
        cols["day_of_week"].append(d.isoweekday())
        cols["month"].append(d.month)
        cols["is_month_end"].append(int(nd.month != d.month))
        cols["is_quarter_end"].append(int((nd.month - 1) // 3 != (d.month - 1) // 3 or nd.year != d.year))
        cols["sessions_gap_next"].append((nd - d).days)
    return pa.table({
        "date": pa.array(cols["date"], pa.date32()),
        "day_idx": pa.array(cols["day_idx"], pa.int64()),
        **{k: pa.array(v, pa.int16()) for k, v in cols.items() if k not in ("date", "day_idx")},
    })


def _next_session(cal: pa.Table, after: date | None) -> dict | None:
    """Known calendar features of the first session after `after`: the decoder row a frozen
    model needs to forecast the day after the panel ends."""
    if after is None:
        return None
    nxt = next((r for r in cal.to_pylist() if r["date"] > after), None)
    return {**nxt, "date": nxt["date"].isoformat()} if nxt else None


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.repo_root(),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


_PANEL_SQL = """
CREATE OR REPLACE TEMP TABLE _panel AS
WITH adj AS (
    SELECT symbol, date, day_idx, transactions, vwap_nx_1d, vwap_pct_prev_day, is_10k, is_10q,
        open * f AS open, high * f AS high, low * f AS low, close * f AS close,
        vwap * f AS vwap, volume / f AS volume
    FROM (SELECT *, COALESCE(split_factor, 1.0) AS f FROM daily_bars WHERE day_idx IS NOT NULL)
),
bars AS (
    SELECT a.*,
        AVG(a.volume) OVER (PARTITION BY a.symbol ORDER BY a.day_idx
                            RANGE BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_ma20
    FROM adj a
),
universe AS (
    SELECT b.symbol
    FROM bars b JOIN symbols s ON s.symbol = b.symbol
    WHERE list_contains(?::VARCHAR[], s.issue_type) AND b.vwap IS NOT NULL
    GROUP BY b.symbol
    HAVING count(*) >= ? AND median(b.vwap) >= ? AND median(b.volume) >= ?
),
ind AS (
    SELECT symbol, min(industry_code) AS industry_code FROM symbol_industry GROUP BY symbol
)
SELECT
    b.vwap_nx_1d                                   AS target_tomorrow_vwap,
    b.vwap_nx_1d / NULLIF(b.vwap, 0) - 1           AS target_return,
    b.symbol,
    COALESCE(i.industry_code, 'unknown')           AS industry_code,
    COALESCE(s.exchange, 'unknown')                AS exchange,
    COALESCE(s.issue_type, 'unknown')              AS issue_type,
    b.day_idx,
    b.date,
    c.day_of_week, c.month, c.is_month_end, c.is_quarter_end, c.sessions_gap_next,
    b.vwap                                         AS today_vwap,
    b.vwap_pct_prev_day,
    (b.high - b.low) / NULLIF(b.vwap, 0)           AS intraday_spread_pct,
    (b.close - b.vwap) / NULLIF(b.vwap, 0)         AS close_position_pct,
    (b.open - b.vwap) / NULLIF(b.vwap, 0)          AS open_position_pct,
    b.volume / NULLIF(b.vol_ma20, 0)               AS volume_velocity,
    b.transactions / NULLIF(b.volume, 0)           AS transactions_per_volume,
    COALESCE(n.mean_score_agg, 0.0)                AS daily_sentiment,
    COALESCE(n.n_articles, 0)::INTEGER             AS sentiment_volume,
    (COALESCE(n.n_articles, 0) > 0)::SMALLINT      AS has_news,
    b.is_10k,
    b.is_10q
FROM bars b
JOIN universe u ON u.symbol = b.symbol
JOIN symbols s ON s.symbol = b.symbol
LEFT JOIN ind i ON i.symbol = b.symbol
LEFT JOIN _cal c ON c.day_idx = b.day_idx
LEFT JOIN {sentiment} n ON n.symbol = b.symbol AND n.date = b.date
WHERE b.vwap IS NOT NULL
  AND (b.vwap_nx_1d IS NOT NULL
       OR (? AND b.day_idx = (SELECT max(day_idx) FROM daily_bars)))
ORDER BY b.symbol, b.day_idx
"""


def export_tft(
    *,
    db_path: Path | None = None,
    news_db_path: Path | None = None,
    out_path: Path | None = None,
    include_inference: bool = False,
    issue_types: tuple[str, ...] = _DEFAULTS["issue_types"],
    min_bars: int = _DEFAULTS["min_bars"],
    min_median_vwap: float = _DEFAULTS["min_median_vwap"],
    min_median_volume: float = _DEFAULTS["min_median_volume"],
    calendar: str = "XNYS",
) -> dict:
    """Write panel.parquet (+ panel.meta.json alongside it). Returns a CLI summary dict.

    include_inference keeps each symbol's bar on the latest session, whose target is NULL,
    so the trained model can predict the next day.
    """
    out = Path(out_path) if out_path is not None else default_out_path()
    news_path = Path(news_db_path) if news_db_path is not None else news_db.db_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    with db.connect(db_path) as con:
        last_date, n_indexed = con.execute(
            "SELECT max(date), count(*) FROM daily_bars WHERE day_idx IS NOT NULL"
        ).fetchone()
        if not n_indexed:
            raise RuntimeError("no bars have a day_idx; run `sourcing-py etrade derive-features` first")

        cal = _calendar_features(last_date, calendar)
        con.register("_cal", cal)
        if news_path.exists():
            con.execute(f"ATTACH '{_sql_str(news_path)}' AS news (READ_ONLY)")
            sentiment = "news.symbol_sentiment_daily"
        else:
            sentiment = (
                "(SELECT NULL::VARCHAR AS symbol, NULL::DATE AS date, "
                "NULL::INTEGER AS n_articles, NULL::DOUBLE AS mean_score_agg WHERE false)"
            )
        try:
            con.execute(
                _PANEL_SQL.format(sentiment=sentiment),
                [list(issue_types), min_bars, min_median_vwap, min_median_volume, include_inference],
            )
            rows, syms, inference_rows, news_rows, di_lo, di_hi, d_lo, d_hi = con.execute(
                "SELECT count(*), count(DISTINCT symbol), "
                "count(*) FILTER (WHERE target_tomorrow_vwap IS NULL), "
                "count(*) FILTER (WHERE has_news = 1), "
                "min(day_idx), max(day_idx), min(date), max(date) FROM _panel"
            ).fetchone()
            tmp = out.with_name(out.name + ".tmp")
            con.execute(f"COPY _panel TO '{_sql_str(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
            os.replace(tmp, out)
        finally:
            con.unregister("_cal")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "rows": rows,
        "symbols": syms,
        "inference_rows": inference_rows,
        "news_rows": news_rows,
        "day_idx_range": [di_lo, di_hi],
        "date_range": [d_lo.isoformat() if d_lo else None, d_hi.isoformat() if d_hi else None],
        "calendar": calendar,
        "next_session": _next_session(cal, d_hi),
        "filters": {
            "issue_types": list(issue_types),
            "min_bars": min_bars,
            "min_median_vwap": min_median_vwap,
            "min_median_volume": min_median_volume,
            "include_inference": include_inference,
        },
    }
    meta_path = out.with_name(out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return {**meta, "out_path": str(out), "meta_path": str(meta_path)}


def _sql_str(p: Path) -> str:
    return str(p).replace("'", "''")
