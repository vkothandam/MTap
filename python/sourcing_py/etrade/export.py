"""Phase 2a — export the TFT training panel (panel.parquet + panel.meta.json).

One row per (symbol, day_idx) over a liquid common-stock/ADR universe, with split-adjusted
(daily_bars.split_factor) but otherwise raw, un-normalized values: TFT's GroupNormalizer does the per-symbol scaling at training time.
The Parquet file plus its meta sidecar is the ONLY contract with the downstream ML code.

Leakage rule: every input column is known at the close of day t. The forward VWAPs other
than the target (vwap_nx_10d / vwap_end_week / vwap_nx_qtr) are deliberately not exported.

The earnings windows come in two forms (see features.py, results_announcements):
  - results_window, days_since_results, days_to_results, days_to_results_is_est: the pre
    windows use the actual date of the next scheduled results release. Companies announce
    those dates weeks ahead, so in practice they are known on day t; where the next release
    is not in the data yet (the latest quarter, the inference row) days_to_results falls
    back to the estimate and days_to_results_is_est = 1.
  - results_window_pit, days_to_results_est: strictly from releases made by day t, the next
    date estimated from the year-ago release.
days_since_results is shared. The day counts are NULL when a symbol has no release yet
(and nothing to estimate from); days_to_results counts to the first release before it.

Schema 4 adds the groups below (all observed on day t unless noted; NULL becomes 0.0 only
after the "known" flags are computed from the raw columns):
  market        mkt_cap_ret, own_sector_cap_ret (the symbol's TRBC sector, else the market),
                sec_50_cap_ret .. sec_63_cap_ret: day t's cap-weighted returns (market_daily)
  macro         ust_* and cpi_* inputs (macro_daily; yields as of the previous session)
  session shape high_position_pct, low_position_pct, gap_open_pct, gap_open_vs_prev_vwap_pct
  after-hours   ah_vwap_pct_rth_vwap, ah_vol_pct_10d, close_pct_rth_vwap, ah_known
  pre-market    pm_vwap_pct_prev_vwap, pm_vol_pct_10d, pm_known: KNOWN-FUTURE, row d holds
                day d's pre-market up to the meta's premarket_cutoff (features.pm_expr)
panel.next.parquet beside the panel holds the inference rows (day_idx of the next session,
pm_* = 0, pm_known = 0, export_id); `etrade export-next` fills them from the live feed.

Schema 2 added results_window_pit and the filing-based day counts. Schema 3 dates the
windows from the 8-K results releases instead of the 10-K/10-Q filings, renames the day
counts and adds results_window, days_to_results and days_to_results_is_est.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..common import config
from ..news import db as news_db
from ..utils import trading_calendar as tc
from . import db, features

SCHEMA_VERSION = 4

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
        results_window, days_since_results, days_to_results, days_to_results_is_est,
        results_window_pit, days_to_results_est,
        open * f AS open, high * f AS high, low * f AS low, close * f AS close,
        vwap * f AS vwap, volume / f AS volume,
        regular_vwap * ef AS rth_vwap, after_hours_vwap * ef AS ah_vwap,
        after_hours_volume / ef AS ah_vol,
        pre_market_tpvwap_{ck} * ef AS pm_tp, pre_market_volume_{ck} / ef AS pm_vol,
        EXISTS (SELECT 1 FROM stock_splits sp WHERE sp.ticker = symbol AND sp.execution_date = date) AS split_today
    FROM (SELECT *, COALESCE(split_factor, 1.0) AS f, COALESCE(ext_split_factor, 1.0) AS ef
          FROM daily_bars WHERE day_idx IS NOT NULL)
),
bars AS (
    SELECT a.*,
        AVG(a.volume) OVER (PARTITION BY a.symbol ORDER BY a.day_idx
                            RANGE BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_ma20,
        AVG(a.volume) OVER (PARTITION BY a.symbol ORDER BY a.day_idx
                            RANGE BETWEEN 10 PRECEDING AND 1 PRECEDING) AS vol_mean10
    FROM adj a
),
mk AS (
    SELECT date, {mk_cols} FROM market_daily GROUP BY date
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
    b.is_10q,
    b.results_window,
    b.days_since_results,
    b.days_to_results,
    b.days_to_results_is_est,
    b.results_window_pit,
    b.days_to_results_est,
    {market_sel},
    {macro_sel},
    COALESCE((b.high - b.vwap) / NULLIF(b.vwap, 0), 0.0)                    AS high_position_pct,
    COALESCE((b.low - b.vwap) / NULLIF(b.vwap, 0), 0.0)                     AS low_position_pct,
    COALESCE(b.open / NULLIF(pv.close, 0) - 1, 0.0)                         AS gap_open_pct,
    COALESCE(b.open / NULLIF(pv.vwap, 0) - 1, 0.0)                          AS gap_open_vs_prev_vwap_pct,
    COALESCE(b.ah_vwap / NULLIF(b.rth_vwap, 0) - 1, 0.0)                    AS ah_vwap_pct_rth_vwap,
    COALESCE(ln(1 + 100 * b.ah_vol / NULLIF(b.vol_mean10, 0)), 0.0)         AS ah_vol_pct_10d,
    COALESCE(b.close / NULLIF(b.rth_vwap, 0) - 1, 0.0)                      AS close_pct_rth_vwap,
    (b.ah_vwap IS NOT NULL AND b.rth_vwap IS NOT NULL)::SMALLINT            AS ah_known,
    {pm_sel}
FROM bars b
JOIN universe u ON u.symbol = b.symbol
JOIN symbols s ON s.symbol = b.symbol
LEFT JOIN bars pv ON pv.symbol = b.symbol AND pv.day_idx = b.day_idx - 1
LEFT JOIN ind i ON i.symbol = b.symbol
LEFT JOIN mk ON mk.date = b.date
LEFT JOIN macro_daily md ON md.date = b.date
LEFT JOIN _cal c ON c.day_idx = b.day_idx
LEFT JOIN {sentiment} n ON n.symbol = b.symbol AND n.date = b.date
WHERE b.vwap IS NOT NULL
  AND (b.vwap_nx_1d IS NOT NULL
       OR (? AND b.day_idx = (SELECT max(day_idx) FROM daily_bars)))
ORDER BY b.symbol, b.day_idx
"""


_MACRO = ("ust_3m_chg_bp", "ust_2y_chg_bp", "ust_10y_chg_bp", "ust_10y_2y_slope", "ust_10y_3m_slope",
          "ust_10y_z252", "cpi_yoy", "core_cpi_yoy", "cpi_yoy_chg", "core_cpi_mom")
_PM_COLS = ("pm_vwap_pct_prev_vwap", "pm_vol_pct_10d", "pm_known")


def _panel_sql(sentiment: str, cutoff: str) -> str:
    """_PANEL_SQL with the schema-4 select lists filled in (SQL braces in the template are
    only the placeholders, so str.format is safe)."""
    sectors = features.SECTORS
    mk_cols = ", ".join(
        [f"max(cap_ret) FILTER (WHERE sector_code = '00') AS mkt"]
        + [f"max(cap_ret) FILTER (WHERE sector_code = '{c}') AS s{c}" for c in sectors])
    own = " ".join(f"WHEN '{c}' THEN mk.s{c}" for c in sectors)
    market_sel = ",\n    ".join(
        ["COALESCE(mk.mkt, 0.0) AS mkt_cap_ret",
         f"COALESCE(CASE left(i.industry_code, 2) {own} END, mk.mkt, 0.0) AS own_sector_cap_ret"]
        + [f"COALESCE(mk.s{c}, mk.mkt, 0.0) AS sec_{c}_cap_ret" for c in sectors])
    macro_sel = ",\n    ".join(f"COALESCE(md.{c}, 0.0) AS {c}" for c in _MACRO)
    pm = features.pm_expr("b.pm_tp", "b.pm_vol", "b.vol_mean10", "pv.vwap",
                          "pv.day_idx IS NOT NULL AND NOT b.split_today")
    pm_sel = ",\n    ".join(f"{pm[c]} AS {c}" for c in _PM_COLS)
    return _PANEL_SQL.format(sentiment=sentiment, ck=features.cutoff_key(cutoff), mk_cols=mk_cols,
                             market_sel=market_sel, macro_sel=macro_sel, pm_sel=pm_sel)


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
    premarket_cutoff: str | None = None,
) -> dict:
    """Write panel.parquet (+ panel.meta.json alongside it). Returns a CLI summary dict.

    include_inference keeps each symbol's bar on the latest session, whose target is NULL,
    so the trained model can predict the next day.
    """
    out = Path(out_path) if out_path is not None else default_out_path()
    news_path = Path(news_db_path) if news_db_path is not None else news_db.db_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    cutoff = premarket_cutoff or features.premarket_cutoff()
    export_id = str(uuid.uuid4())

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
                _panel_sql(sentiment, cutoff),
                [list(issue_types), min_bars, min_median_vwap, min_median_volume, include_inference],
            )
            rows, syms, inference_rows, news_rows, di_lo, di_hi, d_lo, d_hi = con.execute(
                "SELECT count(*), count(DISTINCT symbol), "
                "count(*) FILTER (WHERE target_tomorrow_vwap IS NULL), "
                "count(*) FILTER (WHERE has_news = 1), "
                "min(day_idx), max(day_idx), min(date), max(date) FROM _panel"
            ).fetchone()
            panel_symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM _panel ORDER BY 1").fetchall()]
            # DuckDB locks the file it writes, and SMB shares (the NAS data root) refuse the
            # lock; so write locally, copy beside the target, then rename it into place.
            config.state_root().mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=config.state_root()) as local_dir:
                local = Path(local_dir) / out.name
                con.execute(f"COPY _panel TO '{_sql_str(local)}' (FORMAT parquet, COMPRESSION zstd)")
                tmp = out.with_name(out.name + ".tmp")
                shutil.copyfile(local, tmp)
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
        "export_id": export_id,
        "premarket_cutoff": cutoff,
        "next_session": _next_session(cal, d_hi),
        "filters": {
            "issue_types": list(issue_types),
            "min_bars": min_bars,
            "min_median_vwap": min_median_vwap,
            "min_median_volume": min_median_volume,
            "include_inference": include_inference,
        },
    }
    next_path = None
    if meta["next_session"]:
        next_path = next_panel_path(out)
        _write_parquet_atomic(_next_table(panel_symbols, meta["next_session"]["day_idx"], export_id), next_path)
    meta_path = out.with_name(out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return {**meta, "out_path": str(out), "meta_path": str(meta_path),
            "next_path": str(next_path) if next_path else None}


def next_panel_path(panel: Path) -> Path:
    """panel.parquet -> panel.next.parquet"""
    return panel.with_name(panel.stem + ".next.parquet")


def _next_table(symbols: list[str], day_idx: int, export_id: str, **cols) -> pa.Table:
    """The inference rows: one per symbol at the next session, pre-market columns masked
    (0, pm_known = 0) unless `cols` supplies them."""
    n = len(symbols)
    return pa.table({
        "symbol": pa.array(symbols, pa.string()),
        "day_idx": pa.array([day_idx] * n, pa.int64()),
        "pm_vwap_pct_prev_vwap": pa.array(cols.get("pm_vwap_pct_prev_vwap", [0.0] * n), pa.float64()),
        "pm_vol_pct_10d": pa.array(cols.get("pm_vol_pct_10d", [0.0] * n), pa.float64()),
        "pm_known": pa.array(cols.get("pm_known", [0] * n), pa.int16()),
        "export_id": pa.array([export_id] * n, pa.string()),
    })


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    # same pattern as the panel: write beside the target, then rename into place
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def export_next(
    *,
    date_: date | None = None,
    panel: Path | None = None,
    db_path: Path | None = None,
    live_path: Path | None = None,
) -> dict:
    """Fill panel.next.parquet's pre-market columns from out/premarket_live/premarket-D.jsonl.

    Each live record carries the typical-price VWAP and volume up to the cutoff plus the
    source's own 10-day mean daily volume (`vol_10d_mean_source`): ChartIQ volumes are not
    Massive's, so the ratio is taken within the source. The previous session's VWAP and the
    split check come from DuckDB; the values use the same expressions as the history
    (features.pm_expr). Refuses a file whose cutoff differs from the panel's. Symbols with no
    record stay masked. Returns a summary dict."""
    out = Path(panel) if panel is not None else default_out_path()
    meta_path = out.with_name(out.stem + ".meta.json")
    nxt_path = next_panel_path(out)
    if not meta_path.exists() or not nxt_path.exists():
        raise RuntimeError(f"no {meta_path.name} / {nxt_path.name} beside {out}: run export-tft "
                           "--include-inference first")
    meta = json.loads(meta_path.read_text())
    ns = meta.get("next_session")
    if not ns:
        raise RuntimeError("the panel meta has no next_session")
    d = date_ or date.fromisoformat(ns["date"])
    if d.isoformat() != ns["date"]:
        raise RuntimeError(f"the panel is for the session {ns['date']}, not {d}: re-run export-tft")
    src = Path(live_path) if live_path is not None else config.out_root() / "premarket_live" / f"premarket-{d}.jsonl"
    if not src.exists():
        raise RuntimeError(f"no live pre-market file {src}")
    recs: dict[str, dict] = {}
    for line in src.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["symbol"]] = r
    cutoff = meta.get("premarket_cutoff") or features.DEFAULT_CUTOFF
    bad = {r.get("cutoff") for r in recs.values() if r.get("cutoff") != cutoff}
    if bad:
        raise RuntimeError(f"{src.name} has cutoff {sorted(map(str, bad))}; the panel was exported with {cutoff}")

    nxt = pq.read_table(nxt_path)
    symbols = nxt.column("symbol").to_pylist()
    export_id = nxt.column("export_id")[0].as_py() if nxt.num_rows else meta["export_id"]
    if export_id != meta["export_id"]:
        raise RuntimeError("panel.next.parquet and panel.meta.json come from different exports")
    day_idx = ns["day_idx"]
    live = pa.table({
        "symbol": list(recs),
        "tp": pa.array([recs[s].get("pm_tpvwap") for s in recs], pa.float64()),
        "vol": pa.array([recs[s].get("pm_volume") for s in recs], pa.float64()),
        "mean": pa.array([recs[s].get("vol_10d_mean_source") for s in recs], pa.float64()),
    })
    pm = features.pm_expr("l.tp", "l.vol", "l.mean", "pv.vwap",
                          "pv.day_idx IS NOT NULL AND NOT EXISTS (SELECT 1 FROM stock_splits sp "
                          f"WHERE sp.ticker = l.symbol AND sp.execution_date = DATE '{d.isoformat()}')")
    with db.connect(db_path) as con:
        con.register("_live", live)
        con.register("_nsyms", pa.table({"symbol": symbols}))
        try:
            rows = con.execute(
                f"SELECT n.symbol, {pm['pm_vwap_pct_prev_vwap']}, {pm['pm_vol_pct_10d']}, {pm['pm_known']} "
                "FROM _nsyms n LEFT JOIN _live l ON l.symbol = n.symbol "
                "LEFT JOIN (SELECT symbol, day_idx, vwap * COALESCE(split_factor, 1.0) AS vwap FROM daily_bars) pv "
                f"  ON pv.symbol = n.symbol AND pv.day_idx = {int(day_idx)} - 1").fetchall()
        finally:
            con.unregister("_live")
            con.unregister("_nsyms")
    by = {r[0]: r for r in rows}
    ordered = [by[s] for s in symbols]
    table = _next_table(
        symbols, day_idx, export_id,
        pm_vwap_pct_prev_vwap=[r[1] for r in ordered], pm_vol_pct_10d=[r[2] for r in ordered],
        pm_known=[r[3] for r in ordered])
    _write_parquet_atomic(table, nxt_path)
    known = sum(r[3] for r in ordered)
    return {"date": d.isoformat(), "cutoff": cutoff, "symbols": len(symbols), "live_records": len(recs),
            "pm_known": known, "export_id": export_id, "next_path": str(nxt_path)}


def _sql_str(p: Path) -> str:
    return str(p).replace("'", "''")
