"""Phase 1c — derive feature columns on daily_bars.

Phase 1a loaded raw EOD bars; this step computes the calendar-aligned, point-in-time
features the downstream TFT panel needs, and writes them back onto `daily_bars`:

  day_idx        — global NYSE trading-day index (from the trading_calendar dimension).
  is_10k/is_10q  — 1 on the trading day a 10-K/10-Q was filed (the announcement day, t=0),
                   snapping a filing that lands on a weekend/holiday forward to next session.
  results_window — earnings-window class relative to the NEAREST filing (t):
                     Pre_Earnings_Runup  t-10..t-2   Earnings_Eve  t-1   Earnings_Day  t=0
                     Post_Earnings_Reaction t+1..t+3  Post_Earnings_Drift t+4..t+15
                     Normal_Trading otherwise (offsets counted in trading days).
  vwap_nx_1d / vwap_nx_10d / vwap_end_week / vwap_nx_qtr — forward volume-weighted VWAP
                   aggregates (SUM(vwap*volume)/SUM(volume)) over, respectively: the next
                   trading day; the next 10 trading days; the rest of the current ISO week;
                   and up to the next 10-K/10-Q filing day.
  vwap_pct_prev_day — backward-looking vwap % change vs the previous trading day
                   ((vwap - prev_vwap)/prev_vwap, a fraction; NULL on a symbol's first bar).
  split_factor   — price multiplier from `stock_splits` for splits that executed after both
                   the bar's date and its download (`adjusted=true` already applied earlier
                   ones). All VWAP columns above are computed on split-adjusted prices and
                   volumes (price * factor, volume / factor).

The recompute is FULL and idempotent — every derived column is reset and rewritten each
run — because the forward-looking columns for existing rows change as new bars land, so
`derive-features` must be re-run after each `ingest-eod`. All work is set-based SQL in one
transaction (mirrors industry.map_industries); the trading calendar comes from the generic
utils.trading_calendar helper (the source of truth for day_idx).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pyarrow as pa

from ..utils import trading_calendar as tc
from . import db

# The window buckets, as (label, lo_offset, hi_offset) in trading days relative to t=0.
# Kept here for reference; the SQL CASE below is the executable source of truth.
RESULTS_WINDOWS = (
    ("Pre_Earnings_Runup", -10, -2),
    ("Earnings_Eve", -1, -1),
    ("Earnings_Day", 0, 0),
    ("Post_Earnings_Reaction", 1, 3),
    ("Post_Earnings_Drift", 4, 15),
)


def _refresh_trading_calendar(con, calendar: str) -> int:
    """Rebuild the trading_calendar dimension out to the furthest date we need to index
    (latest bar, latest filing, or today) and upsert it. Returns the session count."""
    end_bar, end_filing = con.execute(
        "SELECT (SELECT max(date) FROM daily_bars), (SELECT max(date_filed) FROM sec_filings)"
    ).fetchone()
    end = max(d for d in (end_bar, end_filing, date.today()) if d is not None)
    rows = tc.trading_calendar(end=end, calendar=calendar)
    tbl = pa.table({
        "date": [d for d, _ in rows],
        "day_idx": [i for _, i in rows],
        "calendar": [calendar] * len(rows),
    })
    con.register("_cal", tbl)
    try:
        con.execute(
            "INSERT INTO trading_calendar (date, day_idx, calendar) "
            "SELECT date, day_idx, calendar FROM _cal "
            "ON CONFLICT (date) DO UPDATE SET day_idx = excluded.day_idx, "
            "calendar = excluded.calendar"
        )
    finally:
        con.unregister("_cal")
    return len(rows)


def derive_features(*, db_path=None, calendar: str = "XNYS", symbols: list[str] | None = None) -> dict:
    """Compute and store the Phase 1c derived columns on daily_bars. Optionally scope to a
    subset of symbols (default: all). Returns a CLI summary dict."""
    now = datetime.now(timezone.utc)
    # Optional symbol scope applied to every daily_bars-touching statement. When scoped, the
    # symbol list is registered as a temp table `_scope_syms`; the fragments below no-op when
    # unscoped. `s(alias)` qualifies a join; `where_scope`/`and_scope` filter a bare statement.
    scoped = bool(symbols)
    s = (lambda a: f" AND {a}.symbol IN (SELECT symbol FROM _scope_syms)") if scoped else (lambda a: "")
    where_scope = " WHERE symbol IN (SELECT symbol FROM _scope_syms)" if scoped else ""
    and_scope = " AND symbol IN (SELECT symbol FROM _scope_syms)" if scoped else ""

    with db.connect(db_path) as con:
        if scoped:
            con.register("_scope_syms", pa.table({"symbol": list(symbols)}))
        con.execute("BEGIN TRANSACTION")
        try:
            # 0. Calendar dimension (must precede day_idx assignment).
            calendar_sessions = _refresh_trading_calendar(con, calendar)

            # 1. day_idx on each bar (join the calendar on the real date).
            con.execute(
                f"UPDATE daily_bars b SET day_idx = c.day_idx "
                f"FROM trading_calendar c WHERE b.date = c.date{s('b')}"
            )

            # 1b. split_factor (reset, then the product of pending splits per bar). Future
            #     (announced, not yet executed) splits are ignored.
            con.execute(f"UPDATE daily_bars SET split_factor = 1.0{where_scope}")
            con.execute(
                f"UPDATE daily_bars b SET split_factor = f.factor FROM ("
                f"  SELECT b.symbol, b.date, exp(sum(ln(sp.split_from / sp.split_to))) AS factor"
                f"  FROM daily_bars b JOIN stock_splits sp ON sp.ticker = b.symbol"
                f"    AND sp.execution_date > b.date"
                f"    AND sp.execution_date > COALESCE(CAST(b.ingested_at AS DATE), b.date)"
                f"    AND sp.execution_date <= current_date"
                f"  WHERE true{s('b')}"
                f"  GROUP BY b.symbol, b.date"
                f") f WHERE b.symbol = f.symbol AND b.date = f.date"
            )
            bars_split_adjusted = con.execute(
                f"SELECT count(*) FROM daily_bars WHERE split_factor <> 1.0{and_scope}"
            ).fetchone()[0]
            con.execute(
                "CREATE OR REPLACE TEMP VIEW _adj_bars AS SELECT symbol, date, day_idx, "
                "vwap * COALESCE(split_factor, 1.0) AS vwap, "
                "volume / COALESCE(split_factor, 1.0) AS volume FROM daily_bars"
            )

            # 2. Snapped filing days: the trading day each filing becomes actionable (t=0).
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE _filing_days AS "
                f"SELECT f.symbol, f.form_type, "
                f"  (SELECT min(c.day_idx) FROM trading_calendar c WHERE c.date >= f.date_filed) AS fday "
                f"FROM sec_filings f WHERE f.date_filed IS NOT NULL{s('f')}"
            )
            filings_snapped = con.execute(
                "SELECT count(*) FROM _filing_days WHERE fday IS NOT NULL"
            ).fetchone()[0]

            # 3. 10-K / 10-Q flags (reset, then mark the filing day).
            con.execute(f"UPDATE daily_bars SET is_10k = 0, is_10q = 0{where_scope}")
            con.execute(
                "UPDATE daily_bars b SET is_10k = 1 FROM _filing_days fd "
                "WHERE fd.form_type = '10-K' AND fd.fday IS NOT NULL "
                "AND b.symbol = fd.symbol AND b.day_idx = fd.fday"
            )
            con.execute(
                "UPDATE daily_bars b SET is_10q = 1 FROM _filing_days fd "
                "WHERE fd.form_type = '10-Q' AND fd.fday IS NOT NULL "
                "AND b.symbol = fd.symbol AND b.day_idx = fd.fday"
            )

            # 4. results_window vs the nearest filing (nearest |offset| wins; tie -> upcoming).
            con.execute(
                f"UPDATE daily_bars b SET results_window = sub.rw FROM ("
                f"  SELECT symbol, date, rw FROM ("
                f"    SELECT b.symbol, b.date,"
                f"      CASE"
                f"        WHEN (b.day_idx - af.fday) BETWEEN -10 AND -2 THEN 'Pre_Earnings_Runup'"
                f"        WHEN (b.day_idx - af.fday) = -1 THEN 'Earnings_Eve'"
                f"        WHEN (b.day_idx - af.fday) = 0 THEN 'Earnings_Day'"
                f"        WHEN (b.day_idx - af.fday) BETWEEN 1 AND 3 THEN 'Post_Earnings_Reaction'"
                f"        WHEN (b.day_idx - af.fday) BETWEEN 4 AND 15 THEN 'Post_Earnings_Drift'"
                f"        ELSE 'Normal_Trading' END AS rw,"
                f"      row_number() OVER (PARTITION BY b.symbol, b.date "
                f"        ORDER BY abs(b.day_idx - af.fday) ASC, af.fday DESC) AS rk"
                f"    FROM daily_bars b"
                f"    JOIN (SELECT DISTINCT symbol, fday FROM _filing_days WHERE fday IS NOT NULL) af"
                f"      ON af.symbol = b.symbol"
                f"    WHERE b.day_idx IS NOT NULL{s('b')}"
                f"  ) WHERE rk = 1"
                f") sub WHERE b.symbol = sub.symbol AND b.date = sub.date"
            )
            # Bars with no filing for their symbol (or otherwise unclassified) -> Normal_Trading.
            con.execute(
                f"UPDATE daily_bars SET results_window = 'Normal_Trading' "
                f"WHERE results_window IS NULL{and_scope}"
            )

            # 5. Forward VWAPs over fixed trading-day windows (RANGE on day_idx is robust to
            #    missing bars; end-of-week partitions by ISO year+week), plus the backward
            #    prev-day % change (LAG over day_idx). Overwrites all scoped rows (NULL where
            #    the forward frame is empty / no prior bar).
            con.execute(
                f"UPDATE daily_bars b SET vwap_nx_1d = v.nx1, vwap_nx_10d = v.nx10, "
                f"  vwap_end_week = v.wk, vwap_pct_prev_day = v.pct FROM ("
                f"  SELECT symbol, date,"
                f"    SUM(vwap*volume) OVER nx1  / NULLIF(SUM(volume) OVER nx1, 0)  AS nx1,"
                f"    SUM(vwap*volume) OVER nx10 / NULLIF(SUM(volume) OVER nx10, 0) AS nx10,"
                f"    SUM(vwap*volume) OVER wk   / NULLIF(SUM(volume) OVER wk, 0)   AS wk,"
                f"    (vwap - LAG(vwap) OVER pv) / NULLIF(LAG(vwap) OVER pv, 0)     AS pct"
                f"  FROM _adj_bars WHERE day_idx IS NOT NULL{and_scope}"
                f"  WINDOW"
                f"    nx1  AS (PARTITION BY symbol ORDER BY day_idx RANGE BETWEEN 1 FOLLOWING AND 1 FOLLOWING),"
                f"    nx10 AS (PARTITION BY symbol ORDER BY day_idx RANGE BETWEEN 1 FOLLOWING AND 10 FOLLOWING),"
                f"    wk   AS (PARTITION BY symbol, extract('isoyear' FROM date), extract('week' FROM date)"
                f"             ORDER BY day_idx ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING),"
                f"    pv   AS (PARTITION BY symbol ORDER BY day_idx)"
                f") v WHERE b.symbol = v.symbol AND b.date = v.date"
            )

            # 6. vwap_nx_qtr: current day .. next filing day (inclusive). Reset first, since
            #    rows with no future filing must end up NULL, not carry a stale value.
            con.execute(f"UPDATE daily_bars SET vwap_nx_qtr = NULL{where_scope}")
            con.execute(
                f"UPDATE daily_bars b SET vwap_nx_qtr = q.v FROM ("
                f"  WITH nf AS ("
                f"    SELECT b.symbol, b.date, b.day_idx,"
                f"      (SELECT min(fd.fday) FROM _filing_days fd"
                f"        WHERE fd.symbol = b.symbol AND fd.fday IS NOT NULL AND fd.fday > b.day_idx) AS next_fday"
                f"    FROM daily_bars b WHERE b.day_idx IS NOT NULL{s('b')}"
                f"  )"
                f"  SELECT nf.symbol, nf.date, SUM(bb.vwap*bb.volume)/NULLIF(SUM(bb.volume), 0) AS v"
                f"  FROM nf JOIN _adj_bars bb ON bb.symbol = nf.symbol"
                f"    AND bb.day_idx >= nf.day_idx AND bb.day_idx <= nf.next_fday"
                f"  WHERE nf.next_fday IS NOT NULL"
                f"  GROUP BY nf.symbol, nf.date"
                f") q WHERE b.symbol = q.symbol AND b.date = q.date"
            )

            # 7. Stamp the rows this run touched, and gather the summary (while _scope_syms lives).
            con.execute(f"UPDATE daily_bars SET features_computed_at = ?{where_scope}", [now])
            bars, syms, missing = con.execute(
                f"SELECT count(*), count(DISTINCT symbol), "
                f"count(*) FILTER (WHERE day_idx IS NULL) FROM daily_bars{where_scope}"
            ).fetchone()

            con.execute("DROP TABLE IF EXISTS _filing_days")
            con.execute("DROP VIEW IF EXISTS _adj_bars")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            if scoped:
                con.unregister("_scope_syms")

    return {
        "bars": bars,
        "symbols": syms,
        "calendar_sessions": calendar_sessions,
        "filings_snapped": filings_snapped,
        "bars_missing_day_idx": missing,
        "bars_split_adjusted": bars_split_adjusted,
    }
