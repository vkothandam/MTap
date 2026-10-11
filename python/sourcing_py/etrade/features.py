"""Phase 1c — derive feature columns on daily_bars.

Phase 1a loaded raw EOD bars; this step computes the calendar-aligned, point-in-time
features the downstream TFT panel needs, and writes them back onto `daily_bars`:

  day_idx        — global NYSE trading-day index (from the trading_calendar dimension).
  is_10k/is_10q  — 1 on the trading day a 10-K/10-Q was filed (E*TRADE sec_filings),
                   snapping a filing that lands on a weekend/holiday forward to next session.
  results_announcements (table) — one row per financial-results release at the session
                   that first reacts to it (t = 0), from SEC EDGAR 8-K item 2.02 acceptance
                   times (see _build_announcements for the rules and fallbacks).
  results_window — earnings-window class relative to the releases, in trading days:
                     Pre_Earnings_Runup  t-10..t-2   Earnings_Eve  t-1   Earnings_Day  t=0
                     Post_Earnings_Reaction t+1..t+3  Post_Earnings_Drift t+4..t+15
                     Normal_Trading otherwise.
                   Post windows count from the latest release on or before the day; pre
                   windows from the next scheduled release (days_to_results). A recent
                   release wins over an upcoming one.
  days_since_results — sessions since the latest release on or before the day (0 = t).
  days_to_results — sessions to the next scheduled release on or after the day. Results
                   dates are announced in advance, so this is known in practice; past the
                   last release in the data it is the estimate below
                   (days_to_results_is_est = 1).
  days_to_results_est / results_window_pit — the strictly point-in-time version: the
                   expected next release from the releases made by the day (the year-ago
                   release + 364 days, same weekday; negative once overdue), and the
                   windows from it (overdue up to OVERDUE_MAX sessions counts as Eve). It
                   doesn't use `scheduled`, which later filings decide.
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

from datetime import date, datetime, timedelta, timezone

import pyarrow as pa

from ..utils import trading_calendar as tc
from . import db, edgar

# The window buckets, as (label, lo_offset, hi_offset) in trading days relative to t=0.
# Kept here for reference; the SQL CASE below is the executable source of truth.
RESULTS_WINDOWS = (
    ("Pre_Earnings_Runup", -10, -2),
    ("Earnings_Eve", -1, -1),
    ("Earnings_Day", 0, 0),
    ("Post_Earnings_Reaction", 1, 3),
    ("Post_Earnings_Drift", 4, 15),
)

# days_to_results_est: an expected release date is the anniversary of a release a year
# earlier, 364 days on (52 weeks, so the weekday matches). Anniversaries less than
# ANNIV_MIN_GAP sessions after the latest release are that quarter's, already out, so they
# are skipped. An overdue expected date (passed, nothing released yet) counts as
# Earnings_Eve for up to OVERDUE_MAX sessions.
ANNIV_DAYS = 364
ANNIV_MIN_GAP = 20
OVERDUE_MAX = 10

# A results release belongs to a 10-K/10-Q filed at most REPORT_MAX sessions after it
# (10-Ks can trail the release by two months); a later report is another quarter's.
REPORT_MAX = 50


def _refresh_trading_calendar(con, calendar: str) -> int:
    """Rebuild the trading_calendar dimension out to the furthest date we need to index
    (latest bar, latest filing, or today) and upsert it. Returns the session count."""
    end_bar, end_filing = con.execute(
        "SELECT (SELECT max(date) FROM daily_bars), greatest((SELECT max(date_filed) FROM sec_filings), "
        "(SELECT max(filing_date) FROM sec_edgar_filings))"
    ).fetchone()
    end = max(d for d in (end_bar, end_filing, date.today()) if d is not None)
    if end_filing is not None:  # far enough ahead to index the latest release's anniversary
        end = max(end, min(end_filing + timedelta(days=ANNIV_DAYS + 7), tc.last_session(calendar)))
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


def _build_announcements(con, s, where_scope: str) -> int:
    """Rebuild results_announcements (for the scoped symbols). Returns its row count.

    Sources, best first:
      - 8-K with item 2.02 (EDGAR): t = 0 is the acceptance day's session if accepted
        before the 16:00 close, else the next session. Without a fetched acceptance time,
        the session on/after the filing date.
      - a 10-K/10-Q (EDGAR) that no 2.02 belongs to: its filing date's session.
      - for symbols with nothing in EDGAR (no CIK): E*TRADE's 10-K/10-Q filing dates.
    A 2.02 belongs to the first 10-K/10-Q filed from one session before it to REPORT_MAX
    sessions after it; the last 2.02 of each such group is the quarter's scheduled results
    release, and any earlier ones (preliminary results, updates) are unscheduled. A 2.02
    with no report in that span is a group of its own (the report is missing), and 2.02s
    after the latest 10-K/10-Q form one group.
    """
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE _edgar AS "
        f"SELECT e.*, (SELECT min(c.day_idx) FROM trading_calendar c WHERE c.date >= e.react) AS day_idx "
        f"FROM (SELECT t.symbol, f.form, f.accession, f.accepted_at, "
        f"    (f.form = '8-K') AS is_release, "
        f"    CASE WHEN f.form = '8-K' AND f.accepted_at IS NOT NULL "
        f"      THEN CAST(f.accepted_at AS DATE) + (CAST(f.accepted_at AS TIME) >= TIME '16:00')::INTEGER "
        f"      ELSE f.filing_date END AS react "
        f"  FROM sec_edgar_filings f JOIN sec_edgar_tickers t ON t.cik = f.cik "
        f"  WHERE (f.form IN ('10-K', '10-Q') OR ({edgar.release_sql('f')})){s('t')}"
        f") e WHERE e.react >= (SELECT min(date) FROM trading_calendar)"  # none before the calendar
    )
    con.execute(f"DELETE FROM results_announcements{where_scope}")
    con.execute(
        f"INSERT INTO results_announcements "
        f"WITH rel AS ("
        f"  SELECT *, CASE WHEN report_day <= day_idx + {REPORT_MAX} THEN report_day "
        f"    WHEN report_day IS NOT NULL THEN -day_idx END AS grp "
        f"  FROM (SELECT r.*, (SELECT min(p.day_idx) FROM _edgar p WHERE p.symbol = r.symbol "
        f"    AND NOT p.is_release AND p.day_idx >= r.day_idx - 1) AS report_day "
        f"    FROM _edgar r WHERE r.is_release AND r.day_idx IS NOT NULL)"
        f"), ev AS ("
        f"  SELECT symbol, day_idx, '8-K 2.02' AS source, "
        f"    day_idx = max(day_idx) OVER (PARTITION BY symbol, grp) AS scheduled, "
        f"    accepted_at, accession, 0 AS pri FROM rel "
        f"  UNION ALL "
        f"  SELECT p.symbol, p.day_idx, p.form, true, NULL, p.accession, 1 FROM _edgar p "
        f"  WHERE NOT p.is_release AND p.day_idx IS NOT NULL AND NOT EXISTS ("
        f"    SELECT 1 FROM rel WHERE rel.symbol = p.symbol AND rel.grp = p.day_idx) "
        f"  UNION ALL "
        f"  SELECT DISTINCT f.symbol, (SELECT min(c.day_idx) FROM trading_calendar c "
        f"    WHERE c.date >= f.date_filed), 'etrade ' || f.form_type, true, NULL, NULL, 2 "
        f"  FROM sec_filings f WHERE f.form_type IN ('10-K', '10-Q') "
        f"    AND f.date_filed >= (SELECT min(date) FROM trading_calendar)"
        f"    AND f.symbol NOT IN (SELECT symbol FROM _edgar){s('f')}"
        f") "
        f"SELECT ev.symbol, ev.day_idx, c.date, ev.source, ev.scheduled, ev.accepted_at, "
        f"  CASE WHEN ev.accepted_at IS NULL THEN NULL "
        f"    WHEN c.date > CAST(ev.accepted_at AS DATE) THEN 'after_close' "  # incl. weekends
        f"    WHEN CAST(ev.accepted_at AS TIME) < TIME '09:30' THEN 'before_open' "
        f"    WHEN CAST(ev.accepted_at AS TIME) < TIME '16:00' THEN 'market_hours' "
        f"    ELSE 'after_close' END, "
        f"  ev.accession "
        f"FROM ev JOIN trading_calendar c ON c.day_idx = ev.day_idx "
        f"QUALIFY row_number() OVER (PARTITION BY ev.symbol, ev.day_idx "
        f"  ORDER BY ev.pri, ev.scheduled DESC, ev.accepted_at DESC, ev.source, ev.accession DESC) = 1"
    )
    return con.execute(f"SELECT count(*) FROM results_announcements{where_scope}").fetchone()[0]


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
        # results_announcements is rebuilt inside the transaction, so its row estimate is
        # tiny and DuckDB would plan the ASOF joins in 4b as nested loops (minutes, not
        # seconds, over all bars). Always use the sort-merge ASOF join.
        con.execute("SET asof_loop_join_threshold = 0")
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
            #     (announced, not yet executed) splits are ignored. ingested_at is naive UTC;
            #     compare on the New York date, since a fetch at 00:54 UTC Monday is still
            #     Sunday evening on the exchange and can't include Monday's split.
            con.execute(f"UPDATE daily_bars SET split_factor = 1.0{where_scope}")
            con.execute(
                f"UPDATE daily_bars b SET split_factor = f.factor FROM ("
                f"  SELECT b.symbol, b.date, exp(sum(ln(sp.split_from / sp.split_to))) AS factor"
                f"  FROM daily_bars b JOIN stock_splits sp ON sp.ticker = b.symbol"
                f"    AND sp.execution_date > b.date"
                f"    AND sp.execution_date > COALESCE(CAST(timezone('America/New_York',"
                f"      timezone('UTC', b.ingested_at)) AS DATE), b.date)"
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

            # 4. results_announcements: one row per financial-results release, at the
            #    session that first reacts to it (t = 0).
            announcements = _build_announcements(con, s, where_scope)

            # 4b. Earnings windows. Per bar: sessions since the latest release on or before
            #     it, to the next scheduled release on or after it (exact, from the data), and
            #     to the expected next one (point in time: the year-ago release + 364 days,
            #     from the releases made by then). Each comes from an ASOF join. The point-in-
            #     time estimate anchors on every announcement, not just the scheduled ones:
            #     whether a release was the scheduled one is only known from later filings.
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE _anniv AS SELECT symbol, day_idx, "
                f"  (SELECT min(c.day_idx) FROM trading_calendar c "
                f"    WHERE c.date >= a.date + INTERVAL {ANNIV_DAYS} DAY) AS aday "
                f"FROM results_announcements a WHERE true{s('a')}"
            )
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE _pit_state AS "
                f"SELECT st.symbol, st.day_idx, "
                f"  (SELECT min(p.aday) FROM _anniv p WHERE p.symbol = st.symbol "
                f"    AND p.day_idx <= st.day_idx AND p.aday > st.day_idx + {ANNIV_MIN_GAP}) AS exp_day "
                f"FROM _anniv st"
            )
            con.execute(
                f"UPDATE daily_bars SET days_since_results = NULL, days_to_results = NULL, "
                f"days_to_results_is_est = NULL, days_to_results_est = NULL{where_scope}"
            )
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE _bars_idx AS SELECT symbol, date, day_idx "
                f"FROM daily_bars b WHERE b.day_idx IS NOT NULL{s('b')}"
            )
            con.execute(
                "UPDATE daily_bars b SET days_since_results = x.since FROM ("
                "  SELECT b.symbol, b.date, b.day_idx - a.day_idx AS since FROM _bars_idx b"
                "  ASOF JOIN results_announcements a ON a.symbol = b.symbol AND b.day_idx >= a.day_idx"
                ") x WHERE b.symbol = x.symbol AND b.date = x.date"
            )
            con.execute(
                "UPDATE daily_bars b SET days_to_results = x.till FROM ("
                "  SELECT b.symbol, b.date, a.day_idx - b.day_idx AS till FROM _bars_idx b"
                "  ASOF JOIN (SELECT symbol, day_idx FROM results_announcements WHERE scheduled) a"
                "    ON a.symbol = b.symbol AND b.day_idx <= a.day_idx"
                ") x WHERE b.symbol = x.symbol AND b.date = x.date"
            )
            con.execute(
                "UPDATE daily_bars b SET days_to_results_est = x.till FROM ("
                "  SELECT b.symbol, b.date, st.exp_day - b.day_idx AS till FROM _bars_idx b"
                "  ASOF JOIN _pit_state st ON st.symbol = b.symbol AND b.day_idx >= st.day_idx"
                ") x WHERE b.symbol = x.symbol AND b.date = x.date"
            )
            # Past the last release in the data, the next one isn't known yet: use the estimate.
            con.execute(
                f"UPDATE daily_bars SET days_to_results_is_est = CASE WHEN days_to_results IS NOT NULL THEN 0 "
                f"  WHEN days_to_results_est IS NOT NULL THEN 1 END, "
                f"days_to_results = COALESCE(days_to_results, days_to_results_est){where_scope}"
            )
            for col, till in (("results_window", "days_to_results"), ("results_window_pit", "days_to_results_est")):
                con.execute(
                    f"UPDATE daily_bars SET {col} = CASE"
                    f"  WHEN days_since_results = 0 THEN 'Earnings_Day'"
                    f"  WHEN days_since_results BETWEEN 1 AND 3 THEN 'Post_Earnings_Reaction'"
                    f"  WHEN days_since_results BETWEEN 4 AND 15 THEN 'Post_Earnings_Drift'"
                    f"  WHEN {till} BETWEEN 2 AND 10 THEN 'Pre_Earnings_Runup'"
                    f"  WHEN {till} BETWEEN -{OVERDUE_MAX} AND 1 THEN 'Earnings_Eve'"
                    f"  ELSE 'Normal_Trading' END{where_scope}"
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
            for tmp in ("_edgar", "_anniv", "_pit_state", "_bars_idx"):
                con.execute(f"DROP TABLE IF EXISTS {tmp}")
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
        "announcements": announcements,
        "bars_missing_day_idx": missing,
        "bars_split_adjusted": bars_split_adjusted,
    }
