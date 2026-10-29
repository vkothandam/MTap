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
  ext_split_factor — the same for the pre-market / after-hours columns, keyed on when
                   those were fetched (ext_fetched_at), which can differ from the bar's.
  vol_10d_mean   — mean split-adjusted daily volume of the 10 sessions before the day
                   (past-only; NULL on a symbol's first bar).
  pre_market_pct_prev_close — last pre-market price vs the previous bar's close; and
  after_hours_pct_close — last after-hours price vs the same day's close. Fractions on
                   split-adjusted prices; NULL with no extended-hours trade (or no prior bar).

  market_daily (table) — per session, the cap-weighted close-to-close return of the whole
                   market (sector_code '00') and of each of the 11 TRBC sectors, from the
                   split-adjusted closes of the CS/DR symbols with bars on both days.
  macro_daily (table) — per session: treasury-yield changes/slopes/z-score (the previous
                   session's yields) and CPI inflation (month M from the first session of M+2).
  The per-bar session-shape / after-hours / pre-market inputs are computed at export time
  from the raw columns (export.py); pm_px_expr() / pm_full_expr() below build the pre-market expressions so
  `export-next` computes live rows the same way.

The recompute is FULL and idempotent — every derived column is reset and rewritten each
run — because the forward-looking columns for existing rows change as new bars land, so
`derive-features` must be re-run after each `ingest-eod`. All work is set-based SQL in one
transaction (mirrors industry.map_industries); the trading calendar comes from the generic
utils.trading_calendar helper (the source of truth for day_idx).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa

from ..common import config
from ..massive_ext.extended import CUTOFFS
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
            # 1c. ext_split_factor: the same rule, for the extended-hours columns' own fetch time.
            con.execute(f"UPDATE daily_bars SET ext_split_factor = 1.0{where_scope}")
            con.execute(
                f"UPDATE daily_bars b SET ext_split_factor = f.factor FROM ("
                f"  SELECT b.symbol, b.date, exp(sum(ln(sp.split_from / sp.split_to))) AS factor"
                f"  FROM daily_bars b JOIN stock_splits sp ON sp.ticker = b.symbol"
                f"    AND sp.execution_date > b.date"
                f"    AND sp.execution_date > COALESCE(CAST(timezone('America/New_York',"
                f"      timezone('UTC', b.ext_fetched_at)) AS DATE), b.date)"
                f"    AND sp.execution_date <= current_date"
                f"  WHERE b.ext_fetched_at IS NOT NULL{s('b')}"
                f"  GROUP BY b.symbol, b.date"
                f") f WHERE b.symbol = f.symbol AND b.date = f.date"
            )
            con.execute(
                "CREATE OR REPLACE TEMP VIEW _adj_bars AS SELECT symbol, date, day_idx, "
                "vwap * COALESCE(split_factor, 1.0) AS vwap, "
                "volume / COALESCE(split_factor, 1.0) AS volume FROM daily_bars"
            )

            # 2. Snapped filing days: the trading day each filing becomes actionable (t=0).
            #    10-K/10-Q come from EDGAR (sec_edgar_filings, by filing date, amendments
            #    excluded: 10-K/A and 10-Q/A are other forms) for symbols mapped to a CIK
            #    (sec_edgar_tickers); E*TRADE's sec_filings (2023-08 on) only for symbols with
            #    no mapping. _build_announcements and vwap_nx_qtr read the same table.
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE _filing_days AS "
                f"SELECT u.symbol, u.form_type, "
                f"  (SELECT min(c.day_idx) FROM trading_calendar c WHERE c.date >= u.date_filed) AS fday "
                f"FROM ("
                f"  SELECT t.symbol, f.form AS form_type, f.filing_date AS date_filed "
                f"  FROM sec_edgar_filings f JOIN sec_edgar_tickers t ON t.cik = f.cik "
                f"  WHERE f.form IN ('10-K', '10-Q') AND f.filing_date IS NOT NULL{s('t')} "
                f"  UNION "
                f"  SELECT f.symbol, f.form_type, f.date_filed FROM sec_filings f "
                f"  WHERE f.date_filed IS NOT NULL AND f.form_type IN ('10-K', '10-Q') "
                f"    AND f.symbol NOT IN (SELECT symbol FROM sec_edgar_tickers){s('f')}"
                f") u"
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

            # 5b. Extended hours vs the regular close, both sides split-adjusted (LAG over
            #     day_idx, as vwap_pct_prev_day). NULL where there was no such trade.
            con.execute(
                f"UPDATE daily_bars b SET pre_market_pct_prev_close = x.pre, after_hours_pct_close = x.post FROM ("
                f"  SELECT symbol, date,"
                f"    pre_market_last * ext_split_factor"
                f"      / NULLIF(LAG(close * COALESCE(split_factor, 1.0)) OVER pv, 0) - 1 AS pre,"
                f"    after_hours_price * ext_split_factor"
                f"      / NULLIF(close * COALESCE(split_factor, 1.0), 0) - 1 AS post"
                f"  FROM daily_bars WHERE day_idx IS NOT NULL{and_scope}"
                f"  WINDOW pv AS (PARTITION BY symbol ORDER BY day_idx)"
                f") x WHERE b.symbol = x.symbol AND b.date = x.date"
            )
            bars_extended = con.execute(
                f"SELECT count(*) FROM daily_bars WHERE ext_fetched_at IS NOT NULL{and_scope}"
            ).fetchone()[0]

            # 5b2. vol_10d_mean: mean split-adjusted daily volume of the 10 sessions before d
            #      (past-only; the pre-market volume ratio's denominator, same source as the
            #      Massive daily volume).
            con.execute(f"UPDATE daily_bars SET vol_10d_mean = NULL{where_scope}")
            con.execute(
                f"UPDATE daily_bars b SET vol_10d_mean = v.m FROM ("
                f"  SELECT symbol, date, AVG(volume) OVER (PARTITION BY symbol ORDER BY day_idx"
                f"    RANGE BETWEEN 10 PRECEDING AND 1 PRECEDING) AS m"
                f"  FROM _adj_bars WHERE day_idx IS NOT NULL{and_scope}"
                f") v WHERE b.symbol = v.symbol AND b.date = v.date"
            )

            # 5c. market and sector returns, macro inputs (whole tables, rewritten each run)
            market_rows, market_mode = _build_market_daily(con)
            macro_rows = _build_macro_daily(con)

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
        "bars_extended": bars_extended,
        "market_daily_rows": market_rows,
        "market_weighting": market_mode,
        "macro_daily_rows": macro_rows,
    }


# --- market and sector returns ---------------------------------------------------------

SECTORS = ("50", "51", "52", "53", "54", "55", "56", "57", "59", "60", "63")
RET_CLIP = 0.5


def _table_exists(con, name: str) -> bool:
    return bool(con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0])


def _build_market_daily(con) -> tuple[int, str]:
    """Rewrite market_daily: Σ cap(t-1)·r(t) / Σ cap(t-1) over CS/DR symbols with bars on
    consecutive sessions, r the split-adjusted close-to-close return clipped at ±50%.

    cap(t-1) = market_cap(S) × adjclose(t-1) / adjclose(S), S the latest share_snapshots
    date <= t-1; a symbol without a snapshot falls back to adjclose(t-1) × the latest
    E*TRADE shares outstanding (QTCO), else it is left out. With no share_snapshots table
    at all the returns are equal-weighted (a warning is printed). Returns (rows, mode)."""
    con.execute("DELETE FROM market_daily")
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _mk_px AS "
        "SELECT b.symbol, b.date, b.day_idx, b.close * COALESCE(b.split_factor, 1.0) AS adj, "
        "  LAG(b.day_idx) OVER w AS p_idx, LAG(b.date) OVER w AS p_date, "
        "  LAG(b.close * COALESCE(b.split_factor, 1.0)) OVER w AS p_adj "
        "FROM daily_bars b JOIN symbols s ON s.symbol = b.symbol "
        "WHERE b.day_idx IS NOT NULL AND b.close > 0 AND s.issue_type = 'CS' "
        "WINDOW w AS (PARTITION BY b.symbol ORDER BY b.day_idx)"
    )
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE _mk_ret AS "
        f"SELECT symbol, date, p_date, p_adj, greatest(-{RET_CLIP}, least({RET_CLIP}, adj / p_adj - 1)) AS r "
        f"FROM _mk_px WHERE p_idx = day_idx - 1"
    )
    if _table_exists(con, "share_snapshots"):
        mode = "cap"
        # S's adjusted close: the bar on/after-before S (latest on or before S, else the
        # symbol's first bar when the snapshot predates the data)
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _mk_snap AS "
            "SELECT sn.symbol, sn.snap_date, sn.market_cap, COALESCE(px.adj, fb.adj) AS s_adj "
            "FROM share_snapshots sn "
            "ASOF LEFT JOIN _mk_px px ON px.symbol = sn.symbol AND sn.snap_date >= px.date "
            "LEFT JOIN (SELECT symbol, arg_min(adj, date) AS adj FROM _mk_px GROUP BY symbol) fb "
            "  ON fb.symbol = sn.symbol "
            "WHERE sn.market_cap > 0"
        )
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _mk_shares AS "
            "SELECT symbol, arg_max(value * CASE WHEN thousand_multiplier THEN 1000 ELSE 1 END, fiscal_end) AS shares "
            "FROM fundamentals WHERE line_item = 'QTCO' AND value > 0 GROUP BY symbol"
        )
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _mk_w AS "
            "SELECT r.symbol, r.date, r.r, "
            "  COALESCE(sn.market_cap * r.p_adj / sn.s_adj, fb.shares * r.p_adj) AS w "
            "FROM _mk_ret r "
            "ASOF LEFT JOIN _mk_snap sn ON sn.symbol = r.symbol AND r.p_date >= sn.snap_date "
            "LEFT JOIN _mk_shares fb ON fb.symbol = r.symbol"
        )
    else:
        mode = "equal"
        print("warning: no share_snapshots table; market and sector returns are equal-weighted")
        con.execute("CREATE OR REPLACE TEMP TABLE _mk_w AS SELECT symbol, date, r, 1.0 AS w FROM _mk_ret")
    secs = ", ".join(f"'{c}'" for c in SECTORS)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _mk_sec AS "
        "SELECT symbol, left(min(industry_code), 2) AS sec FROM symbol_industry GROUP BY symbol"
    )
    con.execute(
        "INSERT INTO market_daily "
        "SELECT date, '00', sum(w * r) / sum(w), count(*) FROM _mk_w WHERE w > 0 GROUP BY date "
        "UNION ALL "
        "SELECT m.date, k.sec, sum(m.w * m.r) / sum(m.w), count(*) FROM _mk_w m "
        f"JOIN _mk_sec k ON k.symbol = m.symbol WHERE m.w > 0 AND k.sec IN ({secs}) GROUP BY m.date, k.sec"
    )
    for t in ("_mk_px", "_mk_ret", "_mk_snap", "_mk_shares", "_mk_w", "_mk_sec"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return con.execute("SELECT count(*) FROM market_daily").fetchone()[0], mode


# --- macro inputs ----------------------------------------------------------------------

Z_WINDOW = 252
Z_MIN = 126  # fewest past sessions for a z-score
CPI_CARRY = 2  # months a missing CPI release carries the previous one


def _month_idx(d: date) -> int:
    return d.year * 12 + d.month - 1


def _as_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def macro_series(sessions: list[date], yields: list[tuple], infl: list[tuple], *, lag: int = 1) -> dict:
    """The macro_daily values per session date. Pure function.

    yields: (date, y_3m, y_2y, y_10y) rows in % (any order). The value used at session i is
    the latest yield dated on or before session i-lag (so bond holidays carry forward, and
    the default lag of 1 is the previous session's yield). Changes are vs the previous
    session's used value, in bp; slopes in % points; z252 is the 10y level vs the mean/std of
    the previous Z_WINDOW used values (today excluded; NULL under Z_MIN).
    infl: (month_start, cpi, cpi_core, cpi_yoy_field) rows. Month M is known from the first
    session of month M+2, i.e. on session d the latest month is month(d)-2; a month that was
    never published carries the one before (up to CPI_CARRY months), and core_cpi_mom across
    such a gap is the per-month rate."""
    ys = sorted((_as_date(r[0]), *r[1:4]) for r in yields if r[0] is not None)
    used: list[tuple | None] = []
    j, cur = 0, None
    for i in range(len(sessions)):
        cutoff = sessions[i - lag] if i - lag >= 0 else None
        while cutoff is not None and j < len(ys) and ys[j][0] <= cutoff:
            cur = ys[j]
            j += 1
        used.append(cur if cutoff is not None else None)
    cpi = {_month_idx(_as_date(r[0])): r for r in infl if r[0] is not None}

    def latest(m: int):
        """The newest published month at or before m, at most CPI_CARRY months back (a
        release that never came, e.g. October 2025 in the shutdown, carries the last one)."""
        return next((x for x in range(m, m - CPI_CARRY - 1, -1) if x in cpi), None)

    def level(m: int, k: int):  # k: 1 = cpi, 2 = core
        r = cpi.get(m)
        return r[k] if r and r[k] else None

    def yoy(m: int, k: int):
        a, b = level(m, k), level(m - 12, k)
        if a is not None and b:
            return a / b - 1
        r = cpi.get(m)
        return r[3] if (k == 1 and r and r[3] is not None) else None

    out: dict[date, dict] = {}
    for i, d in enumerate(sessions):
        u, prev = used[i], used[i - 1] if i else None
        row = dict.fromkeys(("ust_3m_chg_bp", "ust_2y_chg_bp", "ust_10y_chg_bp", "ust_10y_2y_slope",
                             "ust_10y_3m_slope", "ust_10y_z252", "cpi_yoy", "core_cpi_yoy",
                             "cpi_yoy_chg", "core_cpi_mom"))
        if u is not None:
            _, y3m, y2, y10 = u
            if y10 is not None and y2 is not None:
                row["ust_10y_2y_slope"] = y10 - y2
            if y10 is not None and y3m is not None:
                row["ust_10y_3m_slope"] = y10 - y3m
            if prev is not None:
                for key, a, b in (("ust_3m_chg_bp", u[1], prev[1]), ("ust_2y_chg_bp", u[2], prev[2]),
                                  ("ust_10y_chg_bp", u[3], prev[3])):
                    if a is not None and b is not None:
                        row[key] = (a - b) * 100.0
            hist = [x[3] for x in used[max(0, i - Z_WINDOW):i] if x is not None and x[3] is not None]
            if y10 is not None and len(hist) >= Z_MIN:
                mu = sum(hist) / len(hist)
                sd = math.sqrt(sum((h - mu) ** 2 for h in hist) / len(hist))
                row["ust_10y_z252"] = (y10 - mu) / sd if sd > 0 else None
        m = latest(_month_idx(d) - 2)
        if m is not None:
            row["cpi_yoy"], row["core_cpi_yoy"] = yoy(m, 1), yoy(m, 2)
            p = latest(m - 1)
            prev_yoy = yoy(p, 1) if p is not None else None
            if row["cpi_yoy"] is not None and prev_yoy is not None:
                row["cpi_yoy_chg"] = row["cpi_yoy"] - prev_yoy
            c0, c1 = level(m, 2), level(p, 2) if p is not None else None
            if c0 is not None and c1:
                row["core_cpi_mom"] = (c0 / c1) ** (1.0 / (m - p)) - 1  # per month over a gap
        out[d] = row
    return out


def _build_macro_daily(con) -> int:
    """Rewrite macro_daily (one row per session in daily_bars) from treasury_yields and
    inflation; with either table absent its columns stay NULL."""
    con.execute("DELETE FROM macro_daily")
    last = con.execute("SELECT max(date) FROM daily_bars").fetchone()[0]
    if last is None:
        return 0
    lag = int((config.settings().get("macro") or {}).get("yield_lag_sessions", 1))
    sessions = [r[0] for r in con.execute(
        "SELECT date FROM trading_calendar WHERE date <= ? ORDER BY date", [last]).fetchall()]
    yields = (con.execute("SELECT date, yield_3_month, yield_2_year, yield_10_year FROM treasury_yields "
                          "ORDER BY date").fetchall() if _table_exists(con, "treasury_yields") else [])
    infl = []
    if _table_exists(con, "inflation"):
        cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'inflation'").fetchall()}
        yoy_col = "cpi_year_over_year" if "cpi_year_over_year" in cols else "NULL"
        infl = con.execute(f"SELECT date, cpi, cpi_core, {yoy_col} FROM inflation ORDER BY date").fetchall()
    series = macro_series(sessions, yields, infl, lag=lag)
    have = {r[0] for r in con.execute("SELECT DISTINCT date FROM daily_bars").fetchall()}
    names = ["ust_3m_chg_bp", "ust_2y_chg_bp", "ust_10y_chg_bp", "ust_10y_2y_slope", "ust_10y_3m_slope",
             "ust_10y_z252", "cpi_yoy", "core_cpi_yoy", "cpi_yoy_chg", "core_cpi_mom"]
    rows = [d for d in sessions if d in have]
    if not rows:
        return 0
    tbl = pa.table({"date": pa.array(rows, pa.date32()),
                    **{n: pa.array([series[d][n] for d in rows], pa.float64()) for n in names}})
    con.register("_macro", tbl)
    try:
        con.execute("INSERT INTO macro_daily SELECT date, " + ", ".join(names) + " FROM _macro")
    finally:
        con.unregister("_macro")
    return len(rows)


# --- pre-market expressions (shared by export-tft and export-next) ---------------------

PM_CLIP = 0.30
PM_VOL_CLIP = 6.0
PM_RANGE_CLIP = 1.0
DEFAULT_CUTOFF = "09:30"


def cutoff_key(cutoff: str) -> str:
    """'09:30' -> '0930' (a key of massive_ext CUTOFFS)."""
    key = cutoff.replace(":", "")
    if key not in CUTOFFS:
        raise ValueError(f"unknown pre-market cutoff {cutoff!r}; one of "
                         f"{', '.join(k[:2] + ':' + k[2:] for k in CUTOFFS)}")
    return key


def premarket_cutoff() -> str:
    """The [premarket] cutoff setting, e.g. '09:30'."""
    cutoff = str((config.settings().get("premarket") or {}).get("cutoff", DEFAULT_CUTOFF))
    cutoff_key(cutoff)
    return cutoff


def pm_px_expr(last: str, open_: str, pvwap: str, ok: str) -> dict[str, str]:
    """SQL expressions for the pre-market PRICES group (the 09:30 forecast).

    last  the last pre-market trade before 09:30, split-adjusted
    open_ day d's open, split-adjusted
    pvwap the previous session's daily VWAP, split-adjusted
    ok    boolean: the previous session is exactly one session back and no split executes today
    pm_px_known is 1 only with all of them present (an open without a last is not enough);
    otherwise both values are 0 (the masked value). Values are clipped to +-PM_CLIP."""
    known = f"({last} IS NOT NULL AND {open_} IS NOT NULL AND {pvwap} > 0 AND coalesce({ok}, false))"

    def pct(x: str) -> str:
        return (f"CASE WHEN {known} THEN greatest(-{PM_CLIP}, least({PM_CLIP}, {x} / {pvwap} - 1)) "
                "ELSE 0.0 END")

    return {
        "pm_last_pct_prev_vwap": pct(last),
        "open_pct_prev_vwap": pct(open_),
        "pm_px_known": f"CASE WHEN {known} THEN 1 ELSE 0 END::SMALLINT",
    }


def pm_full_expr(vwap: str, vol: str, mean: str, high: str, low: str, pvwap: str, ok: str) -> dict[str, str]:
    """SQL expressions for the pre-market FULL group (the 09:45 forecast).

    vwap  pre-market VWAP (minute-bar vw) up to the cutoff, split-adjusted
    vol   pre-market volume up to the cutoff, split-adjusted
    mean  vol_10d_mean: mean Massive daily volume of the previous 10 sessions (same adjustment)
    high, low  pre-market high / low up to the cutoff, split-adjusted
    pvwap, ok  as in pm_px_expr
    pm_full_known is 1 only with all of them present; otherwise the values are 0. Each group's
    flag reflects its own data only: the trainer masks full when prices are off."""
    known = (f"({vwap} IS NOT NULL AND {vol} IS NOT NULL AND {high} IS NOT NULL AND {low} IS NOT NULL "
             f"AND {mean} > 0 AND {pvwap} > 0 AND coalesce({ok}, false))")
    return {
        "pm_vwap_pct_prev_vwap": (f"CASE WHEN {known} THEN greatest(-{PM_CLIP}, least({PM_CLIP}, "
                                  f"{vwap} / {pvwap} - 1)) ELSE 0.0 END"),
        "pm_vol_pct_10d": (f"CASE WHEN {known} THEN greatest(0.0, least({PM_VOL_CLIP}, "
                           f"ln(1 + 100 * {vol} / {mean}))) ELSE 0.0 END"),
        "pm_range_pct": (f"CASE WHEN {known} THEN greatest(0.0, least({PM_RANGE_CLIP}, "
                         f"({high} - {low}) / {pvwap})) ELSE 0.0 END"),
        "pm_full_known": f"CASE WHEN {known} THEN 1 ELSE 0 END::SMALLINT",
    }
