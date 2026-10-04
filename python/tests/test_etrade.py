"""E*TRADE Phase 1 (symbols) behavior + Phase 2 session gate.

Phase 1 runs fully offline against a small daily_summary fixture. Phase 2's live
scrape is gated on the browser session file, so we only assert the fail-fast guard
here (the normalizers are finalized against a real sampled response once login lands).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sourcing_py.common.errors import ConfigError
from sourcing_py.etrade import db, eod, export, features, fundamentals, industry, symbols
from sourcing_py.etrade.client import EtradeClient
from sourcing_py.news import db as news_db
from sourcing_py.utils import trading_calendar as tc

_FIXTURES = Path(__file__).parent / "fixtures" / "etrade"


def _fixture(name):
    return json.loads((_FIXTURES / name).read_text())


def _write_day(directory, date_str, tickers):
    lines = [json.dumps({"ticker": t, "date": date_str, "close": 1.0}) for t in tickers]
    (directory / f"daily_summary-{date_str}.jsonl").write_text("\n".join(lines) + "\n")


def _write_bars(directory, date_str, bars):
    """bars: {ticker: close}. Writes a full OHLCV daily_summary record per ticker."""
    lines = []
    for t, close in bars.items():
        lines.append(json.dumps({
            "ticker": t, "date": date_str,
            "open": close, "high": close + 1, "low": close - 1, "close": close,
            "volume": 100.5, "vwap": close, "transactions": 42,
            "window_start_ms": 1786737600000,
            "_source": "massive_grouped_daily",
            "_ingested_at": "2026-08-17T00:54:46.548031+00:00",
            "_run_id": "20260817T005446Z",
        }))
    (directory / f"daily_summary-{date_str}.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture
def ds_dir(tmp_path):
    d = tmp_path / "daily_summary"
    d.mkdir()
    _write_day(d, "2026-08-12", ["AAA", "BBB"])
    _write_day(d, "2026-08-13", ["BBB", "CCC"])
    _write_day(d, "2026-08-14", ["AAA", "BBB", "CCC", "DDD"])
    return d


def test_build_symbols_unions_and_tracks_window(ds_dir, tmp_path):
    dbp = tmp_path / "f.duckdb"
    summary = symbols.build_symbols(days=90, directory=ds_dir, db_path=dbp)
    assert summary["symbols_in_window"] == 4  # AAA BBB CCC DDD
    with db.connect(dbp) as con:
        rows = {
            s: (fs.isoformat(), ls.isoformat(), n)
            for s, fs, ls, n in con.execute(
                "SELECT symbol, first_seen, last_seen, days_seen FROM symbols"
            ).fetchall()
        }
    assert rows["AAA"] == ("2026-08-12", "2026-08-14", 2)
    assert rows["BBB"] == ("2026-08-12", "2026-08-14", 3)
    assert rows["DDD"] == ("2026-08-14", "2026-08-14", 1)


def test_build_symbols_is_idempotent(ds_dir, tmp_path):
    dbp = tmp_path / "f.duckdb"
    symbols.build_symbols(days=90, directory=ds_dir, db_path=dbp)
    symbols.build_symbols(days=90, directory=ds_dir, db_path=dbp)  # re-run
    with db.connect(dbp) as con:
        total = con.execute("SELECT count(*) FROM symbols").fetchone()[0]
        dupes = con.execute(
            "SELECT count(*) FROM (SELECT symbol FROM symbols GROUP BY symbol HAVING count(*)>1)"
        ).fetchone()[0]
    assert total == 4 and dupes == 0


def test_ingest_eod_loads_bars_and_refreshes_symbols(tmp_path):
    d = tmp_path / "daily_summary"
    d.mkdir()
    _write_bars(d, "2024-08-16", {"AAA": 10.0, "BBB": 20.0})
    _write_bars(d, "2024-08-19", {"AAA": 11.0, "CCC": 30.0})
    dbp = tmp_path / "f.duckdb"
    summary = eod.ingest_eod(days=None, directory=d, db_path=dbp)  # all files
    assert summary["files"] == 2 and summary["bars_loaded"] == 4 and summary["bars_total"] == 4
    assert not summary["failed_files"]
    with db.connect(dbp) as con:
        # OHLCV landed with correct types/values
        row = con.execute(
            "SELECT open, high, low, close, volume, transactions FROM daily_bars "
            "WHERE symbol='AAA' AND date='2024-08-16'"
        ).fetchone()
        assert row == (10.0, 11.0, 9.0, 10.0, 100.5, 42)
        # symbols master reflects the full ingested span, not just a recent window
        aaa = con.execute(
            "SELECT first_seen, last_seen, days_seen FROM symbols WHERE symbol='AAA'"
        ).fetchone()
        assert (aaa[0].isoformat(), aaa[1].isoformat(), aaa[2]) == ("2024-08-16", "2024-08-19", 2)


def test_ingest_eod_is_idempotent_and_upserts(tmp_path):
    d = tmp_path / "daily_summary"
    d.mkdir()
    _write_bars(d, "2024-08-16", {"AAA": 10.0})
    dbp = tmp_path / "f.duckdb"
    eod.ingest_eod(days=None, directory=d, db_path=dbp)
    # a corrected re-issue of the same day should update in place, not duplicate
    _write_bars(d, "2024-08-16", {"AAA": 99.0})
    eod.ingest_eod(days=None, directory=d, db_path=dbp)
    with db.connect(dbp) as con:
        n = con.execute("SELECT count(*) FROM daily_bars").fetchone()[0]
        close = con.execute("SELECT close FROM daily_bars WHERE symbol='AAA'").fetchone()[0]
    assert n == 1 and close == 99.0


def test_single_file_window(ds_dir, tmp_path):
    files = symbols.select_files(fromdate="2026-08-14", todate="2026-08-14", directory=ds_dir)
    assert [p.name for p in files] == ["daily_summary-2026-08-14.jsonl"]


def test_fetch_requires_session(tmp_path, monkeypatch):
    # Point the session path somewhere that doesn't exist -> fail fast.
    missing = tmp_path / "nope" / "session.json"
    monkeypatch.setattr(
        "sourcing_py.etrade.client.config.etrade_config",
        lambda: {
            "base_url": "https://example.test",
            "stocks_per_sec": 2,
            "db_path": tmp_path / "f.duckdb",
            "session_path": missing,
        },
    )
    with pytest.raises(ConfigError, match="session not found"):
        fundamentals.fetch(symbols=["GOOGL"], db_path=tmp_path / "f.duckdb")


def test_client_gate_and_headers(tmp_path, monkeypatch):
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"cookies": "a=b", "requestHeaders": {"stk1": "x", "stk2": "y"}}))
    monkeypatch.setattr(
        "sourcing_py.etrade.client.config.etrade_config",
        lambda: {
            "base_url": "https://example.test",
            "stocks_per_sec": 2,
            "db_path": tmp_path / "f.duckdb",
            "session_path": session,
        },
    )
    client = EtradeClient()
    headers = client._headers()
    assert headers["cookie"] == "a=b" and headers["stk1"] == "x" and headers["stk2"] == "y"


def _insert_fake_row(con, symbol):
    # line_item = symbol keeps each symbol's PK distinct (real fetch uses real xids)
    con.execute(
        "INSERT INTO fundamentals (xid, symbol, statement, period_type, fiscal_end, "
        "line_item, value, raw_label, period_length, period_length_units, "
        "thousand_multiplier, fetched_at) VALUES (1, ?, 'balance_sheet', 'q', "
        "'2026-06-30', ?, 1.0, 'x', NULL, NULL, false, now())",
        [symbol, symbol],
    )


def test_fetch_aborts_on_auth_error_then_resumes(tmp_path, monkeypatch):
    """An expired token (ConfigError mid-run) stops the run cleanly: symbols committed so
    far persist, the in-flight symbol is rolled back, and --resume continues from there."""
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"accessToken": "tok"}))
    monkeypatch.setattr(
        "sourcing_py.etrade.client.config.etrade_config",
        lambda: {
            "base_url": "https://example.test",
            "stocks_per_sec": 1000,  # no real pacing in the test
            "db_path": tmp_path / "f.duckdb",
            "session_path": session,
        },
    )
    # keep any failure logging inside tmp — never touch the real state/etrade dir
    monkeypatch.setattr("sourcing_py.common.failures.config.repo_root", lambda: tmp_path)
    dbp = tmp_path / "f.duckdb"
    with db.connect(dbp) as con:
        con.executemany("INSERT INTO symbols (symbol) VALUES (?)", [("AAA",), ("BBB",), ("CCC",)])

    calls: list[str] = []

    # network (worker) part: BBB's token check fails; others return an opaque payload
    def fake_payloads(client, symbol):
        calls.append(symbol)
        if symbol == "BBB":
            raise ConfigError("E*TRADE API returned HTTP 401 — token expired")
        return {"symbol": symbol}

    # store (main-thread) part: insert a distinct fundamentals row per symbol
    def fake_store(con, symbol, payloads):
        _insert_fake_row(con, symbol)
        return 1

    monkeypatch.setattr(fundamentals, "_fetch_payloads", fake_payloads)
    monkeypatch.setattr(fundamentals, "_store_payloads", fake_store)
    summary = fundamentals.fetch(fetch_all=True, resume=True, db_path=dbp)
    assert summary["aborted"] is True and summary["ok"] == 1
    assert calls == ["AAA", "BBB"]  # workers=1 (test cfg has no 'workers'): stopped at BBB, CCC never attempted
    with db.connect(dbp) as con:
        done = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM fundamentals").fetchall()]
    assert done == ["AAA"]  # BBB was never stored (its fetch raised before any DB write)

    # resume with a now-working fetch: AAA is skipped, BBB + CCC complete
    calls.clear()

    def ok_payloads(client, symbol):
        calls.append(symbol)
        return {"symbol": symbol}

    monkeypatch.setattr(fundamentals, "_fetch_payloads", ok_payloads)  # _store_payloads stays faked
    summary2 = fundamentals.fetch(fetch_all=True, resume=True, db_path=dbp)
    assert calls == ["BBB", "CCC"] and summary2["aborted"] is False
    with db.connect(dbp) as con:
        done = sorted(r[0] for r in con.execute("SELECT DISTINCT symbol FROM fundamentals").fetchall())
    assert done == ["AAA", "BBB", "CCC"]


def test_fetch_records_no_data_and_resume_skips(tmp_path, monkeypatch):
    """ETFs/funds (556 on statements) and invalid symbols (400 on lookup) are not
    failures: they're recorded as fundamentals_status='none' (committed) so a later
    --resume skips them instead of re-hitting the API and burning the session."""
    from sourcing_py.common.errors import FetchError

    session = tmp_path / "session.json"
    session.write_text(json.dumps({"accessToken": "tok"}))
    monkeypatch.setattr(
        "sourcing_py.etrade.client.config.etrade_config",
        lambda: {
            "base_url": "https://example.test",
            "stocks_per_sec": 1000,
            "db_path": tmp_path / "f.duckdb",
            "session_path": session,
        },
    )
    monkeypatch.setattr("sourcing_py.common.failures.config.repo_root", lambda: tmp_path)
    dbp = tmp_path / "f.duckdb"
    with db.connect(dbp) as con:
        con.executemany("INSERT INTO symbols (symbol) VALUES (?)", [("GOODCO",), ("ETFX",), ("BADX",)])

    bs, isq = _fixture("balance_sheet_q.json"), _fixture("income_statement_q.json")
    filings, empty = _fixture("sec_filings_10k.json"), _fixture("sec_filings_empty.json")

    def fake_resolve(self, symbol):
        if symbol == "BADX":
            raise FetchError("<BADX> Symbol not found", status_code=400)
        return {"XID": 111 if symbol == "GOODCO" else 222, "issueType": "CS"}

    def fake_bs(self, xid, period):
        if xid == 222:  # ETFX — a fund with no balance sheet
            raise FetchError("No Balance Sheet data was returned", status_code=556)
        return bs

    monkeypatch.setattr(EtradeClient, "resolve_xid", fake_resolve)
    monkeypatch.setattr(EtradeClient, "balance_sheet", fake_bs)
    monkeypatch.setattr(EtradeClient, "income_statement", lambda self, xid, period: isq)
    monkeypatch.setattr(EtradeClient, "sec_filings", lambda self, xid, ft: filings if ft == "10-K" else empty)

    summary = fundamentals.fetch(fetch_all=True, resume=True, db_path=dbp)
    assert summary["ok"] == 1 and summary["skipped"] == 2 and summary["failed"] == 0
    assert not summary["aborted"]
    with db.connect(dbp) as con:
        status = dict(con.execute("SELECT symbol, fundamentals_status FROM symbols").fetchall())
        etfx_xid = con.execute("SELECT xid FROM symbols WHERE symbol='ETFX'").fetchone()[0]
    assert status == {"GOODCO": "ok", "ETFX": "none", "BADX": "none"}
    assert etfx_xid == 222  # lookup succeeded, so xid is still enriched despite no statements

    # resume: every symbol has a recorded outcome, so none is re-attempted
    calls: list[str] = []
    orig = fundamentals._fetch_payloads
    monkeypatch.setattr(
        fundamentals, "_fetch_payloads",
        lambda client, symbol: (calls.append(symbol), orig(client, symbol))[1],
    )
    summary2 = fundamentals.fetch(fetch_all=True, resume=True, db_path=dbp)
    assert calls == [] and summary2["ok"] == 0 and summary2["skipped"] == 0


def test_fetch_concurrent_stores_all_symbols(tmp_path, monkeypatch):
    """With workers>1, symbols are fetched concurrently but every write lands on the main
    thread — all symbols must store exactly once with no PK collisions or lost rows."""
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"accessToken": "tok"}))
    monkeypatch.setattr(
        "sourcing_py.etrade.client.config.etrade_config",
        lambda: {
            "base_url": "https://example.test",
            "stocks_per_sec": 1000,
            "workers": 4,  # exercise the thread pool
            "db_path": tmp_path / "f.duckdb",
            "session_path": session,
        },
    )
    monkeypatch.setattr("sourcing_py.common.failures.config.repo_root", lambda: tmp_path)
    dbp = tmp_path / "f.duckdb"
    syms = [f"S{i:02d}" for i in range(12)]
    with db.connect(dbp) as con:
        con.executemany("INSERT INTO symbols (symbol) VALUES (?)", [(s,) for s in syms])

    bs, isq = _fixture("balance_sheet_q.json"), _fixture("income_statement_q.json")
    filings, empty = _fixture("sec_filings_10k.json"), _fixture("sec_filings_empty.json")
    # distinct xid per symbol so the shared fixture data doesn't collide on the fundamentals PK
    monkeypatch.setattr(EtradeClient, "resolve_xid", lambda self, symbol: {"XID": 1000 + int(symbol[1:]), "issueType": "CS"})
    monkeypatch.setattr(EtradeClient, "balance_sheet", lambda self, xid, period: bs)
    monkeypatch.setattr(EtradeClient, "income_statement", lambda self, xid, period: isq)
    monkeypatch.setattr(EtradeClient, "sec_filings", lambda self, xid, ft: filings if ft == "10-K" else empty)

    summary = fundamentals.fetch(fetch_all=True, resume=True, db_path=dbp)
    assert summary["ok"] == 12 and summary["skipped"] == 0 and summary["failed"] == 0
    with db.connect(dbp) as con:
        n_ok = con.execute("SELECT count(*) FROM symbols WHERE fundamentals_status='ok'").fetchone()[0]
        n_syms = con.execute("SELECT count(DISTINCT symbol) FROM fundamentals").fetchone()[0]
    assert n_ok == 12 and n_syms == 12


def _make_trading_db(path):
    """Minimal MBin trading.db: Industries + Symbol_Industry (source of the mapping)."""
    import sqlite3

    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE Industries (Id INTEGER PRIMARY KEY, Sector_Code TEXT, Sector_Name TEXT,
                                 Industry_Code TEXT, Industry_Name TEXT, Source_Url TEXT);
        CREATE TABLE Symbol_Industry (Symbol TEXT, Industry_Id INTEGER, Created_At TEXT);
        """
    )
    con.executemany(
        "INSERT INTO Industries (Id, Sector_Code, Sector_Name, Industry_Code, Industry_Name, Source_Url) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "53", "Consumer Cyclicals", "532040", "Household Goods", "http://x/1"),
            (2, "57", "Technology", "571010", "Semiconductors", "http://x/2"),
        ],
    )
    con.executemany(
        "INSERT INTO Symbol_Industry (Symbol, Industry_Id, Created_At) VALUES (?, ?, ?)",
        # AAA spans TWO industries (in two different sectors) — proves multi-membership;
        # BBB is single; ZZZ isn't in our symbols table.
        [("AAA", 1, "t"), ("AAA", 2, "t"), ("BBB", 2, "t"), ("ZZZ", 2, "t")],
    )
    con.commit()
    con.close()


def test_map_industries_populates_dims_and_memberships(tmp_path, monkeypatch):
    """The sector/industry dims come from MBin and memberships land in the join table. A
    symbol may belong to several industries/sectors; symbols absent from MBin get none, and
    MBin symbols we don't track are reported unmatched."""
    trading = tmp_path / "trading.db"
    _make_trading_db(trading)
    dbp = tmp_path / "f.duckdb"
    with db.connect(dbp) as con:
        con.executemany("INSERT INTO symbols (symbol) VALUES (?)", [("AAA",), ("BBB",), ("CCC",)])

    summary = industry.map_industries(trading_db_path=trading, db_path=dbp)
    assert summary["sectors"] == 2 and summary["industries"] == 2
    assert summary["memberships"] == 3  # AAA×2 + BBB×1 (ZZZ excluded — not in our table)
    assert summary["symbols_matched"] == 2  # distinct: AAA, BBB
    assert summary["symbols_unmatched"] == 1  # ZZZ in MBin but not our table
    assert summary["symbols_with_industry"] == 2

    with db.connect(dbp) as con:
        # AAA resolves to BOTH industries/sectors via the join table
        aaa = {
            (iname, sname)
            for iname, sname in con.execute(
                "SELECT i.industry_name, sec.sector_name FROM symbol_industry si "
                "JOIN industries i ON si.industry_code = i.industry_code "
                "JOIN sectors sec ON i.sector_code = sec.sector_code WHERE si.symbol = 'AAA'"
            ).fetchall()
        }
        assert aaa == {("Household Goods", "Consumer Cyclicals"), ("Semiconductors", "Technology")}
        # CCC (not in MBin) has no memberships
        assert con.execute("SELECT count(*) FROM symbol_industry WHERE symbol='CCC'").fetchone()[0] == 0

    # idempotent: a second run doesn't duplicate dims or memberships
    industry.map_industries(trading_db_path=trading, db_path=dbp)
    with db.connect(dbp) as con:
        assert con.execute("SELECT count(*) FROM industries").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM sectors").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM symbol_industry").fetchone()[0] == 3  # no dupes


def test_map_industries_missing_db_raises(tmp_path):
    with pytest.raises(ConfigError, match="trading DB not found"):
        industry.map_industries(trading_db_path=tmp_path / "nope.db", db_path=tmp_path / "f.duckdb")


def test_client_bearer_auth(tmp_path, monkeypatch):
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"accessToken": "tok123"}))
    monkeypatch.setattr(
        "sourcing_py.etrade.client.config.etrade_config",
        lambda: {
            "base_url": "https://example.test",
            "stocks_per_sec": 2,
            "db_path": tmp_path / "f.duckdb",
            "session_path": session,
        },
    )
    headers = EtradeClient()._headers()
    assert headers["authorization"] == "Bearer tok123"
    assert headers["origin"].startswith("https://") and headers["referer"].startswith("https://")


def test_normalize_balance_sheet():
    rows = fundamentals._normalize_statement(
        _fixture("balance_sheet_q.json"), "balance_sheet", "q"
    )
    # stable codes are the line_item key; human label kept as raw_label
    by_code = {(r["line_item"], r["fiscal_end"].isoformat()): r for r in rows if r["fiscal_end"]}
    atca = by_code[("ATCA", "2026-06-30")]
    assert atca["value"] == 343524 and atca["raw_label"] == "Total current assets"
    assert atca["thousand_multiplier"] is False
    # a nested child whose entry omits `date` still lands on the positional period end
    scsi = by_code[("SCSI", "2026-06-30")]
    assert scsi["value"] == 242474 and scsi["raw_label"] == "Cash and short term inv"
    # balance sheet is a point-in-time snapshot: no period length
    assert atca["period_length"] is None


def test_normalize_income_statement():
    rows = fundamentals._normalize_statement(
        _fixture("income_statement_q.json"), "income_statement", "q"
    )
    rtlr = next(r for r in rows if r["line_item"] == "RTLR" and r["fiscal_end"].isoformat() == "2026-06-30")
    assert rtlr["value"] == 119796 and rtlr["raw_label"] == "Total revenue"
    assert rtlr["period_length"] == 3 and rtlr["period_length_units"] == "M"


def test_normalize_filings():
    rows = fundamentals._normalize_filings(_fixture("sec_filings_10k.json"), "10-K")
    assert rows and rows[0]["form_type"] == "10-K"
    assert rows[0]["date_filed"].isoformat() == "2026-02-05"
    assert rows[0]["html_doc_key"]
    # empty result (e.g. non-primary share class) yields no rows, not a crash
    assert fundamentals._normalize_filings(_fixture("sec_filings_empty.json"), "10-K") == []


def test_store_empty_and_dropped_batches(tmp_path):
    """Empty filings (a share class with none) must not crash; date-less statement rows
    must be reported as dropped rather than silently vanishing."""
    dbp = tmp_path / "f.duckdb"
    with db.connect(dbp) as con:
        # empty filing batch -> no crash, nothing stored, nothing dropped
        empty = fundamentals._normalize_filings(_fixture("sec_filings_empty.json"), "10-K")
        assert fundamentals._store_filings(con, 1, "GOOG", empty) == 0
        # a row with no resolvable period end is counted as dropped, not stored
        dropped = fundamentals._store_fundamentals(con, 1, "GOOG", [
            {"statement": "balance_sheet", "period_type": "q", "fiscal_end": None,
             "line_item": "ZZZ", "value": 1.0, "raw_label": "x", "period_length": None,
             "period_length_units": None, "thousand_multiplier": False},
        ])
        assert dropped == 1
        assert con.execute("SELECT count(*) FROM fundamentals").fetchone()[0] == 0

    # a full quarterly balance sheet stores every period the API returned (no truncation)
    with db.connect(dbp) as con:
        rows = fundamentals._normalize_statement(_fixture("balance_sheet_q.json"), "balance_sheet", "q")
        fundamentals._store_fundamentals(con, 1, "GOOG", rows)
        periods = con.execute("SELECT count(DISTINCT fiscal_end) FROM fundamentals").fetchone()[0]
        assert periods == 2  # the 2-period trimmed fixture -> both periods kept


# --- Phase 1c: derived feature columns (features.derive_features) ------------------------
# These run fully offline: exchange_calendars computes NYSE sessions deterministically, and
# derive_features touches no network. Bars are inserted straight into daily_bars so tests can
# pin exact vwap/volume; real 2024 session dates are used so they line up with the calendar.


def _bar(con, symbol, d, vwap, volume=100.0, close=None):
    con.execute(
        "INSERT INTO daily_bars (symbol, date, close, vwap, volume) VALUES (?, ?, ?, ?, ?)",
        [symbol, d, vwap if close is None else close, vwap, volume],
    )


def _insert_filing(con, symbol, form_type, date_filed, key):
    con.execute(
        "INSERT INTO sec_filings (xid, symbol, form_type, date_filed, html_doc_key) "
        "VALUES (1, ?, ?, ?, ?)",
        [symbol, form_type, date_filed, key],
    )


def _seed_ramp(con, symbol, sess, base=10.0, volume=100.0):
    """One bar per session, vwap ramping base, base+1, ... so forward means are predictable."""
    for i, d in enumerate(sess):
        _bar(con, symbol, d, base + i, volume=volume)


def test_trading_calendar_excludes_weekends_and_holidays():
    days = {d.isoformat() for d in tc.sessions(date(2024, 1, 1), date(2024, 1, 20))}
    assert "2024-01-01" not in days  # New Year (holiday)
    assert "2024-01-15" not in days  # MLK Day (holiday)
    assert "2024-01-06" not in days and "2024-01-07" not in days  # weekend
    assert "2024-01-16" in days
    # day_idx is contiguous from the anchor, and gap-free
    idxs = [i for _, i in tc.trading_calendar(end=date(2024, 1, 20))]
    assert idxs == list(range(1, len(idxs) + 1))
    assert tc.day_idx_for(date(2020, 1, 2)) == 1  # first session on/after 2020-01-01
    assert tc.day_idx_for(date(2020, 1, 1)) is None  # holiday -> not a session
    assert tc.next_trading_day(date(2024, 1, 20)) == date(2024, 1, 22)  # Sat -> Mon


def test_derive_features_assigns_consecutive_day_idx(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess)
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        rows = con.execute("SELECT date, day_idx FROM daily_bars ORDER BY date").fetchall()
        cal = dict(con.execute("SELECT date, day_idx FROM trading_calendar").fetchall())
    idxs = [di for _, di in rows]
    assert idxs == list(range(idxs[0], idxs[0] + len(idxs)))  # gap-free over consecutive sessions
    assert all(cal[d] == di for d, di in rows)  # bars match the calendar dimension


def test_filing_flag_snaps_to_next_session(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess)
        _insert_filing(con, "AAA", "10-K", "2024-01-20", "k1")  # Saturday
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        flagged = [d.isoformat() for (d,) in con.execute(
            "SELECT date FROM daily_bars WHERE is_10k = 1"
        ).fetchall()]
        total_k, total_q = con.execute(
            "SELECT sum(is_10k), sum(is_10q) FROM daily_bars"
        ).fetchone()
    assert flagged == ["2024-01-22"]  # snapped forward to Monday
    assert total_k == 1 and total_q == 0


def test_results_window_classification(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 3, 1))
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess)
        _insert_filing(con, "AAA", "10-Q", "2024-01-22", "k1")  # a session (Mon)
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        rw = {d.isoformat(): w for d, w in con.execute(
            "SELECT date, results_window FROM daily_bars"
        ).fetchall()}
    assert rw["2024-01-16"] == "Pre_Earnings_Runup"        # t-4
    assert rw["2024-01-19"] == "Earnings_Eve"              # t-1
    assert rw["2024-01-22"] == "Earnings_Day"              # t0
    assert rw["2024-01-24"] == "Post_Earnings_Reaction"    # t+2
    assert rw["2024-01-29"] == "Post_Earnings_Drift"       # t+5
    assert rw["2024-02-29"] == "Normal_Trading"            # far out
    assert all(v is not None for v in rw.values())         # every bar classified


def test_results_window_nearest_filing_wins(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 2, 1))
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess)
        _insert_filing(con, "AAA", "10-Q", "2024-01-22", "k1")  # t0 for the earlier window
        _insert_filing(con, "AAA", "10-K", "2024-01-26", "k2")  # 4 sessions later
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        mid = con.execute(
            "SELECT results_window FROM daily_bars WHERE date = '2024-01-24'"
        ).fetchone()[0]
    # 01-24 is +2 from the first filing and -2 from the second: a tie in |offset| that the
    # tie-break resolves toward the upcoming filing -> classified relative to 01-26 (t-2).
    assert mid == "Pre_Earnings_Runup"


def test_forward_vwap_values_and_nulls(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 2, 15))  # > 10 sessions
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess, base=10.0, volume=100.0)  # vwap 10,11,12,...
        _insert_filing(con, "AAA", "10-Q", "2024-01-22", "k1")
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        first = con.execute(
            "SELECT vwap_nx_1d, vwap_nx_10d, vwap_end_week, vwap_nx_qtr "
            "FROM daily_bars WHERE date = '2024-01-16'"
        ).fetchone()
        last = con.execute(
            "SELECT vwap_nx_1d, vwap_nx_10d, vwap_nx_qtr FROM daily_bars WHERE date = ?",
            [sess[-1]],
        ).fetchone()
    # equal volumes -> volume-weighted mean collapses to the simple mean
    assert first[0] == 11.0   # vwap_nx_1d: the next session's vwap
    assert first[1] == 15.5   # vwap_nx_10d: mean of the next 10 (11..20)
    assert first[2] == 11.5   # vwap_end_week: 2024-01-16..19 incl (10,11,12,13)
    assert first[3] == 12.0   # vwap_nx_qtr: 01-16..01-22 incl (10..14)
    assert last[0] is None and last[1] is None  # no forward bars
    assert last[2] is None                       # no future filing


def test_forward_vwap_is_volume_weighted(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 19))  # Tue..Fri, one ISO week
    prices = [10.0, 20.0, 30.0, 40.0]
    vols = [10.0, 100.0, 100.0, 100.0]
    with db.connect(dbp) as con:
        for d, p, v in zip(sess, prices, vols):
            _bar(con, "AAA", d, p, volume=v)
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        end_week_first = con.execute(
            "SELECT vwap_end_week FROM daily_bars WHERE date = '2024-01-16'"
        ).fetchone()[0]
    # includes the current (low-volume) day: (10*10 + 20*100 + 30*100 + 40*100) / 310
    assert end_week_first == pytest.approx((100 + 2000 + 3000 + 4000) / 310)


def test_vwap_pct_prev_day(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess, base=10.0)  # vwap 10, 11, 12, ...
        # BBB's first bar has vwap 0 so its next bar's prev-day change divides by zero
        _bar(con, "BBB", sess[0], 0.0)
        _bar(con, "BBB", sess[1], 5.0)
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        pct = {d.isoformat(): p for d, p in con.execute(
            "SELECT date, vwap_pct_prev_day FROM daily_bars WHERE symbol = 'AAA' ORDER BY date"
        ).fetchall()}
        bbb = {d.isoformat(): p for d, p in con.execute(
            "SELECT date, vwap_pct_prev_day FROM daily_bars WHERE symbol = 'BBB' ORDER BY date"
        ).fetchall()}
    assert pct["2024-01-16"] is None                        # first bar: no prior day
    assert pct["2024-01-17"] == pytest.approx((11 - 10) / 10)  # +10%
    assert pct["2024-01-18"] == pytest.approx((12 - 11) / 11)
    assert bbb["2024-01-16"] is None                        # first bar
    assert bbb["2024-01-17"] is None                        # prev vwap 0 -> divide-by-zero guard


def test_derive_features_idempotent_and_recomputes(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess)
    features.derive_features(db_path=dbp)
    cols = "date, day_idx, results_window, vwap_nx_1d, vwap_nx_10d"
    with db.connect(dbp) as con:
        snap1 = con.execute(f"SELECT {cols} FROM daily_bars ORDER BY date").fetchall()
        last_before = con.execute(
            "SELECT vwap_nx_1d FROM daily_bars WHERE date = ?", [sess[-1]]
        ).fetchone()[0]
    features.derive_features(db_path=dbp)  # re-run: must be byte-for-byte identical
    with db.connect(dbp) as con:
        snap2 = con.execute(f"SELECT {cols} FROM daily_bars ORDER BY date").fetchall()
    assert snap1 == snap2
    assert last_before is None  # the last bar had no forward session yet

    # append the next session and recompute: the previously-last row now has a forward day
    nxt = tc.sessions(sess[-1], date(2024, 2, 1))[1]  # first session strictly after the old last
    with db.connect(dbp) as con:
        _bar(con, "AAA", nxt, 99.0)
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        last_after = con.execute(
            "SELECT vwap_nx_1d FROM daily_bars WHERE date = ?", [sess[-1]]
        ).fetchone()[0]
    assert last_after == 99.0  # forward-looking column refreshed as new data landed


def test_derive_features_scoped_to_symbols(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(dbp) as con:
        _seed_ramp(con, "AAA", sess)
        _seed_ramp(con, "BBB", sess)
    features.derive_features(db_path=dbp, symbols=["AAA"])
    with db.connect(dbp) as con:
        aaa = con.execute(
            "SELECT count(*) FROM daily_bars WHERE symbol='AAA' AND day_idx IS NOT NULL"
        ).fetchone()[0]
        bbb = con.execute(
            "SELECT count(*) FROM daily_bars WHERE symbol='BBB' AND day_idx IS NOT NULL"
        ).fetchone()[0]
    assert aaa == len(sess)  # in scope -> derived
    assert bbb == 0          # out of scope -> untouched


# --- Phase 2a: TFT panel export (export.export_tft) ---------------------------------------

_LOOSE = {"min_bars": 1, "min_median_vwap": 0.0, "min_median_volume": 0.0}


def _ohlc_bar(con, symbol, d, vwap, volume=100.0, transactions=50):
    con.execute(
        "INSERT INTO daily_bars (symbol, date, open, high, low, close, vwap, volume, transactions) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [symbol, d, vwap - 0.5, vwap + 1.0, vwap - 1.0, vwap + 0.25, vwap, volume, transactions],
    )


def _symbol(con, symbol, issue_type="CS", exchange="NASDAQ"):
    con.execute(
        "INSERT INTO symbols (symbol, issue_type, exchange) VALUES (?, ?, ?)",
        [symbol, issue_type, exchange],
    )


def _export(tmp_path, dbp, **kw):
    out = tmp_path / "out" / "panel.parquet"
    kw = {**_LOOSE, "news_db_path": tmp_path / "absent.duckdb", **kw}
    summary = export.export_tft(db_path=dbp, out_path=out, **kw)
    return summary, pq.read_table(out).to_pylist()


def test_export_tft_columns_formulas_and_fills(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 2, 9))
    with db.connect(dbp) as con:
        _symbol(con, "AAA")
        _symbol(con, "BBB", issue_type="DR", exchange=None)
        con.execute("INSERT INTO symbol_industry (symbol, industry_code) VALUES ('BBB', '532040')")
        for i, d in enumerate(sess):
            _ohlc_bar(con, "AAA", d, 10.0 + i)
            _ohlc_bar(con, "BBB", d, 20.0)
    features.derive_features(db_path=dbp)
    summary, rows = _export(tmp_path, dbp)

    assert list(rows[0]) == [
        "target_tomorrow_vwap", "target_return", "symbol", "industry_code", "exchange",
        "issue_type", "day_idx", "date", "day_of_week", "month", "is_month_end",
        "is_quarter_end", "sessions_gap_next", "today_vwap", "vwap_pct_prev_day",
        "intraday_spread_pct", "close_position_pct", "open_position_pct", "volume_velocity",
        "transactions_per_volume", "daily_sentiment", "sentiment_volume", "has_news",
        "is_10k", "is_10q",
    ]  # exact set: no leaky forward VWAPs / results_window
    by = {(r["symbol"], r["date"].isoformat()): r for r in rows}
    r = by[("AAA", "2024-01-16")]  # vwap 10, next session 11
    assert r["target_tomorrow_vwap"] == 11.0
    assert r["target_return"] == pytest.approx(0.1)
    assert r["intraday_spread_pct"] == pytest.approx(2.0 / 10)
    assert r["close_position_pct"] == pytest.approx(0.25 / 10)
    assert r["open_position_pct"] == pytest.approx(-0.5 / 10)
    assert r["transactions_per_volume"] == pytest.approx(0.5)
    assert r["volume_velocity"] == pytest.approx(1.0)  # constant volume
    assert (r["industry_code"], r["exchange"], r["issue_type"]) == ("unknown", "NASDAQ", "CS")
    assert (r["daily_sentiment"], r["sentiment_volume"], r["has_news"]) == (0.0, 0, 0)
    assert by[("BBB", "2024-01-16")]["industry_code"] == "532040"
    assert by[("BBB", "2024-01-16")]["exchange"] == "unknown"

    fri = by[("AAA", "2024-01-19")]
    assert (fri["day_of_week"], fri["sessions_gap_next"]) == (5, 3)  # Fri -> Mon
    assert by[("AAA", "2024-01-31")]["is_month_end"] == 1
    assert by[("AAA", "2024-01-30")]["is_month_end"] == 0
    assert by[("AAA", "2024-01-31")]["is_quarter_end"] == 0

    last = sess[-1].isoformat()
    assert ("AAA", last) not in by  # NULL target dropped
    assert summary["rows"] == 2 * (len(sess) - 1) and summary["inference_rows"] == 0


def test_export_tft_volume_velocity_trailing_window_over_gap(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 2), date(2024, 2, 29))[:25]
    gap = sess[10]
    with db.connect(dbp) as con:
        _symbol(con, "AAA")
        for i, d in enumerate(sess):
            if d != gap:
                _ohlc_bar(con, "AAA", d, 10.0, volume=300.0 if i == 22 else 100.0)
    features.derive_features(db_path=dbp)
    _, rows = _export(tmp_path, dbp)
    vv = {r["date"]: r["volume_velocity"] for r in rows}
    # 20-session trailing window ending on sess[22] covers sess[3..22]; sess[10] is missing,
    # so it averages 19 bars: 18 x 100 + 300 (the current day is included).
    assert vv[sess[22]] == pytest.approx(300.0 / ((18 * 100 + 300) / 19))
    assert vv[sess[1]] == pytest.approx(1.0)  # partial window at the start of history


def test_export_tft_universe_filter(tmp_path):
    dbp = tmp_path / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 2, 9))
    with db.connect(dbp) as con:
        for sym, it in (("GOOD", "CS"), ("ETF1", "ETF"), ("PENNY", "CS"), ("THIN", "CS"), ("NEW", "CS")):
            _symbol(con, sym, issue_type=it)
        for d in sess:
            _ohlc_bar(con, "GOOD", d, 10.0, volume=1000.0)
            _ohlc_bar(con, "ETF1", d, 10.0, volume=1000.0)
            _ohlc_bar(con, "PENNY", d, 0.5, volume=1000.0)
            _ohlc_bar(con, "THIN", d, 10.0, volume=10.0)
        for d in sess[:3]:
            _ohlc_bar(con, "NEW", d, 10.0, volume=1000.0)
    features.derive_features(db_path=dbp)
    _, rows = _export(tmp_path, dbp, min_bars=5, min_median_vwap=1.0, min_median_volume=500.0)
    assert {r["symbol"] for r in rows} == {"GOOD"}


def test_export_tft_inference_rows_news_join_and_meta(tmp_path):
    dbp = tmp_path / "f.duckdb"
    ndb = tmp_path / "news.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(dbp) as con:
        _symbol(con, "AAA")
        for i, d in enumerate(sess):
            _ohlc_bar(con, "AAA", d, 10.0 + i)
    with news_db.connect(ndb) as con:
        con.execute(
            "INSERT INTO symbol_sentiment_daily (symbol, date, n_articles, mean_score_agg) "
            "VALUES ('AAA', ?, 3, 0.4)", [sess[2]]
        )
    features.derive_features(db_path=dbp)
    summary, rows = _export(tmp_path, dbp, news_db_path=ndb, include_inference=True)

    by = {r["date"]: r for r in rows}
    assert by[sess[-1]]["target_tomorrow_vwap"] is None  # latest session kept for inference
    assert summary["inference_rows"] == 1 and summary["rows"] == len(sess)
    assert (by[sess[2]]["daily_sentiment"], by[sess[2]]["sentiment_volume"], by[sess[2]]["has_news"]) \
        == (0.4, 3, 1)
    assert by[sess[3]]["has_news"] == 0

    meta = json.loads(Path(summary["meta_path"]).read_text())
    assert meta["schema_version"] == export.SCHEMA_VERSION
    assert meta["rows"] == len(sess) and meta["news_rows"] == 1
    assert meta["date_range"] == [sess[0].isoformat(), sess[-1].isoformat()]
    assert meta["filters"]["include_inference"] is True


def test_export_tft_requires_derive_features(tmp_path):
    dbp = tmp_path / "f.duckdb"
    with db.connect(dbp) as con:
        _symbol(con, "AAA")
        _ohlc_bar(con, "AAA", date(2024, 1, 16), 10.0)
    with pytest.raises(RuntimeError, match="derive-features"):
        _export(tmp_path, dbp)
