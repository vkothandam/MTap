"""E*TRADE Phase 1 (symbols) behavior + Phase 2 session gate.

Phase 1 runs fully offline against a small daily_summary fixture. Phase 2's live
scrape is gated on the browser session file, so we only assert the fail-fast guard
here (the normalizers are finalized against a real sampled response once login lands).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sourcing_py.common.errors import ConfigError
from sourcing_py.etrade import db, eod, fundamentals, symbols
from sourcing_py.etrade.client import EtradeClient

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
