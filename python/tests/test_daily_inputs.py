"""Daily-model inputs (schema 4): massive-ext VWAPs and bars, survivorship, market/macro
tables, pre-market columns, panel.next and export-next. Offline, no API calls."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sourcing_py.common import config
from sourcing_py.etrade import db, edgar, eod, export, features
from sourcing_py.massive_1s import prepare
from sourcing_py.massive_ext import extended as ext
from sourcing_py.utils import trading_calendar as tc

NY = ext.NY
_LOOSE = {"min_bars": 1, "min_median_vwap": 0.0, "min_median_volume": 0.0}


@pytest.fixture
def roots(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    return tmp_path


def _bar(d, hhmm, o, h, lo, c, v, vw=None):
    t = datetime.fromisoformat(f"{d}T{hhmm}:00").replace(tzinfo=NY)
    b = {"t": int(t.timestamp() * 1000), "o": o, "h": h, "l": lo, "c": c, "v": v, "n": 3}
    if vw is not None:
        b["vw"] = vw
    return b


# --- massive_ext: VWAP sums ------------------------------------------------------------


def test_vwap_sums_at_each_cutoff_and_session():
    day = "2026-10-14"
    bars = [
        _bar(day, "04:00", 10, 12, 8, 10, 100, vw=10.0),     # in every cutoff
        _bar(day, "08:00", 10, 13, 10, 13, 100, vw=12.0),    # not in 0800 (starts at it)
        _bar(day, "09:14", 10, 10, 10, 10, 200, vw=20.0),    # 0915+ only
        _bar(day, "09:24", 10, 11, 10, 11, 100),             # 0925+; vw missing -> typical
        _bar(day, "09:29", 10, 10, 10, 10, 50, vw=30.0),     # 0930 only
        _bar(day, "09:30", 20, 20, 20, 20, 1000, vw=20.0),
        _bar(day, "10:00", 30, 30, 30, 30, 1000, vw=30.0),
        _bar(day, "16:00", 5, 5, 5, 5, 100, vw=5.0),
        _bar(day, "17:00", 7, 7, 7, 7, 300, vw=7.0),
    ]
    (r,) = ext.summarize("ABC", [bars], "x")
    assert r["regular_vwap"] == pytest.approx(25.0)
    assert r["after_hours_vwap"] == pytest.approx((5 * 100 + 7 * 300) / 400)
    assert r["pre_market_volume_0800"] == 100 and r["pre_market_vwap_0800"] == pytest.approx(10.0)
    assert r["pre_market_volume_0900"] == 200
    assert r["pre_market_vwap_0900"] == pytest.approx(11.0)
    assert r["pre_market_volume_0915"] == 400
    assert r["pre_market_vwap_0915"] == pytest.approx((1000 + 1200 + 4000) / 400)
    tp_last = (11 + 10 + 11) / 3
    assert r["pre_market_volume_0925"] == 500
    assert r["pre_market_vwap_0925"] == pytest.approx((1000 + 1200 + 4000 + tp_last * 100) / 500)
    assert r["pre_market_volume_0930"] == 550 and r["pre_market_volume"] == 550
    assert r["pre_market_vwap_0930"] == pytest.approx((1000 + 1200 + 4000 + tp_last * 100 + 1500) / 550)
    # typical-price VWAP: bar 1 (12+8+10)/3=10, bar 2 (13+10+13)/3=12, then 10, 10.667, 10
    assert r["pre_market_tpvwap_0800"] == pytest.approx(10.0)
    assert r["pre_market_tpvwap_0900"] == pytest.approx(11.0)
    assert r["pre_market_tpvwap_0925"] == pytest.approx((10 * 100 + 12 * 100 + 10 * 200 + tp_last * 100) / 500)


def test_half_day_vwaps_use_the_early_close():
    half = "2025-12-24"
    bars = [_bar(half, "12:59", 5, 5, 5, 5, 10, vw=5.0), _bar(half, "13:00", 6, 6, 6, 6, 30, vw=6.0)]
    (r,) = ext.summarize("ABC", [bars], "x")
    assert r["regular_vwap"] == 5.0 and r["after_hours_vwap"] == 6.0
    assert r["pre_market_vwap_0925"] is None and r["pre_market_volume_0925"] == 0


def test_old_and_new_rows_mix_through_finalize(roots):
    d = date(2026, 10, 14)
    old = {"ticker": "OLD", "date": d.isoformat(), "pre_market_price": 1, "pre_market_last": 1,
           "pre_market_volume": 5.0, "regular_volume": 1.0, "after_hours_price": None,
           "after_hours_volume": 0.0, "minute_bars": 2, "fetched_at": "2026-10-15T00:00:00+00:00"}
    out = ext.out_path(d)
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(old) + "\n")
    job = ext.job_dir(d, d)
    (job / "staging").mkdir(parents=True)
    (job / "staging" / "NEW.jsonl").write_text(
        json.dumps(ext.summarize("NEW", [[_bar(d, "08:00", 1, 1, 1, 1, 5, vw=1.0)]], "f")[0]) + "\n")
    assert ext.finalize(job) == 1
    rows = {r["ticker"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert "regular_vwap" not in rows["OLD"] and rows["NEW"]["pre_market_vwap_0930"] == 1.0
    # a staged file without the new columns (an old job) still finalizes
    (job / "staging" / "NEW.jsonl").write_text(json.dumps({**old, "ticker": "NEW"}) + "\n")
    ext.finalize(job)
    rows = {r["ticker"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert rows["NEW"]["regular_vwap"] is None and rows["OLD"]["pre_market_volume"] == 5.0


def test_allow_failures_finalizes_and_keeps_the_job(roots, monkeypatch):
    def fake(client, cfg, ticker, d0, d1, sink=None):
        if ticker == "BAD":
            raise RuntimeError("HTTP 502")
        return ext.summarize(ticker, [[_bar("2026-10-14", "08:00", 1, 1, 1, 1, 5)]], "f", sink)

    monkeypatch.setattr(ext, "fetch_ticker", fake)
    d = date(2026, 10, 14)
    assert ext.fetch(d, d, tickers=["AAA", "BAD"], workers=1) == 1
    assert not ext.out_path(d).exists()
    assert ext.fetch(d, d, tickers=["AAA", "BAD"], workers=1, allow_failures=1) == 0
    assert [json.loads(x)["ticker"] for x in ext.out_path(d).read_text().splitlines()] == ["AAA"]
    fails = (ext.job_dir(d, d) / "failures.jsonl").read_text()
    assert "BAD" in fails  # left for a later retry


def test_existing_date_file_is_never_replaced_by_fewer_tickers(roots, monkeypatch):
    d = date(2026, 10, 14)
    monkeypatch.setattr(ext, "fetch_ticker", lambda c, cfg, t, a, b, sink=None:
                        ext.summarize(t, [[_bar("2026-10-14", "08:00", 1, 1, 1, 1, 5)]], "f"))
    ext.fetch(d, d, tickers=["AAA", "BBB", "CCC"], workers=1)
    ext.fetch(d, d, tickers=["AAA"], workers=1)
    assert len(ext.out_path(d).read_text().splitlines()) == 3


def test_quarters_split_and_chunk_runs_each_as_its_own_job(roots, monkeypatch):
    assert ext.quarters(date(2026, 2, 10), date(2026, 8, 5)) == [
        (date(2026, 2, 10), date(2026, 3, 31)), (date(2026, 4, 1), date(2026, 6, 30)),
        (date(2026, 7, 1), date(2026, 8, 5))]
    assert ext.quarters(date(2025, 12, 1), date(2026, 1, 3)) == [
        (date(2025, 12, 1), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 1, 3))]
    jobs = []
    monkeypatch.setattr(ext, "_fetch_one", lambda d0, d1, **kw: jobs.append((d0, d1)) or 0)
    assert ext.fetch(date(2026, 2, 10), date(2026, 8, 5), chunk="quarter") == 0
    assert len(jobs) == 3
    with pytest.raises(ValueError):
        ext.fetch(date(2026, 2, 10), date(2026, 8, 5), chunk="year")


def test_save_bars_writes_monthly_parquet(roots, monkeypatch):
    bars = {"AAA": [_bar("2026-09-30", "10:00", 1, 2, 1, 2, 10, vw=1.5), _bar("2026-10-01", "10:00", 2, 3, 2, 3, 20)],
            "BBB": [_bar("2026-10-01", "11:00", 5, 5, 5, 5, 7, vw=5.0)]}

    def fake_pages(client, cfg, ticker, d0, d1):
        yield bars[ticker]

    monkeypatch.setattr(ext, "_pages", fake_pages)
    dest = roots / "bars_1m"
    assert ext.fetch(date(2026, 9, 30), date(2026, 10, 1), tickers=["AAA", "BBB"], workers=1,
                     save_bars=dest) == 0
    sep, octo = sorted(p.parent.name for p in dest.glob("month=*/*.parquet"))
    assert (sep, octo) == ("month=2026-09", "month=2026-10")
    t = pq.read_table(next(dest.glob("month=2026-10/*.parquet"))).to_pylist()
    assert [(r["ticker"], r["v"]) for r in t] == [("AAA", 20.0), ("BBB", 7.0)]
    assert list(t[0]) == ["ticker", "t", "o", "h", "l", "c", "v", "vw", "n"]
    assert t[0]["vw"] is None and not list(dest.rglob(".*tmp"))
    # off by default: no bars written
    monkeypatch.setattr(ext, "fetch_ticker", lambda c, cfg, t, a, b, sink=None: [])
    ext.fetch(date(2026, 9, 30), date(2026, 10, 1), tickers=["AAA"], workers=1)
    assert not (roots / "state" / "massive_ext" / "jobs" / "x").exists()


def test_universe_takes_cs_and_adrc_and_1s_stays_cs(roots):
    d = roots / "out" / "tickers_cs"
    d.mkdir(parents=True)
    rows = [{"ticker": "CSX", "name": "Cs Inc", "active": True, "type": "CS"},
            {"ticker": "ADR", "name": "Adr plc", "active": True, "type": "ADRC"},
            {"ticker": "OLDF", "name": "Old file", "active": True}]  # no type: CS
    (d / "tickers_cs-2026-10-15.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ext.universe(date(2021, 10, 15)) == ["ADR", "CSX", "OLDF"]
    assert [s["ticker"] for s in prepare.stock_list(rows)] == ["CSX", "OLDF"]


def test_tickers_cs_source_fetches_both_types(roots, monkeypatch):
    from sourcing_py.sources.massive_tickers_cs import source as tk

    asked = []

    def fake_pages(client, cfg, url, params):
        asked.append((params["type"], params["active"]))
        yield [{"ticker": f"{params['type']}{params['active']}"}]

    monkeypatch.setattr(tk, "pages", fake_pages)
    src = tk.MassiveTickersCsSource()
    got = src.fetch({})
    assert {a[0] for a in asked} == {"CS", "ADRC"}
    rows = src.extract(got)
    assert {r["type"] for r in rows} == {"CS", "ADRC"}


# --- EDGAR survivorship ----------------------------------------------------------------


def _symbol(con, symbol, issue_type="CS"):
    con.execute("INSERT INTO symbols (symbol, issue_type) VALUES (?, ?) ON CONFLICT DO NOTHING",
                [symbol, issue_type])


def test_store_keeps_delisted_mappings_and_cik_falls_back_to_tickers_cs(roots):
    import httpx

    dbp = roots / "f.duckdb"
    with db.connect(dbp) as con:
        for s in ("LIVE", "GONE"):
            _symbol(con, s)
        con.execute("INSERT INTO sec_edgar_tickers VALUES ('GONE', 77, 'GONE', 'Gone Inc', now())")
    tdir = roots / "out" / "tickers_cs"
    tdir.mkdir(parents=True)
    (tdir / "tickers_cs-2026-10-15.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"ticker": "GONE", "active": False, "cik": "0000000077", "delisted_utc": "2024-01-01T00:00:00Z"},
        {"ticker": "DEAD", "active": False, "cik": "0000000088", "delisted_utc": "2023-01-01T00:00:00Z"},
        {"ticker": "DEAD", "active": False, "cik": "0000000099", "delisted_utc": "2025-01-01T00:00:00Z"},
    ]) + "\n")
    with db.connect(dbp) as con:
        _symbol(con, "DEAD")

    def handler(request):
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 1, "ticker": "LIVE", "title": "Live"}})
        return httpx.Response(200, json={"filings": {"recent": {"form": []}, "files": []}})

    import os
    os.environ["EDGAR_REQUESTS_PER_SEC"] = "0"
    os.environ["EDGAR_WORKERS"] = "1"
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            summary = edgar.fetch_edgar(since="2024-01-01", db_path=dbp, client=client)
    finally:
        del os.environ["EDGAR_REQUESTS_PER_SEC"], os.environ["EDGAR_WORKERS"]
    with db.connect(dbp) as con:
        got = dict(con.execute("SELECT symbol, cik FROM sec_edgar_tickers").fetchall())
    # DEAD: the most recent listing row's CIK wins; GONE (not listed by SEC) keeps its mapping
    assert got == {"LIVE": 1, "GONE": 77, "DEAD": 99}
    assert summary["symbols_mapped"] == 3


def test_issue_type_filled_from_tickers_cs_and_recycled_tickers_counted(roots):
    rows = [{"ticker": "A", "type": "CS", "cik": "1", "active": True},
            {"ticker": "B", "type": "ADRC", "cik": "2", "active": False},
            {"ticker": "R", "type": "CS", "cik": "3", "active": False},
            {"ticker": "R", "type": "CS", "cik": "4", "active": True}]
    assert eod.recycled_tickers(rows) == ["R"]
    with db.connect(roots / "f.duckdb") as con:
        con.execute("INSERT INTO symbols (symbol, issue_type) VALUES ('A', NULL), ('B', NULL), ('R', 'ETF'), ('Z', NULL)")
        assert eod.fill_issue_types(con, rows) == 2
        got = dict(con.execute("SELECT symbol, issue_type FROM symbols").fetchall())
    assert got == {"A": "CS", "B": "DR", "R": "ETF", "Z": None}


def test_ingest_eod_extended_only_updates_vwap_columns(roots):
    daily = roots / "out" / "daily_summary"
    daily.mkdir(parents=True)
    (daily / "daily_summary-2026-10-14.jsonl").write_text(json.dumps(
        {"ticker": "AAA", "date": "2026-10-14", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10,
         "vwap": 1.5, "transactions": 1, "window_start_ms": 1, "_source": "t",
         "_ingested_at": "2026-10-15T00:00:00+00:00", "_run_id": "r"}) + "\n")
    dbp = roots / "f.duckdb"
    eod.ingest_eod(directory=daily, db_path=dbp)
    (summ,) = ext.summarize("AAA", [[_bar("2026-10-14", "09:00", 1, 1, 1, 1, 5, vw=1.0),
                                     _bar("2026-10-14", "10:00", 2, 2, 2, 2, 5, vw=2.0)]], "2026-10-15T00:00:00+00:00")
    ext_dir = roots / "out" / "extended_hours"
    ext_dir.mkdir()
    (ext_dir / "extended_hours-2026-10-14.jsonl").write_text(json.dumps({**summ, "_source": "x"}) + "\n")
    out = eod.ingest_eod(directory=daily, db_path=dbp, extended_only=True, fromdate="2026-10-14")
    assert out["bars_loaded"] == 0 and out["bars_extended"] == 1
    with db.connect(dbp) as con:
        got = con.execute("SELECT regular_vwap, pre_market_volume_0925, pre_market_tpvwap_0930 "
                          "FROM daily_bars").fetchone()
    assert got == (2.0, 5.0, 1.0)


# --- derive-features: filings, market, macro --------------------------------------------


def _bars(con, symbol, sessions, closes, volume=100.0, split_factors=None):
    for i, (d, c) in enumerate(zip(sessions, closes, strict=True)):
        con.execute(
            "INSERT INTO daily_bars (symbol, date, open, high, low, close, vwap, volume, transactions) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 10)", [symbol, d, c, c + 0.5, c - 0.5, c, c, volume])


def test_is_10k_from_edgar_with_etrade_fallback_for_unmapped(roots):
    dbp = roots / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 2, 9))
    with db.connect(dbp) as con:
        for s in ("MAP", "UNM"):
            _symbol(con, s)
            _bars(con, s, sess, [10.0] * len(sess))
        con.execute("INSERT INTO sec_edgar_tickers VALUES ('MAP', 5, 'MAP', 'Map', now())")
        for acc, form, filed in (("a1", "10-K", "2024-01-19"), ("a2", "10-K/A", "2024-01-23"),
                                 ("a3", "10-Q", "2024-02-02")):
            con.execute("INSERT INTO sec_edgar_filings (cik, accession, form, filing_date) VALUES (5, ?, ?, ?)",
                        [acc, form, filed])
        # E*TRADE rows: ignored for MAP (it is mapped), used for UNM
        for sym in ("MAP", "UNM"):
            con.execute("INSERT INTO sec_filings (xid, symbol, form_type, date_filed, html_doc_key) "
                        "VALUES (?, ?, '10-K', DATE '2024-01-30', 'k')", [ord(sym[0]), sym])
    features.derive_features(db_path=dbp)
    with db.connect(dbp) as con:
        rows = con.execute("SELECT symbol, date, is_10k, is_10q FROM daily_bars "
                           "WHERE is_10k = 1 OR is_10q = 1 ORDER BY 1, 2").fetchall()
        nx = con.execute("SELECT vwap_nx_qtr IS NOT NULL FROM daily_bars WHERE symbol = 'MAP' "
                         "AND date = DATE '2024-01-16'").fetchone()[0]
    assert rows == [("MAP", date(2024, 1, 19), 1, 0), ("MAP", date(2024, 2, 2), 0, 1),
                    ("UNM", date(2024, 1, 30), 1, 0)]
    assert nx  # vwap_nx_qtr reads the same filing days


def _market_db(path, *, snapshots: bool):
    """Two CS symbols in sectors 53 and 54; B has a 2-for-1 split on day 3 that the bar
    history has not applied (raw close halves)."""
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(path) as con:
        for s in ("A", "B"):
            _symbol(con, s)
        con.execute("INSERT INTO symbol_industry (symbol, industry_code) VALUES ('A', '531010'), ('B', '541010')")
        a = [100.0, 110.0, 110.0, 99.0, 99.0, 99.0, 99.0, 99.0]
        b = [50.0, 50.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
        _bars(con, "A", sess[:8], a)
        _bars(con, "B", sess[:8], b)
        con.execute("INSERT INTO stock_splits VALUES ('s', 'B', ?, 1, 2, now())", [sess[2]])
        con.execute("UPDATE daily_bars SET ingested_at = TIMESTAMP '2024-01-17 12:00:00'")
        if snapshots:
            con.execute("CREATE TABLE share_snapshots (symbol TEXT, snap_date DATE, market_cap DOUBLE, "
                        "weighted_shares_outstanding DOUBLE, close_at_snap DOUBLE)")
            con.execute("INSERT INTO share_snapshots VALUES ('A', DATE '2024-01-10', 1000, 10, 100), "
                        "('B', DATE '2024-01-10', 3000, 60, 50)")
    return sess


def test_market_and_sector_returns_with_a_split(roots):
    dbp = roots / "f.duckdb"
    sess = _market_db(dbp, snapshots=True)
    out = features.derive_features(db_path=dbp)
    assert out["market_weighting"] == "cap"
    with db.connect(dbp) as con:
        got = {(d, sc): (r, n) for d, sc, r, n in con.execute(
            "SELECT date, sector_code, cap_ret, n FROM market_daily").fetchall()}
    # day 2 (sess[1]): A +10%, B 0%; caps at t-1 (sess[0]): A 1000, B 3000 -> (100)/4000
    assert got[(sess[1], "00")][0] == pytest.approx(100 / 4000)
    assert got[(sess[1], "53")] == (pytest.approx(0.10), 1)
    assert got[(sess[1], "54")][0] == pytest.approx(0.0)
    # day 3: B halves raw but the split factor makes it flat (adjusted close 50 -> 50)
    assert got[(sess[2], "54")][0] == pytest.approx(0.0)
    # cap at t-1 follows adjusted price: A 1000*110/100 = 1100, B 3000 -> A flat on day 3
    assert got[(sess[2], "00")][0] == pytest.approx(0.0)
    # day 4: A -10%; weights A 1100, B 3000
    assert got[(sess[3], "00")][0] == pytest.approx(-0.10 * 1100 / 4100)
    assert (sess[0], "00") not in got  # no previous session
    assert got[(sess[1], "00")][1] == 2


def test_market_returns_fall_back_to_equal_weight_without_share_snapshots(roots, capsys):
    dbp = roots / "f.duckdb"
    sess = _market_db(dbp, snapshots=False)
    out = features.derive_features(db_path=dbp)
    assert out["market_weighting"] == "equal"
    assert "equal-weighted" in capsys.readouterr().out
    with db.connect(dbp) as con:
        r = con.execute("SELECT cap_ret FROM market_daily WHERE sector_code = '00' AND date = ?",
                        [sess[1]]).fetchone()[0]
    assert r == pytest.approx(0.05)


def test_macro_series_availability_and_bond_holiday_ffill():
    sessions = [date(2024, 2, 29), date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6),
                date(2024, 5, 1)]
    # no yield on 03-04 (bond holiday); yields dated by session
    yields = [(date(2024, 2, 29), 5.0, 4.0, 4.2), (date(2024, 3, 1), 5.1, 4.1, 4.4),
              (date(2024, 3, 5), 5.1, 4.3, 4.6)]
    infl = [(date(2023, m, 1), 100.0 + m, 200.0 + m, None) for m in range(1, 13)]
    infl += [(date(2024, 1, 1), 110.0, 210.0, None), (date(2024, 2, 1), 111.0, 212.0, None),
             (date(2024, 3, 1), 112.0, 214.0, None)]
    out = features.macro_series(sessions, yields, infl, lag=1)
    s0, s1, s2, s3, s4, s5 = sessions
    assert out[s0]["ust_10y_2y_slope"] is None  # nothing before it
    # 03-01 uses the 02-29 yield; no previous used value, so no change yet
    assert out[s1]["ust_10y_chg_bp"] is None
    assert out[s1]["ust_10y_2y_slope"] == pytest.approx(0.2)
    # 03-04 uses 03-01: +20bp on the 10y
    assert out[s2]["ust_10y_chg_bp"] == pytest.approx(20.0)
    assert out[s2]["ust_10y_3m_slope"] == pytest.approx(4.4 - 5.1)
    # 03-05 uses 03-04's yield: no row (bond holiday), so 03-01 carries and nothing changes
    assert out[s3]["ust_10y_chg_bp"] == pytest.approx(0.0)
    # 03-06 uses the 03-05 yield: 2y +20bp
    assert out[s4]["ust_2y_chg_bp"] == pytest.approx(20.0)
    assert out[s4]["ust_10y_z252"] is None  # too little history
    # CPI: January is first usable on the first session of March
    assert out[s0]["cpi_yoy"] is None
    assert out[s1]["cpi_yoy"] == pytest.approx(110.0 / 101.0 - 1)
    assert out[s1]["core_cpi_mom"] == pytest.approx(210.0 / 212.0 - 1)  # Jan vs Dec core
    assert out[s5]["cpi_yoy"] == pytest.approx(112.0 / 103.0 - 1)  # March data on May 1


def test_cpi_is_absent_before_month_plus_two():
    sessions = [date(2024, 2, 29), date(2024, 3, 1)]
    infl = [(date(2023, 1, 1), 100.0, 200.0, None), (date(2024, 1, 1), 110.0, 210.0, None)]
    out = features.macro_series(sessions, [], infl)
    assert out[sessions[0]]["cpi_yoy"] is None  # January is not usable in February
    assert out[sessions[1]]["cpi_yoy"] == pytest.approx(0.10)
    assert out[sessions[1]]["ust_10y_chg_bp"] is None  # no yields table rows


def test_macro_z252_is_past_only():
    sessions = [date(2020, 1, 1) + __import__("datetime").timedelta(days=i) for i in range(300)]
    yields = [(d, 1.0, 1.0, 1.0 + (i % 5) * 0.1) for i, d in enumerate(sessions)]
    out = features.macro_series(sessions, yields, [])
    i = 200
    hist = [1.0 + ((k - 1) % 5) * 0.1 for k in range(max(1, i - 252), i)]  # used values up to i-1
    y = 1.0 + ((i - 1) % 5) * 0.1
    prior = [1.0 + ((k - 1) % 5) * 0.1 for k in range(max(1, i - 252), i)]
    mu = sum(prior) / len(prior)
    sd = math.sqrt(sum((v - mu) ** 2 for v in prior) / len(prior))
    assert hist and out[sessions[i]]["ust_10y_z252"] == pytest.approx((y - mu) / sd)


def test_derive_features_writes_macro_daily_from_tables(roots):
    dbp = roots / "f.duckdb"
    sess = tc.sessions(date(2024, 3, 1), date(2024, 3, 8))
    with db.connect(dbp) as con:
        _symbol(con, "A")
        _bars(con, "A", sess, [10.0] * len(sess))
        con.execute("CREATE TABLE treasury_yields (date DATE, yield_1_month DOUBLE, yield_3_month DOUBLE, "
                    "yield_6_month DOUBLE, yield_1_year DOUBLE, yield_2_year DOUBLE, yield_3_year DOUBLE, "
                    "yield_5_year DOUBLE, yield_7_year DOUBLE, yield_10_year DOUBLE, yield_20_year DOUBLE, "
                    "yield_30_year DOUBLE)")
        for i, d in enumerate(sess):
            con.execute("INSERT INTO treasury_yields (date, yield_3_month, yield_2_year, yield_10_year) "
                        "VALUES (?, 5.0, 4.0, ?)", [d, 4.0 + i * 0.1])
    out = features.derive_features(db_path=dbp)
    assert out["macro_daily_rows"] == len(sess)
    with db.connect(dbp) as con:
        rows = con.execute("SELECT date, ust_10y_chg_bp, ust_10y_2y_slope, cpi_yoy FROM macro_daily "
                           "ORDER BY date").fetchall()
    assert rows[0][1] is None and rows[2][1] == pytest.approx(10.0)
    assert rows[2][2] == pytest.approx(4.1 - 4.0)
    assert all(r[3] is None for r in rows)  # no inflation table: NULL


# --- pre-market columns, panel.next, export-next ---------------------------------------


def _pm_db(path):
    """AAA: five sessions with pre-market data at 09:25. A split executes on sessions[3];
    sessions[4] has no previous bar problem; BBB is missing sessions[1] (gap before [2])."""
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(path) as con:
        for s in ("AAA", "BBB"):
            _symbol(con, s)
        _bars(con, "AAA", sess, [10.0] * len(sess), volume=100.0)
        _bars(con, "BBB", [d for i, d in enumerate(sess) if i != 1], [20.0] * (len(sess) - 1), volume=100.0)
        con.execute("UPDATE daily_bars SET pre_market_tpvwap_0925 = close * 1.02, "
                    "pre_market_volume_0925 = 10.0, regular_vwap = close, after_hours_vwap = close * 1.01, "
                    "after_hours_volume = 5.0, ext_fetched_at = TIMESTAMP '2024-03-01 00:00:00'")
        con.execute("UPDATE daily_bars SET pre_market_tpvwap_0925 = NULL, pre_market_volume_0925 = NULL "
                    "WHERE symbol = 'AAA' AND date = ?", [sess[5]])
        con.execute("INSERT INTO stock_splits VALUES ('s1', 'AAA', ?, 1, 1, now())", [sess[3]])
    return sess


def _export(tmp_path, dbp, **kw):
    out = tmp_path / "out" / "tft" / "panel.parquet"
    summary = export.export_tft(db_path=dbp, out_path=out, news_db_path=tmp_path / "none.duckdb",
                                include_inference=True, **{**_LOOSE, **kw})
    return summary, pq.read_table(out).to_pylist(), out


def test_premarket_after_hours_and_shape_columns(roots):
    dbp = roots / "f.duckdb"
    sess = _pm_db(dbp)
    features.derive_features(db_path=dbp)
    summary, rows, out = _export(roots, dbp)
    by = {(r["symbol"], r["date"]): r for r in rows}
    r = by[("AAA", sess[2])]
    assert r["pm_known"] == 1
    assert r["pm_vwap_pct_prev_vwap"] == pytest.approx(0.02)
    assert r["pm_vol_pct_10d"] == pytest.approx(math.log(1 + 100 * 10 / 100))
    assert r["ah_known"] == 1 and r["ah_vwap_pct_rth_vwap"] == pytest.approx(0.01)
    assert r["close_pct_rth_vwap"] == pytest.approx(0.0)
    assert r["high_position_pct"] == pytest.approx(0.05) and r["low_position_pct"] == pytest.approx(-0.05)
    assert r["gap_open_pct"] == pytest.approx(0.0) and r["gap_open_vs_prev_vwap_pct"] == pytest.approx(0.0)
    # a split executes on sessions[3]: pm_known = 0 and the values are masked
    s = by[("AAA", sess[3])]
    assert (s["pm_known"], s["pm_vwap_pct_prev_vwap"], s["pm_vol_pct_10d"]) == (0, 0.0, 0.0)
    # no pre-market data that day
    n = by[("AAA", sess[5])]
    assert n["pm_known"] == 0 and n["pm_vwap_pct_prev_vwap"] == 0.0
    # BBB lacks sessions[1], so sessions[2] has no previous session exactly one back
    g = by[("BBB", sess[2])]
    assert g["pm_known"] == 0 and g["gap_open_pct"] == 0.0
    assert by[("BBB", sess[3])]["pm_known"] == 1
    # the first bar has no previous session
    assert by[("AAA", sess[0])]["pm_known"] == 0
    # market/macro without data: zeros, not NULL
    assert r["mkt_cap_ret"] is not None and r["ust_10y_chg_bp"] == 0.0


def test_pm_split_factor_uses_the_ext_fetch_time(roots):
    dbp = roots / "f.duckdb"
    sess = tc.sessions(date(2024, 1, 16), date(2024, 1, 26))
    with db.connect(dbp) as con:
        _symbol(con, "AAA")
        _bars(con, "AAA", sess, [1.0] * len(sess), volume=100.0)
        # 1-for-10 reverse split on sessions[4] never applied to the bars (factor 10), but the
        # extended-hours rows were fetched after it (already adjusted: factor 1)
        con.execute("UPDATE daily_bars SET close = CASE WHEN date >= ? THEN 10 ELSE 1 END, "
                    "vwap = CASE WHEN date >= ? THEN 10 ELSE 1 END, "
                    "ingested_at = TIMESTAMP '2024-01-10 00:00:00'", [sess[4], sess[4]])
        con.execute("UPDATE daily_bars SET pre_market_tpvwap_0925 = 10.2, pre_market_volume_0925 = 1.0, "
                    "ext_fetched_at = TIMESTAMP '2024-02-01 00:00:00'")
        con.execute("INSERT INTO stock_splits VALUES ('s', 'AAA', ?, 10, 1, now())", [sess[4]])
    features.derive_features(db_path=dbp)
    _, rows, _ = _export(roots, dbp)
    by = {r["date"]: r for r in rows}
    assert by[sess[2]]["pm_known"] == 1
    assert by[sess[2]]["pm_vwap_pct_prev_vwap"] == pytest.approx(0.02)  # 10.2 vs adjusted 10


def test_schema4_meta_export_id_and_panel_next(roots):
    dbp = roots / "f.duckdb"
    sess = _pm_db(dbp)
    features.derive_features(db_path=dbp)
    summary, rows, out = _export(roots, dbp)
    meta = json.loads(Path(summary["meta_path"]).read_text())
    assert meta["schema_version"] == 4 == export.SCHEMA_VERSION
    assert meta["premarket_cutoff"] == "09:25"
    cols = set(rows[0])
    for c in ("mkt_cap_ret", "own_sector_cap_ret", "sec_50_cap_ret", "sec_63_cap_ret", "ust_10y_z252",
              "core_cpi_mom", "ah_known", "pm_known"):
        assert c in cols
    nxt = pq.read_table(export.next_panel_path(out)).to_pylist()
    assert {r["symbol"] for r in nxt} == {"AAA", "BBB"}
    assert {r["export_id"] for r in nxt} == {meta["export_id"]}
    assert {r["day_idx"] for r in nxt} == {meta["next_session"]["day_idx"]}
    assert all(r["pm_known"] == 0 and r["pm_vwap_pct_prev_vwap"] == 0.0 and r["pm_vol_pct_10d"] == 0.0
               for r in nxt)
    # a second export gets a new id
    again, _, _ = _export(roots, dbp)
    assert again["export_id"] != meta["export_id"]


def test_market_columns_fall_back_to_the_market_for_own_sector(roots):
    dbp = roots / "f.duckdb"
    sess = _market_db(dbp, snapshots=True)
    with db.connect(dbp) as con:
        _symbol(con, "C")
        _bars(con, "C", sess[:8], [10.0] * 8)  # no industry
    features.derive_features(db_path=dbp)
    _, rows, _ = _export(roots, dbp)
    by = {(r["symbol"], r["date"]): r for r in rows}
    a, c = by[("A", sess[1])], by[("C", sess[1])]
    assert a["own_sector_cap_ret"] == pytest.approx(a["sec_53_cap_ret"])
    assert c["own_sector_cap_ret"] == pytest.approx(c["mkt_cap_ret"])
    assert a["sec_50_cap_ret"] == pytest.approx(a["mkt_cap_ret"])  # sector with no stocks: market


def _live_file(root, d, recs):
    p = root / "out" / "premarket_live" / f"premarket-{d}.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_export_next_fills_pm_columns_from_the_live_file(roots):
    dbp = roots / "f.duckdb"
    sess = _pm_db(dbp)
    features.derive_features(db_path=dbp)
    summary, _, out = _export(roots, dbp)
    meta = json.loads(Path(summary["meta_path"]).read_text())
    d = meta["next_session"]["date"]
    base = {"date": d, "cutoff": "09:25", "source": "etrade", "fetched_at": "x"}
    _live_file(roots, d, [
        {**base, "symbol": "AAA", "pm_tpvwap": 10.3, "pm_volume": 5.0, "vol_10d_mean_source": 50.0},
        {**base, "symbol": "BBB", "pm_tpvwap": None, "pm_volume": None, "vol_10d_mean_source": 50.0},
    ])
    with config_db(dbp):
        res = export.export_next(panel=out, db_path=dbp)
    assert res["pm_known"] == 1 and res["export_id"] == meta["export_id"]
    nxt = {r["symbol"]: r for r in pq.read_table(export.next_panel_path(out)).to_pylist()}
    # vwap(d-1) is the last session's 10.0; volume ratio uses the file's own 10-day mean
    assert nxt["AAA"]["pm_vwap_pct_prev_vwap"] == pytest.approx(10.3 / 10.0 - 1)
    assert nxt["AAA"]["pm_vol_pct_10d"] == pytest.approx(math.log(1 + 100 * 5 / 50))
    assert nxt["AAA"]["pm_known"] == 1 and nxt["AAA"]["export_id"] == meta["export_id"]
    assert nxt["BBB"]["pm_known"] == 0 and nxt["BBB"]["pm_vwap_pct_prev_vwap"] == 0.0


def test_export_next_refuses_a_cutoff_mismatch_and_masks_a_split_day(roots):
    dbp = roots / "f.duckdb"
    _pm_db(dbp)
    features.derive_features(db_path=dbp)
    summary, _, out = _export(roots, dbp)
    d = json.loads(Path(summary["meta_path"]).read_text())["next_session"]["date"]
    rec = {"symbol": "AAA", "date": d, "cutoff": "09:00", "pm_tpvwap": 10.3, "pm_volume": 5.0,
           "vol_10d_mean_source": 50.0, "source": "etrade", "fetched_at": "x"}
    p = _live_file(roots, d, [rec])
    with pytest.raises(RuntimeError, match="cutoff"):
        export.export_next(panel=out, db_path=dbp)
    p.write_text(json.dumps({**rec, "cutoff": "09:25"}) + "\n")
    with db.connect(dbp) as con:  # a split executes on the forecast day
        con.execute("INSERT INTO stock_splits VALUES ('s2', 'AAA', ?, 1, 2, now())", [date.fromisoformat(d)])
    export.export_next(panel=out, db_path=dbp)
    nxt = {r["symbol"]: r for r in pq.read_table(export.next_panel_path(out)).to_pylist()}
    assert nxt["AAA"]["pm_known"] == 0 and nxt["AAA"]["pm_vwap_pct_prev_vwap"] == 0.0


class config_db:  # noqa: N801 - tiny context manager; keeps the settings cache untouched
    def __init__(self, _path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_premarket_cutoff_setting_is_validated():
    assert features.cutoff_key("09:25") == "0925"
    with pytest.raises(ValueError):
        features.cutoff_key("09:20")
    assert config.settings is not None
