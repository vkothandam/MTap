"""Pre-market / after-hours summaries (sourcing_py/massive_ext): offline, no API calls."""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from sourcing_py.common import config, schema
from sourcing_py.massive_ext import extended as ext


def _bar(d: str, hhmm: str, o: float, c: float, v: float) -> dict:
    t = datetime.fromisoformat(f"{d}T{hhmm}:00").replace(tzinfo=ext.NY)
    return {"t": int(t.timestamp() * 1000), "o": o, "h": max(o, c), "l": min(o, c), "c": c, "v": v}


@pytest.fixture
def roots(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    return tmp_path


def test_sessions_split_at_the_open_and_the_close_minute():
    day = "2026-10-14"
    bars = [_bar(day, "04:00", 10.0, 10.1, 100), _bar(day, "09:29", 10.4, 10.5, 50),
            _bar(day, "09:30", 10.6, 10.7, 1000), _bar(day, "15:59", 11.0, 11.1, 900),
            _bar(day, "16:00", 11.1, 11.2, 400), _bar(day, "19:59", 11.3, 11.4, 25)]
    (r,) = ext.summarize("ABC", [bars[:3], bars[3:]], "2026-10-15T04:00:00+00:00")  # two pages
    assert r["date"] == day and r["minute_bars"] == 6
    assert (r["pre_market_price"], r["pre_market_last"], r["pre_market_volume"]) == (10.0, 10.5, 150)
    assert r["regular_volume"] == 1900
    assert (r["after_hours_price"], r["after_hours_volume"]) == (11.4, 425)  # the 16:00 minute counts


def test_a_half_day_closes_at_one_and_a_quiet_day_has_no_prices():
    half = "2025-12-24"  # 13:00 close
    (r,) = ext.summarize("ABC", [[_bar(half, "12:59", 5, 5, 10), _bar(half, "13:00", 5, 5.5, 7)]], "x")
    assert r["regular_volume"] == 10 and r["after_hours_volume"] == 7 and r["after_hours_price"] == 5.5
    (q,) = ext.summarize("ABC", [[_bar("2026-10-14", "10:00", 5, 5, 10)]], "x")
    assert q["pre_market_price"] is None and q["after_hours_price"] is None
    assert q["pre_market_volume"] == 0 and q["after_hours_volume"] == 0
    schema.validate_records([r, q], schema.load_schema(config.load_source(ext.SOURCE).schema_path))


def test_universe_keeps_listed_and_recently_delisted(roots):
    d = roots / "out" / "tickers_cs"
    d.mkdir(parents=True)
    rows = [{"ticker": "OLD", "active": False, "delisted_utc": "2019-01-02T00:00:00Z"},
            {"ticker": "GONE", "active": False, "delisted_utc": "2023-05-01T00:00:00Z"},
            {"ticker": "LIVE", "active": True, "delisted_utc": None}]
    (d / "tickers_cs-2026-10-01.jsonl").write_text(json.dumps({"ticker": "STALE", "active": True}) + "\n")
    (d / "tickers_cs-2026-10-15.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ext.universe(date(2021, 10, 15)) == ["GONE", "LIVE"]


def test_fetch_stages_resumes_and_writes_per_date_files(roots, monkeypatch):
    calls: list[str] = []

    def fake(client, cfg, ticker, d0, d1):
        calls.append(ticker)
        if ticker == "BAD":
            raise RuntimeError("HTTP 502 apiKey=secret")
        return [{**ext.summarize(ticker, [[_bar(d, "16:30", 1, 2, 3)]], "f")[0]}
                for d in ("2026-10-13", "2026-10-14")] if ticker != "EMPTY" else []

    monkeypatch.setattr(ext, "fetch_ticker", fake)
    d0, d1 = date(2026, 10, 13), date(2026, 10, 14)
    assert ext.fetch(d0, d1, tickers=["AAA", "BAD", "EMPTY"], workers=2) == 1
    assert not (roots / "out" / "extended_hours").exists()  # nothing written until all are in
    monkeypatch.setattr(ext, "fetch_ticker", lambda c, cfg, t, a, b: [] if t == "BAD" else fake(c, cfg, t, a, b))
    calls.clear()
    assert ext.fetch(d0, d1, tickers=["AAA", "BAD", "EMPTY"], workers=2) == 0
    assert calls == []  # AAA and EMPTY were staged; only BAD went again (through the lambda)
    out = sorted((roots / "out" / "extended_hours").iterdir())
    assert [p.name for p in out] == ["extended_hours-2026-10-13.jsonl", "extended_hours-2026-10-14.jsonl"]
    rows = [json.loads(line) for line in out[0].read_text().splitlines()]
    assert [r["ticker"] for r in rows] == ["AAA"] and rows[0]["after_hours_volume"] == 3
    assert rows[0]["_source"] == ext.SOURCE
    assert not ext.job_dir(d0, d1).exists()


def test_finalize_merges_into_an_existing_date_file(roots, monkeypatch):
    monkeypatch.setattr(ext, "fetch_ticker",
                        lambda c, cfg, t, a, b: ext.summarize(t, [[_bar("2026-10-14", "08:00", 1, 1, 5)]], "f"))
    d = date(2026, 10, 14)
    ext.fetch(d, d, tickers=["AAA", "BBB"], workers=1)
    monkeypatch.setattr(ext, "fetch_ticker",
                        lambda c, cfg, t, a, b: ext.summarize(t, [[_bar("2026-10-14", "08:00", 1, 1, 9)]], "f"))
    ext.fetch(d, d, tickers=["BBB"], workers=1)
    rows = {r["ticker"]: r for r in map(json.loads, ext.out_path(d).read_text().splitlines())}
    assert rows["AAA"]["pre_market_volume"] == 5 and rows["BBB"]["pre_market_volume"] == 9


def test_gaps_are_sessions_with_daily_bars_but_no_extended_file(roots):
    daily = roots / "out" / "daily_summary"
    daily.mkdir(parents=True)
    for d in ("2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14"):
        (daily / f"daily_summary-{d}.jsonl").write_text("{}\n")
    (roots / "out" / "extended_hours").mkdir()
    (roots / "out" / "extended_hours" / "extended_hours-2026-10-12.jsonl").write_text("")
    gaps = ext.missing_sessions(date(2026, 10, 15), days=7)
    assert gaps == [date(2026, 10, 9), date(2026, 10, 13), date(2026, 10, 14)]  # 10-15 is today
    assert ext.missing_sessions(date(2026, 10, 15), days=7, cap=1) == [date(2026, 10, 14)]


def test_tickers_list_is_refreshed_only_when_a_week_old(roots, monkeypatch):
    from sourcing_py.sources.massive_tickers_cs import source as tickers

    runs = []
    monkeypatch.setattr(tickers.MassiveTickersCsSource, "run", lambda self, params: runs.append(1) or "new")
    assert ext.refresh_tickers(date(2026, 10, 15)) == "new"  # none yet
    d = roots / "out" / "tickers_cs"
    d.mkdir(parents=True)
    (d / "tickers_cs-2026-10-08.jsonl").write_text("")
    assert ext.refresh_tickers(date(2026, 10, 15)) is None  # 7 days: still fresh
    assert ext.refresh_tickers(date(2026, 10, 16)) == "new"
    assert len(runs) == 2
