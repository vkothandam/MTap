"""Live pre-market (etrade/premarket.py): parsing, cutoff, VWAP, auth stop, Massive fallback."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from sourcing_py.etrade import premarket as pm

D = date(2026, 10, 27)
UTC = timezone.utc
BUCKET = timedelta(seconds=86.4)


def _ny(hh: int, mm: int, d: date = D) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=pm.NY).astimezone(UTC)


def minute_fixture() -> dict:
    """Synthetic probe-shaped response: 86.4 s buckets from the previous session's close,
    through 04:00 (D) and past the 09:25 cutoff. Close = 100 before 04:00, then 10/20/30..."""
    series = []
    t = _ny(15, 0, D - timedelta(days=1))
    end = _ny(9, 35)
    while t < end:
        pre = t >= _ny(4, 0)
        series.append({"DT": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "Open": 1, "High": 12 if pre else 101,
                       "Low": 6 if pre else 99, "Close": 9 if pre else 100, "Volume": 100 if pre else 5000})
        t += BUCKET
    return {"data": {"chartData": {"series": series}}}


def day_fixture() -> dict:
    days = [D - timedelta(days=i) for i in range(14, -1, -1) if (D - timedelta(days=i)).weekday() < 5]
    series = [{"DT": x.isoformat(), "Open": 1, "High": 2, "Low": 1, "Close": 2, "Volume": 1000 + i}
              for i, x in enumerate(days)]
    return {"data": {"chartData": {"series": series}}}


def test_parse_and_cutoff_window():
    bars = pm.parse_series(minute_fixture())
    assert bars == sorted(bars, key=lambda b: b.dt)
    st = pm.premarket_stats(bars, D, "09:25")
    lo, hi = pm.window(D, "09:25")
    inwin = [b for b in bars if lo <= b.dt < hi]
    assert st["n_bars"] == len(inwin) and 224 <= len(inwin) <= 227  # 5h25m / 86.4 s = 225.7
    assert st["last_bar_dt"] < hi and st["last_bar_dt"] >= hi - BUCKET
    assert st["pm_volume"] == 100 * len(inwin)
    # typical price = (12 + 6 + 9) / 3 = 9, constant, so the VWAP is 9; yesterday's bars excluded
    assert st["pm_tpvwap"] == pytest.approx(9.0)
    # an earlier cutoff drops later bars
    assert pm.premarket_stats(bars, D, "08:00")["n_bars"] < st["n_bars"]


def test_typical_price_weighting():
    t0 = _ny(5, 0)
    bars = [pm.Bar(t0, 11, 9, 10, 100), pm.Bar(t0 + BUCKET, 22, 18, 20, 300)]
    st = pm.premarket_stats(bars, D, "09:25")
    assert st["pm_tpvwap"] == pytest.approx((10 * 100 + 20 * 300) / 400)
    assert pm.premarket_stats([], D, "09:25")["pm_tpvwap"] is None


def test_mean_volume_excludes_day_d_and_takes_10():
    bars = pm.parse_series(day_fixture())
    prior = [b for b in bars if b.dt.date() < D]
    expect = sum(b.volume for b in prior[-10:]) / 10
    assert pm.mean_volume_before(bars, D) == pytest.approx(expect)
    assert pm.mean_volume_before([], D) is None


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    monkeypatch.setenv("MASSIVE_RATE_LIMIT_PER_MIN", "0")
    return tmp_path


def test_token_file_then_session_fallback(env):
    tok = env / "state" / "etrade" / "bearer.json"
    sess = env / "state" / "etrade" / "session.json"
    tok.parent.mkdir(parents=True)
    with pytest.raises(pm.AuthStop):
        pm.load_token(tok, sess)
    sess.write_text(json.dumps({"requestHeaders": {"authorization": "Bearer SESS"}}))
    assert pm.load_token(tok, sess) == "Bearer SESS"
    tok.write_text(json.dumps({"token": "Bearer FILE", "captured_at": "x"}))
    assert pm.load_token(tok, sess) == "Bearer FILE"


def _etrade_handler(requests: list, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.headers["authorization"], body))
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json=minute_fixture() if body["dataPeriod"] == "Minute" else day_fixture())
    return handler


NOW = _ny(9, 28)


def test_run_etrade_writes_records(env):
    reqs: list = []
    client = httpx.Client(transport=httpx.MockTransport(_etrade_handler(reqs)))
    s = pm.run_premarket(D, {"AAA": 11, "BBB": 22}, source="etrade", token="Bearer T", client=client, now=NOW)
    assert s["written"] == 2 and s["by_source"] == {"etrade_chartiq": 2} and s["auth_error"] is None
    assert len(reqs) == 4 and {r[0] for r in reqs} == {"Bearer T"}
    minute = [b for _, b in reqs if b["dataPeriod"] == "Minute"][0]
    assert minute["extendedHours"] is True
    assert [e["symbol"] for e in minute["elements"]] == [minute["elements"][0]["symbol"]] * 2
    assert minute["dateStart"].startswith("2026-10-27T08:00:00")  # 04:00 EDT in UTC
    rec = [json.loads(x) for x in open(s["path"])][0]
    assert rec["pm_tpvwap"] == pytest.approx(9.0) and rec["vol_10d_mean_source"] > 1000
    assert rec["source"] == "etrade_chartiq" and rec["cutoff"] == "09:25" and rec["n_bars"] > 200
    assert s["newest_bar_lag_s"] == pytest.approx((NOW - _ny(9, 25)).total_seconds(), abs=90)
    assert s["latency_s"]["calls"] == 4


def test_auth_failure_stops_whole_job_without_retry(env):
    reqs: list = []
    client = httpx.Client(transport=httpx.MockTransport(_etrade_handler(reqs, status=401)))
    s = pm.run_premarket(D, {f"S{i}": i for i in range(40)}, source="etrade", token="Bearer T",
                         client=client, now=NOW)
    assert s["written"] == 0 and s["auth_error"] and s["path"] is None
    assert len(reqs) <= 4  # at most the in-flight calls; no retry, nothing after the stop


def test_auth_failure_falls_back_to_massive(env):
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "getChartiqFeed" in str(request.url):
            return httpx.Response(403)
        seen.append(str(request.url))
        if "/range/1/minute/" in str(request.url):
            results = [{"t": int(_ny(5, 0).timestamp() * 1000), "h": 11, "l": 9, "c": 10, "v": 200},
                       {"t": int(_ny(9, 25).timestamp() * 1000), "h": 99, "l": 99, "c": 99, "v": 999}]
        else:
            results = [{"t": 0, "h": 1, "l": 1, "c": 1, "v": 500}]
        return httpx.Response(200, json={"results": results})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    s = pm.run_premarket(D, {"AAA": 1}, source="auto", token="Bearer T", client=client, now=NOW)
    assert s["fallback_used"] is True and s["by_source"] == {"massive": 1}
    rec = [json.loads(x) for x in open(s["path"])][0]
    assert rec["source"] == "massive" and rec["pm_volume"] == 200 and rec["pm_tpvwap"] == pytest.approx(10.0)
    assert rec["xid"] == 1


def test_missing_token_in_auto_mode_uses_massive(env, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    s = pm.run_premarket(D, {"AAA": 1}, source="auto", client=client, now=NOW)  # no token files
    assert s["auth_error"] and s["fallback_used"] and s["written"] == 1
    rec = [json.loads(x) for x in open(s["path"])][0]
    assert rec["n_bars"] == 0 and rec["pm_tpvwap"] is None


def test_xids_file(env, tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("AAA,11\nBBB,22\nbad,x\n")
    assert pm.read_xids_file(f) == {"AAA": 11, "BBB": 22}
    assert pm.load_universe(["BBB", "ZZZ"], f) == {"BBB": 22}


def test_delayed_feed_before_cutoff_is_incomplete():
    bars = [pm.Bar(_ny(5, 0), 11, 9, 10, 100)]
    early = [pm._record("AAA", 1, D, "09:25", bars, [], "massive", NOW)]
    pm.mark_complete(early, D, "09:25", _ny(9, 30), 15)  # Massive at 09:30 reaches only 09:15
    assert early[0]["complete"] is False and "_newest" not in early[0]
    late = [pm._record("AAA", 1, D, "09:25", bars, [], "massive", NOW)]
    pm.mark_complete(late, D, "09:25", _ny(9, 41), 15)
    assert late[0]["complete"] is True
    # E*TRADE: judged by the newest bar in the responses (a delayed token stops short)
    et = [pm._record(s, 1, D, "09:25", [pm.Bar(_ny(9, 12), 1, 1, 1, 1)], [], "etrade_chartiq", NOW)
          for s in ("A", "B", "C")]
    until = pm.mark_complete(et, D, "09:25", _ny(9, 28), 15)
    assert not any(r["complete"] for r in et) and until["etrade_chartiq"].startswith("2026-10-27T13:12")
