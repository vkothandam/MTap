"""Morning live prices (etrade/premarket.py): token, multi-symbol parsing, watchlist -> ChartIQ
fallback, open, auth stop. Fixtures only; no network."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from sourcing_py.common import config, schema
from sourcing_py.etrade import premarket as pm

D = date(2026, 10, 27)
UTC = timezone.utc


def _ny(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(D.year, D.month, D.day, hh, mm, ss, tzinfo=pm.NY).astimezone(UTC)


def _dt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def batch_fixture(xids: list[int]) -> dict:
    """Probe-shaped batched response: the first symbol at the top level of each bar, the rest
    nested under their XID. Pre-market closes 10.0, 10.1 ..; regular opens at 09:30."""
    series = []
    for i, t in enumerate([_ny(9, 27), _ny(9, 28, 26), _ny(9, 29, 53), _ny(9, 31, 20), _ny(9, 32, 46)]):
        row = {"DT": _dt(t), "Open": 10 + i, "Close": 10.5 + i}
        for k, x in enumerate(xids[1:], start=1):
            row[str(x)] = {"Open": 100 * k + i, "Close": 100 * k + i + 0.5}
        series.append(row)
    return {"data": {"chartData": {"series": series}}}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    tok = tmp_path / "state" / "etrade"
    tok.mkdir(parents=True)
    (tok / "bearer.json").write_text(json.dumps({"token": "abc"}))
    return tmp_path


def test_parse_prices_top_level_and_nested():
    out = pm.parse_prices(batch_fixture([1, 2, 3]), [1, 2, 3])
    assert [b.close for b in out[1]] == [10.5, 11.5, 12.5, 13.5, 14.5]
    assert [b.close for b in out[2]] == [100.5, 101.5, 102.5, 103.5, 104.5]
    assert out[3][0].open == 200
    # a bare number nested under the XID is read as a close
    resp = {"data": {"chartData": {"series": [{"DT": _dt(_ny(9, 0)), "Close": 1.0, "7": 5.5}]}}}
    assert pm.parse_prices(resp, [1, 7])[7][0].close == 5.5


def test_last_before_cutoff_and_first_regular():
    bars = pm.parse_prices(batch_fixture([1]), [1])[1]
    now = _ny(9, 33)
    last = pm.last_before(bars, pm.cutoff_utc(D, "09:30"), now)
    assert last.close == 12.5 and last.dt == _ny(9, 29, 53)  # the bucket starting before 09:30
    first = pm.first_regular(bars, D, now)
    assert first.open == 13 and first.dt == _ny(9, 31, 20)
    assert pm.first_regular(bars, D, _ny(9, 30)) is None      # nothing printed yet
    assert pm.last_before(bars, pm.cutoff_utc(D, "09:30"), _ny(9, 28)).close == 10.5  # not the future


def test_watchlist_mid_and_open_rules():
    assert pm.watchlist_price({"bid": 10.0, "ask": 10.2, "price": 9.0})[0] == pytest.approx(10.1)
    assert pm.watchlist_price({"price": 9.0}) is None                      # prev close before the open
    assert pm.watchlist_price({"price": 9.5, "day_open": 9.4})[0] == 9.5   # opened: price is live
    assert pm.watchlist_open({"day_open": 9.4})[0] == 9.4
    assert pm.watchlist_open({"price": 9.0}) is None


def _handler(requests: list, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.headers["authorization"], body))
        if status != 200:
            return httpx.Response(status)
        xids = [int(e["symbol"]) for e in body["elements"]]
        assert all(e["type"] == "price" for e in body["elements"])  # never a Volume element
        return httpx.Response(200, json=batch_fixture(xids))
    return handler


def _syms(n: int) -> dict[str, int]:
    return {f"S{i}": 1000 + i for i in range(n)}


def test_watchlist_first_chartiq_for_the_rest_batched_by_20(env):
    xids = _syms(45)
    rows = {"S0": {"bid": 20.0, "ask": 20.2}, "S1": {"bid": 30.0, "ask": 30.4}}
    reqs: list = []
    client = httpx.Client(transport=httpx.MockTransport(_handler(reqs)))
    s = pm.run_prices(D, list(xids), rows=rows, client=client, now=_ny(9, 30), xids=xids)
    assert len(reqs) == 3 and all(len(b["elements"]) <= 20 for _, b in reqs)   # 43 left over -> 20+20+3
    assert reqs[0][0] == "Bearer abc"
    recs = {r["symbol"]: r for r in map(json.loads, open(s["path"]))}
    assert recs["S0"]["source"] == "watchlist" and recs["S0"]["last"] == pytest.approx(20.1)
    assert recs["S2"]["source"] == "etrade_chartiq" and recs["S2"]["last"] == 12.5  # <= 09:30 only
    assert recs["S3"]["last"] == 102.5  # nested symbol (2nd of its batch)
    assert recs["S2"]["open"] is None and s["with_last"] == 45
    schema.validate_records(list(recs.values()), schema.load_schema(config.repo_root() / "schemas" / "premarket_prices.schema.json"))


def test_no_chartiq_call_when_the_watchlist_covers_everything(env):
    reqs: list = []
    client = httpx.Client(transport=httpx.MockTransport(_handler(reqs)))
    rows = {"A": {"bid": 1.0, "ask": 1.1}}
    s = pm.run_prices(D, ["A"], rows=rows, client=client, now=_ny(9, 30), xids={"A": 1})
    assert reqs == [] and s["by_source"] == {"watchlist": 1}


def test_open_pass_keeps_lasts_and_fills_opens(env):
    xids = _syms(3)
    reqs: list = []
    client = httpx.Client(transport=httpx.MockTransport(_handler(reqs)))
    pm.run_prices(D, list(xids), rows={}, client=client, now=_ny(9, 30), xids=xids)
    rows = {"S0": {"day_open": 12.0, "ts_utc": "2026-10-27T13:30:01Z"}}
    s = pm.run_prices(D, list(xids), open_pass=True, rows=rows, client=client, now=_ny(9, 33), xids=xids)
    recs = {r["symbol"]: r for r in map(json.loads, open(s["path"]))}
    assert recs["S0"]["open"] == 12.0 and recs["S0"]["last"] == 12.5   # last kept from the 09:30 pass
    assert recs["S1"]["open"] == 13.0                                   # ChartIQ's first bucket >= 09:30
    assert recs["S1"]["open_at"].endswith("Z") and s["missing_open"] == 0
    # before any regular bucket prints, the open stays null and is reported
    __import__("os").unlink(s["path"])
    s2 = pm.run_prices(D, ["S2"], open_pass=True, rows={}, client=client, now=_ny(9, 30, 5), xids=xids)
    assert s2["missing_open"] == 1


@pytest.mark.parametrize("status", [401, 403, 429])
def test_auth_failure_stops_without_retry_and_keeps_watchlist(env, status):
    xids = _syms(60)
    reqs: list = []
    client = httpx.Client(transport=httpx.MockTransport(_handler(reqs, status)))
    rows = {"S0": {"bid": 5.0, "ask": 5.2}}
    s = pm.run_prices(D, list(xids), rows=rows, client=client, now=_ny(9, 30), xids=xids)
    assert s["auth_error"] and "Bearer" not in s["auth_error"]
    assert len(reqs) <= pm.config.premarket_config()["max_concurrency"]   # in-flight only, no retry
    assert s["with_last"] == 1                                            # the watchlist price stands


def test_missing_token_never_logs_in(env):
    (env / "state" / "etrade" / "bearer.json").unlink()
    s = pm.run_prices(D, ["A"], rows={}, xids={"A": 1}, now=_ny(9, 30))
    assert s["auth_error"] and s["with_last"] == 0


def test_token_file_then_session_fallback(env):
    assert pm.load_token() == "Bearer abc"
    (env / "state" / "etrade" / "bearer.json").write_text(json.dumps({"token": "Bearer zzz"}))
    assert pm.load_token() == "Bearer zzz"
    (env / "state" / "etrade" / "bearer.json").unlink()
    with pytest.raises(pm.AuthStop):
        pm.load_token(session_path=env / "none.json")


def test_unknown_live_source_is_refused(env):
    with pytest.raises(pm.ConfigError):
        pm.run_prices(D, ["A"], source="nope", rows={})


def test_xids_file_and_symbols_file(env, tmp_path):
    f = tmp_path / "xids.csv"
    f.write_text("AAPL,1\nMSFT,2\nbad,x\n")
    assert pm.load_universe(["AAPL", "ZZZ"], f) == {"AAPL": 1}
    s = tmp_path / "short.csv"
    s.write_text("symbol,score,reasons\nAAPL,0.9,top\nMSFT,0.5,move\n")
    assert pm.read_symbols_file(s) == ["AAPL", "MSFT"]
    plain = tmp_path / "plain.txt"
    plain.write_text("A\n\nB\n")
    assert pm.read_symbols_file(plain) == ["A", "B"]
