"""premarket full (etrade/premarket_full.py): Massive minute-bar aggregation, the 09:45 refusal."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import httpx
import pytest

from sourcing_py.common import schema
from sourcing_py.etrade import premarket_full as pf
from sourcing_py.massive_ext import extended as ext

D = date(2026, 10, 27)
UTC = timezone.utc


def _ms(hh: int, mm: int) -> int:
    return int(datetime(D.year, D.month, D.day, hh, mm, tzinfo=ext.NY).timestamp() * 1000)


BARS = {
    "AAA": [  # 04:00 trades, 08:00 trades, 09:29 trades, 09:30 regular (excluded), 09:10 zero volume
        {"t": _ms(4, 0), "o": 10, "h": 10.5, "l": 9.5, "c": 10.0, "v": 100, "vw": 10.0},
        {"t": _ms(8, 0), "o": 10, "h": 12.0, "l": 10, "c": 11.0, "v": 300, "vw": 11.0},
        {"t": _ms(9, 10), "o": 11, "h": 11, "l": 11, "c": 99.0, "v": 0, "vw": 0},
        {"t": _ms(9, 29), "o": 11, "h": 11.5, "l": 8.0, "c": 11.4, "v": 100, "vw": 11.2},
        {"t": _ms(9, 30), "o": 50, "h": 60, "l": 40, "c": 55, "v": 9999, "vw": 55},
    ],
    "BBB": [],
}


def _client(fail: set[str] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        sym = request.url.path.split("/")[4]
        if sym in (fail or set()):
            return httpx.Response(404)
        return httpx.Response(200, json={"results": BARS.get(sym, [])})
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MASSIVE_API_KEY", "k")
    return tmp_path


NOW_OK = datetime(2026, 10, 27, 13, 46, tzinfo=UTC)      # 09:46 ET
NOW_EARLY = datetime(2026, 10, 27, 13, 40, tzinfo=UTC)   # 09:40 ET


def test_aggregation_matches_training_definitions(env):
    s = pf.run_full(D, ["AAA", "BBB"], now=NOW_OK, client=_client())
    recs = {r["symbol"]: r for r in map(json.loads, open(s["path"]))}
    a = recs["AAA"]
    assert a["pm_volume"] == 500
    assert a["pm_vwap"] == pytest.approx((10 * 100 + 11 * 300 + 11.2 * 100) / 500)   # vw-weighted
    assert a["pm_last"] == 11.4 and a["pm_last_t"] == _ms(9, 29)    # last bar that TRADED, not the 99 at v=0
    assert (a["pm_high"], a["pm_low"]) == (12.0, 8.0)               # the 09:30 bar is not in the window
    assert a["complete"] is True and a["cutoff"] == "09:30" and a["date"] == "2026-10-27"
    b = recs["BBB"]
    assert b["pm_vwap"] is None and b["pm_volume"] == 0 and b["pm_last"] is None and b["n_bars"] == 0
    assert s["written"] == 2 and s["with_bars"] == 1 and s["complete"] == 2
    schema.validate_records(list(recs.values()), schema.load_schema(pf.config.repo_root() / "schemas" / "etrade_premarket.schema.json"))
    assert (env / "out" / "premarket_live" / "premarket-2026-10-27.summary.json").exists()


def test_same_result_as_massive_ext_summary(env):
    day = ext.summarize("AAA", [BARS["AAA"]], "x")[0]
    rec = pf.record("AAA", D, "09:30", day, "x")
    assert rec["pm_vwap"] == day["pre_market_vwap_0930"] and rec["pm_last"] == day["pm_last_0930"]


def test_refuses_before_cutoff_plus_delay_unless_forced(env):
    with pytest.raises(pf.TooEarly) as e:
        pf.run_full(D, ["AAA"], now=NOW_EARLY, client=_client())
    assert "09:45" in str(e.value)
    assert not (env / "out" / "premarket_live").exists()                 # nothing written
    s = pf.run_full(D, ["AAA"], now=NOW_EARLY, force=True, client=_client())
    rec = json.loads(open(s["path"]).read())
    assert rec["complete"] is False and s["forced"] is True and s["complete"] == 0
    # exactly at 09:45 it runs
    pf.run_full(D, ["AAA"], now=datetime(2026, 10, 27, 13, 45, tzinfo=UTC), client=_client())


def test_a_failed_symbol_is_counted_not_fatal(env):
    s = pf.run_full(D, ["AAA", "BBB"], now=NOW_OK, client=_client(fail={"BBB"}))
    assert s["written"] == 1 and s["errors"] == 1


def test_unsupported_cutoff_is_refused(env):
    with pytest.raises(Exception):
        pf.run_full(D, ["AAA"], cutoff="09:20", now=NOW_OK, client=_client())
