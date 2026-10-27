"""The 09:29:30 pre-check (etrade/precheck.py): Redis rows (a fake), snapshot scoring, shortlist."""

from __future__ import annotations

import json
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sourcing_py.common import bus, config
from sourcing_py.etrade import precheck as pc

D = date(2026, 10, 27)


class FakeRedis:
    def __init__(self, entries):
        self.entries = entries  # oldest first, like a stream
        self.calls = []

    def xrevrange(self, key, count=None):
        self.calls.append((key, count))
        return [(f"{i}-0", {"data": json.dumps(e) if not isinstance(e, str) else e})
                for i, e in reversed(list(enumerate(self.entries)))][:count]


class DownRedis:
    def xrevrange(self, *a, **k):
        raise ConnectionError("redis is down")


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    return tmp_path


def test_redis_rows_newest_non_null_wins_and_bid_ask_pair():
    old = {"rows": [{"symbol": "A", "price": 10.0, "volume_cum": 100, "bid": 9.9, "ask": 10.1},
                    {"symbol": "B", "price": 5.0}]}
    new = {"rows": [{"symbol": "A", "volume_cum": 250, "bid": 10.0},          # ask missing: pair not taken
                    {"symbol": "C", "day_open": 7.0}]}
    fake = FakeRedis([old, new, "not json"])
    rows = pc.read_watchlist(client=fake)
    assert fake.calls == [(bus.stream_key("INTRADAY_FEED"), 300)]            # the contract stream
    assert rows["A"]["volume_cum"] == 250 and rows["A"]["price"] == 10.0
    assert (rows["A"]["bid"], rows["A"]["ask"]) == (9.9, 10.1)
    assert rows["B"]["price"] == 5.0 and rows["C"]["day_open"] == 7.0
    assert pc.read_watchlist(client=DownRedis()) == {}                       # Redis down: no rows, no error


def test_snapshot_features_are_defensive():
    f = pc.snapshot_features({"day": {"c": 0, "v": 0}, "min": {"c": 11.0, "av": 5000},
                              "prevDay": {"c": 10.0, "vw": 9.8}, "todaysChangePerc": 2.0})
    assert f == {"price": 11.0, "volume": 5000.0, "prev_vwap": 9.8, "change_pct": 2.0}
    assert pc.snapshot_features({})["price"] is None


def _world(n=300):
    syms = [f"S{i:03d}" for i in range(n)]
    snap = {s: {"min": {"c": 10.0 * (1 + 0.0001 * i), "av": 1000}, "prevDay": {"vw": 10.0}} for i, s in enumerate(syms)}
    mean10 = {s: 1_000_000.0 for s in syms}
    quant = {s: {"q05": -0.01 - 0.0001 * i, "q50": 0.0, "q95": 0.01 + 0.0001 * i} for i, s in enumerate(syms)}
    return syms, snap, mean10, quant


def test_score_prefers_big_moves_volume_and_wide_night_range():
    syms, snap, mean10, quant = _world(50)
    snap["S010"]["min"]["c"] = 11.0                      # +10% move
    rows = {"S020": {"volume_cum": 9_000_000, "bid": 10.0, "ask": 10.0}}   # 9x its mean
    feats = pc.build_features(syms, snap, rows, quant, mean10)
    sc = pc.score(feats)
    assert feats["S010"]["move"] == pytest.approx(0.10)
    assert feats["S020"]["vol_ratio"] == 9.0
    assert sc["S010"] > sc["S011"] and sc["S020"] > sc["S019"]   # neighbours differ only by the feature


def test_shortlist_is_fixed_size_plus_threshold_pulls():
    syms, snap, mean10, quant = _world(300)
    snap["S000"]["min"]["c"] = 10.5                       # +5%: over move_pct 2.0 but a tiny night range
    quant["S000"] = {"q05": -0.0001, "q50": 0.0, "q95": 0.0001}
    rows = {"S001": {"volume_cum": 6_000_000}}            # 6x: over vol_ratio 5.0
    feats = pc.build_features(syms, snap, rows, quant, mean10)
    sc = pc.score(feats)
    out = pc.shortlist(feats, sc, size=200, move_pct=2.0, vol_ratio=5.0)
    names = [s for s, _, _ in out]
    assert len(names) == len(set(names)) and len(names) >= 200
    reasons = {s: w for s, _, w in out}
    assert "move" in reasons["S000"] and "volume" in reasons["S001"]
    assert all(sc[names[i]] >= sc[names[i + 1]] for i in range(len(names) - 1))     # best first
    assert sum(1 for _, _, w in out if w.startswith("top")) == 200


def test_run_precheck_night_only_fallback_writes_csv(env):
    syms, _, mean10, quant = _world(30)
    night = env / "forecast_2026-10-27.parquet"
    pq.write_table(pa.table({"symbol": syms, "q05": [quant[s]["q05"] for s in syms],
                             "q50": [0.0] * 30, "q95": [quant[s]["q95"] for s in syms]}), night)
    nxt = env / "panel.next.parquet"
    pq.write_table(pa.table({"symbol": syms, "vol_10d_mean": [mean10[s] for s in syms]}), nxt)
    s = pc.run_precheck(D, night=night, panel_next=nxt, snapshot={}, rows={}, use_snapshot=False)
    assert s["snapshot"] == 0 and s["night"] == 30 and s["shortlist"] == 30     # fewer than 200: all
    lines = open(s["path"]).read().splitlines()
    assert lines[0] == "symbol,score,reasons" and lines[1].startswith("S029,")  # widest night range first


def test_snapshot_failure_degrades_instead_of_failing(env, monkeypatch):
    def boom(client=None):
        raise RuntimeError("HTTP 500")
    monkeypatch.setattr(pc, "fetch_snapshot", boom)
    nxt = env / "panel.next.parquet"
    pq.write_table(pa.table({"symbol": ["A", "B"], "vol_10d_mean": [1.0, 1.0]}), nxt)
    s = pc.run_precheck(D, panel_next=nxt, rows={})
    assert "500" in s["snapshot_error"] and s["shortlist"] == 2


def test_fetch_snapshot_reads_the_tickers(monkeypatch):
    import httpx

    monkeypatch.setenv("MASSIVE_API_KEY", "k")
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json={"tickers": [{"ticker": "A", "day": {"c": 1}}, {"ticker": "B"}]})
    got = pc.fetch_snapshot(httpx.Client(transport=httpx.MockTransport(handler)))
    assert seen["path"] == "/v2/snapshot/locale/us/markets/stocks/tickers" and set(got) == {"A", "B"}
