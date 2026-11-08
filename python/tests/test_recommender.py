"""The contract recommender (`mtap/recommender`): schema pins, message building, RESYNC,
problem reporting, outcomes. Protocol: ../MBin/docs/contracts/mtap-mbin-protocol.md."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

import pytest

from sourcing_py.common import bus
from sourcing_py.recommender import cli, contract
from sourcing_py.recommender import service as svc

T0 = datetime.fromisoformat("2026-11-02T15:05:13.044Z").timestamp()  # 10:05 ET


class FakeRedis:
    """xadd only: records (stream, payload, maxlen)."""

    def __init__(self):
        self.added: list[tuple[str, dict, int]] = []

    def xadd(self, key, fields, maxlen=None, approximate=True):
        self.added.append((key, json.loads(fields["data"]), maxlen))
        return f"{len(self.added)}-0"

    def of(self, msg_type):
        return [m for k, m, _ in self.added if k == f"mtap:evt:{msg_type}"]


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def rec(tmp_path):
    return svc.Recommender(FakeRedis(), clock=Clock(), root=tmp_path, instance_id="mtap-rec-1")


def candidate(**over):
    c = {
        "symbol": "AAPL", "name": "Apple Inc", "data_ts_utc": "2026-11-02T15:05:12.912Z",
        "score": 0.62, "probability": {"value": 0.66, "target_pct": 1.0, "horizon": "60m"},
        "target_pct": 1.0, "stop_pct": 0.7, "ttl_ms": 300000,
        "price_ref": {"price": 231.42, "ts_utc": "2026-11-02T15:05:12.912Z"},
        "market_cap": 3.61e12, "sector": "Technology", "industry": "Phones & Handheld Devices",
        "strategy_id": "intraday.pressure", "strategy_version": "IA1",
        "rationale": "66% chance of +1.0% within 60 min", "inputs": {"lean": 1.4},
    }
    c.update(over)
    return c


def mbin_fixture(name):
    path = contract.mbin_schemas_dir().parent / "fixtures" / name
    if not path.exists():
        pytest.skip("MBin checkout not next to MTap")
    return json.loads(path.read_text())


PARAMS = {
    "type": "TRADING_PARAMETERS", "schema_version": 1, "producer": "mbin/signals",
    "ts_utc": "2026-11-02T14:02:01.105Z", "_key": "tp:mbin-trading-1:1", "_ts": 1,
    "params_version": 17,
    "session": {"date_et": "2026-11-02", "is_trading_day": True,
                "open_ts_utc": "2026-11-02T14:30:00.000Z", "close_ts_utc": "2026-11-02T21:00:00.000Z",
                "is_half_day": False},
    "ttl": {"default_ms": 300000, "max_ms": 600000},
    "thresholds": {"default_target_pct": 1.0, "default_stop_pct": 0.7},
    "accepted": {"instruments": ["EQ"], "entry_actions": ["BUY"], "sell_policy": "held_only_as_exit"},
    "trading_state": "NORMAL", "autonomy_phase": "manual", "hints": None,
}


# -- vendored schemas ---------------------------------------------------------------------

def test_vendored_schemas_match_the_pins():
    for name, pin in contract.PINNED.items():
        assert hashlib.sha256(contract.schema_bytes(name)).hexdigest() == pin, name


def test_vendored_schemas_match_mbin_when_checked_out():
    if not contract.mbin_schemas_dir().exists():
        pytest.skip("MBin checkout not next to MTap")
    assert cli.check_schemas() == []


def test_mbin_fixtures_validate_with_the_vendored_copy():
    for name in ("trading_parameters.v1.example.json", "recommendation.v1.example.json",
                 "contract_status.v1.example-mtap-heartbeat.json", "resync_request.v1.example.json",
                 "recommendation_outcome.v1.example-closed.json"):
        assert contract.validate(mbin_fixture(name)) is None, name
    assert contract.validate(mbin_fixture("recommendation.v1.invalid-score-range.json"))


# -- ids and fields ------------------------------------------------------------------------

def test_rec_id_is_deterministic_and_matches_the_schema():
    a = contract.rec_id("AAPL", "2026-11-02T15:05:12.912Z", "intraday.pressure", 1.0, "60m")
    assert a == contract.rec_id("AAPL", "2026-11-02T15:05:12.912Z", "intraday.pressure", 1.0, "60m")
    assert a.startswith("r-20261102-150512-AAPL-") and len(a.rsplit("-", 1)[1]) == 4
    assert a != contract.rec_id("AAPL", "2026-11-02T15:05:12.912Z", "intraday.pressure", 1.5, "60m")
    assert contract.rec_id("BRK.B", "2026-11-02T15:05:12.912Z", "s", 1.0, "close").startswith(
        "r-20261102-150512-BRK.B-")


def test_cap_bands():
    assert [contract.cap_band(x) for x in (3e12, 200e9, 50e9, 2e9, 300e6, 1e6, None)] == [
        "mega", "mega", "large", "mid", "small", "micro", None]


def test_ts_utc_has_milliseconds_and_z():
    assert contract.ts_utc(datetime.fromtimestamp(T0).astimezone()) == "2026-11-02T15:05:13.044Z"


# -- publishing ----------------------------------------------------------------------------

def test_candidate_becomes_a_valid_recommendation(rec):
    msg = rec.recommend(candidate())
    assert msg is not None and contract.validate(msg) is None
    assert msg["_key"] == f"rec:{msg['rec_id']}" and msg["producer"] == "mtap/recommender"
    assert msg["cap_band"] == "mega" and msg["action"] == "BUY" and msg["instrument"] == "EQ"
    key, sent, maxlen = rec.client.added[-1]
    assert key == "mtap:evt:RECOMMENDATION" and sent == msg and maxlen == 20000
    audit = (rec.root / "published" / "2026-11-02.jsonl").read_text().splitlines()
    assert json.loads(audit[-1])["msg"]["rec_id"] == msg["rec_id"]


def test_same_decision_is_sent_once_and_invalid_is_not_sent(rec):
    assert rec.recommend(candidate()) is not None
    assert rec.recommend(candidate()) is None
    assert rec.recommend(candidate(symbol="MSFT", score=1.5)) is None      # schema: score ≤ 1
    assert rec.recommend(candidate(symbol="MSFT", account="x")) is None    # unknown field
    assert len(rec.client.of("RECOMMENDATION")) == 1


def test_entries_still_sent_when_not_normal(rec):
    # REVIEW item 12: MTap keeps producing in CLOSE_ONLY / FLATTENING; MBin filters.
    rec.on_params({**PARAMS, "trading_state": "CLOSE_ONLY"})
    assert rec.recommend(candidate()) is not None


def test_defaults_record_params_version(rec):
    rec.on_params(PARAMS)
    msg = rec.recommend(candidate(target_pct=None, stop_pct=None))
    assert msg["inputs"]["params_version"] == 17


def test_startup_sends_status_and_resync(rec):
    rec.start()
    (status,) = rec.client.of("CONTRACT_STATUS")
    (resync,) = rec.client.of("RESYNC_REQUEST")
    assert contract.validate(status) is None and contract.validate(resync) is None
    assert status["reason"] == "startup" and status["_key"] == "cs:mtap:mtap-rec-1:1"
    assert {p["type"] for p in status["produces"]} == set(contract.PRODUCES)
    assert status["produces"][0]["schema_sha256"] == contract.PINNED["recommendation.v1"]
    assert resync["request_id"] == "mtap:mtap-rec-1:2" and resync["since_ts_utc"] is None
    assert "RECOMMENDATION_OUTCOME" in resync["requested_types"]


# -- RESYNC (§8.7) -------------------------------------------------------------------------

def resync_from_mbin(types=("RECOMMENDATION", "CONTRACT_STATUS"), n=1):
    return {
        "type": "RESYNC_REQUEST", "schema_version": 1, "producer": "mbin/signals",
        "ts_utc": "2026-11-02T15:06:00.000Z", "_key": f"rs:mbin:mbin-trading-1:{n}", "_ts": 1,
        "request_id": f"mbin:mbin-trading-1:{n}", "party": "mbin", "requested_types": list(types),
        "since_ts_utc": None, "reason": "startup",
    }


def test_resync_republishes_active_recs_with_the_original_key(rec):
    first = rec.recommend(candidate())
    rec.clock.t += 60
    assert rec.on_resync(resync_from_mbin()) == 2
    recs = rec.client.of("RECOMMENDATION")
    assert len(recs) == 2 and recs[1]["_key"] == first["_key"] and recs[1]["rec_id"] == first["rec_id"]
    assert recs[1]["correlation_id"] == "mbin:mbin-trading-1:1"
    assert rec.client.of("CONTRACT_STATUS")[-1]["correlation_id"] == "mbin:mbin-trading-1:1"


def test_resync_skips_expired_and_rate_limits(rec):
    rec.recommend(candidate())                         # ttl 300 s from 15:05:12.9
    rec.clock.t += 400
    assert rec.on_resync(resync_from_mbin(types=["RECOMMENDATION"])) == 0
    rec.clock.t += 1
    rec.recommend(candidate(symbol="MSFT", data_ts_utc="2026-11-02T15:11:50.000Z"))
    assert rec.on_resync(resync_from_mbin(types=["RECOMMENDATION"], n=2)) == 0   # < 10 s
    rec.clock.t += 10
    assert rec.on_resync(resync_from_mbin(types=["RECOMMENDATION"], n=3)) == 1


def test_expiry_is_clamped_to_ttl_max_and_the_close(rec):
    rec.on_params({**PARAMS, "ttl": {"default_ms": 300000, "max_ms": 120000}})
    msg = rec.recommend(candidate())
    assert rec.expires_ms(msg) == contract.epoch_ms(msg["data_ts_utc"]) + 120000
    late = candidate(data_ts_utc="2026-11-02T20:59:00.000Z", ttl_ms=600000)
    assert rec.expires_ms(late) == contract.epoch_ms("2026-11-02T21:00:00.000Z")


def test_restart_restores_todays_active_recs(rec, tmp_path):
    first = rec.recommend(candidate())
    again = svc.Recommender(FakeRedis(), clock=Clock(T0 + 30), root=tmp_path, instance_id="mtap-rec-2")
    assert list(again.active) == [first["rec_id"]]


# -- incoming checks (§6.4) ----------------------------------------------------------------

def test_problems_are_recorded_and_reported(rec):
    assert rec.check("TRADING_PARAMETERS", "{nope") is None
    assert rec.check("TRADING_PARAMETERS", json.dumps({**PARAMS, "schema_version": 2})) is None
    assert rec.check("TRADING_PARAMETERS", json.dumps({**PARAMS, "params_version": "x"})) is None
    assert rec.check("X", json.dumps({"type": "SOMETHING"})) is None
    future = {**PARAMS, "ts_utc": "2026-11-02T15:06:00.000Z"}
    assert rec.check("TRADING_PARAMETERS", json.dumps(future)) is not None     # still used
    assert set(rec.problems) == {
        ("TRADING_PARAMETERS", "unparseable"), ("TRADING_PARAMETERS", "unsupported_version"),
        ("TRADING_PARAMETERS", "validation_failed"), ("SOMETHING", "unknown_type"),
        ("TRADING_PARAMETERS", "clock_skew"),
    }
    statuses = rec.client.of("CONTRACT_STATUS")
    assert all(s["reason"] == "problem" and contract.validate(s) is None for s in statuses)
    # One report per type per 10 s: TRADING_PARAMETERS once, SOMETHING once.
    assert len(statuses) == 2
    detail = rec.problems[("TRADING_PARAMETERS", "validation_failed")].detail
    assert detail.startswith("params_version:")


def test_schema_mismatch_from_mbin_status(rec):
    status = mbin_fixture("contract_status.v1.example-mbin-problem.json")
    rec.on_mbin_status(status)
    assert rec.mbin_online()
    # The fixture's hashes are illustrative (protocol §8.6), so they differ from ours.
    assert any(p == "schema_mismatch" for _t, p in rec.problems)
    rec.clock.t += svc.OFFLINE_S + 1
    assert not rec.mbin_online()


def test_params_latest_wins(rec):
    rec.on_params(PARAMS)
    rec.on_params({**PARAMS, "ts_utc": "2026-11-02T14:00:00.000Z", "params_version": 3})
    assert rec.params["params_version"] == 17


# -- outcomes ------------------------------------------------------------------------------

def test_outcomes_dedup_and_set_since(rec):
    out = mbin_fixture("recommendation_outcome.v1.example-closed.json")
    assert rec.on_outcome(out) and not rec.on_outcome(out)
    assert rec.last_outcome_ts == out["outcome_ts_utc"]
    rec.start()
    assert rec.client.of("RESYNC_REQUEST")[-1]["since_ts_utc"] == out["outcome_ts_utc"]


# -- bus ------------------------------------------------------------------------------------

def test_publish_stamps_key_ts_and_per_stream_maxlen():
    fake = FakeRedis()
    bus.publish("", "CONTRACT_STATUS", {"a": 1}, key="cs:x", client=fake)
    bus.publish("", "INTRADAY_FEED", {"a": 1}, client=fake)
    bus.publish("", "RECOMMENDATION_OUTCOME", {"a": 1}, prefix="mbin:evt:", client=fake)
    (k1, m1, n1), (k2, m2, n2), (k3, _m3, n3) = fake.added
    assert (k1, m1["_key"], n1) == ("mtap:evt:CONTRACT_STATUS", "cs:x", 200) and m1["_ts"] > 0
    assert (k2, n2) == ("mtap:evt:INTRADAY_FEED", 4000) and "_key" not in m2
    assert (k3, n3) == ("mbin:evt:RECOMMENDATION_OUTCOME", 20000)


@pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="needs REDIS_URL")
def test_redis_end_to_end(tmp_path, monkeypatch):
    """Over a real Redis: MBin's params + RESYNC in, a candidate through, outcomes back."""
    import redis

    url = os.environ["REDIS_URL"]
    r = redis.Redis.from_url(url, decode_responses=True)
    r.flushdb()
    monkeypatch.setattr(svc, "state_dir", lambda: tmp_path)
    now = contract.ts_utc()
    bus.publish(url, "TRADING_PARAMETERS", {**PARAMS, "ts_utc": now}, key=PARAMS["_key"],
                prefix="mbin:evt:")
    bus.publish(url, svc.CANDIDATE_EVENT,
                candidate(data_ts_utc=now, price_ref={"price": 231.42, "ts_utc": now}),
                prefix=svc.CANDIDATE_PREFIX)
    out = {**mbin_fixture("recommendation_outcome.v1.example-closed.json"), "ts_utc": now,
           "outcome_ts_utc": now}
    bus.publish(url, "RECOMMENDATION_OUTCOME", out, key=out["_key"], prefix="mbin:evt:")
    assert svc.run(url, stop_after_s=2.5) == 0

    def entries(key):
        return [json.loads(f["data"]) for _id, f in r.xrange(key)]

    statuses = entries("mtap:evt:CONTRACT_STATUS")
    assert [s["reason"] for s in statuses] == ["startup", "shutdown"]
    (rq,) = entries("mtap:evt:RESYNC_REQUEST")
    (rc,) = entries("mtap:evt:RECOMMENDATION")
    assert contract.validate(rq) is None and contract.validate(rc) is None
    assert rc["inputs"] == {"lean": 1.4}
    assert len(list((tmp_path / "outcomes").glob("*.jsonl"))) == 1
    groups = {g["name"]: g for g in r.xinfo_groups("mbin:evt:RECOMMENDATION_OUTCOME")}
    assert groups["mtap-outcomes"]["pending"] == 0
    r.flushdb()
