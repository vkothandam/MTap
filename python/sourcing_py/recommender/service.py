"""The `mtap/recommender` service: MTap's phase-1 side of the MTap↔MBin contract.

Streams (protocol §7; `mtap:evt:` is MTap's, `mbin:evt:` is MBin's):

| stream | direction | how it is read here |
| --- | --- | --- |
| `mtap:int:REC_CANDIDATE` | scorer → recommender | durable group `recommender` (MTap-internal) |
| `mtap:evt:RECOMMENDATION` | published | — |
| `mtap:evt:CONTRACT_STATUS` | published | startup, every 60 s, shutdown, on a problem |
| `mtap:evt:RESYNC_REQUEST` | published | at startup |
| `mbin:evt:TRADING_PARAMETERS`, `mbin:evt:CONTRACT_STATUS` | read | state: newest entry, then tail |
| `mbin:evt:RESYNC_REQUEST` | read | tail-only |
| `mbin:evt:RECOMMENDATION_OUTCOME` | read | durable group `mtap-outcomes`, new group at 0 (REVIEW M11), ack after the write |

A candidate is a RECOMMENDATION without its header and `rec_id` (the live scorer writes it;
`sourcing-py recommender submit` writes one by hand). Every message is validated against
the vendored schema before it is published; one that fails is logged and dropped.

The audit trail (§5) is `state/recommender/published/<date>.jsonl` (every message MTap
publishes, heartbeats excepted); outcomes go to `state/recommender/outcomes/<date>.jsonl`.
Today's published file also restores the active recommendations after a restart, so a
RESYNC still re-publishes them.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..common import bus, config
from . import contract as c

log = logging.getLogger(__name__)

CANDIDATE_PREFIX = "mtap:int:"
CANDIDATE_EVENT = "REC_CANDIDATE"
CANDIDATE_GROUP = "recommender"
OUTCOME_GROUP = "mtap-outcomes"

HEARTBEAT_S = 60
OFFLINE_S = 180               # no CONTRACT_STATUS from MBin for this long = offline (§5)
PROBLEM_REPEAT_S = 10         # a continuing problem is re-reported at most this often (§6.4)
RESYNC_MIN_GAP_S = 10         # one RESYNC answer per requester per 10 s (§8.7)
CLOCK_SKEW_S = 5
DEFAULT_TTL_MS = 300_000      # until MBin's TRADING_PARAMETERS say otherwise
RECONNECT_S = 5

KNOWN_TYPES = {
    "RECOMMENDATION", "EXIT_RECOMMENDATION", "INTRADAY_FEED", "TRADING_PARAMETERS",
    "POSITIONS_SNAPSHOT", "RECOMMENDATION_OUTCOME", "CONTRACT_STATUS", "RESYNC_REQUEST",
}
CANDIDATE_FIELDS = {
    "symbol", "name", "data_ts_utc", "action", "score", "probability", "target_pct",
    "stop_pct", "ttl_ms", "price_ref", "cap_band", "market_cap", "sector", "industry",
    "strategy_id", "strategy_version", "rationale", "inputs",
}


def state_dir() -> Path:
    return config.state_root() / "recommender"


@dataclass
class Problem:
    count: int
    first_ts_utc: str
    last_ts_utc: str
    received_version: int | None = None
    sample_key: str | None = None
    detail: str | None = None


@dataclass
class Recommender:
    """All contract logic, driven by `run()` (or directly by tests). `client` is a
    redis-py client with decode_responses=True; `clock` returns epoch seconds."""

    client: Any
    clock: Callable[[], float] = time.time
    root: Path = field(default_factory=lambda: state_dir())   # looked up per instance
    instance_id: str = ""
    seq: int = 0
    started_ts_utc: str = ""
    params: dict | None = None
    mbin_status: dict | None = None
    mbin_seen_s: float | None = None
    active: dict[str, dict] = field(default_factory=dict)       # rec_id -> message
    problems: dict[tuple[str, str], Problem] = field(default_factory=dict)
    _problem_sent: dict[str, float] = field(default_factory=dict)
    _resync_answered: dict[str, float] = field(default_factory=dict)
    _outcome_keys: set[str] = field(default_factory=set)
    last_outcome_ts: str | None = None

    def __post_init__(self) -> None:
        now = self.clock()
        self.instance_id = self.instance_id or f"mtap-rec-{int(now)}"
        self.started_ts_utc = self.started_ts_utc or c.ts_utc(_dt(now))
        self._restore()

    # -- helpers ------------------------------------------------------------------------

    def _now_ts(self) -> str:
        return c.ts_utc(_dt(self.clock()))

    def _header(self, msg_type: str, key: str) -> dict[str, Any]:
        return {
            "type": msg_type, "schema_version": 1, "producer": c.PRODUCER, "ts_utc": self._now_ts(),
            "_key": key, "_ts": int(self.clock() * 1000),
        }

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _day(self) -> str:
        return _dt(self.clock()).strftime("%Y-%m-%d")

    def _append(self, sub: str, record: dict) -> None:
        path = self.root / sub / f"{self._day()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _send(self, msg: dict[str, Any], *, audit: bool = True) -> str | None:
        """Validate, publish, audit. Returns the stream entry id, or None when invalid."""
        err = c.validate(msg)
        if err:
            log.error("not publishing invalid %s %s: %s", msg.get("type"), msg.get("_key"), err)
            return None
        entry = bus.publish("", msg["type"], msg, key=msg["_key"], client=self.client)
        if audit:
            self._append("published", {"entry_id": entry, "msg": msg})
        return entry

    def _restore(self) -> None:
        """Today's still-valid recommendations and processed outcomes, after a restart."""
        pub = self.root / "published" / f"{self._day()}.jsonl"
        for line in _lines(pub):
            msg = line.get("msg") or {}
            if msg.get("type") == "RECOMMENDATION":
                self.active[msg["rec_id"]] = msg
        self._expire()
        for line in _lines(self.root / "outcomes" / f"{self._day()}.jsonl"):
            self._outcome_keys.add(line.get("_key", ""))
            ts = line.get("outcome_ts_utc")
            if ts and (self.last_outcome_ts is None or ts > self.last_outcome_ts):
                self.last_outcome_ts = ts

    # -- expiry (§8.1 step 2, as MBin computes it) --------------------------------------

    def expires_ms(self, msg: dict) -> int:
        ttl = (self.params or {}).get("ttl") or {}
        ttl_ms = msg.get("ttl_ms") or ttl.get("default_ms") or DEFAULT_TTL_MS
        if ttl.get("max_ms"):
            ttl_ms = min(ttl_ms, ttl["max_ms"])
        expires = c.epoch_ms(msg["data_ts_utc"]) + ttl_ms
        close = ((self.params or {}).get("session") or {}).get("close_ts_utc")
        return min(expires, c.epoch_ms(close)) if close else expires

    def _expire(self) -> None:
        now_ms = int(self.clock() * 1000)
        for rid in [r for r, m in self.active.items() if self.expires_ms(m) <= now_ms]:
            del self.active[rid]

    # -- outgoing -----------------------------------------------------------------------

    def recommend(self, candidate: dict[str, Any]) -> dict | None:
        """Turn a scorer candidate into a RECOMMENDATION and publish it. Returns the message,
        or None when it is invalid or a repeat of a decision already sent. It is sent in every
        trading_state: MBin filters, so its outcomes record what was missed (REVIEW item 12)."""
        extra = set(candidate) - CANDIDATE_FIELDS
        if extra:
            log.error("candidate has unknown fields %s; dropped", sorted(extra))
            return None
        state = (self.params or {}).get("trading_state")
        if state and state != "NORMAL":
            log.info("trading_state %s: candidate %s sent, MBin filters", state, candidate.get("symbol"))
        body = {"instrument": "EQ", "action": "BUY", **candidate}
        if body.get("cap_band") is None and body.get("market_cap") is not None:
            body["cap_band"] = c.cap_band(body["market_cap"])
        if self.params and (body.get("target_pct") is None or body.get("stop_pct") is None):
            # The score is against MBin's defaults; record which ones (§8.1 `score`).
            body["inputs"] = {**(body.get("inputs") or {}), "params_version": self.params["params_version"]}
        prob = body.get("probability") or {}
        rid = c.rec_id(body["symbol"], body["data_ts_utc"], body["strategy_id"],
                       body.get("target_pct") or prob.get("target_pct"), prob.get("horizon"))
        if rid in self.active:
            return None
        msg = {**self._header("RECOMMENDATION", c.rec_key(rid)), "rec_id": rid, **body}
        if self._send(msg) is None:
            return None
        self.active[rid] = msg
        return msg

    def status_message(self, reason: str, correlation_id: str | None = None) -> dict:
        def consumes(t: str) -> dict:
            return {"type": t, "versions": [{"version": 1, "schema_sha256": c.schema_sha256(t)}]}

        msg = {
            **self._header("CONTRACT_STATUS", f"cs:mtap:{self.instance_id}:{self._next_seq()}"),
            "party": c.PARTY, "instance_id": self.instance_id, "started_ts_utc": self.started_ts_utc,
            "protocol_version": c.PROTOCOL_VERSION, "reason": reason,
            "consumes": [consumes(t) for t in c.CONSUMES],
            "produces": [{"type": t, "version": 1, "schema_sha256": c.schema_sha256(t)} for t in c.PRODUCES],
            "problems": [
                {"type": t, "problem": p, **{k: v for k, v in vars(pr).items()}}
                for (t, p), pr in sorted(self.problems.items())
            ],
        }
        if correlation_id:
            msg["correlation_id"] = correlation_id
        return msg

    def send_status(self, reason: str, correlation_id: str | None = None) -> str | None:
        return self._send(self.status_message(reason, correlation_id), audit=reason != "heartbeat")

    def request_resync(self, reason: str = "startup") -> str | None:
        """§5 MTap start sequence. POSITIONS_SNAPSHOT joins in phase 2."""
        seq = self._next_seq()
        msg = {
            **self._header("RESYNC_REQUEST", f"rs:mtap:{self.instance_id}:{seq}"),
            "request_id": f"mtap:{self.instance_id}:{seq}", "party": c.PARTY,
            "requested_types": ["TRADING_PARAMETERS", "CONTRACT_STATUS", "RECOMMENDATION_OUTCOME"],
            "since_ts_utc": self.last_outcome_ts, "reason": reason,
        }
        return self._send(msg)

    def start(self) -> None:
        self.send_status("startup")
        self.request_resync("startup")

    def stop(self) -> None:
        self.send_status("shutdown")

    # -- incoming -----------------------------------------------------------------------

    def _problem(self, msg_type: str, problem: str, *, version: int | None = None,
                 key: str | None = None, detail: str | None = None) -> None:
        now = self._now_ts()
        pr = self.problems.get((msg_type, problem))
        if pr is None:
            self.problems[(msg_type, problem)] = Problem(1, now, now, version, key, detail)
        else:
            pr.count += 1
            pr.last_ts_utc = now
            pr.received_version, pr.sample_key, pr.detail = version, key, detail
        log.warning("contract problem %s on %s (%s): %s", problem, msg_type, key, detail)
        last = self._problem_sent.get(msg_type)
        if last is None or self.clock() - last >= PROBLEM_REPEAT_S:
            self._problem_sent[msg_type] = self.clock()
            self.send_status("problem")

    def check(self, stream_type: str, raw: str | None) -> dict | None:
        """§6.4 checks on one entry from an `mbin:evt:` stream. The message, or None after
        recording the problem (the caller still acks it)."""
        try:
            msg = json.loads(raw) if raw is not None else None
        except ValueError as exc:
            self._problem(stream_type, "unparseable", detail=str(exc)[:300])
            return None
        if not isinstance(msg, dict):
            self._problem(stream_type, "unparseable", detail="not a JSON object")
            return None
        msg_type, version, key = msg.get("type"), msg.get("schema_version"), msg.get("_key")
        key = key if isinstance(key, str) else None
        if msg_type not in KNOWN_TYPES:
            self._problem(str(msg_type)[:64], "unknown_type", key=key)
            return None
        if msg_type not in c.CONSUMES or not isinstance(version, int) or not c.has_schema(msg_type, version):
            self._problem(msg_type, "unsupported_version",
                          version=version if isinstance(version, int) else None, key=key)
            return None
        err = c.validate(msg)
        if err:
            self._problem(msg_type, "validation_failed", version=version, key=key, detail=err)
            return None
        if c.epoch_ms(msg["ts_utc"]) > (self.clock() + CLOCK_SKEW_S) * 1000:
            self._problem(msg_type, "clock_skew", version=version, key=key,
                          detail=f"ts_utc {msg['ts_utc']} is ahead of {self._now_ts()}")
        return msg

    def on_params(self, msg: dict) -> None:
        if self.params is None or msg["ts_utc"] >= self.params["ts_utc"]:  # latest wins (§4)
            self.params = msg
            self._expire()

    def on_mbin_status(self, msg: dict) -> None:
        if self.mbin_status is not None and msg["ts_utc"] < self.mbin_status["ts_utc"]:
            return
        self.mbin_status, self.mbin_seen_s = msg, self.clock()
        for p in msg["produces"]:
            known = p["type"] in c.CONSUMES and p["schema_sha256"] and c.has_schema(p["type"], p["version"])
            if known and p["schema_sha256"] != c.schema_sha256(p["type"], p["version"]):
                self._problem(p["type"], "schema_mismatch", version=p["version"],
                              key=msg["_key"], detail="MBin built against other schema bytes")
        ours = {t: c.schema_sha256(t) for t in c.PRODUCES}
        for entry in msg["consumes"]:
            if entry["type"] in ours:
                hashes = {v["schema_sha256"] for v in entry["versions"] if v["version"] == 1}
                if hashes and ours[entry["type"]] not in hashes:
                    log.error("MBin consumes %s v1 with other schema bytes", entry["type"])
        for pr in msg["problems"]:
            if pr["type"] in c.PRODUCES:
                log.error("MBin reports %s on our %s (x%s): %s", pr["problem"], pr["type"],
                          pr["count"], pr.get("detail"))

    def mbin_online(self) -> bool:
        return self.mbin_seen_s is not None and self.clock() - self.mbin_seen_s <= OFFLINE_S

    def on_resync(self, msg: dict) -> int:
        """Answer MBin's RESYNC_REQUEST (§8.7). Returns the number of messages re-sent."""
        requester = msg["request_id"].rsplit(":", 1)[0]
        last = self._resync_answered.get(requester)
        if last is not None and self.clock() - last < RESYNC_MIN_GAP_S:
            log.info("RESYNC %s within %s s of the last answer; skipped", msg["request_id"], RESYNC_MIN_GAP_S)
            return 0
        self._resync_answered[requester] = self.clock()
        sent = 0
        if "RECOMMENDATION" in msg["requested_types"]:
            self._expire()
            for rec in list(self.active.values()):
                again = {**rec, "ts_utc": self._now_ts(), "_ts": int(self.clock() * 1000),
                         "correlation_id": msg["request_id"]}
                sent += self._send(again) is not None
        if "CONTRACT_STATUS" in msg["requested_types"]:
            sent += self.send_status("heartbeat", correlation_id=msg["request_id"]) is not None
        log.info("RESYNC %s (%s): re-sent %d", msg["request_id"], msg["reason"], sent)
        return sent

    def on_outcome(self, msg: dict) -> bool:
        """Store one RECOMMENDATION_OUTCOME (dedup on `_key`). True when it is new."""
        if msg["_key"] in self._outcome_keys:
            return False
        self._append("outcomes", msg)
        self._outcome_keys.add(msg["_key"])
        if self.last_outcome_ts is None or msg["outcome_ts_utc"] > self.last_outcome_ts:
            self.last_outcome_ts = msg["outcome_ts_utc"]
        return True

    def status_file(self) -> dict:
        return {
            "ts_utc": self._now_ts(), "instance_id": self.instance_id,
            "mbin_online": self.mbin_online(),
            "mbin_instance": (self.mbin_status or {}).get("instance_id"),
            "params_version": (self.params or {}).get("params_version"),
            "trading_state": (self.params or {}).get("trading_state"),
            "active_recs": len(self.active), "last_outcome_ts_utc": self.last_outcome_ts,
            "problems": [{"type": t, "problem": p, "count": pr.count} for (t, p), pr in self.problems.items()],
        }


def _dt(epoch_s: float) -> datetime:
    return datetime.fromtimestamp(epoch_s, tz=UTC)


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            log.warning("skipping a bad line in %s", path)
    return out


# -- the loop ---------------------------------------------------------------------------

STATE_TYPES = ("TRADING_PARAMETERS", "CONTRACT_STATUS")


def run(url: str, *, consumer: str = "recommender-1", stop_after_s: float | None = None) -> int:
    """The service loop. SIGTERM / SIGINT publish the shutdown CONTRACT_STATUS and exit."""
    from redis.exceptions import RedisError

    client = bus._client(url)
    rec = Recommender(client)
    stopping = {"flag": False}

    def _stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    candidates = bus.StreamConsumer(CANDIDATE_EVENT, CANDIDATE_GROUP, consumer, url,
                                    prefix=CANDIDATE_PREFIX, client=client)
    outcomes = bus.StreamConsumer("RECOMMENDATION_OUTCOME", OUTCOME_GROUP, consumer, url,
                                  prefix=bus.MBIN_PREFIX, client=client)
    next_beat = time.time() + HEARTBEAT_S
    began = time.time()
    status_path = rec.root / "status.json"
    tail: dict[str, str] | None = None
    first = True
    while not stopping["flag"]:
        try:
            if tail is None:
                tail = _connect(rec, candidates, outcomes, reason="startup" if first else "gap_detected")
                first = False
                log.info("recommender %s up: %s", rec.instance_id, url.split("@")[-1])
            reply = client.xread(tail, count=50, block=1000) or []
            for key, entries in reply:
                t = key.removeprefix(bus.MBIN_PREFIX)
                for entry_id, fields in entries:
                    tail[key] = entry_id
                    _dispatch(rec, t, (fields or {}).get("data"))
            _drain_candidates(candidates, rec)
            _drain_outcomes(outcomes, rec)
            if time.time() >= next_beat:
                rec.send_status("heartbeat")
                next_beat = time.time() + HEARTBEAT_S
        except (RedisError, OSError) as exc:  # Redis down (the daily stop, a restart), NOGROUP
            if tail is not None or first:
                log.warning("Redis unavailable (%s); retrying every %s s", exc, RECONNECT_S)
            tail = None
            time.sleep(RECONNECT_S)
        if time.time() - began >= HEARTBEAT_S or stop_after_s is not None:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(rec.status_file(), indent=1))
        if stop_after_s is not None and time.time() - began >= stop_after_s:
            break
    try:
        rec.stop()
    except (RedisError, OSError) as exc:
        log.warning("shutdown CONTRACT_STATUS not sent: %s", exc)
    return 0


def _connect(rec: Recommender, candidates: bus.StreamConsumer, outcomes: bus.StreamConsumer,
             *, reason: str) -> dict[str, str]:
    """(Re)create the groups, run the §5 start sequence, and return the XREAD positions:
    state streams from their newest entry (handled now, §4), RESYNC_REQUEST from the
    current end (tail-only). After a Redis loss the RESYNC carries `gap_detected` (§5)."""
    candidates.ensure_group()
    outcomes.ensure_group()   # a new group starts at 0 (REVIEW M11)
    client = rec.client
    if reason == "startup":
        rec.start()           # CONTRACT_STATUS first, then the RESYNC_REQUEST
    else:
        rec.send_status("heartbeat")
        rec.request_resync(reason)
    tail: dict[str, str] = {}
    for t in STATE_TYPES:
        key = bus.stream_key(t, bus.MBIN_PREFIX)
        newest = client.xrevrange(key, count=1)
        tail[key] = newest[0][0] if newest else "0-0"
        if newest:
            _dispatch(rec, t, newest[0][1].get("data"))
    key = bus.stream_key("RESYNC_REQUEST", bus.MBIN_PREFIX)
    newest = client.xrevrange(key, count=1)
    # A concrete id, not `$`: `$` on every XREAD would drop a request that lands between reads.
    tail[key] = newest[0][0] if newest else "0-0"
    return tail


def _dispatch(rec: Recommender, stream_type: str, raw: str | None) -> None:
    msg = rec.check(stream_type, raw)
    if msg is None:
        return
    if msg["type"] == "TRADING_PARAMETERS":
        rec.on_params(msg)
    elif msg["type"] == "CONTRACT_STATUS":
        rec.on_mbin_status(msg)
    elif msg["type"] == "RESYNC_REQUEST":
        rec.on_resync(msg)


def _drain_candidates(consumer: bus.StreamConsumer, rec: Recommender) -> None:
    """Candidates: handle each, then ack it whatever happened (a bad one must not block)."""
    entries = consumer.read(count=50, block_ms=1)
    for entry_id, payload in entries:
        if "_trimmed" in payload or "_undecodable" in payload:
            log.error("candidate %s unreadable: %s", entry_id, payload)
            continue
        payload.pop("_ts", None)   # stamped by the internal publish, not a candidate field
        payload.pop("_key", None)
        try:
            rec.recommend(payload)
        except Exception:
            log.exception("candidate %s failed", entry_id)
    consumer.ack([i for i, _ in entries])


def _drain_outcomes(consumer: bus.StreamConsumer, rec: Recommender) -> None:
    """Outcomes: §6.4 checks, store, then ack (ack after the write, protocol §4)."""
    entries = consumer.read(count=50, block_ms=1)
    for entry_id, payload in entries:
        if "_trimmed" in payload:
            log.warning("outcome %s was trimmed before it was read", entry_id)
            continue
        if "_undecodable" in payload:
            rec.check("RECOMMENDATION_OUTCOME", "{")   # records `unparseable`
            continue
        msg = rec.check("RECOMMENDATION_OUTCOME", json.dumps(payload))
        if msg is not None and msg["type"] == "RECOMMENDATION_OUTCOME":
            rec.on_outcome(msg)
    consumer.ack([i for i, _ in entries])
