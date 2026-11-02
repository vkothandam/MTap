"""Redis Streams consumer, wire-compatible with node/src/common/bus/RedisBus.ts.

The wire format is the contract, not shared code: one stream per event
(`<prefix><EVENT>`, `mtap:evt:` by default), the JSON envelope in a `data` field, one
consumer group per logical consumer (each group sees every entry). Only what the Python
consumers need is here: a durable group reader with explicit acks, and a publisher.

The prefix is per call (MTap↔MBin protocol §2, REVIEW item M5): MTap writes `mtap:evt:`,
and reads MBin's messages from `mbin:evt:`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

STREAM_PREFIX = "mtap:evt:"
MBIN_PREFIX = "mbin:evt:"

# Per-stream length caps (MAXLEN ~), the Python twin of RedisBus.ts EVENT_MAXLEN. The
# contract streams follow protocol §4: durable ones keep a full trading day, state ones
# about 200 entries.
DEFAULT_MAXLEN = 4000
EVENT_MAXLEN: dict[str, int] = {
    "INTRADAY_FEED": 4000,
    "RECOMMENDATION": 20000,
    "EXIT_RECOMMENDATION": 20000,
    "RECOMMENDATION_OUTCOME": 20000,
    "TRADING_PARAMETERS": 200,
    "POSITIONS_SNAPSHOT": 200,
    "CONTRACT_STATUS": 200,
    "RESYNC_REQUEST": 200,
}


def stream_key(event: str, prefix: str = STREAM_PREFIX) -> str:
    return f"{prefix}{event}"


def _client(url: str):
    import redis  # imported lazily so tests without Redis never need the module loaded

    return redis.Redis.from_url(url, decode_responses=True)


class StreamConsumer:
    """A durable consumer-group reader.

    Durable means: the group is created at the start of the stream (`0`), so entries
    published before the consumer first ran are read, and a restart resumes after the
    last ack. A stable `consumer` name lets a restarted process re-read its own
    delivered-but-unacked entries first (`pending()`).
    """

    def __init__(
        self, event: str, group: str, consumer: str, url: str, *, prefix: str = STREAM_PREFIX,
        client=None,
    ):
        self.key = stream_key(event, prefix)
        self.group = group
        self.consumer = consumer
        self.r = client if client is not None else _client(url)

    def ensure_group(self) -> None:
        try:
            self.r.xgroup_create(self.key, self.group, id="0", mkstream=True)
        except Exception as exc:  # redis.ResponseError: BUSYGROUP when it already exists
            if "BUSYGROUP" not in str(exc):
                raise

    def _decode(self, reply) -> list[tuple[str, dict[str, Any]]]:
        """Entries as (id, payload). Never raises on one bad entry: an entry trimmed from
        the stream while pending comes back as {"_trimmed": True}, undecodable JSON as
        {"_undecodable": ...}, so the consumer can count and ack them."""
        out: list[tuple[str, dict[str, Any]]] = []
        for _stream, entries in reply or []:
            for entry_id, fields in entries:
                if not fields or "data" not in fields:
                    out.append((entry_id, {"_trimmed": True}))
                    continue
                try:
                    payload = json.loads(fields["data"])
                except ValueError as exc:
                    payload = {"_undecodable": str(exc)}
                out.append((entry_id, payload if isinstance(payload, dict) else {"_undecodable": "not an object"}))
        return out

    def pending(self, count: int = 100) -> Iterator[tuple[str, dict[str, Any]]]:
        """This consumer's delivered-but-unacked entries (from before a restart)."""
        last = "0"
        while True:
            reply = self.r.xreadgroup(self.group, self.consumer, {self.key: last}, count=count)
            entries = self._decode(reply)
            if not entries:
                return
            yield from entries
            last = entries[-1][0]

    def read(self, count: int = 50, block_ms: int = 1000) -> list[tuple[str, dict[str, Any]]]:
        reply = self.r.xreadgroup(
            self.group, self.consumer, {self.key: ">"}, count=count, block=block_ms
        )
        return self._decode(reply)

    def ack(self, ids: list[str]) -> None:
        if ids:
            self.r.xack(self.key, self.group, *ids)

    def lag(self) -> dict[str, Any]:
        """Group lag + pending count, for the heartbeat and `storage check`."""
        for g in self.r.xinfo_groups(self.key):
            if g.get("name") == self.group:
                return {"pending": g.get("pending"), "lag": g.get("lag")}
        return {}


def publish(
    url: str, event: str, payload: dict[str, Any], *, key: str | None = None,
    prefix: str = STREAM_PREFIX, maxlen: int | None = None, client=None,
) -> str:
    """XADD one envelope (the Python twin of Bus.publish + RedisBus.publish).

    Stamps `_ts` (epoch ms, unless the caller already set it, e.g. to validate the exact
    message first) and, when `key` is given, `_key`, as the Node Bus does. There is no
    producer-side dedup here, so a RESYNC re-publish needs no bypass (REVIEW M7).
    """
    r = client if client is not None else _client(url)
    msg = {**payload}
    msg.setdefault("_ts", int(time.time() * 1000))
    if key is not None:
        msg["_key"] = key
    cap = maxlen if maxlen is not None else EVENT_MAXLEN.get(event, DEFAULT_MAXLEN)
    return r.xadd(stream_key(event, prefix), {"data": json.dumps(msg)}, maxlen=cap, approximate=True)
