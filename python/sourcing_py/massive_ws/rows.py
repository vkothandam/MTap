"""Massive `A` (per-second aggregate) messages -> the agg_1s Arrow table.

Its own compact schema, not the canonical intraday_feed one: Massive bars carry no
quotes, and `z` is the average trade size (not a trade count). Column names match
intraday_feed where the meaning is the same. Fractional-share volumes (`dv`, `dav`)
are dropped; `v`/`av` are the whole-share volumes. See schemas/intraday_feed.md.
"""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1
SOURCE = "massive_ws"
SORT_BY = ("symbol", "ts_utc")

AGG_1S_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),            # sym
        ("ts_utc", pa.timestamp("ms", tz="UTC")),  # s: bar start (end is s + 1 s)
        ("price", pa.float64()),            # c: bar close
        ("open_i", pa.float64()),           # o
        ("high_i", pa.float64()),           # h
        ("low_i", pa.float64()),            # l
        ("volume_i", pa.int64()),           # v: bar volume
        ("volume_cum", pa.int64()),         # av: today's volume so far
        ("vwap_i", pa.float64()),           # vw: bar VWAP
        ("vwap_session", pa.float64()),     # a: today's VWAP
        ("day_open", pa.float64()),         # op: official open (null before it prints)
        ("avg_trade_size", pa.int64()),     # z
        ("otc", pa.bool_()),                # otc (omitted by Massive when false)
    ],
    metadata={"agg_1s_schema_version": str(SCHEMA_VERSION), "source": SOURCE},
)

_FIELDS = (
    ("price", "c", float), ("open_i", "o", float), ("high_i", "h", float), ("low_i", "l", float),
    ("volume_i", "v", int), ("volume_cum", "av", int), ("vwap_i", "vw", float),
    ("vwap_session", "a", float), ("day_open", "op", float), ("avg_trade_size", "z", int),
)


class Columns:
    """Accumulates bars column-wise until the next flush."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cols: dict[str, list] = {f.name: [] for f in AGG_1S_SCHEMA}
        self.n = 0
        self.max_s: int | None = None  # newest bar start (ms) in the buffer

    def add(self, msg: dict) -> None:
        """Append one `A` message. Raises KeyError/TypeError/ValueError on a malformed
        one, before anything is appended."""
        sym, s = msg["sym"], int(msg["s"])
        if not isinstance(sym, str):
            raise TypeError(f"sym is {type(sym).__name__}")
        values = [(name, None if msg.get(key) is None else cast(msg[key])) for name, key, cast in _FIELDS]
        c = self.cols
        c["symbol"].append(sym)
        c["ts_utc"].append(s)
        for name, v in values:
            c[name].append(v)
        c["otc"].append(bool(msg.get("otc")))
        self.n += 1
        if self.max_s is None or s > self.max_s:
            self.max_s = s

    def table(self) -> pa.Table:
        return pa.table(self.cols, schema=AGG_1S_SCHEMA)
