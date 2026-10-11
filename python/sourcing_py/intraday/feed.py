"""The canonical intraday row (schemas/intraday_feed.schema.json) as an Arrow table."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pyarrow as pa

NY = ZoneInfo("America/New_York")
SCHEMA_VERSION = 1

_FLOATS = (
    "price", "open_i", "high_i", "low_i", "volume_cum", "volume_i", "vwap_session", "vwap_i",
    "day_open", "prev_close", "bid", "ask", "bid_size", "ask_size", "vol_10d_src",
)

ARROW_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("ts_utc", pa.timestamp("ms", tz="UTC")),
        ("source", pa.string()),
        ("kind", pa.string()),
        ("feed_seq", pa.int64()),
        *[(name, pa.float64()) for name in _FLOATS],
        ("trades_i", pa.int64()),
        ("source_meta", pa.string()),  # JSON text
        ("stream_id", pa.string()),  # Redis entry id, for audit
    ],
    metadata={"intraday_feed_schema_version": str(SCHEMA_VERSION)},
)


def parse_ts(value: str) -> datetime:
    """ISO-8601 (with a trailing Z) -> aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def partition(ts: datetime) -> tuple[str, int]:
    """(New York date, New York hour) of a UTC time: the segment a row belongs to."""
    local = ts.astimezone(NY)
    return local.date().isoformat(), local.hour


def rows_to_columns(rows: list[dict], stream_id: str) -> dict[str, list]:
    """Canonical row dicts (missing fields = null) -> column lists for ARROW_SCHEMA."""
    cols: dict[str, list] = {f.name: [] for f in ARROW_SCHEMA}
    for row in rows:
        cols["symbol"].append(row["symbol"])
        cols["ts_utc"].append(parse_ts(row["ts_utc"]))
        cols["source"].append(row["source"])
        cols["kind"].append(row["kind"])
        cols["feed_seq"].append(row.get("feed_seq"))
        for name in _FLOATS:
            v = row.get(name)
            cols[name].append(float(v) if v is not None else None)
        cols["trades_i"].append(row.get("trades_i"))
        meta = row.get("source_meta")
        cols["source_meta"].append(json.dumps(meta, sort_keys=True) if meta else None)
        cols["stream_id"].append(stream_id)
    return cols


def to_table(cols: dict[str, list]) -> pa.Table:
    return pa.table(cols, schema=ARROW_SCHEMA)
