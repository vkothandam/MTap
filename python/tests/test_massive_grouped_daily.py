"""Massive grouped-daily extraction + schema conformance, using a sample payload.

Uses a canned API response so it runs offline (no key, no network).
"""

from __future__ import annotations

from sourcing_py.common import config, schema
from sourcing_py.sources.massive_grouped_daily.source import MassiveGroupedDailySource

# Shape of a real /v2/aggs/grouped response (trimmed to two tickers).
SAMPLE = {
    "adjusted": True,
    "resultsCount": 2,
    "results": [
        {"T": "AAPL", "o": 224.5, "h": 225.35, "l": 222.76, "c": 224.72,
         "v": 46414013, "vw": 224.13, "n": 512345, "t": 1723694400000},
        {"T": "MSFT", "o": 415.1, "h": 418.0, "l": 413.2, "c": 416.9,
         "v": 18234100, "vw": 416.02, "n": 210987, "t": 1723694400000},
    ],
}


def _extract():
    source = MassiveGroupedDailySource()
    source._date = "2024-08-15"  # normally set by fetch()
    return source, source.extract(SAMPLE)


def test_extract_maps_fields():
    _, records = _extract()
    assert len(records) == 2
    aapl = records[0]
    assert aapl["ticker"] == "AAPL"
    assert aapl["open"] == 224.5
    assert aapl["close"] == 224.72
    assert aapl["volume"] == 46414013
    assert aapl["vwap"] == 224.13
    assert aapl["transactions"] == 512345
    assert aapl["window_start_ms"] == 1723694400000
    assert aapl["date"] == "2024-08-15"


def test_records_conform_to_schema():
    source, records = _extract()
    declared = schema.load_schema(config.load_source(source.name).schema_path)
    schema.validate_records(records, declared)  # raises on failure
