"""Economy / shares sources + the reference loader, offline."""

from __future__ import annotations

import json
from datetime import date

import duckdb
import pytest

from sourcing_py.common import config, schema
from sourcing_py.etrade.reference_ingest import ingest_reference
from sourcing_py.sources.massive_inflation import source as infl
from sourcing_py.sources.massive_inflation.source import MassiveInflationSource
from sourcing_py.sources.massive_shares import source as shares
from sourcing_py.sources.massive_shares.source import MassiveSharesSource
from sourcing_py.sources.massive_treasury_yields.source import MassiveTreasuryYieldsSource


@pytest.fixture
def roots(monkeypatch, tmp_path):
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    return tmp_path


def _conform(source, records):
    schema.validate_records(records, schema.load_schema(config.load_source(source.name).schema_path))


def test_yields_extract_conforms():
    s = MassiveTreasuryYieldsSource()
    recs = s.extract([{"date": "2026-10-23", "yield_1_month": 4.11, "yield_10_year": 5.0, "extra": 1}, {"x": 1}])
    assert recs[0]["yield_10_year"] == 5.0 and recs[0]["yield_2_year"] is None and len(recs) == 1
    _conform(s, recs)


def test_inflation_extract_conforms_and_due_logic(roots):
    s = MassiveInflationSource()
    recs = s.extract([{"date": "2026-08-01", "cpi": 334.1, "cpi_core": 337.7, "pce": 131.5},
                      {"date": "2026-09-01", "cpi": 336.3, "cpi_core": 338.9}])
    _conform(s, recs)
    d = roots / "out" / "inflation"
    d.mkdir(parents=True)
    today = date(2026, 10, 27)
    assert infl.expected_month(today) == date(2026, 8, 1)
    assert infl.expected_month(date(2026, 1, 5)) == date(2025, 11, 1)
    assert infl.is_due(d, today) is True  # nothing stored
    (d / "inflation-a.jsonl").write_text(json.dumps({"date": "2026-07-01", "cpi": 1}) + "\n")
    assert infl.is_due(d, today) is True
    (d / "inflation-b.jsonl").write_text(json.dumps({"date": "2026-08-01", "cpi": 1}) + "\n"
                                         + json.dumps({"date": "2026-09-01", "cpi": None}) + "\n")
    assert infl.is_due(d, today) is False


def test_shares_snapshot_dates_and_universe(roots):
    snaps = shares.snapshot_dates(date(2026, 10, 27))
    assert snaps[0] == date(2021, 9, 30) and snaps[-1] == date(2026, 9, 30)
    assert date(2022, 12, 30) in snaps  # 12-31 is a Saturday: last session before it
    t = roots / "out" / "tickers_cs"
    t.mkdir(parents=True)
    rows = [{"ticker": "AAA", "active": True, "type": "CS"},
            {"ticker": "ETF1", "active": True, "type": "ETF"},
            {"ticker": "ADR", "active": True, "type": "ADRC"},
            {"ticker": "GONE", "active": False, "delisted_utc": "2023-01-05T00:00:00Z", "type": "CS"},
            {"ticker": "LATE", "active": False, "delisted_utc": "2022-02-01T00:00:00Z", "type": "CS"},
            {"ticker": "OLDFMT", "active": True}]
    (t / "tickers_cs-2026-10-27.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert shares.universe(date(2022, 6, 30)) == ["AAA", "ADR", "GONE", "OLDFMT"]
    assert shares.universe(date(2023, 3, 31)) == ["AAA", "ADR", "OLDFMT"]


def test_shares_fetch_idempotent_and_conforms(roots, monkeypatch):
    t = roots / "out" / "tickers_cs"
    t.mkdir(parents=True)
    (t / "tickers_cs-2026-10-27.jsonl").write_text(
        "\n".join(json.dumps({"ticker": x, "active": True, "type": "CS"}) for x in ("AAA", "BBB")))
    calls = []

    def fake_one(self, client, cfg, symbol, snap):
        calls.append((symbol, snap))
        return None if symbol == "BBB" else {
            "symbol": symbol, "snap_date": snap, "market_cap": 1e9,
            "weighted_shares_outstanding": 5e7, "share_class_shares_outstanding": 5e7, "close_at_snap": None}

    monkeypatch.setattr(MassiveSharesSource, "_one", fake_one)
    s = MassiveSharesSource()
    p = s.run(params={"date": "2024-03-28"})
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert [r["symbol"] for r in rows] == ["AAA"] and rows[0]["market_cap"] == 1e9
    assert s.run(params={"date": "2024-03-28"}) is None  # file exists: skipped, no calls
    assert len(calls) == 2
    _conform(s, s.extract(rows[:0] or [{k: v for k, v in rows[0].items() if not k.startswith("_")}]))


def _write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_ingest_reference_upserts_and_first_release_inflation(roots):
    out = roots / "out"
    _write(out / "treasury_yields" / "treasury_yields-20261020T000000Z.jsonl",
           [{"date": "2026-10-19", "yield_10_year": 4.9, "_ingested_at": "2026-10-20T00:00:00+00:00"}])
    _write(out / "treasury_yields" / "treasury_yields-20261027T000000Z.jsonl",
           [{"date": "2026-10-19", "yield_10_year": 4.95, "_ingested_at": "2026-10-27T00:00:00+00:00"},
            {"date": "2026-10-23", "yield_10_year": 5.0, "yield_2_year": 4.69, "_ingested_at": "2026-10-27T00:00:00+00:00"}])
    _write(out / "inflation" / "inflation-20260915T000000Z.jsonl",
           [{"date": "2026-08-01", "cpi": 334.0, "cpi_core": 337.0, "_ingested_at": "2026-09-15T00:00:00+00:00"},
            {"date": "2026-07-01", "cpi": 332.0, "_ingested_at": "2026-09-15T00:00:00+00:00"}])
    _write(out / "inflation" / "inflation-20261027T000000Z.jsonl",
           [{"date": "2026-08-01", "cpi": 334.131, "cpi_core": 337.765, "_ingested_at": "2026-10-27T00:00:00+00:00"},
            {"date": "2026-09-01", "cpi": 336.3, "_ingested_at": "2026-10-27T00:00:00+00:00"}])
    _write(out / "shares" / "shares-2024-03-28.jsonl",
           [{"symbol": "AAA", "snap_date": "2024-03-28", "market_cap": 1e9, "weighted_shares_outstanding": 5e7}])
    con = duckdb.connect()
    counts = ingest_reference(con)
    ingest_reference(con)  # idempotent

    y = dict(con.execute("SELECT date, yield_10_year FROM treasury_yields").fetchall())
    assert y[date(2026, 10, 19)] == 4.95 and len(y) == 2  # latest vintage wins
    cpi = dict(con.execute("SELECT date, cpi FROM inflation").fetchall())
    assert cpi[date(2026, 8, 1)] == 334.0  # first release, not the revision
    assert cpi[date(2026, 9, 1)] == 336.3 and len(cpi) == 3
    assert con.execute("SELECT count(*) FROM inflation_vintages").fetchone()[0] == 4
    assert con.execute("SELECT market_cap, close_at_snap FROM share_snapshots").fetchone() == (1e9, None)
    assert counts["share_snapshots"] == 1
    cols = [r[0] for r in con.execute("DESCRIBE treasury_yields").fetchall()]
    assert cols[0] == "date" and cols[-1] == "fetched_at" and "yield_30_year" in cols
    # a later run adds a new month and still keeps the first release of the old ones
    _write(out / "inflation" / "inflation-20261114T000000Z.jsonl",
           [{"date": "2026-08-01", "cpi": 999.0}, {"date": "2026-10-01", "cpi": 340.0}])
    ingest_reference(con)
    cpi = dict(con.execute("SELECT date, cpi FROM inflation").fetchall())
    assert cpi[date(2026, 8, 1)] == 334.0 and cpi[date(2026, 10, 1)] == 340.0
