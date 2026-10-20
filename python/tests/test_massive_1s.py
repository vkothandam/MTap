"""Inputs for the NAS 1-second-bar fetcher (sourcing_py/massive_1s) and the Rust contract."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sourcing_py.common import config
from sourcing_py.massive_1s import prepare as prep
from sourcing_py.massive_1s.cli import main
from sourcing_py.massive_ws.rows import AGG_1S_SCHEMA

FIXTURE = Path(__file__).parent / "fixtures" / "agg_1s" / "massive_rest_sample.parquet"


@pytest.fixture
def roots(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / config.DATA_ROOT_MARKER).write_text("")
    monkeypatch.setenv("MTAP_DATA_ROOT", str(data))
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    return tmp_path


@pytest.mark.parametrize("name, reason", [
    ("Apple Inc.", None),
    ("Energy Transfer LP Common Units", None),
    ("Realty Income Corporation", None),
    ("Agnc Investment Corp. Depositary Shares", "preferred"),
    ("ADAMAS Inc 8.00% Senior Notes Due 2030", "note"),
    ("Blackstone Secured Lending Fund", "fund"),
    ("Some Acquisition Corp Rights", "rights"),
    (None, None),
])
def test_the_name_filter(name, reason):
    assert prep.exclusion(name) == reason


def test_stock_list_is_listed_only_sorted_and_marked():
    rows = [
        {"ticker": "ZZZ", "name": "Zed Inc", "active": True, "primary_exchange": "XNYS"},
        {"ticker": "GONE", "name": "Gone Inc", "active": False},
        {"ticker": "AAA", "name": "A Corp 6% Notes due 2031", "active": True, "primary_exchange": "XNAS"},
    ]
    out = prep.stock_list(rows)
    assert [(s["ticker"], s["kept"], s["reason"]) for s in out] == [("AAA", 0, "note"), ("ZZZ", 1, "")]


def test_windows_run_from_nine_to_close_plus_seven_minutes():
    (full, half) = prep.windows(date(2025, 11, 26), date(2025, 11, 28))  # 27th: Thanksgiving
    def ny(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, prep.NY).strftime("%Y-%m-%d %H:%M")
    assert (full[0], ny(full[1]), ny(full[2])) == ("2025-11-26", "2025-11-26 09:00", "2025-11-26 16:07")
    assert (half[0], ny(half[1]), ny(half[2])) == ("2025-11-28", "2025-11-28 09:00", "2025-11-28 13:07")


def test_prepare_writes_both_inputs(roots, capsys):
    d = roots / "out" / "tickers_cs"
    d.mkdir(parents=True)
    rows = [{"ticker": "AAPL", "name": "Apple Inc.", "active": True, "primary_exchange": "XNAS"},
            {"ticker": "BXSL", "name": "Blackstone Secured Lending Fund", "active": True, "primary_exchange": "XNYS"}]
    (d / "tickers_cs-2026-10-15.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert main(["prepare", "--fromdate", "2025-11-26", "--todate", "2025-11-28"]) == 0
    assert "1 kept of 2 listed" in capsys.readouterr().out
    inputs = prep.inputs_dir()
    with (inputs / "universe.csv").open() as f:
        assert [(r["ticker"], r["kept"]) for r in csv.DictReader(f)] == [("AAPL", "1"), ("BXSL", "0")]
    with (inputs / "windows.csv").open() as f:
        assert [r["date"] for r in csv.DictReader(f)] == ["2025-11-26", "2025-11-28"]


def test_prepare_refuses_a_data_root_without_its_marker(roots):
    (Path(config.data_root()) / config.DATA_ROOT_MARKER).unlink()
    assert main(["prepare", "--fromdate", "2025-11-26", "--todate", "2025-11-28"]) == 1


def test_the_rust_day_file_has_the_agg_1s_schema():
    """nas/massive1s writes this (cargo test with MASSIVE1S_FIXTURE_OUT); the column set,
    types and order must match AGG_1S_SCHEMA so both sources read as one table."""
    t = pq.read_table(FIXTURE)
    assert t.schema.equals(AGG_1S_SCHEMA, check_metadata=False)
    assert t.schema.metadata[b"source"] == b"massive_rest"
    assert t.schema.metadata[b"agg_1s_schema_version"] == b"2"
    assert t.column("symbol").to_pylist() == ["AAA", "BBB", "BBB"]
    for name in ("volume_cum", "volume_cum_dec", "vwap_session", "day_open"):
        assert t.column(name).null_count == t.num_rows
