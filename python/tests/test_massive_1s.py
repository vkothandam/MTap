"""Inputs for the NAS 1-second-bar fetcher (sourcing_py/massive_1s) and the Rust contract."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
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
    assert "2 kept of 3 listed" in capsys.readouterr().out
    inputs = prep.inputs_dir()
    with (inputs / "universe.csv").open() as f:
        got = [(r["ticker"], r["kept"]) for r in csv.DictReader(f)]
    assert got == [("AAPL", "1"), ("BXSL", "0"), ("SPY", "1")]
    with (inputs / "windows.csv").open() as f:
        assert [r["date"] for r in csv.DictReader(f)] == ["2025-11-26", "2025-11-28"]


def _tickers_file(roots, day: str, rows: list[dict]) -> None:
    d = roots / "out" / "tickers_cs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"tickers_cs-{day}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _bars_month(month: str, rows: list[tuple[str, str]]) -> None:
    """rows = (ticker, New York 'YYYY-MM-DD HH:MM')."""
    d = Path(config.data_root()) / "bars_1m" / f"month={month}"
    d.mkdir(parents=True, exist_ok=True)
    ms = [int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=prep.NY).timestamp() * 1000) for _, s in rows]
    pq.write_table(pa.table({"ticker": [t for t, _ in rows], "t": pa.array(ms, pa.int64()),
                             "c": [1.0] * len(rows), "v": [100.0] * len(rows)}), d / "bars-2022-01-01_2022-03-31.parquet")


def test_traded_reads_new_york_dates_across_month_folders(roots):
    _bars_month("2022-02", [("EARLY", "2022-02-28 09:31"), ("FEB", "2022-02-28 15:59")])
    _bars_month("2022-03", [("MAR", "2022-03-01 04:00"), ("MAR", "2022-03-02 10:00"), ("LATE", "2022-03-31 19:59")])
    assert prep.traded(date(2022, 3, 1), date(2022, 3, 31)) == {"MAR", "LATE"}
    assert prep.traded(date(2022, 2, 28), date(2022, 3, 1)) == {"EARLY", "FEB", "MAR"}
    with pytest.raises(FileNotFoundError):
        prep.traded(date(2021, 11, 1), date(2021, 11, 30))


def test_stock_list_from_bars_keeps_delisted_and_old_tickers_and_drops_adrs():
    rows = {
        "AAPL": [{"ticker": "AAPL", "name": "Apple Inc.", "active": True, "type": "CS", "primary_exchange": "XNAS"}],
        "FI": [{"ticker": "FI", "name": "Fiserv, Inc.", "active": False, "type": "CS", "primary_exchange": "XNYS"}],
        "GONE": [{"ticker": "GONE", "name": "Gone Corp", "active": False, "type": "CS"}],
        "TSM": [{"ticker": "TSM", "name": "Taiwan Semiconductor ADR", "active": True, "type": "ADRC"}],
        "PFD": [{"ticker": "PFD", "name": "X Corp Depositary Shares", "active": True, "type": "CS"}],
        # an old file without `type`, then a newer one that renamed the name: both CS, the clean name wins
        "OLD": [{"ticker": "OLD", "name": "Old Co 5% Notes due 2030", "active": False},
                {"ticker": "OLD", "name": "Old Co", "active": False}],
    }
    out = {s["ticker"]: (s["kept"], s["reason"], s["listed"])
           for s in prep.stock_list_from_bars({"AAPL", "FI", "GONE", "TSM", "PFD", "OLD", "WHO"}, rows, add=["SPY"])}
    assert out == {
        "AAPL": (1, "", 1), "FI": (1, "", 0), "GONE": (1, "", 0), "OLD": (1, "", 0),
        "TSM": (0, "adr", 1), "PFD": (0, "preferred", 1), "WHO": (0, "no tickers_cs row", 0), "SPY": (1, "", 1),
    }
    assert list(out) == sorted(out)


def test_all_tickers_finds_a_ticker_only_an_older_file_has(roots):
    _tickers_file(roots, "2025-01-06", [{"ticker": "FI", "name": "Fiserv, Inc.", "active": True}])
    _tickers_file(roots, "2026-10-27", [{"ticker": "FISV", "name": "Fiserv, Inc.", "active": True, "type": "CS"}])
    rows = prep.all_tickers()
    assert set(rows) == {"FI", "FISV"} and rows["FI"][0]["active"] is True


def test_prepare_from_bars_writes_the_traded_universe(roots, capsys):
    _tickers_file(roots, "2026-10-27", [
        {"ticker": "AAPL", "name": "Apple Inc.", "active": True, "type": "CS", "primary_exchange": "XNAS"},
        {"ticker": "FI", "name": "Fiserv, Inc.", "active": False, "type": "CS", "primary_exchange": "XNYS"},
        {"ticker": "NEW", "name": "Listed Later Inc", "active": True, "type": "CS", "primary_exchange": "XNAS"}])
    _bars_month("2022-03", [("AAPL", "2022-03-01 10:00"), ("FI", "2022-03-01 10:00")])
    assert main(["prepare", "--fromdate", "2022-03-01", "--todate", "2022-03-31", "--from-bars"]) == 0
    out = capsys.readouterr().out
    assert "3 kept of 3 traded in bars_1m" in out and "1 kept stock(s) are no longer listed" in out
    with (prep.inputs_dir() / "universe.csv").open() as f:
        got = [(r["ticker"], r["kept"], r["listed"]) for r in csv.DictReader(f)]
    assert got == [("AAPL", "1", "1"), ("FI", "1", "0"), ("SPY", "1", "1")]  # NEW never traded then


def test_prepare_with_a_name_writes_its_own_folder(roots, capsys):
    _tickers_file(roots, "2026-11-04", [
        {"ticker": "AAPL", "name": "Apple Inc.", "active": True, "type": "CS", "primary_exchange": "XNAS"},
        {"ticker": "TSM", "name": "Taiwan Semiconductor", "active": True, "type": "ADRC"}])
    assert main(["prepare", "--fromdate", "2026-10-19", "--todate", "2026-11-06", "--name", "d2"]) == 0
    assert "2 kept of 2 listed" in capsys.readouterr().out
    assert not prep.inputs_dir().exists(), "inputs/ (a running job's) is not touched"
    with (prep.inputs_dir("d2") / "universe.csv").open() as f:
        assert [(r["ticker"], r["kept"]) for r in csv.DictReader(f)] == [("AAPL", "1"), ("SPY", "1")]
    with (prep.inputs_dir("d2") / "windows.csv").open() as f:
        dates = [r["date"] for r in csv.DictReader(f)]
    assert (dates[0], dates[-1], len(dates)) == ("2026-10-19", "2026-11-06", 15)
    for bad in (["--name"], ["--name", "../x"]):
        assert main(["prepare", "--fromdate", "2026-10-19", "--todate", "2026-11-06", *bad]) == 2


def test_prepare_rejects_an_unknown_option(roots, capsys):
    assert main(["prepare", "--fromdate", "2022-03-01", "--todate", "2022-03-31", "--frombars"]) == 2
    assert "unknown option(s): frombars" in capsys.readouterr().err


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
