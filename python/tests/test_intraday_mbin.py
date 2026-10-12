"""ingest-mbin: MBin watchlist CSVs -> canonical intraday feed rows."""

from __future__ import annotations

import json
from datetime import date

import pyarrow.parquet as pq
import pytest

from sourcing_py.common import config
from sourcing_py.intraday import feed, mbin

DAY = date(2025, 12, 19)
MAIN_HEADER = ("Symbol, Volume, Ask, Bid, Ask Size, Bid Size, Qty, Open, Previous Close, Change %, "
               "10 Day Volume, Last Trade, Day's Gain %, VWAP, sysdate")
MC_HEADER = ("Symbol, Last Price, Change $, Change %, Price Paid, Date Acquired, "
             "Change since Added %, Volume, VWAP, sysdate")


def main_line(sym, vol, open_, last, ts, prev=100.0):
    return f"{sym}, {vol}, 101.5, 101.4, 100, 300, 100, {open_}, {prev}, 0, 4133172, {last}, 0, 99.9, {ts}"


def mc_line(sym, last, vol, ts):
    return f"{sym}, {last}, 0, 0, 14.96, 12/15/2025, 4.3, {vol}, 15.61, {ts}"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tmp_path / "nas"))
    root = tmp_path / "mbin"
    d = root / DAY.isoformat()
    d.mkdir(parents=True)
    (d / "AAPL_06-16-05.csv").write_text("\n".join([
        MAIN_HEADER,
        main_line("AAPL", 1000, 0, 100.0, "2025-12-19T14:16:25.988Z"),
        main_line("AAPL", 1500, 101.0, 101.2, "2025-12-19T14:30:03.100Z"),
        main_line("AAPL", 1500, 101.0, 101.2, "2025-12-19T14:30:03.100Z"),  # duplicate poll
    ]) + "\n")
    # A second session file for the same symbol overlapping the first one.
    (d / "AAPL_09-00-00.csv").write_text("\n".join([
        MAIN_HEADER,
        main_line("AAPL", 1500, 101.0, 101.2, "2025-12-19T14:30:03.100Z"),
        main_line("AAPL", "1700", 101.0, "--", "2025-12-19T14:30:05.200Z"),
    ]) + "\n")
    (d / "MSFT_06-16-05.csv").write_text("\n".join([
        MAIN_HEADER, main_line("MSFT", 50, 0, 480.0, "2025-12-19T14:16:25.988Z", prev=480.0)]) + "\n")
    (d / "MC_AAL_06-16-05.csv").write_text("\n".join([
        MC_HEADER,
        mc_line("AAL", 15.61, 89149, "2025-12-19T14:16:27.536Z"),
        mc_line("AAL", 15.70, 99149, "2025-12-19T14:30:10.000Z"),
        mc_line("AAL", 15.80, 99999, "2025-12-20T06:00:00.000Z"),  # the next NY day: dropped
    ]) + "\n")
    return root


def test_split_files_keeps_m_symbols_in_main(tree):
    main, mc = mbin.split_files(tree / DAY.isoformat())
    assert [p.name for p in main] == ["AAPL_06-16-05.csv", "AAPL_09-00-00.csv", "MSFT_06-16-05.csv"]
    assert [p.name for p in mc] == ["MC_AAL_06-16-05.csv"]


def test_read_day_maps_like_the_live_adapter(tree):
    t = mbin.read_day(tree / DAY.isoformat(), DAY)
    assert t.schema.equals(feed.ARROW_SCHEMA, check_metadata=False)
    rows = t.to_pylist()
    aapl = [r for r in rows if r["symbol"] == "AAPL"]
    assert len(aapl) == 3  # deduped on (symbol, ts) across both session files
    first, opened, late = aapl
    assert first["day_open"] is None  # Open 0 before the open -> null
    assert opened["day_open"] == 101.0 and opened["price"] == 101.2
    assert first["prev_close"] == 100.0 and first["vol_10d_src"] == 4133172
    assert first["bid"] == 101.4 and first["bid_size"] == 300
    assert late["volume_cum"] == 1700 and late["price"] is None  # "--" -> null
    assert first["vwap_session"] is None  # MBin's VWAP is dropped
    assert json.loads(first["source_meta"]) == {
        "origin": "mbin", "watchlist": "Experimental", "watchlist_id": "215565674306"}
    assert [r["feed_seq"] for r in aapl] == [0, 1, 2]

    aal = [r for r in rows if r["symbol"] == "AAL"]
    assert len(aal) == 2  # the row on the next New York date is dropped
    assert aal[0]["price"] == 15.61 and aal[0]["volume_cum"] == 89149
    assert aal[0]["prev_close"] is None and aal[0]["bid"] is None
    assert json.loads(aal[0]["source_meta"])["watchlist"] == "MidCap"
    assert [r["symbol"] for r in rows if r["symbol"] == "MSFT"] == ["MSFT"]


def test_older_header_without_days_gain_is_read_by_name(tree):
    d = tree / DAY.isoformat()
    old = MAIN_HEADER.replace(" Day's Gain %,", "")
    (d / "MSFT_09-00-00.csv").write_text(
        old + "\nMSFT, 70, 481.5, 481.4, 100, 300, 100, 480.5, 480.0, 0, 4133172, 481.0, 480.9, 2025-12-19T14:30:03.100Z\n")
    msft = [r for r in mbin.read_day(d, DAY).to_pylist() if r["symbol"] == "MSFT"]
    assert [(r["volume_cum"], r["price"], r["day_open"]) for r in msft] == [(50, 480.0, None), (70, 481.0, 480.5)]
    seqs = {r["ts_utc"]: r["feed_seq"] for r in mbin.read_day(d, DAY).to_pylist() if "Experimental" in r["source_meta"]}
    assert sorted(seqs.values()) == list(range(len(seqs)))


def test_midcap_rows_under_a_main_header_are_read_by_width(tree):
    d = tree / DAY.isoformat()
    old = MAIN_HEADER.replace(" Day's Gain %,", "")
    (d / "MC_AAON_06-24-35.csv").write_text(old + "\n" + mc_line("AAON", 77.93, 1168, "2025-12-19T14:30:51.246Z") + "\n")
    (d / "MC_EMPTY_06-24-35.csv").write_text(old + "\n")
    aaon = [r for r in mbin.read_day(d, DAY).to_pylist() if r["symbol"] == "AAON"]
    assert [(r["price"], r["volume_cum"]) for r in aaon] == [(77.93, 1168)]


def test_ingest_writes_atomically_and_leaves_the_source_alone(tree):
    before = {p: p.read_bytes() for p in tree.rglob("*") if p.is_file()}
    done = mbin.ingest(tree)
    assert [d["date"] for d in done] == [DAY.isoformat()]
    assert done[0]["main"]["symbols"] == 2 and done[0]["mc"]["symbols"] == 1
    out = config.data_root() / "intraday_feed" / f"date={DAY}" / "source=etrade_watchlist" / "mbin.parquet"
    assert out.exists() and not list(out.parent.glob(".*tmp"))
    assert pq.read_table(out).num_rows == done[0]["rows"]
    assert {p: p.read_bytes() for p in tree.rglob("*") if p.is_file()} == before


def test_ingest_skips_holidays_and_refuses_to_write_into_the_source(tree, monkeypatch):
    (tree / "2025-12-25").mkdir()
    assert [d["date"] for d in mbin.ingest(tree)] == [DAY.isoformat()]
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tree))
    with pytest.raises(ValueError):
        mbin.ingest(tree)


def test_summary_reads_the_backfill(tree):
    from sourcing_py.intraday import summary

    mbin.ingest(tree)
    t = summary.load_day(DAY)
    assert t.num_rows == 6
    assert set(summary.summarize(t, DAY)) == {"Experimental", "MidCap"}


def test_session_open_and_close_include_half_days():
    from sourcing_py.utils import trading_calendar as tc

    assert tc.session_open_utc(DAY).isoformat() == "2025-12-19T14:30:00+00:00"
    assert tc.session_close_utc(DAY).isoformat() == "2025-12-19T21:00:00+00:00"
    assert tc.session_close_utc(date(2025, 12, 24)).isoformat() == "2025-12-24T18:00:00+00:00"


def test_cli_ingest_needs_root():
    from sourcing_py.intraday import cli

    assert cli.main(["ingest-mbin"]) == 2

