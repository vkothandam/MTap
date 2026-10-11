"""Intraday feed storage: canonical rows -> Arrow segments -> parquet -> NAS."""

from __future__ import annotations

import copy
import json
import threading
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sourcing_py.common import config, storage
from sourcing_py.intraday import feed, store, summary, upload

FIXTURE = Path(__file__).resolve().parents[2] / "schemas" / "fixtures" / "intraday_feed_batch.etrade.json"


def golden() -> dict:
    return json.loads(FIXTURE.read_text())


def batch_at(ts: str, symbols=("AAPL", "MSFT"), watchlist="Experimental") -> dict:
    b = golden()
    b["ts_utc"] = ts
    rows = []
    for sym in symbols:
        r = copy.deepcopy(b["rows"][0])
        r.update(symbol=sym, ts_utc=ts)
        r["source_meta"]["watchlist"] = watchlist
        rows.append(r)
    b["rows"] = rows
    return b


class FakeConsumer:
    """Stands in for common.bus.StreamConsumer."""

    def __init__(self, entries=(), pending=(), stop: threading.Event | None = None):
        self.entries = list(entries)
        self._pending = list(pending)
        self.acked: list[str] = []
        self.stop = stop

    def ensure_group(self):
        pass

    def pending(self):
        yield from self._pending

    def read(self, count=50, block_ms=1000):
        if self.entries:
            out, self.entries = self.entries[:count], self.entries[count:]
            return out
        if self.stop is not None:
            self.stop.set()
        return []

    def ack(self, ids):
        self.acked.extend(ids)

    def lag(self):
        return {"pending": 0, "lag": 0}


@pytest.fixture
def roots(monkeypatch, tmp_path):
    for var in ("MTAP_DATA_ROOT", "MTAP_STATE_ROOT", "OUT_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "settings", lambda: {})
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "local"))
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tmp_path / "nas"))
    (tmp_path / "nas").mkdir()
    return tmp_path


CFG = {"flush_rows": 20000, "flush_s": 0, "idle_close_s": 600, "upload_retry_s": 300, "redis_url": ""}


# -- canonical rows ----------------------------------------------------------------------

def test_golden_batch_to_arrow():
    b = golden()
    table = feed.to_table(feed.rows_to_columns(b["rows"], "1-0"))
    assert table.schema == feed.ARROW_SCHEMA
    rows = table.to_pylist()
    assert [r["symbol"] for r in rows] == ["AAPL", "MSFT"]
    aapl, msft = rows
    assert aapl["price"] == 273.06 and aapl["bid"] == 273.05 and aapl["stream_id"] == "1-0"
    assert json.loads(aapl["source_meta"]) == {"watchlist": "Experimental", "watchlist_id": "215565674306"}
    # Fields the feed left out are null, not 0.
    assert msft["day_open"] is None and msft["bid"] is None and msft["open_i"] is None
    assert msft["volume_cum"] == 0
    assert feed.partition(feed.parse_ts(b["ts_utc"])) == ("2026-10-14", 10)


def test_partition_uses_new_york_time():
    # 03:30 UTC on the 15th is still the 14th in New York (EDT).
    assert feed.partition(feed.parse_ts("2026-10-15T03:30:00Z")) == ("2026-10-14", 23)
    # After the DST change (EST, UTC-5): 14:30 UTC is 09:30 ET.
    assert feed.partition(feed.parse_ts("2026-11-02T14:30:00Z")) == ("2026-11-02", 9)


# -- the store ---------------------------------------------------------------------------

def test_store_acks_only_after_the_segment_is_written(roots):
    consumer = FakeConsumer()
    st = store.Store(consumer, root=store.segments_root(), cfg=CFG)
    st.add("1-0", golden())
    assert consumer.acked == []  # buffered, not yet on disk
    st.flush()
    assert consumer.acked == ["1-0"]
    seg = st.open["etrade_watchlist"]
    assert seg.path.suffix == ".arrows" and seg.path.parent.name == "date=2026-10-14"
    assert seg.path.name.startswith("h10-")
    st.close_all()
    files = list(store.segments_root().rglob("*.parquet"))
    assert len(files) == 1 and not list(store.segments_root().rglob("*.arrows"))
    table = pq.read_table(files[0])
    assert table.num_rows == 2 and table.column("stream_id").to_pylist() == ["1-0", "1-0"]


def test_store_closes_a_segment_when_the_hour_changes(roots):
    closed = []
    st = store.Store(FakeConsumer(), root=store.segments_root(), cfg=CFG, on_close=closed.append)
    st.add("1-0", batch_at("2026-10-14T14:59:58.000Z"))
    st.flush()
    st.add("2-0", batch_at("2026-10-14T15:00:01.000Z"))
    st.flush()
    assert [p.name[:3] for p in closed] == ["h10"]
    st.close_all()
    names = sorted(p.name[:3] for p in store.segments_root().rglob("*.parquet"))
    assert names == ["h10", "h11"]


def test_bad_entries_are_acked_and_counted(roots):
    consumer = FakeConsumer()
    st = store.Store(consumer, root=store.segments_root(), cfg=CFG)
    st.add("1-0", {"schema_version": 99, "rows": []})
    st.add("2-0", {"schema_version": 1, "rows": [{"symbol": "X"}]})  # no ts_utc / source
    assert st.due()
    st.flush()
    assert consumer.acked == ["1-0", "2-0"]
    assert st.stats["bad_entries"] == 2 and st.stats["rows"] == 0
    assert not list(store.segments_root().rglob("*.arrows"))


def test_recover_a_crashed_segment(roots):
    root = store.segments_root()
    seg = store.Segment.open(root, "etrade_watchlist", "2026-10-14", 10)
    seg.write(feed.to_table(feed.rows_to_columns(golden()["rows"], "1-0")))
    seg.write(feed.to_table(feed.rows_to_columns(golden()["rows"], "2-0")))
    seg._sink.close()  # the process died: no end-of-stream marker
    size = seg.path.stat().st_size
    with open(seg.path, "r+b") as fh:  # and the last batch was half written
        fh.truncate(size - 40)
    out = store.recover(root)
    assert len(out) == 1 and not seg.path.exists()
    assert pq.read_table(out[0]).column("stream_id").to_pylist() == ["1-0", "1-0"]


# -- upload ------------------------------------------------------------------------------

def _closed_segment() -> Path:
    st = store.Store(FakeConsumer(), root=store.segments_root(), cfg=CFG)
    st.add("1-0", golden())
    st.close_all()
    (local,) = store.segments_root().rglob("*.parquet")
    return local


def test_upload_waits_for_the_marker_then_moves(roots):
    local = _closed_segment()
    done, failed, reason = upload.upload_all(store.segments_root())
    assert (done, failed) == (0, 0) and "marker" in reason
    assert local.exists()

    assert storage.init() == 0
    done, failed, reason = upload.upload_all(store.segments_root())
    assert (done, failed, reason) == (1, 0, None)
    assert not local.exists()
    (nas,) = (roots / "nas" / "intraday_feed").rglob("*.parquet")
    assert nas.parent.name == "source=etrade_watchlist" and nas.parent.parent.name == "date=2026-10-14"
    assert pq.read_metadata(nas).num_rows == 2
    assert not list((roots / "nas").rglob("*.tmp"))


def test_upload_after_a_crash_between_copy_and_delete(roots):
    storage.init()
    local = _closed_segment()
    dest = upload.destination(local, store.segments_root())
    dest.parent.mkdir(parents=True)
    dest.write_bytes(local.read_bytes())  # already on the NAS, local copy not yet deleted
    assert upload.upload_all(store.segments_root())[:2] == (1, 0)
    assert not local.exists() and len(list(dest.parent.glob("*.parquet"))) == 1


# -- the whole loop ----------------------------------------------------------------------

def test_follow_end_to_end(roots):
    storage.init()
    stop = threading.Event()
    entries = [("1-0", batch_at("2026-10-14T14:31:00.000Z")), ("2-0", batch_at("2026-10-14T14:31:02.000Z"))]
    pending = [("0-1", batch_at("2026-10-14T14:30:58.000Z"))]
    consumer = FakeConsumer(entries, pending, stop)
    t = threading.Thread(target=store.follow, kwargs={"stop": stop, "consumer": consumer})
    t.start()
    t.join(timeout=30)
    assert not t.is_alive()
    assert consumer.acked == ["0-1", "1-0", "2-0"]
    assert not list(store.segments_root().rglob("*.parquet")) and not list(store.segments_root().rglob("*.arrows"))
    (nas,) = (roots / "nas" / "intraday_feed").rglob("*.parquet")
    assert pq.read_metadata(nas).num_rows == 6
    hb = json.loads((roots / "local" / "intraday" / "store.heartbeat").read_text())
    assert hb["stopped"] and hb["rows"] == 6


def test_a_second_store_refuses_to_start(roots):
    lock = store._singleton_lock(roots / "local" / "intraday" / "store.lock")
    try:
        with pytest.raises(SystemExit):
            store.follow(stop=threading.Event(), consumer=FakeConsumer())
    finally:
        lock.close()


# -- summary -----------------------------------------------------------------------------

def test_summary_reports_rth_gaps(roots):
    st = store.Store(FakeConsumer(), root=store.segments_root(), cfg=CFG)
    # Polls at 09:30:05, 09:31:05, 09:45:00 and 10:00:00 ET, then nothing to the close.
    for i, ts in enumerate(["13:30:05", "13:31:05", "13:45:00", "14:00:00"]):
        st.add(f"{i}-0", batch_at(f"2026-10-14T{ts}.000Z", watchlist="MAIN"))
        st.flush()
    out = summary.summarize(summary.load_day(date(2026, 10, 14)), date(2026, 10, 14))
    main = out["MAIN"]
    assert main["polls"] == 4 and main["rows"] == 8 and main["symbols"] == 2
    assert [g[:2] for g in main["gaps"]] == [("09:31:05", "09:45:00"), ("09:45:00", "10:00:00"), ("10:00:00", "16:00:00")]
    # Open .arrows segments are read too (the shutdown summary runs before they close).
    assert st.open


# -- Redis wire format (needs a running redis-server) -------------------------------------

@pytest.mark.skipif(not __import__("os").environ.get("REDIS_URL"), reason="needs REDIS_URL")
def test_redis_round_trip():
    import os
    import uuid

    from sourcing_py.common import bus

    url = os.environ["REDIS_URL"]
    event = f"TEST_INTRADAY_{uuid.uuid4().hex[:8]}"
    consumer = bus.StreamConsumer(event, "storage", "storage-1", url)
    try:
        consumer.ensure_group()
        consumer.ensure_group()  # BUSYGROUP is fine
        eid = bus.publish(url, event, golden())
        got = consumer.read(block_ms=500)
        assert [e for e, _ in got] == [eid] and got[0][1]["rows"][0]["symbol"] == "AAPL"
        # Unacked: a restarted consumer with the same name sees it again.
        again = bus.StreamConsumer(event, "storage", "storage-1", url)
        assert [e for e, _ in again.pending()] == [eid]
        consumer.ack([eid])
        assert list(again.pending()) == []
        assert consumer.lag()["pending"] == 0
    finally:
        consumer.r.delete(bus.stream_key(event))


def test_failed_ack_is_retried_not_lost(roots):
    class Flaky(FakeConsumer):
        fail = True

        def ack(self, ids):
            if self.fail:
                raise ConnectionError("redis down")
            super().ack(ids)

    consumer = Flaky()
    st = store.Store(consumer, root=store.segments_root(), cfg=CFG)
    st.add("1-0", golden())
    st.flush()  # written, ack failed
    assert consumer.acked == [] and st.due()
    consumer.fail = False
    st.add("2-0", golden())
    st.flush()
    assert consumer.acked == ["1-0", "2-0"]
    st.close_all()
    (seg,) = store.segments_root().rglob("*.parquet")
    assert pq.read_metadata(seg).num_rows == 4


# -- failure paths found in review -------------------------------------------------------

def test_decode_flags_trimmed_and_undecodable_entries():
    from sourcing_py.common import bus

    c = bus.StreamConsumer("X", "g", "c", "", client=object())
    reply = [("k", [("1-0", None), ("2-0", {"data": "{not json"}), ("3-0", {"data": "[1]"}),
                    ("4-0", {"data": json.dumps(golden())})])]
    got = dict(c._decode(reply))
    assert got["1-0"] == {"_trimmed": True}
    assert "_undecodable" in got["2-0"] and "_undecodable" in got["3-0"]
    assert got["4-0"]["rows"][0]["symbol"] == "AAPL"


def test_trimmed_and_unbuildable_entries_are_acked_without_breaking_the_flush(roots):
    consumer = FakeConsumer()
    st = store.Store(consumer, root=store.segments_root(), cfg=CFG)
    bad = golden()
    bad["rows"][0]["price"] = {"not": "a number"}
    st.add("1-0", {"_trimmed": True})
    st.add("2-0", bad)
    st.add("3-0", {"_undecodable": "x"})
    st.add("4-0", golden())
    st.flush()
    assert consumer.acked == ["1-0", "2-0", "3-0", "4-0"]
    assert st.stats["trimmed_entries"] == 1 and st.stats["bad_entries"] == 2 and st.stats["rows"] == 2


def test_follow_recreates_the_group_after_a_redis_restart(roots):
    storage.init()
    stop = threading.Event()

    class Restarted(FakeConsumer):
        groups = 0
        failed = False

        def ensure_group(self):
            self.groups += 1

        def read(self, count=50, block_ms=1000):
            if not self.failed:
                self.failed = True
                raise RuntimeError("NOGROUP No such key 'mtap:evt:INTRADAY_FEED' or consumer group")
            return super().read(count, block_ms)

    consumer = Restarted([("1-0", batch_at("2026-10-14T14:31:00.000Z"))], (), stop)
    t = threading.Thread(target=store.follow, kwargs={"stop": stop, "consumer": consumer})
    t.start()
    t.join(timeout=30)
    assert not t.is_alive()
    assert consumer.groups == 2 and consumer.acked == ["1-0"]


def test_uploader_stop_does_not_wait_on_a_hung_nas(roots, monkeypatch):
    storage.init()
    _closed_segment()
    release = threading.Event()
    monkeypatch.setattr(upload, "upload_all", lambda root: (release.wait(), (0, 0, None))[1])
    u = upload.Uploader(store.segments_root(), retry_s=300)
    u.start()
    import time as _t

    t0 = _t.monotonic()
    monkeypatch.setattr(u, "join", lambda timeout=None: None)  # don't actually wait 30 s
    u.stop(drain=True)
    assert _t.monotonic() - t0 < 5 and u.is_alive()
    release.set()
