"""Massive per-second collector: A messages -> agg_1s segments -> parquet -> NAS agg_1s/."""

from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sourcing_py.common import config
from sourcing_py.intraday.feed import NY
from sourcing_py.intraday.upload import upload_all
from sourcing_py.massive_ws import collector as mc
from sourcing_py.massive_ws.client import FeedError
from sourcing_py.massive_ws.rows import AGG_1S_SCHEMA, Columns
from sourcing_py.sources.massive_grouped_daily.gaps import missing_sessions


def ms(hhmmss: str, day: str = "2026-10-14") -> int:
    return int(datetime.fromisoformat(f"{day}T{hhmmss}").replace(tzinfo=NY).timestamp() * 1000)


def bar(sym: str, s: int, **kw) -> dict:
    # Shape of a real delayed-feed message (dv/dav included, op absent before the open).
    return {"ev": "A", "sym": sym, "v": 957, "dv": "957.0", "av": 25794291, "dav": "25794291.5",
            "vw": 3.8, "o": 3.8, "c": 3.81, "h": 3.82, "l": 3.79, "a": 3.7832, "z": 478,
            "s": s, "e": s + 1000, **kw}


CFG = {"ws_url": "ws://unused", "channels": "A.*", "start_et": "09:00", "stop_et": "16:21",
       "roll_min": 5, "flush_s": 0, "flush_rows": 50000, "max_local_gb": 10, "min_free_gb": 0,
       "upload_retry_s": 300}


@pytest.fixture
def roots(monkeypatch, tmp_path):
    for var in ("MTAP_DATA_ROOT", "MTAP_STATE_ROOT", "OUT_ROOT", "MASSIVE_WS_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "settings", dict)
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "local"))
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tmp_path / "nas"))
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    (tmp_path / "nas").mkdir()
    (tmp_path / "nas" / config.DATA_ROOT_MARKER).write_text("x")
    return tmp_path


def collector(**cfg) -> mc.Collector:
    return mc.Collector({**CFG, **cfg}, root=mc.segments_root(), key="k")


# -- rows ----------------------------------------------------------------------------------
def test_a_message_maps_to_the_agg_1s_columns():
    cols = Columns()
    cols.add(bar("TCRT", ms("09:31:07")))
    cols.add(bar("ABCD", ms("09:31:08"), op=3.7, otc=True, v=12.0))
    t = cols.table()
    assert t.schema == AGG_1S_SCHEMA
    r = t.to_pylist()
    assert r[0]["symbol"] == "TCRT" and r[0]["price"] == 3.81 and r[0]["volume_i"] == 957
    assert r[0]["volume_cum"] == 25794291 and r[0]["vwap_session"] == 3.7832
    assert r[0]["avg_trade_size"] == 478 and r[0]["day_open"] is None and r[0]["otc"] is False
    assert r[1]["day_open"] == 3.7 and r[1]["otc"] is True and r[1]["volume_i"] == 12
    assert cols.max_s == ms("09:31:08")


def test_a_malformed_message_appends_nothing():
    cols = Columns()
    for broken in ({"ev": "A", "s": 1}, {"ev": "A", "sym": "X", "s": 1, "v": "lots"}):
        with pytest.raises((KeyError, TypeError, ValueError)):
            cols.add(broken)
    assert cols.n == 0 and all(not v for v in cols.cols.values())


def test_slot_rolls_every_five_minutes_of_bar_time():
    assert mc.slot(ms("09:34:59"), 5) == ("2026-10-14", 9, "h09-m30")
    assert mc.slot(ms("09:35:00"), 5) == ("2026-10-14", 9, "h09-m35")
    assert mc.slot(ms("16:00:01"), 5) == ("2026-10-14", 16, "h16-m00")


def test_window_is_new_york_time_on_trading_days():
    at = lambda d, t: datetime.fromisoformat(f"{d}T{t}").replace(tzinfo=NY)
    assert mc.in_window(at("2026-10-14", "09:00"), CFG)
    assert mc.in_window(at("2026-10-14", "16:20:59"), CFG)
    assert not mc.in_window(at("2026-10-14", "16:21"), CFG)
    assert not mc.in_window(at("2026-10-14", "08:59"), CFG)
    assert not mc.in_window(at("2026-10-17", "10:00"), CFG)  # Saturday
    assert not mc.in_window(at("2026-11-26", "10:00"), CFG)  # Thanksgiving


# -- segments and upload -------------------------------------------------------------------
def test_frames_become_sorted_five_minute_parquet_then_move_to_the_nas(roots):
    c = collector()
    c.q.put(json.dumps([bar("MSFT", ms("09:34:58")), bar("AAPL", ms("09:34:59"))]))
    c.q.put(json.dumps([{"ev": "status", "status": "success", "message": "subscribed to: A.*"}]))
    c.tick(want=False)  # parses, flushes into h09-m30
    c.q.put(json.dumps([bar("AAPL", ms("09:35:00")), bar("AAPL", ms("09:35:01"))]))
    c.tick(want=False)  # rolls: h09-m30 closes; window closed, so h09-m35 closes too
    assert c.seg is None and c.stats["segments_closed"] == 2 and c.stats["bars"] == 4
    assert c.stats["last_status"] == "success: subscribed to: A.*"
    local = sorted(mc.segments_root().rglob("*.parquet"))
    assert [p.name.split("-")[0] + "-" + p.name.split("-")[1] for p in local] == ["h09-m30", "h09-m35"]
    first = pq.read_table(local[0])
    assert first.column("symbol").to_pylist() == ["AAPL", "MSFT"]  # sorted by symbol, ts

    done, failed, reason = upload_all(mc.segments_root(), mc.nas_dir())
    assert (done, failed, reason) == (2, 0, None)
    nas = sorted((roots / "nas" / "agg_1s" / "date=2026-10-14" / "source=massive_ws").glob("*.parquet"))
    assert len(nas) == 2 and sum(pq.read_metadata(p).num_rows for p in nas) == 4
    assert not list(mc.segments_root().rglob("*.parquet"))  # local copies gone
    assert not (roots / "nas" / "intraday_feed").exists()  # E*TRADE tree untouched


def test_a_reconnect_keeps_the_open_segment(roots):
    c = collector()
    c.q.put(json.dumps([bar("AAPL", ms("09:31:00"))]))
    c.tick(want=True)
    c.disconnect()  # what a dropped session looks like to tick(): no session
    c.next_connect = time.monotonic() + 3600
    c.tick(want=True)
    assert c.seg is not None and c.stats["segments_closed"] == 0
    c.close_segment()


def test_the_local_guard_pauses_and_resumes(roots):
    c = collector(max_local_gb=1e-6)  # 1 kB
    seg_dir = mc.segments_root() / "massive_ws" / "date=2026-10-14"
    seg_dir.mkdir(parents=True)
    (seg_dir / "h09-m30-x.parquet").write_bytes(b"0" * 5000)
    c.check_guard()
    assert c.paused
    c.tick(want=True)
    assert c.session is None and c.state(True) == "paused"
    (seg_dir / "h09-m30-x.parquet").unlink()
    c.check_guard()
    assert not c.paused


# -- end to end against a local WebSocket server -------------------------------------------
class FakeMassive:
    """Speaks the Massive handshake; each connection sends the next list of frames, then closes."""

    def __init__(self, sessions: list[list[list[dict]]], auth_ok: bool = True):
        from websockets.sync.server import serve

        self.sessions, self.auth_ok = list(sessions), auth_ok
        self.auths: list[str] = []
        self.subs: list[str] = []
        self.server = serve(self.handler, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def handler(self, ws):
        ws.send(json.dumps([{"ev": "status", "status": "connected", "message": "Connected Successfully"}]))
        self.auths.append(json.loads(ws.recv())["params"])
        if not self.auth_ok:
            ws.send(json.dumps([{"ev": "status", "status": "auth_failed", "message": "authentication failed"}]))
            return
        ws.send(json.dumps([{"ev": "status", "status": "auth_success", "message": "authenticated"}]))
        self.subs.append(json.loads(ws.recv())["params"])
        for frame in self.sessions.pop(0) if self.sessions else []:
            ws.send(json.dumps(frame))
        if not self.sessions:
            for _ in ws:  # last session: stay open until the client leaves
                pass

    def close(self):
        self.server.shutdown()


def test_follow_end_to_end_with_a_reconnect(roots, monkeypatch):
    server = FakeMassive([
        [[bar("AAPL", ms("09:31:00")), bar("MSFT", ms("09:31:00"))]],  # then the server drops us
        [[bar("AAPL", ms("09:31:01"))], [bar("AAPL", ms("09:36:00"))]],
    ])
    monkeypatch.setenv("MASSIVE_WS_URL", server.url)
    monkeypatch.setenv("MASSIVE_WS_FLUSH_S", "0.1")
    try:
        assert mc.follow(window=False, duration_s=4) == 0
    finally:
        server.close()
    assert server.auths == ["test-key", "test-key"] and server.subs == ["A.*", "A.*"]
    nas = list((roots / "nas" / "agg_1s").rglob("*.parquet"))
    assert sum(pq.read_metadata(p).num_rows for p in nas) == 4
    assert sorted(p.name[:7] for p in nas) == ["h09-m30", "h09-m35"]
    hb = json.loads(mc.heartbeat_path().read_text())
    assert hb["stopped"] and hb["bars"] == 4 and hb["disconnects"] == 1 and hb["state"] == "idle"
    assert hb["pending_upload"]["files"] == 0


def test_auth_failure_backs_off_for_ten_minutes(roots):
    server = FakeMassive([], auth_ok=False)
    c = mc.Collector({**CFG, "ws_url": server.url}, root=mc.segments_root(), key="bad")
    try:
        c.tick(want=True)
        c.session.join(timeout=5)
        c.tick(want=True)
    finally:
        server.close()
    assert isinstance(c.stats["last_error"], str) and "auth_failed" in c.stats["last_error"]
    assert c.session is None and c.next_connect - time.monotonic() > 590


def test_a_second_collector_refuses_to_start(roots):
    from sourcing_py.intraday.store import _singleton_lock

    held = _singleton_lock(mc.lock_path(), "massive-ws follow")
    try:
        with pytest.raises(SystemExit, match="massive-ws follow"):
            mc.follow(window=False, duration_s=0)
    finally:
        held.close()


def test_feed_error_carries_the_status():
    assert FeedError("max_connections", "x").status == "max_connections"


# -- daily gaps ----------------------------------------------------------------------------
def test_daily_gaps_skip_holidays_weekends_and_today(tmp_path: Path):
    (tmp_path / "daily_summary-2026-09-03.jsonl").write_text("")
    # 2026-09-07 is Labor Day; 09-05/06 a weekend; 09-09 is "today".
    got = missing_sessions(tmp_path, date(2026, 9, 9), days=7, cap=10)
    assert got == [date(2026, 9, 2), date(2026, 9, 4), date(2026, 9, 8)]
    assert missing_sessions(tmp_path, date(2026, 9, 9), days=7, cap=2) == [date(2026, 9, 4), date(2026, 9, 8)]
    for d in ("02", "04", "08"):
        (tmp_path / f"daily_summary-2026-09-{d}.jsonl").write_text("")
    assert missing_sessions(tmp_path, date(2026, 9, 9), days=7) == []


# -- end-of-day combine ----------------------------------------------------------------------
def _part(name: str, bars: list[dict]) -> Path:
    from sourcing_py.massive_ws.compact import day_dir

    cols = Columns()
    for b in bars:
        cols.add(b)
    path = day_dir("2026-10-14") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(cols.table(), path)
    return path


def test_compact_combines_a_day_into_one_sorted_file_and_is_rerunnable(roots):
    from sourcing_py.massive_ws.compact import compact, day_dir

    _part("h09-m30-a.parquet", [bar("ZZZ", ms("09:30:00")), bar("AAA", ms("09:30:01"))])
    _part("h09-m35-b.parquet", [bar("AAA", ms("09:35:00")), bar("MMM", ms("09:35:02"))])
    # A copy left behind by an interrupted run: its bars duplicate the first file's.
    _part("h09-m30-a-dup1.parquet", [bar("ZZZ", ms("09:30:00"))])
    after_close = datetime(2026, 10, 14, 16, 30, tzinfo=NY)

    done, errors = compact(now=after_close)
    assert errors == [] and len(done) == 1
    s = done[0]
    assert (s["files"], s["rows"], s["duplicates"], s["symbols"]) == (3, 4, 1, 3)
    files = sorted(p.name for p in day_dir("2026-10-14").iterdir())
    assert files == ["day-2026-10-14.parquet"]
    t = pq.read_table(day_dir("2026-10-14") / files[0])
    assert t.schema.equals(AGG_1S_SCHEMA, check_metadata=True)
    assert [(r["symbol"], r["ts_utc"].minute) for r in t.to_pylist()] == [
        ("AAA", 30), ("AAA", 35), ("MMM", 35), ("ZZZ", 30)]

    # A late upload is merged into the day file next time; nothing else is redone.
    _part("h15-m55-c.parquet", [bar("AAA", ms("15:55:00"))])
    done, errors = compact(now=after_close)
    assert errors == [] and done[0]["files"] == 2 and done[0]["rows"] == 5
    assert compact(now=after_close) == ([], [])


def test_compact_leaves_today_alone_until_the_window_closes(roots):
    from sourcing_py.massive_ws.compact import compact, day_dir

    _part("h09-m30-a.parquet", [bar("AAA", ms("09:30:00"))])
    _part("h09-m35-b.parquet", [bar("AAA", ms("09:35:00"))])
    assert compact(now=datetime(2026, 10, 14, 12, 0, tzinfo=NY)) == ([], [])
    assert len(list(day_dir("2026-10-14").iterdir())) == 2
    # The next day, yesterday is due whatever the time.
    done, _ = compact(now=datetime(2026, 10, 15, 8, 0, tzinfo=NY))
    assert [d["date"] for d in done] == ["2026-10-14"]


def test_market_gap_finds_the_longest_silent_stretch():
    import pyarrow as pa

    from sourcing_py.massive_ws.compact import market_gap

    seconds = [ms("09:30:00") + i * 1000 for i in range(23_400) if not 3600 <= i < 3690]  # 10:30:00-10:31:29 silent
    t = pa.table({"ts_utc": pa.array(seconds, pa.timestamp("ms", tz="UTC"))})
    assert market_gap(t, "2026-10-14") == (90, "10:30:00")
    full = pa.table({"ts_utc": pa.array([ms("09:30:00") + i * 1000 for i in range(23_400)], pa.timestamp("ms", tz="UTC"))})
    assert market_gap(full, "2026-10-14") == (0, None)


def test_uploads_after_a_compaction_are_never_skipped(roots):
    """A local segment whose NAS copy was combined away is uploaded again (not treated as
    already there), an in-flight .tmp is ignored, and the next compaction drops the repeat."""
    from sourcing_py.massive_ws.compact import compact, day_dir

    def local_segment(name: str, bars: list[dict]) -> Path:
        cols = Columns()
        for b in bars:
            cols.add(b)
        path = mc.segments_root() / "massive_ws" / "date=2026-10-14" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(cols.table(), path)
        return path

    after_close = datetime(2026, 10, 14, 16, 30, tzinfo=NY)
    first = [bar("AAA", ms("09:30:00")), bar("BBB", ms("09:30:01"))]
    local_segment("h09-m30-x.parquet", first)
    local_segment("h09-m35-y.parquet", [bar("AAA", ms("09:35:00"))])
    assert upload_all(mc.segments_root(), mc.nas_dir())[:2] == (2, 0)
    compact(now=after_close)
    assert [p.name for p in day_dir("2026-10-14").iterdir()] == ["day-2026-10-14.parquet"]

    # The same segment is still local (power cut before the local delete): uploaded again.
    again = local_segment("h09-m30-x.parquet", first)
    (day_dir("2026-10-14") / "h09-m40-z.parquet.tmp").write_bytes(b"half written")
    assert upload_all(mc.segments_root(), mc.nas_dir())[:2] == (1, 0)
    assert not again.exists() and (day_dir("2026-10-14") / "h09-m30-x.parquet").exists()

    done, errors = compact(now=after_close)
    assert errors == [] and done[0]["duplicates"] == 2 and done[0]["rows"] == 3
    assert sorted(p.name for p in day_dir("2026-10-14").iterdir()) == [
        "day-2026-10-14.parquet", "h09-m40-z.parquet.tmp"]
