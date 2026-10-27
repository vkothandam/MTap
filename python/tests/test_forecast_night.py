"""forecast-night: the claim file (O_EXCL, heartbeat, stale takeover, done skip) and the run."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from sourcing_py import forecast_night as fn

D = date(2026, 10, 28)   # Wednesday; t = Tuesday 10-27
UTC = timezone.utc
T0 = datetime(2026, 10, 28, 1, 0, tzinfo=UTC)  # 21:00 ET on the 27th


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MTAP_DATA_ROOT", raising=False)
    return tmp_path


def test_first_host_claims_second_stands_aside(env):
    c, why = fn.acquire(D, host="nas", now=T0)
    assert why == "new" and c is not None
    data = json.loads(fn.claim_path(D).read_text())
    assert data["host"] == "nas" and data["status"] == "running" and data["started_at"] and data["heartbeat"]
    c2, why2 = fn.acquire(D, host="mac", now=T0 + timedelta(minutes=5))
    assert c2 is None and why2 == "running"


def test_done_is_skipped_forever(env):
    c, _ = fn.acquire(D, host="nas", now=T0)
    c.done()
    assert json.loads(fn.claim_path(D).read_text())["status"] == "done"
    c2, why = fn.acquire(D, host="mac", now=T0 + timedelta(days=3))
    assert c2 is None and why == "done"


def test_heartbeat_keeps_a_claim_fresh_and_a_stale_one_is_taken_over(env):
    c, _ = fn.acquire(D, host="nas", now=T0)
    c.beat(T0 + timedelta(minutes=29))
    assert fn.acquire(D, host="mac", now=T0 + timedelta(minutes=58))[1] == "running"     # 29 min since beat
    c3, why = fn.acquire(D, host="mac", now=T0 + timedelta(minutes=29 + 31))
    assert why == "takeover" and c3 is not None and c3.host == "mac"
    assert json.loads(fn.claim_path(D).read_text())["host"] == "mac"
    assert not [p for p in fn.claim_path(D).parent.iterdir() if "stale" in p.name]        # grave removed


def test_unreadable_claim_counts_as_stale(env):
    fn.claim_path(D).parent.mkdir(parents=True)
    fn.claim_path(D).write_text("{broken")
    c, why = fn.acquire(D, host="mac", now=T0)
    assert why == "takeover" and c is not None


def test_concurrent_acquire_has_exactly_one_winner(env):
    results, barrier = [], threading.Barrier(12)

    def go(i):
        barrier.wait()
        results.append(fn.acquire(D, host=f"h{i}", now=T0)[0])

    threads = [threading.Thread(target=go, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(1 for r in results if r is not None) == 1


def test_concurrent_stale_takeover_has_exactly_one_winner(env):
    c, _ = fn.acquire(D, host="old", now=T0)
    later = T0 + timedelta(hours=2)
    results, barrier = [], threading.Barrier(12)

    def go(i):
        barrier.wait()
        results.append(fn.acquire(D, host=f"h{i}", now=later)[0])

    threads = [threading.Thread(target=go, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(1 for r in results if r is not None) == 1


def test_release_removes_the_claim_so_the_next_host_can_run(env):
    c, _ = fn.acquire(D, host="nas", now=T0)
    c.release()
    assert not fn.claim_path(D).exists()
    assert fn.acquire(D, host="mac", now=T0)[1] == "new"


def test_heartbeat_thread_updates_the_file(env):
    c, _ = fn.acquire(D, host="nas", now=T0)
    c.start_heartbeat(0.05)
    import time

    time.sleep(0.3)
    c.stop_heartbeat()
    assert json.loads(fn.claim_path(D).read_text())["heartbeat"] != fn._iso(T0)   # the thread beat at real time


def test_earliest_start_is_2020_et_on_the_previous_session(env):
    assert fn.previous_session(D) == date(2026, 10, 27)
    assert fn.previous_session(date(2026, 10, 26)) == date(2026, 10, 23)    # over the weekend
    assert fn.earliest_start(D) == datetime(2026, 10, 28, 0, 20, tzinfo=UTC)   # 20:20 EDT = 00:20Z


def test_refuses_before_2020_unless_forced(env):
    calls = []
    early = datetime(2026, 10, 27, 23, 0, tzinfo=UTC)   # 19:00 ET
    assert fn.run_night(D, now=early, runner=lambda a, cwd=None: calls.append(a) or 0, run_dir="r") == 2
    assert calls == [] and not fn.claim_path(D).exists()
    assert fn.run_night(D, now=early, force=True, runner=lambda a, cwd=None: calls.append(a) or 0, run_dir="r") == 0


def test_run_executes_steps_then_predict_and_marks_done(env):
    calls = []
    rc = fn.run_night(D, now=T0, runner=lambda a, cwd=None: calls.append(a) or 0, run_dir="runs/x", host="nas")
    assert rc == 0
    flat = [" ".join(a) for a in calls]
    assert flat[0] == "sourcing-py run massive_grouped_daily --date 2026-10-27"
    assert any(f.startswith("sourcing-py massive-ext gaps") for f in flat)
    assert "sourcing-py etrade universe --date 2026-10-28" in flat
    assert not any("export-next" in f for f in flat)          # no pre-market in the night state
    assert flat.index("sourcing-py etrade export-tft --include-inference") < len(flat) - 1
    assert flat[-1].endswith("runs/x --state night") and "tft-vwap-predict" in flat[-1]
    assert json.loads(fn.claim_path(D).read_text())["status"] == "done"
    # a second run on any host does nothing
    calls.clear()
    assert fn.run_night(D, now=T0, runner=lambda a, cwd=None: calls.append(a) or 0, run_dir="runs/x", host="mac") == 0
    assert calls == []


def test_a_failed_hard_step_releases_the_claim(env):
    def runner(a, cwd=None):
        return 1 if a[:3] == ["sourcing-py", "etrade", "export-tft"] else 0
    assert fn.run_night(D, now=T0, runner=runner, run_dir="r", host="nas") == 1
    assert not fn.claim_path(D).exists()


def test_a_failed_soft_step_does_not_stop_the_run(env):
    def runner(a, cwd=None):
        return 1 if a[:3] == ["sourcing-py", "massive-ext", "gaps"] else 0
    assert fn.run_night(D, now=T0, runner=runner, run_dir="r", host="nas") == 0


def test_predict_needs_a_run_dir(env):
    with pytest.raises(fn.ConfigError):
        fn.predict_argv("")
