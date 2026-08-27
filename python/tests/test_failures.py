"""Failure-log record / load / rewrite behavior."""

from __future__ import annotations

import pytest

from sourcing_py.common import failures
from sourcing_py.__main__ import _parse_params


@pytest.fixture
def tmp_state(monkeypatch, tmp_path):
    # Point repo_root at a temp dir so we don't touch the real state/ folder.
    monkeypatch.setattr(failures.config, "repo_root", lambda: tmp_path)
    return tmp_path


def test_record_and_load_dedups(tmp_state):
    failures.record("src", "2024-11-05", "boom")
    failures.record("src", "2024-11-05", "boom again")  # same date
    failures.record("src", "2024-11-06", "boom")
    assert failures.load_dates("src") == ["2024-11-05", "2024-11-06"]


def test_rewrite_prunes_and_deletes(tmp_state):
    failures.record("src", "2024-11-05", "boom")
    failures.rewrite("src", [{"date": "2024-11-06", "error": "x", "at": "t"}])
    assert failures.load_dates("src") == ["2024-11-06"]
    failures.rewrite("src", [])  # empty -> file removed
    assert failures.load_dates("src") == []
    assert not failures.log_path("src").exists()


def test_parse_boolean_flag():
    assert _parse_params(["--retry-failed"]) == {"retry-failed": True}
    assert _parse_params(["--fromdate", "2024-08-16", "--retry-failed"]) == {
        "fromdate": "2024-08-16",
        "retry-failed": True,
    }
