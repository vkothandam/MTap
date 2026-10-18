"""Storage roots (common/config.py) and the data-root guard (common/storage.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sourcing_py.common import config, storage
from sourcing_py.common.errors import ConfigError

ROOT_VARS = ("MTAP_DATA_ROOT", "MTAP_STATE_ROOT", "OUT_ROOT", "MTAP_NAS_URL")


@pytest.fixture
def clean(monkeypatch):
    for var in ROOT_VARS:
        monkeypatch.delenv(var, raising=False)
    settings: dict = {}
    monkeypatch.setattr(config, "settings", lambda: settings)
    return settings


def test_defaults_keep_the_repo_layout(clean):
    repo = config.repo_root()
    assert config.data_root() == repo
    assert config.state_root() == repo / "state"
    assert config.out_root() == repo / "out"
    assert config.state_path("./state/etrade/fundamentals.duckdb") == repo / "state/etrade/fundamentals.duckdb"
    assert storage.data_root_ready() == (True, "data root is the repo (MTAP_DATA_ROOT not set)")


def test_env_roots(clean, monkeypatch, tmp_path):
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tmp_path / "nas"))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "local"))
    assert config.data_root() == tmp_path / "nas"
    assert config.out_root() == tmp_path / "nas" / "out"
    assert config.state_path("./state/etrade/session.json") == tmp_path / "local/etrade/session.json"
    assert config.state_path("state/news.duckdb") == tmp_path / "local/news.duckdb"
    # Paths outside ./state stay repo-relative; absolute paths stand.
    assert config.state_path("data/x.db") == config.repo_root() / "data/x.db"
    assert config.state_path(tmp_path / "abs.db") == tmp_path / "abs.db"


def test_settings_roots_and_env_wins(clean, monkeypatch, tmp_path):
    clean["storage"] = {"data_root": str(tmp_path / "s-data"), "state_root": str(tmp_path / "s-state")}
    assert config.data_root() == tmp_path / "s-data"
    assert config.state_root() == tmp_path / "s-state"
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tmp_path / "env-data"))
    assert config.data_root() == tmp_path / "env-data"


def test_out_root_override(clean, monkeypatch, tmp_path):
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tmp_path / "nas"))
    clean["destination"] = {"out_root": "exports"}
    assert config.out_root() == tmp_path / "nas" / "exports"
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "elsewhere"))
    assert config.out_root() == tmp_path / "elsewhere"


def test_marker_guards_the_data_root(clean, monkeypatch, tmp_path):
    nas = tmp_path / "nas"
    monkeypatch.setenv("MTAP_DATA_ROOT", str(nas))
    monkeypatch.setenv("MTAP_STATE_ROOT", str(tmp_path / "local"))
    ok, reason = storage.data_root_ready()
    assert not ok and "does not exist" in reason
    nas.mkdir()
    ok, reason = storage.data_root_ready()
    assert not ok and "marker" in reason
    with pytest.raises(ConfigError):
        storage.require_data_root()
    assert storage.init() == 0
    assert (nas / config.DATA_ROOT_MARKER).exists()
    assert (nas / "intraday_feed").is_dir() and (tmp_path / "local/segments").is_dir()
    assert storage.require_data_root() == nas


def testunmounted_volume_is_refused(clean, monkeypatch):
    monkeypatch.setenv("MTAP_DATA_ROOT", "/Volumes/mtap-test-not-mounted/data")
    ok, reason = storage.data_root_ready()
    assert not ok and "not mounted" in reason
    assert storage.unmounted_volume(Path("/Volumes/mtap-test-not-mounted"))
    assert not storage.unmounted_volume(Path("/tmp/x"))


def test_dotenv_parse():
    text = '# c\nA=1\nexport B="two words"\nC=\'x=y\'\n\nbad line\nD=  spaced  \n'
    assert config._parse_dotenv(text) == {"A": "1", "B": "two words", "C": "x=y", "D": "spaced"}


def test_writer_refuses_a_data_root_that_is_not_ready(clean, monkeypatch, tmp_path):
    from sourcing_py.common import writer

    monkeypatch.setenv("MTAP_DATA_ROOT", "/Volumes/mtap-test-not-mounted/data")
    with pytest.raises(ConfigError, match="not mounted"):
        writer.write([{"a": 1}], source="s", dataset="d", fmt="jsonl")
    assert not Path("/Volumes/mtap-test-not-mounted").exists()
    # an OUT_ROOT outside the data root is not the NAS's business
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    assert writer.write([{"a": 1}], source="s", dataset="d", fmt="jsonl").exists()


def test_writer_replaces_atomically_and_leaves_no_temp_file(clean, monkeypatch, tmp_path):
    from datetime import date

    from sourcing_py.common import writer
    from sourcing_py.etrade import symbols

    monkeypatch.setenv("OUT_ROOT", str(tmp_path))
    d = date(2026, 10, 14)
    path = writer.write([{"ticker": "A"}], source="s", dataset="daily_summary", fmt="jsonl", dt=d)
    assert path.name == "daily_summary-2026-10-14.jsonl"

    def boom(records, p):
        p.write_text("half")
        raise OSError("disk gone")

    monkeypatch.setattr(writer, "_write_jsonl", boom)
    with pytest.raises(OSError):
        writer.write([{"ticker": "B"}], source="s", dataset="daily_summary", fmt="jsonl", dt=d)
    assert '"A"' in path.read_text()  # the old file is untouched
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    (path.parent / f".{path.name}.tmp").write_text("partial")  # e.g. the other machine mid-write
    assert symbols.select_files(directory=path.parent) == [path]
