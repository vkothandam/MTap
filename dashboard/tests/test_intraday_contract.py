import json

import pytest
from conftest import ID_TARGETS, write_intraday_run, write_run

from forecast_dash import intraday_contract as ic


def test_discover_newest_first_and_only_intraday_runs(tmp_path):
    write_intraday_run(tmp_path, "20260101-a")
    write_intraday_run(tmp_path, "20260102-b")
    write_run(tmp_path, "daily-run")  # a daily run next to them is not an intraday run
    other = write_intraday_run(tmp_path, "other")
    cfg = json.loads((other / "config.json").read_text())
    (other / "config.json").write_text(json.dumps({**cfg, "model": "something_else"}))
    (tmp_path / "empty").mkdir()
    assert [p.name for p in ic.discover_intraday(tmp_path)] == ["20260102-b", "20260101-a"]
    assert ic.discover_intraday(tmp_path / "missing") == []


def test_load_config_and_targets(tmp_path):
    run = write_intraday_run(tmp_path)
    cfg = ic.load_config(run)
    assert cfg["model"] == "tft_intraday" and len(cfg["folds"]) == 2
    assert ic.targets_of(run) == ID_TARGETS


def test_load_intraday_masks_and_adds_fold(tmp_path):
    run = write_intraday_run(tmp_path)
    valid = ic.load_intraday(run, "r_5m")
    allrows = ic.load_intraday(run, "r_5m", valid_only=False)
    assert len(valid) < len(allrows) and (valid["mask"] == 1).all()
    assert sorted(valid["fold"].unique()) == [0, 1]
    assert set(valid["date"]) == {"2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06"}
    assert all(c in valid for c in ic.QCOLS + ic.MOMCOLS + ic.CLIMCOLS)
    assert "daily_q50" in ic.load_intraday(run, "r_close") and "daily_q50" not in valid


def test_load_intraday_skips_columns_an_older_run_lacks(tmp_path):
    run = write_intraday_run(tmp_path, with_clim=False)
    d = ic.load_intraday(run, "r_5m", columns=["symbol", *ic.QCOLS, *ic.CLIMCOLS])
    assert list(d.columns) == ["fold", "symbol", *ic.QCOLS] and len(d)


def test_load_intraday_columns_and_missing_target(tmp_path):
    run = write_intraday_run(tmp_path)
    d = ic.load_intraday(run, "r_1m", columns=["symbol", "actual"])
    assert list(d.columns) == ["fold", "symbol", "actual"]
    with pytest.raises(FileNotFoundError):
        ic.load_intraday(run, "r_nope")


def test_load_report(tmp_path):
    assert ic.load_report(write_intraday_run(tmp_path, "a")) == {}
    rep = ic.load_report(write_intraday_run(tmp_path, "b", report=True))
    assert set(rep["by_target"]["model"]) == {"tft", "climatology", "momentum"}
