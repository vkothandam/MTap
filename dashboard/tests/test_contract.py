import json

import numpy as np
import pandas as pd
import pytest
from conftest import write_run

from forecast_dash import contract


def test_load_run_converts_to_moves(run_dir):
    run = contract.load_run(run_dir)
    assert run.name == "run-a" and len(run.levels) == 19 and run.levels[0] == 0.05
    assert run.r_q.shape == (len(run.frame), 19) and (np.diff(run.r_q, axis=1) >= 0).all()
    f = run.frame
    assert sorted(f["fold"].unique()) == [0, 1] and run.has_dates
    assert set(f["industry_code"]) == {"A", "B"}
    i = 0
    assert run.r_act[i] == pytest.approx(f["actual"][i] / f["prev_vwap"][i] - 1)
    assert [p.name for p in run.forecasts()] == ["forecast_2025-03-03.parquet"]


def test_discover_root_or_single(tmp_path):
    a = write_run(tmp_path / "runs", "20260101-a")
    b = write_run(tmp_path / "runs", "20260102-b")
    assert contract.discover(tmp_path / "runs") == [b, a]
    assert contract.discover(a) == [a]
    assert contract.discover(tmp_path / "nothing") == []


def test_missing_columns_raise(run_dir):
    p = run_dir / "fold_0" / "test_predictions.parquet"
    pd.read_parquet(p).drop(columns="actual").to_parquet(p)
    with pytest.raises(ValueError, match="actual"):
        contract.load_run(run_dir)
    with pytest.raises(ValueError, match="quantile"):
        contract.quantile_columns(["q50", "symbol"])


def test_keys_joined_from_panel_when_missing(tmp_path):
    panel = tmp_path / "panel.parquet"
    days = np.arange(1000, 1060)
    pd.DataFrame({"symbol": np.repeat([f"S{i}" for i in range(8)], 60),
                  "day_idx": np.tile(days, 8),
                  "date": np.tile(pd.date_range("2025-01-01", periods=60), 8),
                  "industry_code": "X"}).to_parquet(panel)
    run = contract.load_run(write_run(tmp_path / "runs", panel=panel, with_keys=False))
    assert run.has_dates and set(run.frame["industry_code"]) == {"X"}
    # no panel either: day_idx stands in for the date
    bare = write_run(tmp_path / "runs", "bare", with_keys=False)
    (bare / "config.json").write_text(json.dumps({}))
    r = contract.load_run(bare)
    assert not r.has_dates and (r.frame["date"] == r.frame["day_idx"]).all()
    assert set(r.frame["industry_code"]) == {"unknown"}


def test_load_forecast(run_dir):
    keys, r_q, levels = contract.load_forecast(run_dir / "forecast_2025-03-03.parquet")
    assert len(keys) == 8 and r_q.shape == (8, 19) and levels[-1] == 0.95
    assert "up_70" not in keys.columns and "today_vwap" in keys.columns
