"""The daily -> intraday handoff table (tft_vwap/handoff.py)."""

import numpy as np
import pandas as pd
import pytest

from tft_vwap import handoff

LEVELS = [round(k * 0.05, 2) for k in range(1, 20)]
QCOLS = [f"q{round(lv * 100):02d}" for lv in LEVELS]


def _fold(run, k, dates, symbols, seed):
    """test_predictions.parquet in price units, like evaluate.score_frame writes it."""
    rng = np.random.default_rng(seed)
    rows = []
    for sym in symbols:
        prev = 100.0
        for d in dates:
            z = np.sort(rng.normal(0, 0.02, len(LEVELS)))
            act = prev * np.exp(rng.normal(0, 0.02))
            rows.append({"symbol": sym, "day_idx": 0, "date": pd.Timestamp(d),
                         "industry_code": "531010" if sym != "C" else "571010",
                         "prev_vwap": prev, "actual": act,
                         **{c: prev * np.exp(v) for c, v in zip(QCOLS, z)}})
            prev = act
    d = run / f"fold_{k}"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "test_predictions.parquet", index=False)


@pytest.fixture
def run(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=30)
    _fold(tmp_path, 0, days[:15], ["A", "B", "C", "A~1"], 0)
    _fold(tmp_path, 1, days[15:], ["A", "B", "C"], 1)
    return tmp_path


def test_one_row_per_symbol_day_and_sorted_quantiles(run):
    h = handoff.build(run)
    assert not h.duplicated(["symbol", "date"]).any()
    assert "A~1" not in set(h["symbol"])
    dq = h[[c for c in h.columns if c.startswith("dq")]].to_numpy()
    assert (np.diff(dq, axis=1) >= 0).all()
    assert set(h["fold"]) == {0, 1} and set(h["source"]) == {"oof"}


def test_summaries_match_the_quantiles(run):
    h = handoff.build(run)
    assert np.allclose(h["d_med"], h["dq50"])
    assert np.allclose(h["d_width"], h["dq90"] - h["dq10"])
    assert np.allclose(h["d_width90"], h["dq95"] - h["dq05"])
    assert h["d_p_up"].between(0, 1).all() and h["d_width_rank"].between(0, 1).all()


def test_history_uses_previous_sessions_only(run):
    h = handoff.build(run).sort_values(["symbol", "date"])
    a = h[h["symbol"] == "A"].reset_index(drop=True)
    assert a.loc[0, "hist_n"] == 0 and np.isnan(a.loc[0, "hist_mae"])
    # day 2's record is day 1's error alone: changing day 2's actual must not move it
    raw = pd.read_parquet(run / "fold_0" / "test_predictions.parquet")
    one = raw[(raw["symbol"] == "A")].iloc[0]
    err = abs(np.log(one["actual"] / one["prev_vwap"]) - np.log(one["q50"] / one["prev_vwap"]))
    assert a.loc[1, "hist_n"] == 1
    assert a.loc[1, "hist_mae"] == pytest.approx(err, rel=1e-5)
    assert a["hist_n"].max() == handoff.HIST


def test_sector_mean_leaves_the_stock_out(run):
    h = handoff.build(run)
    day = h[h["date"] == h["date"].min()].set_index("symbol")
    assert day.loc["A", "sec_d_med"] == pytest.approx(day.loc["B", "d_med"], rel=1e-5)
    assert np.isnan(day.loc["C", "sec_d_med"])  # alone in its sector
    assert day.loc["A", "mkt_d_med"] == pytest.approx(day["d_med"].mean(), rel=1e-5)


def test_unknown_industry_is_no_sector(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=3)
    _fold(tmp_path, 0, days, ["A", "B", "D", "E"], 0)
    p = tmp_path / "fold_0" / "test_predictions.parquet"
    f = pd.read_parquet(p)
    f.loc[f["symbol"].isin(["D", "E"]), "industry_code"] = "unknown"
    f.to_parquet(p, index=False)
    h = handoff.build(tmp_path)
    assert h.loc[h["symbol"].isin(["D", "E"]), "sec_d_med"].isna().all()
    assert h.loc[h["symbol"] == "A", "sec_d_med"].notna().all()


def test_live_forecasts_only_after_the_last_fold(run):
    last = pd.read_parquet(run / "fold_1" / "test_predictions.parquet")["date"].max()
    rows = []
    for d in (last, last + pd.offsets.BDay(1)):
        for sym in ("A", "B"):
            rows.append({"date": d.date().isoformat(), "asof": "x", "fold": 1, "symbol": sym,
                         "industry_code": "531010", "today_vwap": 100.0,
                         **{c: float(v) for c, v in zip(QCOLS, np.linspace(-0.03, 0.03, 19))}})
        pd.DataFrame(rows[-2:]).to_parquet(run / f"forecast_{d.date()}.parquet", index=False)
    h = handoff.build(run, live=True)
    live = h[h["source"] == "live"]
    assert set(live["date"]) == {last + pd.offsets.BDay(1)}
    assert (live["fold"] == -1).all()
    assert live["dq50"].iloc[0] == pytest.approx(np.log1p(0.0), abs=1e-6)
    assert not h.duplicated(["symbol", "date"]).any()


def test_main_writes_table_and_meta(run):
    assert handoff.main([str(run)]) == 0
    assert (run / "handoff.parquet").exists()
    meta = pd.read_json(run / "handoff.meta.json", typ="series")
    assert meta["rows"] == len(pd.read_parquet(run / "handoff.parquet"))
