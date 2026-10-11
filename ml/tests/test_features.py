import json

import numpy as np
import pandas as pd
import pytest

from tft_vwap.config import (
    EARNINGS_REALS,
    EARNINGS_SOURCES,
    FILING_REALS,
    KNOWN_REALS,
    PRICE_VOLUME_REALS,
    SENTIMENT_REALS,
    STATIC_CATEGORICALS,
    TrainConfig,
    from_run,
)
from tft_vwap.dataset import prepare, read_meta


def test_prepare_earnings_inputs(panel_df):
    raw = panel_df.copy()
    raw.loc[raw.index[:3], "days_to_results_est"] = [-30, 200, np.nan]
    df = prepare(raw)
    cols = list(dict.fromkeys([*EARNINGS_REALS["exact"], *EARNINGS_REALS["pit"]]))
    assert df[cols].notna().all().all()
    for c in ("days_to_results", "days_to_results_est"):
        assert df[c].between(-10, 63).all()
    assert df["days_since_results"].between(0, 63).all()
    s0 = df[df["symbol"] == "S0"].set_index("day_idx")
    assert s0.loc[1000, "days_since_results"] == 63  # before the first release: far
    for mode, source in EARNINGS_SOURCES.items():
        flags = df[[c for c in EARNINGS_REALS[mode] if c.startswith("is_")]]
        assert set(np.unique(flags)) == {0.0, 1.0} and (flags.sum(axis=1) <= 1).all()
        eve = df[source] == "Earnings_Eve"
        assert eve.any() and (flags.loc[eve].iloc[:, 1] == 1).all()  # is_earnings_eve[_pit]
    assert (df["is_earnings_eve"] != df["is_earnings_eve_pit"]).any()


def test_inputs_by_category():
    cfg = TrainConfig()
    assert cfg.earnings == "exact"
    assert cfg.observed_reals == [*PRICE_VOLUME_REALS[1:], *FILING_REALS, *EARNINGS_REALS["exact"]]
    assert TrainConfig(earnings="pit").observed_reals[-7:] == list(EARNINGS_REALS["pit"])
    assert TrainConfig(earnings="off").observed_reals == [*PRICE_VOLUME_REALS[1:], *FILING_REALS]
    with pytest.raises(ValueError, match="earnings"):
        TrainConfig(earnings="yes")
    with_news = TrainConfig(use_sentiment=True).observed_reals
    assert with_news[-len(SENTIMENT_REALS):] == list(SENTIMENT_REALS)
    for mode in ("exact", "pit"):
        groups = [STATIC_CATEGORICALS, KNOWN_REALS, PRICE_VOLUME_REALS, FILING_REALS,
                  EARNINGS_REALS[mode], SENTIMENT_REALS]
        names = [c for g in groups for c in g]
        assert len(names) == len(set(names))  # every input in exactly one category


def test_run_saved_before_earnings_inputs_trained_without_them(tmp_path):
    saved = {k: v for k, v in TrainConfig().to_dict().items() if k != "earnings"}
    (tmp_path / "config.json").write_text(json.dumps({"config": saved}))
    assert from_run(tmp_path).earnings == "off"
    saved["earnings"] = "pit"
    (tmp_path / "config.json").write_text(json.dumps({"config": saved}))
    assert from_run(tmp_path).earnings == "pit"


def test_series_breaks_split_a_symbol(panel_df):
    raw = panel_df.copy()
    s2 = raw["symbol"] == "S2"
    raw = raw[~(s2 & raw["day_idx"].between(1030, 1060))]  # 31 sessions without a bar
    s2 = raw["symbol"] == "S2"
    at = s2 & (raw["day_idx"] == 1061)
    raw.loc[at, "vwap_pct_prev_day"] = 0.5  # the move across the gap
    jump = s2 & (raw["day_idx"] == 1100)
    raw.loc[jump, "vwap_pct_prev_day"] = 15.0  # a missed 1:16 reverse split
    df = prepare(raw).set_index(["symbol", "day_idx"])
    parts = df.loc[df.index.get_level_values("symbol").str.startswith("S2")].reset_index()
    span = parts.groupby("symbol")["day_idx"].agg(["min", "max"])
    assert span.to_dict("index") == {"S2~2": {"min": 1000, "max": 1029},
                                     "S2~1": {"min": 1061, "max": 1099},
                                     "S2": {"min": 1100, "max": 1149}}
    assert df.loc[("S2~1", 1061), "vwap_pct_prev_day"] == 0  # unknown, like a first bar
    assert df.loc[("S2", 1100), "vwap_pct_prev_day"] == 0
    assert df.index.get_level_values("symbol").str.contains("~").sum() == len(parts) - 50
    same = prepare(raw, series_breaks=False)
    assert set(same["symbol"]) == set(raw["symbol"])
    assert same.loc[same["symbol"] == "S2", "vwap_pct_prev_day"].max() == 15.0


def test_run_saved_before_series_breaks_trained_without_them(tmp_path):
    saved = {k: v for k, v in TrainConfig().to_dict().items() if k != "series_breaks"}
    (tmp_path / "config.json").write_text(json.dumps({"config": saved}))
    assert from_run(tmp_path).series_breaks is False
    assert TrainConfig().series_breaks is True


def test_schema_1_panel_is_rejected(tmp_path):
    path = tmp_path / "panel.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)
    path.with_name("panel.meta.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="re-export"):
        read_meta(path)
