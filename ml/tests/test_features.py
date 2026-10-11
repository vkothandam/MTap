import json

import numpy as np
import pandas as pd
import pytest

from tft_vwap.config import (
    EARNINGS_REALS,
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
    raw.loc[raw.index[:3], "days_to_filing_est"] = [-30, 200, np.nan]
    df = prepare(raw)
    assert df[list(EARNINGS_REALS)].notna().all().all()
    assert df["days_to_filing_est"].between(-10, 63).all()
    assert df["days_since_filing"].between(0, 63).all()
    s0 = df[df["symbol"] == "S0"].set_index("day_idx")
    assert s0.loc[1000, "days_since_filing"] == 63  # before the first filing: far
    flags = df[["is_pre_earnings_runup", "is_earnings_eve", "is_post_earnings_reaction",
                "is_post_earnings_drift"]]
    assert set(np.unique(flags)) == {0.0, 1.0} and (flags.sum(axis=1) <= 1).all()
    eve = df["results_window_pit"] == "Earnings_Eve"
    assert eve.any() and (df.loc[eve, "is_earnings_eve"] == 1).all()


def test_inputs_by_category():
    cfg = TrainConfig()
    assert cfg.observed_reals == [*PRICE_VOLUME_REALS[1:], *FILING_REALS, *EARNINGS_REALS]
    assert not set(EARNINGS_REALS) & set(TrainConfig(use_earnings=False).observed_reals)
    with_news = TrainConfig(use_sentiment=True).observed_reals
    assert with_news[-len(SENTIMENT_REALS):] == list(SENTIMENT_REALS)
    groups = [STATIC_CATEGORICALS, KNOWN_REALS, PRICE_VOLUME_REALS, FILING_REALS,
              EARNINGS_REALS, SENTIMENT_REALS]
    names = [c for g in groups for c in g]
    assert len(names) == len(set(names))  # every input in exactly one category


def test_run_saved_before_earnings_inputs_trained_without_them(tmp_path):
    saved = {k: v for k, v in TrainConfig().to_dict().items() if k != "use_earnings"}
    (tmp_path / "config.json").write_text(json.dumps({"config": saved}))
    assert from_run(tmp_path).use_earnings is False
    saved["use_earnings"] = True
    (tmp_path / "config.json").write_text(json.dumps({"config": saved}))
    assert from_run(tmp_path).use_earnings is True


def test_schema_1_panel_is_rejected(tmp_path):
    path = tmp_path / "panel.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)
    path.with_name("panel.meta.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="re-export"):
        read_meta(path)
