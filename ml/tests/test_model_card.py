"""model_card: MODEL.md written by every trainer and read by the dashboard."""

import argparse
import json

import model_card


def test_write_keeps_label_summary_purpose_across_rewrites(tmp_path):
    run = tmp_path / "20261029-170229Z-gbm-10sym"
    model_card.write(run, kind="GBM", forecast="09:45", inputs="HGB inputs.", folds="2 folds.",
                     label="R7f", summary="HGB + full pre-market", purpose="Does HGB beat R6f?",
                     command="tft-vwap-gbm --label R7f")
    first = model_card.read(run)
    assert first["label"] == "R7f" and first["status"] == "running"
    assert first["purpose"] == "Does HGB beat R6f?" and first["command"] == "tft-vwap-gbm --label R7f"
    assert first["result"].startswith("Not finished")
    model_card.write(run, kind="GBM", forecast="09:45", inputs="HGB inputs.", folds="2 folds.",
                     status="done", result="Skill 40%.")
    done = model_card.read(run)
    assert (done["label"], done["summary"], done["status"], done["result"]) == (
        "R7f", "HGB + full pre-market", "done", "Skill 40%.")
    assert done["purpose"] == "Does HGB beat R6f?" and done["command"] == "tft-vwap-gbm --label R7f"
    text = (run / "MODEL.md").read_text()
    assert text.startswith("---\nlabel: R7f\nforecast: 09:45\n") and text.count("**Purpose.**") == 1


def test_default_label_and_summary(tmp_path):
    run = tmp_path / "20261029-170229Z-return-10sym"
    model_card.write(run, kind="TFT", forecast="night", inputs="TFT, target return.", folds="x")
    card = model_card.read(run)
    assert card["label"] == "TFT 1029-1702" and card["summary"] == "TFT: TFT, target return."
    assert model_card.default_label("TFT", tmp_path / "odd-name") == "odd-name"
    assert model_card.label_of(tmp_path / "none") == "none"


def test_contract_result_pools_rows(tmp_path):
    for k, (rows, loss, naive) in enumerate([(100, 0.01, 0.02), (300, 0.02, 0.02)]):
        d = tmp_path / f"fold_{k}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"test": {
            "rows": rows, "quantile_loss": loss, "naive_quantile_loss": naive,
            "interval_coverage": 0.9, "directional_accuracy": 0.6}}))
    text = model_card.contract_result(tmp_path)
    # pooled loss (100*0.01 + 300*0.02)/400 = 0.0175 vs naive 0.02: skill 12.5%
    assert "Skill 12.5%" in text and "400 test forecasts in 2 folds" in text and "60.0%" in text
    assert model_card.contract_result(tmp_path / "empty") == "No test fold scored."


def test_cli_args():
    p = argparse.ArgumentParser()
    model_card.add_args(p)
    a = p.parse_args(["--label", "R9", "--about", "x y"])
    card = model_card.from_args(a, "tft-vwap", ["--label", "R9", "--about", "x y"])
    assert card == {"label": "R9", "summary": "x y", "purpose": None,
                    "command": "tft-vwap --label R9 --about 'x y'"}
