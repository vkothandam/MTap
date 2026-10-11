import ast
import json
from pathlib import Path

from conftest import write_run
from streamlit.testing.v1 import AppTest

PKG = Path(__file__).parents[1] / "forecast_dash"


def test_app_renders_every_tab(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    write_run(root, "20260101-a")
    write_run(root, "20260102-b", seed=1)
    monkeypatch.setenv("FORECAST_DASH_RUNS", str(root))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 10  # 8 pages + Forecast's two subtabs
    assert all(len(t.get("plotly_chart")) >= 1 for t in at.tabs), [t.label for t in at.tabs]
    in_short = {t.label: [m.value for m in t.markdown if m.value.startswith("**In short**")]
                for t in at.tabs}
    assert all(in_short.values()), in_short
    assert "metrics" in in_short["Compare runs"][0]
    assert any("band held" in m for m in in_short["Forecast vs actual"])
    assert any("Live forecast for 2025-02-28" in m for m in in_short["Summary"])
    _assert_explained(at)
    # a graded day in Price ranges adds the actuals and follows into Forecast vs actual
    at.selectbox(key="tm_day").set_value("2025-02-28")
    at.slider(key="tm_conf").set_value(0.0)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    ranges = next(t for t in at.tabs if t.label == "Price ranges")
    assert any("has happened" in m.value for m in ranges.markdown)
    assert "actual_vwap" in ranges.dataframe[0].value.columns
    assert at.selectbox(key="lv_day").value == "2025-02-28"
    _assert_explained(at)
    # change filters: one side, a custom move, 5% buckets, one stock
    at.sidebar.radio[0].set_value("up")
    at.sidebar.number_input[0].set_value(1.3)
    at.sidebar.select_slider[0].set_value(0.05)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def _assert_explained(at):
    """Every chart and table has its ⓘ popover, and every table column a header tooltip."""
    for t in at.tabs:
        about = [p for p in t.get("popover") if p.proto.popover.label != "ℹ️ About this page"]
        assert len(about) >= len(t.get("plotly_chart")) + len(t.dataframe), t.label
        for df in t.dataframe:
            cfg = json.loads(df.proto.columns)
            missing = [c for c in df.value.columns if not cfg.get(str(c), {}).get("help")]
            assert not missing, (t.label, missing)


def test_app_picks_up_edited_modules(run_dir, monkeypatch):
    """A long-running server must use the code on disk, not the modules of its first run."""
    import sys

    monkeypatch.setenv("FORECAST_DASH_RUNS", str(run_dir.parent))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120).run()
    del sys.modules["forecast_dash.charts"].call_heatmap  # as if charts.py predates the app
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_with_no_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECAST_DASH_RUNS", str(tmp_path))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=60).run()
    assert not at.exception and at.warning


def test_no_imports_from_model_or_sourcing_code():
    """The dashboard must stay movable: it may only read run files, never import ml/ code."""
    banned = {"tft_vwap", "sourcing_py", "torch", "pytorch_forecasting", "lightning"}
    for f in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not {n.split(".")[0] for n in names} & banned, f"{f.name} imports {names}"



def test_every_panel_and_column_has_help():
    """Each _chart/_table call names a PANELS entry (some only render with enough data)."""
    from forecast_dash.help_text import PANELS, column_help

    used = {n.args[1].value for n in ast.walk(ast.parse((PKG / "app.py").read_text()))
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("_chart", "_table")}
    assert len(used) >= 30 and used <= set(PANELS), used - set(PANELS)
    assert "Wilson" in column_help("ci_lo")
    assert column_help("low (P10)").endswith("10% chance the VWAP ends at or below it")
    assert column_help("up_said") and column_help("P(>= +1.0%)")
    assert column_help("came_true", "lv_table") != column_help("came_true")
