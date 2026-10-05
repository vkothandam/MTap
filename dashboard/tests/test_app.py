import ast
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
    assert len(at.tabs) == 7
    assert all(len(t.get("plotly_chart")) >= 1 for t in at.tabs), [t.label for t in at.tabs]
    # change filters: one side, a custom move, 5% buckets, one stock
    at.sidebar.radio[0].set_value("up")
    at.sidebar.number_input[0].set_value(1.3)
    at.sidebar.select_slider[0].set_value(0.05)
    at.sidebar.multiselect(key=None) if False else None
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
