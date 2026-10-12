import ast
import json
from pathlib import Path

from conftest import write_intraday_run, write_run
from streamlit.testing.v1 import AppTest

PKG = Path(__file__).parents[1] / "forecast_dash"


def test_app_renders_every_tab(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    write_run(root, "20260101-a")
    write_run(root, "20260102-b", seed=1)
    monkeypatch.setenv("FORECAST_DASH_RUNS", str(root))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120)
    at.query_params["section"] = "cross-day"
    at.run()
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
    at.radio(key="side").set_value("up")
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
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120)
    at.query_params["section"] = "cross-day"
    at.run()
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

    def panels(name):
        return {n.args[1].value for n in ast.walk(ast.parse((PKG / name).read_text()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("_chart", "_table")}

    daily, intraday, home = (panels(f) for f in ("app.py", "intraday_page.py", "home.py"))
    used = daily | intraday | home
    assert len(daily) >= 30 and len(intraday) >= 10 and home, (daily, intraday, home)
    assert used <= set(PANELS), used - set(PANELS)
    assert all(k.startswith("id_") for k in intraday)
    assert "Wilson" in column_help("ci_lo")
    assert column_help("low (P10)").endswith("10% chance the VWAP ends at or below it")
    assert column_help("up_said") and column_help("P(>= +1.0%)")
    assert column_help("coverage80") and column_help("skill_mom") and column_help("horizon")
    assert column_help("skill_clim")
    assert column_help("came_true", "lv_table") != column_help("came_true")


def _section(root, section, monkeypatch, intraday=None, **kw):
    monkeypatch.setenv("FORECAST_DASH_RUNS", str(root))
    if intraday is not None:
        monkeypatch.setenv("FORECAST_DASH_INTRADAY_RUNS", str(intraday))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120)
    at.query_params["section"] = section
    return at.run()


def test_home_shows_both_cards(tmp_path, monkeypatch):
    write_run(tmp_path / "runs", "20260101-a")
    write_intraday_run(tmp_path / "runs" / "intraday", "id-a", report=False)
    at = _section(tmp_path / "runs", "home", monkeypatch)  # intraday found under <runs>/intraday
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning
    heads = [s.value for s in at.subheader]
    assert heads == ["Cross-day", "Intraday"]
    assert {m.label for m in at.metric} == {"P10–P90 coverage", "Pinball skill", "Direction hit"}
    assert "Latest forecast day: **2025-03-03**" in [m.value for m in at.markdown]
    assert len(at.dataframe) == 1 and list(at.dataframe[0].value["horizon"]) == [
        "1 min", "5 min", "15 min", "60 min"]
    assert {b.key for b in at.button} == {"open_cross_day", "open_intraday"}
    assert not at.info


def test_home_uses_the_report_when_present(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    write_intraday_run(tmp_path / "runs" / "intraday", report=True)
    at = _section(tmp_path / "runs", "home", monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    h = at.dataframe[0].value
    assert (h["skill_mom"] == 0.123).all() and (h["skill_clim"] == 0.045).all()
    assert list(h.columns) == ["horizon", "n", "coverage80", "skill_clim", "skill_mom", "dir_last"]


def test_home_with_a_run_from_before_climatology(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    write_intraday_run(tmp_path / "runs" / "intraday", with_clim=False)
    at = _section(tmp_path / "runs", "home", monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert "skill_clim" not in at.dataframe[0].value and "skill_mom" in at.dataframe[0].value


def test_intraday_section_with_a_run_from_before_climatology(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    write_intraday_run(tmp_path / "id", with_clim=False)
    at = _section(tmp_path / "runs", "intraday", monkeypatch, intraday=tmp_path / "id")
    assert not at.exception, [e.value for e in at.exception]
    assert all(len(t.get("plotly_chart")) >= 1 for t in at.tabs)


def test_home_without_an_intraday_run(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "home", monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert any("isn't trained yet" in i.value for i in at.info)
    assert not at.dataframe and len(at.button) == 2


def test_home_buttons_switch_section(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    write_intraday_run(tmp_path / "runs" / "intraday")
    at = _section(tmp_path / "runs", "home", monkeypatch)
    at.button(key="open_intraday").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="section").value == "Intraday"
    assert at.query_params["section"] in ("intraday", ["intraday"])
    assert len(at.tabs) == 6
    at.radio(key="section").set_value("Home").run()
    at.button(key="open_cross_day").click().run()
    assert at.radio(key="section").value == "Cross-day" and len(at.tabs) == 10


def test_intraday_section_renders_six_explained_tabs(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    write_intraday_run(tmp_path / "id", "id-a")
    write_intraday_run(tmp_path / "id", "id-b", seed=3)
    at = _section(tmp_path / "runs", "intraday", monkeypatch, intraday=tmp_path / "id")
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Summary", "Calibration", "Time of day",
                                          "Session replay", "Across stocks", "Intraday vs daily"]
    assert at.selectbox(key="id_run").value == "id-b"  # newest first
    assert all(len(t.get("plotly_chart")) >= 1 for t in at.tabs), [t.label for t in at.tabs]
    in_short = {t.label: [m.value for m in t.markdown if m.value.startswith("**In short**")]
                for t in at.tabs}
    assert all(in_short.values()), in_short
    _assert_explained(at)
    # other targets and references
    for target in ("r_1m", "r_rod", "r_close"):
        for ref in ("last", "session", "trail"):
            at.selectbox(key="id_target").set_value(target)
            at.selectbox(key="id_ref").set_value(ref)
            at.run()
            assert not at.exception, (target, ref, [e.value for e in at.exception])
            assert all(len(t.get("plotly_chart")) >= 1 for t in at.tabs)
    _assert_explained(at)
    # the replay follows the day and stock pickers
    at.selectbox(key="id_day").set_value("2025-03-05")
    at.selectbox(key="id_sym").set_value("S3")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    replay = next(t for t in at.tabs if t.label == "Session replay")
    assert any("S3 on 2025-03-05" in m.value for m in replay.markdown)
    # the stock table follows the min-n filter
    at.number_input(key="id_min_n").set_value(10**5)
    at.run()
    assert not at.exception and any("No stocks" in i.value for i in at.info)


def test_intraday_section_without_a_run(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "intraday", monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert any("tft-intraday" in i.value and "Training hasn't run" in i.value for i in at.info)
    assert not at.tabs
