import ast
import json
from pathlib import Path

from conftest import write_intraday_run, write_pockets_run, write_run, write_space_lib
from streamlit.testing.v1 import AppTest

PKG = Path(__file__).parents[1] / "forecast_dash"


def test_app_renders_every_tab(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    write_run(root, "20260101-a")
    b = write_run(root, "20260102-b", seed=1)
    (b / "MODEL.md").write_text("---\nlabel: R9\nforecast: night\nsummary: test model\n---\n\nbody\n")
    monkeypatch.setenv("FORECAST_DASH_RUNS", str(root))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120)
    at.query_params["section"] = "cross-day"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    cmp = next(t for t in at.tabs if t.label == "Compare runs")
    assert cmp.dataframe[0].value["model"].tolist() == ["R9", "20260101-a"]
    assert "R9" in cmp.dataframe[1].value.index
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

    daily, intraday, home, space, pockets = (panels(f) for f in (
        "app.py", "intraday_page.py", "home.py", "space_page.py", "pockets_page.py"))
    used = daily | intraday | home | space | pockets
    assert len(daily) >= 30 and len(intraday) >= 10 and home, (daily, intraday, home)
    assert len(space) >= 10, space
    assert used <= set(PANELS), used - set(PANELS)
    assert all(k.startswith("id_") for k in intraday)
    assert all(k.startswith("sp_") for k in space)
    assert len(pockets) >= 10 and all(k.startswith("pk_") for k in pockets), pockets
    assert "Wilson" in column_help("ci_lo")
    assert column_help("low (P10)").endswith("10% chance the VWAP ends at or below it")
    assert column_help("up_said") and column_help("P(>= +1.0%)")
    assert column_help("coverage80") and column_help("skill_mom") and column_help("horizon")
    assert column_help("skill_clim")
    assert column_help("came_true", "lv_table") != column_help("came_true")


def _section(root, section, monkeypatch, intraday=None, space=None, view=None, pockets=None,
             **params):
    monkeypatch.setenv("FORECAST_DASH_RUNS", str(root))
    monkeypatch.setenv("FORECAST_DASH_POCKETS_RUNS", str(pockets or Path(root) / "no-pockets"))
    if intraday is not None:
        monkeypatch.setenv("FORECAST_DASH_INTRADAY_RUNS", str(intraday))
    monkeypatch.setenv("FORECAST_DASH_SPACE_LIB", str(space or Path(root) / "no-space"))
    at = AppTest.from_file(str(PKG / "app.py"), default_timeout=120)
    at.query_params["section"] = section
    if view:
        at.query_params["view"] = view
    for k, v in params.items():
        at.query_params[k] = v
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
    at = _section(tmp_path / "runs", "intraday", monkeypatch, intraday=tmp_path / "id",
                  view="runs")
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
    assert at.radio(key="id_view").value == "Model runs"  # Home opens the model runs
    assert len(at.tabs) == 6
    at.radio(key="section").set_value("Home").run()
    at.button(key="open_cross_day").click().run()
    assert at.radio(key="section").value == "Cross-day" and len(at.tabs) == 10


def test_intraday_section_renders_six_explained_tabs(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    write_intraday_run(tmp_path / "id", "id-a")
    write_intraday_run(tmp_path / "id", "id-b", seed=3)
    (tmp_path / "id" / "id-b" / "MODEL.md").write_text(
        "---\nlabel: I1\nforecast: intraday\nsummary: tiny intraday\n---\n\nbody\n")
    at = _section(tmp_path / "runs", "intraday", monkeypatch, intraday=tmp_path / "id",
                  view="runs")
    assert not at.exception, [e.value for e in at.exception]
    # the picker shows the card's label, and the sidebar has the card's popover
    assert at.selectbox(key="id_run").options[0] == "I1 · id-b"
    assert any(p.proto.popover.label.startswith("ℹ️ I1") for p in at.sidebar.get("popover"))
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
    at = _section(tmp_path / "runs", "intraday", monkeypatch, view="runs")
    assert not at.exception, [e.value for e in at.exception]
    assert any("tft-intraday" in i.value and "Training hasn't run" in i.value for i in at.info)
    assert not at.tabs


# ---- Intraday → Space ---------------------------------------------------------------------------

def test_space_is_the_default_intraday_view(tmp_path, monkeypatch, space_lib):
    write_run(tmp_path / "runs")   # no intraday run: Space still works
    at = _section(tmp_path / "runs", "intraday", monkeypatch, space=space_lib.parent)
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="id_view").value == "Space (pockets)"
    assert at.query_params["view"] in ("space", ["space"])
    assert [t.label for t in at.tabs] == ["Compare", "Matches", "Dial curve", "Day path"]
    assert all(len(t.get("plotly_chart")) >= 1 for t in at.tabs if t.label != "Matches")
    assert any(m.value.startswith("**In short**") for m in at.markdown)
    s3d = [e for e in at.main if getattr(e, "type", None) == "bidi_component"]
    assert len(s3d) == 1 and s3d[0].proto.component_name == "forecast-dash.space3d"
    assert s3d[0].proto.js_source_path == "space3d.js"     # served as a file, not inlined
    assert any(p.proto.popover.label == "The pocket space (3-D)" for p in at.get("popover"))
    ev = at.dataframe[0].value
    assert [v.removeprefix("⚠️ ") for v in ev["view"]] == [
        "close match", "loose match", "everything", "what it did"]
    _assert_explained(at)
    # the URL keeps the moment
    tk, date = at.selectbox(key="sp_stock").value, at.selectbox(key="sp_date").value
    assert at.query_params["sp_stock"] in (tk, [tk]) and at.query_params["sp_date"] in (
        date, [date])
    assert at.query_params["sp_time"] in ("11:00", ["11:00"])


def test_space_follows_every_control(tmp_path, monkeypatch, space_lib):
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "intraday", monkeypatch, space=space_lib.parent)
    assert not at.exception, [e.value for e in at.exception]
    stocks = at.selectbox(key="sp_stock").options
    at.selectbox(key="sp_stock").set_value(stocks[2].split(" ")[0])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    dates = at.selectbox(key="sp_date").options
    at.selectbox(key="sp_date").set_value(dates[3])
    at.select_slider(key="sp_time").set_value(240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "13:30 ET" in " ".join(m.value for m in at.markdown)
    at.multiselect(key="sp_params").set_value(["vwap_vs_prev", "rvol", "from_high"])
    at.slider(key="sp_dials").set_value((1.0, 2.5))
    at.radio(key="sp_window").set_value(20 if 20 in at.radio(key="sp_window").options else 25)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    _assert_explained(at)
    # one parameter twice as loose, the pool filters, ±15 min
    at.number_input(key="sp_tol_rvol").set_value(2.0)
    at.radio(key="sp_tod").set_value(15)
    at.checkbox(key="sp_own").check()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # no parameters: every pool moment is a match
    at.checkbox(key="sp_own").uncheck()
    at.multiselect(key="sp_params").set_value([])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    ev = at.dataframe[0].value.set_index("view")
    assert ev.loc["loose match", "matches"] == ev.loc["everything", "matches"]
    # a picked cell and the difference of the close match
    at.select_slider(key="sp_cell_x").set_value(20)
    at.select_slider(key="sp_cell_y").set_value(5)
    at.radio(key="sp_diff_of").set_value("close")
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_space_picks_a_match_row(tmp_path, monkeypatch, space_lib):
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "intraday", monkeypatch, space=space_lib.parent)
    at.slider(key="sp_dials").set_value((1.5, 3.0))
    at.run()
    matches = next(t for t in at.tabs if t.label == "Matches")
    assert len(matches.dataframe) == 2 and not matches.get("plotly_chart")
    at.session_state["sp_match_pick"] = {"selection": {"rows": [0], "columns": []}}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    matches = next(t for t in at.tabs if t.label == "Matches")
    assert len(matches.get("plotly_chart")) == 1
    _assert_explained(at)


def test_space_opens_a_moment_from_the_url(tmp_path, monkeypatch, space_lib):
    from mtap_space.library import Library

    lib = Library(space_lib)
    tk = lib.stocks().tk.iloc[1]
    date = lib.days_of(tk).date.iloc[2]
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "intraday", monkeypatch, space=space_lib.parent,
                  view="space", sp_stock=tk, sp_date=date, sp_time="13:15")
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="sp_stock").value == tk
    assert at.selectbox(key="sp_date").value == date
    assert at.select_slider(key="sp_time").value == 225
    # an unknown stock in the URL falls back to the most traded one
    at2 = _section(tmp_path / "runs", "intraday", monkeypatch, space=space_lib.parent,
                   sp_stock="NOPE", sp_time="25:99")
    assert not at2.exception, [e.value for e in at2.exception]
    assert at2.selectbox(key="sp_stock").value == lib.stocks().tk.iloc[0]


def test_space_without_a_library(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "intraday", monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert any("space-build" in i.value for i in at.info)
    at.radio(key="id_view").set_value("Model runs").run()
    assert any("Training hasn't run" in i.value for i in at.info)
    assert at.query_params["view"] in ("runs", ["runs"])


def test_space_with_a_moment_before_any_history(tmp_path, monkeypatch):
    lib_dir = write_space_lib(tmp_path, sessions=25)
    from mtap_space.library import Library

    lib = Library(lib_dir)
    tk = lib.stocks().tk.iloc[0]
    first = lib.days_of(tk).date.iloc[-1]   # its oldest library day: nothing before it
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "intraday", monkeypatch, space=lib_dir.parent,
                  sp_stock=tk, sp_date=first)
    assert not at.exception, [e.value for e in at.exception]
    assert any("No earlier moments" in i.value for i in at.info)


# ---- Intraday → Pockets model ---------------------------------------------------------------------

PK_TABS = ["Go rule", "Pocket shape", "By group", "Overlays", "Stock day", "Moves"]


def _pockets(tmp_path, monkeypatch, **kw):
    write_run(tmp_path / "runs")
    root = tmp_path / "pk"
    write_pockets_run(root, "20261107-000000-ft-B", **kw)
    return _section(tmp_path / "runs", "intraday", monkeypatch, view="pockets", pockets=root)


def test_pockets_renders_five_explained_tabs(tmp_path, monkeypatch):
    write_pockets_run(tmp_path / "pk", "20261106-000000-ft-A", fold="A", label="IA1A", seed=2)
    at = _pockets(tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="id_view").value == "Pockets model"
    assert at.selectbox(key="pk_run").value == "20261107-000000-ft-B"     # newest first
    assert any(p.proto.popover.label.startswith("ℹ️ IA1") for p in at.sidebar.get("popover"))
    assert [t.label for t in at.tabs] == PK_TABS
    assert all(len(t.get("plotly_chart")) >= 1 for t in at.tabs), [t.label for t in at.tabs]
    in_short = {t.label: [m.value for m in t.markdown if m.value.startswith("**In short**")]
                for t in at.tabs if t.label not in ("Stock day",)}
    assert all(in_short.values()), in_short
    go = next(t for t in at.tabs if t.label == "Go rule")
    assert go.dataframe[0].value["passes"].all()
    assert sorted(r.split(" · ")[0] for r in go.dataframe[1].value["run"]) == ["IA1", "IA1A"]
    _assert_explained(at)
    raw = next(t for t in at.tabs if t.label == "Pocket shape").dataframe[0].value
    assert (raw["p50_hit"] < 0.45).all()                                 # the synthetic shift
    # recalibrated: the shift is fixed on the warm days
    at.radio(key="pk_mode").set_value("recal")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    cal = next(t for t in at.tabs if t.label == "Pocket shape").dataframe[0].value
    assert (abs(cal["p50_hit"] - 0.5) < 0.06).all() and (cal["n"] < raw["n"]).all()
    _assert_explained(at)


def test_pockets_follows_every_control(tmp_path, monkeypatch):
    at = _pockets(tmp_path, monkeypatch)
    for fam in ("cm_dn", "bulge", "sag", "rvol"):
        at.selectbox(key="pk_shape_family").set_value(fam)
        at.selectbox(key="pk_shape_bin").set_value(1)
        at.run()
        assert not at.exception, (fam, [e.value for e in at.exception])
    for key in ("sector", "industry", "minute_bucket", "symbol", "live_set"):
        at.selectbox(key="pk_group_by").set_value(key)
        at.selectbox(key="pk_ov_by").set_value(key)
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
        grp = next(t for t in at.tabs if t.label == "By group").dataframe[0].value
        assert key in grp and grp["n"].sum() > 0
    for ev in ("p_bulge60", "p_sag60"):
        at.selectbox(key="pk_event").set_value(ev)
        at.selectbox(key="pk_day_event").set_value(ev)
        at.run()
        assert not at.exception, (ev, [e.value for e in at.exception])
    _assert_explained(at)
    # the stock day follows its pickers
    at.selectbox(key="pk_day").set_value("2026-01-06")
    at.selectbox(key="pk_sym").set_value("S3")
    at.selectbox(key="pk_day_bin").set_value(2)
    at.select_slider(key="pk_when").set_value("10:04")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    day = next(t for t in at.tabs if t.label == "Stock day")
    assert any("S3 · 2026-01-06" in c.value for c in day.caption)
    assert len(day.dataframe[0].value) == 5
    # the filters cut the rows; an empty selection says so
    at.multiselect(key="pk_sector").set_value(["10"])  # not a TRBC code: shown as is
    at.checkbox(key="pk_live").check()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.caption[-1].value.startswith("1,014 of 4,056 decisions · 3 stocks")
    at.multiselect(key="pk_stocks").set_value(["S11"])                  # S11 is not live
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No rows" in i.value for i in at.info)


def test_pockets_without_side_files_or_report(tmp_path, monkeypatch):
    at = _pockets(tmp_path, monkeypatch, side_files=False, report=False)
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == PK_TABS
    assert any("clim.parquet" in w.value for w in at.sidebar.warning)
    at.radio(key="pk_mode").set_value("recal")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("no recal.parquet" in w.value for w in at.sidebar.warning)
    assert any("clim_rates.parquet" in i.value for i in at.info)


def test_pockets_without_a_run(tmp_path, monkeypatch):
    write_run(tmp_path / "runs")
    at = _section(tmp_path / "runs", "intraday", monkeypatch, view="pockets")
    assert not at.exception, [e.value for e in at.exception]
    assert any("intraday-ft-train" in i.value for i in at.info)
    assert not at.tabs


def _moves(at):
    return next(t for t in at.tabs if t.label == "Moves")


def test_pockets_moves_filter(tmp_path, monkeypatch):
    at = _pockets(tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert _moves(at).slider(key="pk_mv_min_p").value == 0.4
    at.slider(key="pk_mv_min_p").set_value(0.2)      # the synthetic run's calibrated P stays < 0.4
    at.run()
    mv = _moves(at)
    assert any(m.value.startswith("**In short**") for m in mv.markdown)
    summ = mv.dataframe[0].value
    assert list(summ.side) == ["up", "dn", "both"] and summ.predictions.iloc[2] > 0
    days = mv.dataframe[1].value
    assert days.predictions.sum() == summ.predictions.iloc[2]       # per day adds up to the period
    assert days.date.nunique() == 10                                 # the last 10 days by default
    lst = next(d.value for d in mv.dataframe if "real_centre" in d.value)
    assert (lst.p >= 0.2).all() and lst.high.notna().all() and lst.vwap.notna().all()
    assert lst.window.str.match(r"b[1-5]: ").all()                    # the window whose centre is shown
    assert lst.best_x.notna().all() and {"mean_best_x", "median_best_x"} <= set(summ.columns)
    assert (days.max_said.dropna() >= days.said.dropna()).all()
    assert days.touched_only.sum() == int(((lst.reached != 1) & (lst.touched == 1)).sum()) or len(lst) == 500
    _assert_explained(at)
    # every control
    at.select_slider(key="pk_mv_dates").set_value(("2026-01-05", "2026-02-09"))
    for x in (0.005, 0.015, 0.02, 0.03):
        at.selectbox(key="pk_mv_x").set_value(x)
        at.run()
        assert not at.exception, (x, [e.value for e in at.exception])
    at.selectbox(key="pk_mv_x").set_value(0.005)
    for side in ("up", "dn", "both"):
        at.selectbox(key="pk_mv_side").set_value(side)
        at.run()
        assert not at.exception, (side, [e.value for e in at.exception])
    for b in (1, 3, 4, 5):
        at.selectbox(key="pk_mv_bin").set_value(b)
        at.run()
        assert not at.exception, (b, [e.value for e in at.exception])
    assert at.toggle(key="pk_mv_cum").value                          # by the window's end: on by default
    n_cum = _moves(at).dataframe[0].value.predictions.iloc[-1]
    at.toggle(key="pk_mv_cum").set_value(False)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert _moves(at).dataframe[0].value.predictions.iloc[-1] <= n_cum   # one window alone: fewer
    at.toggle(key="pk_mv_cum").set_value(True)
    at.radio(key="pk_mv_mode").set_value("raw")
    at.slider(key="pk_mv_min_p").set_value(0.3)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    n_all = _moves(at).dataframe[0].value.predictions.iloc[-1]
    at.checkbox(key="pk_mv_first").check()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert _moves(at).dataframe[0].value.predictions.iloc[-1] < n_all  # runs merged
    day = min(_moves(at).selectbox(key="pk_mv_day").options)
    at.selectbox(key="pk_mv_day").set_value(day)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    _assert_explained(at)
    # no match
    at.slider(key="pk_mv_min_p").set_value(0.95)
    at.selectbox(key="pk_mv_x").set_value(0.03)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No decision" in m.value for m in _moves(at).markdown)


def test_pockets_moves_without_extremes(tmp_path, monkeypatch):
    at = _pockets(tmp_path, monkeypatch, extremes=False)
    at.slider(key="pk_mv_min_p").set_value(0.2)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    mv = _moves(at)
    assert any("extremes.parquet" in w.value for w in mv.warning)
    lst = next(d.value for d in mv.dataframe if "real_centre" in d.value)
    assert lst.high.isna().all() and lst.vwap.isna().all() and lst.real_centre.notna().any()
