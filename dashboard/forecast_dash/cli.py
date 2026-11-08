"""`forecast-dash [--runs DIR] [--intraday-runs DIR] [--pockets-runs DIR]
[--space-lib DIR] [streamlit options]`:
launch the dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast-dash", description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="../ml/runs",
                   help="a run directory, or a directory of runs (default: ../ml/runs)")
    p.add_argument("--intraday-runs", default=None,
                   help="directory of intraday runs (default: <runs>/intraday, or "
                        "$FORECAST_DASH_INTRADAY_RUNS)")
    p.add_argument("--pockets-runs", default=None,
                   help="directory of intraday-ft runs for Intraday → Pockets model (default: "
                        "<runs>/intraday_ft, or $FORECAST_DASH_POCKETS_RUNS)")
    p.add_argument("--space-lib", default=None,
                   help="directory of moment libraries for Intraday → Space (default: "
                        "../out/space, or $FORECAST_DASH_SPACE_LIB)")
    p.add_argument("--port", type=int, default=8501)
    a, rest = p.parse_known_args(argv)
    from streamlit.web import cli as stcli

    # headless when not on a terminal: else streamlit's first-run email prompt blocks forever
    headless = "false" if sys.stdin.isatty() else "true"
    sys.argv = ["streamlit", "run", str(APP), "--server.port", str(a.port),
                "--server.headless", headless, "--browser.gatherUsageStats", "false", *rest,
                "--", "--runs", str(Path(a.runs).resolve())]
    if a.intraday_runs:
        sys.argv += ["--intraday-runs", str(Path(a.intraday_runs).resolve())]
    if a.pockets_runs:
        sys.argv += ["--pockets-runs", str(Path(a.pockets_runs).resolve())]
    if a.space_lib:
        sys.argv += ["--space-lib", str(Path(a.space_lib).resolve())]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
