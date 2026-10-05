"""`forecast-dash [--runs DIR] [streamlit options]`: launch the dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast-dash", description=__doc__.split("\n")[0])
    p.add_argument("--runs", default="../ml/runs",
                   help="a run directory, or a directory of runs (default: ../ml/runs)")
    p.add_argument("--port", type=int, default=8501)
    a, rest = p.parse_known_args(argv)
    from streamlit.web import cli as stcli

    # headless when not on a terminal: else streamlit's first-run email prompt blocks forever
    headless = "false" if sys.stdin.isatty() else "true"
    sys.argv = ["streamlit", "run", str(APP), "--server.port", str(a.port),
                "--server.headless", headless, "--browser.gatherUsageStats", "false", *rest,
                "--", "--runs", str(Path(a.runs).resolve())]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
