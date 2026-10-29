"""CLI for pre-market / after-hours bars (sourcing_py/massive_ext/extended.py).

Dispatched from `sourcing_py.__main__` when the first arg is `massive-ext`:

    sourcing-py massive-ext fetch --fromdate D1 --todate D2 [--tickers A,B] [--workers N] [--keep]
    sourcing-py massive-ext gaps [--days 14] [--max 10] [--list]

`fetch` is the backfill: rerun the same command to resume it or retry failed tickers.
--allow-failures N finalizes even with up to N failed tickers; --chunk quarter runs each
calendar quarter as its own resumable job (finished quarters are skipped on a rerun) and
--parallel N runs N quarters at a time, one process each; --save-bars DIR also writes the
raw minute bars as DIR/month=YYYY-MM/*.parquet (New York month); --bars-universe
PANEL.parquet limits the saved bars to the symbols of that panel.
`gaps` fetches the recent sessions that have a daily_summary file but no extended_hours
file (scripts/schedule/daily-sourcing.sh), refreshing the common-stock list first when it
is over a week old; --list only prints them.
"""

from __future__ import annotations

import sys
from datetime import date, datetime

from ..common.errors import ConfigError
from ..intraday.cli import _parse

_USAGE = (
    "usage:\n"
    "  sourcing-py massive-ext fetch --fromdate D1 --todate D2 [--tickers A,B] [--workers N] [--keep]\n"
    "      [--allow-failures N] [--chunk quarter [--parallel N]] [--save-bars DIR]\n"
    "      [--bars-universe PANEL.parquet]\n"
    "  sourcing-py massive-ext gaps [--days 14] [--max 10] [--list]"
)


def main(argv: list[str]) -> int:
    from . import extended as ext

    if not argv or argv[0] not in ("fetch", "gaps"):
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        params = _parse(argv[1:])
        workers = int(params["workers"]) if params.get("workers") else None
        tickers = [t.strip() for t in params["tickers"].split(",")] if params.get("tickers") else None
        if argv[0] == "fetch":
            d0, d1 = date.fromisoformat(params["fromdate"]), date.fromisoformat(params["todate"])
            if d1 < d0:
                raise ValueError(f"todate {d1} is before fromdate {d0}")
            return ext.fetch(d0, d1, tickers=tickers, workers=workers, keep=bool(params.get("keep")),
                             allow_failures=int(params.get("allow-failures", 0)),
                             chunk=params.get("chunk") or None,
                             parallel=int(params.get("parallel", 1)),
                             save_bars=params.get("save-bars") or None,
                             bars_universe=(ext.panel_symbols(params["bars-universe"])
                                            if params.get("bars-universe") else None))
        gaps = ext.missing_sessions(datetime.now(ext.NY).date(), days=int(params.get("days", 14)),
                                    cap=int(params.get("max", 10)))
        if params.get("list"):
            for d in gaps:
                print(d.isoformat())
            return 0
        if not gaps:
            print("massive-ext gaps: nothing missing")
            return 0
        print(f"massive-ext gaps: {' '.join(d.isoformat() for d in gaps)}")
        if tickers is None and (path := ext.refresh_tickers(datetime.now(ext.NY).date())):
            print(f"refreshed the common-stock list: {path}")
        return ext.fetch(gaps[0], gaps[-1], tickers=tickers, workers=workers, only_dates=set(gaps))
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
