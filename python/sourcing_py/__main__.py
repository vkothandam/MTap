"""CLI entry point: `sourcing-py run <source> [options]`.

Date handling (for date-based sources like massive_grouped_daily):
  --date D                  run a single day D
  --fromdate D1 --todate D2 run every day in [D1, D2] inclusive
  --fromdate D1             run [D1, today] inclusive
  --retry-failed            rerun only the days in this source's failure log

Each day is fetched and written independently: a day that errors is logged to
state/failures/<source>.jsonl and the run continues. Rerun with --retry-failed to
reattempt just those days; entries that succeed are pruned from the log.
"""

from __future__ import annotations

import importlib
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .common import config, failures
from .common.errors import ConfigError


def _load_source_class(name: str):
    """Import sources.<name>.source and return its single Source subclass."""
    from .common.source import Source

    module = importlib.import_module(f"sourcing_py.sources.{name}.source")
    for attr in vars(module).values():
        if isinstance(attr, type) and issubclass(attr, Source) and attr is not Source:
            return attr
    raise LookupError(f"No Source subclass found in sources.{name}.source")


def _parse_params(args: list[str]) -> dict[str, Any]:
    """Parse --key value pairs and bare --flags (value True) into a dict."""
    params: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise ValueError(f"expected --key value or --flag, got {token!r}")
        key = token[2:]
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            params[key] = args[i + 1]
            i += 2
        else:
            params[key] = True  # boolean flag
            i += 1
    return params


def _date_range(params: dict[str, Any], *, weekdays_only: bool = False) -> list[date | None]:
    """Resolve params into the list of logical dates to run.

    Returns [None] when no date options are given (non-date-based source). An explicit
    single --date is always honored; weekend filtering applies only to a from/to range.
    """
    if params.get("date"):
        return [date.fromisoformat(params["date"])]
    if params.get("fromdate"):
        start = date.fromisoformat(params["fromdate"])
        end = (
            date.fromisoformat(params["todate"])
            if params.get("todate")
            else datetime.now(timezone.utc).date()
        )
        if end < start:
            raise ValueError(f"todate {end} is before fromdate {start}")
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        if weekdays_only:
            days = [d for d in days if d.weekday() < 5]  # Mon–Fri
        return days
    return [None]


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "etrade":
        from .etrade.cli import main as etrade_main

        return etrade_main(argv[1:])
    if argv and argv[0] == "news":
        from .news.cli import main as news_main

        return news_main(argv[1:])
    if len(argv) < 2 or argv[0] != "run":
        print(
            "usage: sourcing-py run <source> "
            "[--date D | --fromdate D1 [--todate D2] | --retry-failed]\n"
            "       sourcing-py etrade <build-symbols|ingest-eod|fetch|derive-features> [options]\n"
            "       sourcing-py news <aggregate|analyze|build-daily|run> [options]",
            file=sys.stderr,
        )
        return 2

    name = argv[1]
    try:
        params = _parse_params(argv[2:])
        config.load_source(name)  # validate it's registered before importing code
        source = _load_source_class(name)()
        if params.get("retry-failed"):
            dates: list[date | None] = [date.fromisoformat(d) for d in failures.load_dates(name)]
            if not dates:
                print(f"no failed days logged for {name}")
                return 0
            print(f"retrying {len(dates)} previously failed day(s)")
        else:
            dates = _date_range(params, weekdays_only=source.weekdays_only)
    except (ValueError, KeyError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Fail fast on config problems (missing key) rather than failing every day.
    try:
        source.preflight()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    hint = source.cadence_hint()
    if hint and len(dates) > 1:
        print(hint)

    retrying = bool(params.get("retry-failed"))
    written = 0
    still_failed: list[dict] = []
    for d in dates:
        run_params = dict(params)
        if d is not None:
            run_params["date"] = d.isoformat()
        try:
            path = source.run(params=run_params)
        except ConfigError as exc:
            # A config problem mid-run is fatal for every remaining day.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # per-day failure: log and keep going
            msg = f"{type(exc).__name__}: {exc}"
            print(f"{d}: FAILED — {msg}", file=sys.stderr)
            entry = {"date": d.isoformat(), "error": msg, "at": datetime.now(timezone.utc).isoformat()}
            if retrying:
                still_failed.append(entry)
            else:
                failures.record(name, d.isoformat(), msg)
            continue

        if path is None:
            print(f"{d}: no data, skipped")
        else:
            print(f"{d}: wrote {path}")
            written += 1

    if retrying:
        failures.rewrite(name, still_failed)  # drain succeeded days; keep the rest

    print(f"done — {written}/{len(dates)} day(s) written{_failure_summary(name, retrying, still_failed)}")
    return 1 if (retrying and still_failed) else 0


def _failure_summary(source: str, retrying: bool, still_failed: list[dict]) -> str:
    if retrying:
        return f"; {len(still_failed)} still failing" if still_failed else "; all retried days recovered"
    remaining = failures.load_dates(source)
    if remaining:
        return f"; {len(remaining)} day(s) logged as failed — rerun with --retry-failed"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
