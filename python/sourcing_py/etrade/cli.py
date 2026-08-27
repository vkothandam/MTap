"""CLI for the E*TRADE fundamentals subcommands.

Dispatched from `sourcing_py.__main__` when the first arg is `etrade`:

    sourcing-py etrade build-symbols [--days 90 | --fromdate D1 [--todate D2]]
    sourcing-py etrade ingest-eod [--days N | --fromdate D1 [--todate D2]]
    sourcing-py etrade fetch [--symbols A,B | --all] [--resume] [--retry-failed]
"""

from __future__ import annotations

import sys

from ..common.errors import ConfigError
from . import eod, fundamentals, symbols

_USAGE = (
    "usage:\n"
    "  sourcing-py etrade build-symbols [--days N | --fromdate D1 [--todate D2]]\n"
    "  sourcing-py etrade ingest-eod [--days N | --fromdate D1 [--todate D2]]  (default: all files)\n"
    "  sourcing-py etrade fetch [--symbols A,B | --all] [--resume] [--retry-failed]"
)


def _parse(args: list[str]) -> dict:
    params: dict = {}
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
            params[key] = True
            i += 1
    return params


def _cmd_build_symbols(p: dict) -> int:
    days = int(p["days"]) if p.get("days") else (None if (p.get("fromdate") or p.get("todate")) else 90)
    summary = symbols.build_symbols(
        days=days, fromdate=p.get("fromdate"), todate=p.get("todate")
    )
    lo, hi = summary["window"]
    print(
        f"built symbols from {summary['files']} file(s) [{lo} .. {hi}]: "
        f"{summary['symbols_in_window']} in window, {summary['symbols_total']} total in DB"
    )
    return 0


def _cmd_ingest_eod(p: dict) -> int:
    # default: all files (days=None); an explicit --days or --fromdate/--todate narrows it
    days = int(p["days"]) if p.get("days") else None
    summary = eod.ingest_eod(days=days, fromdate=p.get("fromdate"), todate=p.get("todate"))
    lo, hi = summary["window"]
    print(
        f"ingested EOD bars from {summary['files']} file(s) [{lo} .. {hi}]: "
        f"{summary['bars_loaded']:,} loaded, {summary['bars_total']:,} bars total, "
        f"{summary['symbols_total']:,} symbols in DB"
    )
    if summary["failed_files"]:
        print(f"  {len(summary['failed_files'])} file(s) failed to parse", file=sys.stderr)
        return 1
    return 0


def _cmd_fetch(p: dict) -> int:
    syms = [s.strip() for s in str(p["symbols"]).split(",")] if p.get("symbols") else None
    try:
        summary = fundamentals.fetch(
            symbols=syms,
            fetch_all=bool(p.get("all")),
            resume=bool(p.get("resume")),
            retry_failed=bool(p.get("retry-failed")),
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    status = "aborted (resume to continue)" if summary.get("aborted") else "done"
    print(
        f"{status} — {summary['ok']}/{summary['targets']} fetched; "
        f"{summary.get('skipped', 0)} no-data (ETF/fund/invalid); {summary['failed']} failed"
    )
    return 1 if (summary.get("aborted") or summary["failed"]) else 0


def main(argv: list[str]) -> int:
    """argv is everything AFTER the `etrade` verb."""
    if not argv:
        print(_USAGE, file=sys.stderr)
        return 2
    sub, rest = argv[0], argv[1:]
    try:
        params = _parse(rest)
    except ValueError as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2

    if sub == "build-symbols":
        return _cmd_build_symbols(params)
    if sub == "ingest-eod":
        return _cmd_ingest_eod(params)
    if sub == "fetch":
        return _cmd_fetch(params)
    print(f"unknown etrade subcommand {sub!r}\n{_USAGE}", file=sys.stderr)
    return 2
