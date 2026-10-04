"""CLI for the E*TRADE fundamentals subcommands.

Dispatched from `sourcing_py.__main__` when the first arg is `etrade`:

    sourcing-py etrade build-symbols [--days 90 | --fromdate D1 [--todate D2]]
    sourcing-py etrade ingest-eod [--days N | --fromdate D1 [--todate D2]]
    sourcing-py etrade login [--once]
    sourcing-py etrade map-industries [--trading-db PATH]
    sourcing-py etrade fetch [--symbols A,B | --all] [--resume] [--retry-failed]
    sourcing-py etrade derive-features [--symbols A,B]
    sourcing-py etrade export-tft [--out PATH] [--include-inference] [--min-bars N] ...
"""

from __future__ import annotations

import sys

from ..common.errors import ConfigError
from . import eod, export, features, fundamentals, industry, symbols
from . import login as login_mod

_USAGE = (
    "usage:\n"
    "  sourcing-py etrade build-symbols [--days N | --fromdate D1 [--todate D2]]\n"
    "  sourcing-py etrade ingest-eod [--days N | --fromdate D1 [--todate D2]]  (default: all files)\n"
    "  sourcing-py etrade login [--once]   (puppeteer login -> state/etrade/session.json)\n"
    "  sourcing-py etrade map-industries [--trading-db PATH]  (sector/industry from MBin)\n"
    "  sourcing-py etrade fetch [--symbols A,B | --all] [--resume] [--retry-failed]\n"
    "  sourcing-py etrade derive-features [--symbols A,B]  (day_idx, filing flags, VWAP; re-run after ingest-eod)\n"
    "  sourcing-py etrade export-tft [--out PATH] [--include-inference] [--issue-types CS,DR]\n"
    "      [--min-bars 120] [--min-median-vwap 1] [--min-median-volume 50000]  (TFT panel.parquet)"
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


def _cmd_login(p: dict) -> int:
    try:
        return login_mod.login(once=bool(p.get("once")))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_map_industries(p: dict) -> int:
    trading_db = p["trading-db"] if p.get("trading-db") and p["trading-db"] is not True else None
    try:
        summary = industry.map_industries(trading_db_path=trading_db)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"mapped industries from {summary['trading_db']}: "
        f"{summary['sectors']} sectors, {summary['industries']} industries; "
        f"{summary['memberships']} memberships across {summary['symbols_matched']} symbols "
        f"({summary['symbols_matched']}/{summary['source_symbols']} source symbols matched, "
        f"{summary['symbols_unmatched']} not in our table)"
    )
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


def _cmd_derive_features(p: dict) -> int:
    syms = [s.strip() for s in str(p["symbols"]).split(",")] if p.get("symbols") else None
    summary = features.derive_features(symbols=syms)
    print(
        f"derived features for {summary['bars']:,} bars across {summary['symbols']:,} symbols; "
        f"{summary['calendar_sessions']:,} calendar sessions, "
        f"{summary['filings_snapped']:,} filings mapped"
    )
    if summary["bars_missing_day_idx"]:
        print(
            f"  {summary['bars_missing_day_idx']:,} bar(s) had a non-session date (no day_idx)",
            file=sys.stderr,
        )
    return 0


def _cmd_export_tft(p: dict) -> int:
    kwargs: dict = {"include_inference": bool(p.get("include-inference"))}
    if p.get("out"):
        kwargs["out_path"] = p["out"]
    if p.get("issue-types"):
        kwargs["issue_types"] = tuple(t.strip() for t in str(p["issue-types"]).split(","))
    for flag, key, cast in (
        ("min-bars", "min_bars", int),
        ("min-median-vwap", "min_median_vwap", float),
        ("min-median-volume", "min_median_volume", float),
    ):
        if p.get(flag):
            kwargs[key] = cast(p[flag])
    try:
        summary = export.export_tft(**kwargs)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    lo, hi = summary["date_range"]
    news_share = summary["news_rows"] / summary["rows"] if summary["rows"] else 0.0
    print(
        f"exported {summary['rows']:,} rows across {summary['symbols']:,} symbols "
        f"[{lo} .. {hi}, day_idx {summary['day_idx_range'][0]}..{summary['day_idx_range'][1]}] "
        f"-> {summary['out_path']}"
    )
    print(
        f"  {summary['inference_rows']:,} inference row(s); "
        f"{news_share:.2%} of rows have news sentiment"
    )
    return 0


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
    if sub == "login":
        return _cmd_login(params)
    if sub == "map-industries":
        return _cmd_map_industries(params)
    if sub == "fetch":
        return _cmd_fetch(params)
    if sub == "derive-features":
        return _cmd_derive_features(params)
    if sub == "export-tft":
        return _cmd_export_tft(params)
    print(f"unknown etrade subcommand {sub!r}\n{_USAGE}", file=sys.stderr)
    return 2
