"""CLI for the news processing subcommands.

Dispatched from `sourcing_py.__main__` when the first arg is `news`:

    sourcing-py news aggregate [--symbols A,B | --all] [--days N] [--no-resume]
    sourcing-py news analyze [--no-resume]                 # FinBERT + verified attribution
    sourcing-py news build-daily [--fromdate D1 --todate D2]   # segment 3, on-demand
    sourcing-py news run [--symbols A,B | --all] [--days N]     # aggregate -> analyze
"""

from __future__ import annotations

import sys

from ..common.errors import ConfigError
from . import aggregator, daily, sentiment

_USAGE = (
    "usage:\n"
    "  sourcing-py news aggregate [--symbols A,B | --all] [--days N] [--no-resume]\n"
    "  sourcing-py news analyze [--no-resume]   (FinBERT sentiment + verified symbol matching)\n"
    "  sourcing-py news build-daily [--fromdate D1 --todate D2]   (segment 3, on-demand)\n"
    "  sourcing-py news run [--symbols A,B | --all] [--days N]     (aggregate -> analyze)"
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


def _symbols(p: dict) -> list[str] | None:
    return [s.strip() for s in str(p["symbols"]).split(",")] if p.get("symbols") else None


def _cmd_aggregate(p: dict) -> int:
    days = int(p["days"]) if p.get("days") else None
    try:
        summary = aggregator.aggregate(
            symbols=_symbols(p),
            fetch_all=bool(p.get("all")),
            days=days,
            resume=not p.get("no-resume"),
        )
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2 if isinstance(exc, ValueError) else 1
    print(
        f"aggregated {summary['targets']} symbol(s): {summary['new_articles']} new article(s), "
        f"{summary['skipped_seen']} already seen, {summary['failed_feeds']} feed fetch(es) failed; "
        f"{summary['total_articles']:,} articles total in DB"
    )
    return 1 if summary["failed_feeds"] else 0


def _cmd_analyze(p: dict) -> int:
    try:
        summary = sentiment.analyze(resume=not p.get("no-resume"))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"scored {summary['scored']} article(s); "
        f"{summary['attributions']} verified attribution(s) across "
        f"{summary['articles_attributed']} article(s)"
    )
    return 0


def _cmd_build_daily(p: dict) -> int:
    fromdate = p["fromdate"] if p.get("fromdate") and p["fromdate"] is not True else None
    todate = p["todate"] if p.get("todate") and p["todate"] is not True else None
    summary = daily.build_daily(fromdate=fromdate, todate=todate)
    print(f"built symbol_sentiment_daily: {summary['rows_total']:,} row(s) total in DB")
    return 0


def _cmd_run(p: dict) -> int:
    rc = _cmd_aggregate(p)
    if rc > 1:  # usage/config error — don't proceed to analyze
        return rc
    return _cmd_analyze(p) or rc


def main(argv: list[str]) -> int:
    """argv is everything AFTER the `news` verb."""
    if not argv:
        print(_USAGE, file=sys.stderr)
        return 2
    sub, rest = argv[0], argv[1:]
    try:
        params = _parse(rest)
    except ValueError as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2

    if sub == "aggregate":
        return _cmd_aggregate(params)
    if sub == "analyze":
        return _cmd_analyze(params)
    if sub == "build-daily":
        return _cmd_build_daily(params)
    if sub == "run":
        return _cmd_run(params)
    print(f"unknown news subcommand {sub!r}\n{_USAGE}", file=sys.stderr)
    return 2
