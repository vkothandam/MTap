"""CLI for the news processing subcommands.

Dispatched from `sourcing_py.__main__` when the first arg is `news`:

    sourcing-py news aggregate [--symbols A,B | --all] [--days N] [--no-resume]
    sourcing-py news analyze [--no-resume]                 # FinBERT + verified attribution
    sourcing-py news build-daily [--fromdate D1 --todate D2]   # segment 3, on-demand
    sourcing-py news run [--symbols A,B | --all] [--days N]     # aggregate -> analyze
    sourcing-py news scrape-yahoo [--source stock-market-news,most-active] [--since YYYY-MM-DD]
                                  [--pages N] [--no-resume] [--no-raw]   # Yahoo Finance pages
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from ..common.errors import ConfigError, FetchError
from . import aggregator, daily, sentiment, yahoo

_USAGE = (
    "usage:\n"
    "  sourcing-py news aggregate [--symbols A,B | --all] [--days N] [--no-resume]\n"
    "  sourcing-py news analyze [--no-resume]   (FinBERT sentiment + verified symbol matching)\n"
    "  sourcing-py news build-daily [--fromdate D1 --todate D2]   (segment 3, on-demand)\n"
    "  sourcing-py news run [--symbols A,B | --all] [--days N]     (aggregate -> analyze)\n"
    "  sourcing-py news scrape-yahoo [--source stock-market-news,most-active] [--since YYYY-MM-DD]\n"
    "                               [--pages N] [--no-resume] [--no-raw]\n"
    "      (Yahoo Finance topic pages + most-active news stream -> full articles with exact\n"
    "       timestamps; --since bounds most-active, default 00:00 New York yesterday)"
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


def _cmd_scrape_yahoo(p: dict) -> int:
    try:
        pages = int(p["pages"]) if p.get("pages") else None
        sources = [x.strip() for x in str(p["source"]).split(",") if x.strip()] if p.get("source") else None
        since = None
        if p.get("since"):
            day = date.fromisoformat(str(p["since"]))
            since = datetime(day.year, day.month, day.day, tzinfo=ZoneInfo("America/New_York"))
        summary = yahoo.scrape(
            sources=sources, pages=pages, since=since,
            resume=not p.get("no-resume"), keep_raw=not p.get("no-raw"),
        )
    except (ConfigError, FetchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ValueError) else 1
    for err in summary["errors"]:
        print(f"  failed: {err}", file=sys.stderr)
    for msg in summary["age_mismatch"]:
        print(f"  warning: {msg}", file=sys.stderr)
    if summary["capped"]:
        print(f"  note: the most-active stream ends at Yahoo's cap ({summary['capped']:%Y-%m-%d %H:%M}Z), "
              "after the cutoff; older stories in it were not reachable", file=sys.stderr)
    dropped = ", ".join(f"{n} {r}" for r, n in sorted(summary["dropped"].items())) or "none"
    head = ["listed", "stored", "new", "duplicate", "failed"]
    print(f"{'':<20}" + "".join(f"{h:>10}" for h in head))
    for name, counts in summary["report"].items():
        print(f"{name:<20}" + "".join(f"{counts[h]:>10}" for h in head))
    print(
        f"stored = by an earlier run; duplicate = same story under another url, dropped. "
        f"{summary['pages']} topic page(s), most-active since {summary['since']:%Y-%m-%d %H:%M}Z "
        f"(not listed: {dropped})\n"
        f"{summary['fetched']} fetched: {summary['ok']} ok, {summary['no_timestamp']} without timestamp, "
        f"{summary['failed']} failed; {summary['paywalled']} paywalled teaser(s), "
        f"{summary['short']} short body(ies); {summary['total_articles']:,} yahoo-topic articles in DB"
    )
    # an empty listing means the page layout changed under the selectors
    bad = summary["listed"] == 0 or summary["failed"] or summary["no_timestamp"] or summary["errors"]
    return 1 if bad else 0


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
    if sub == "scrape-yahoo":
        return _cmd_scrape_yahoo(params)
    print(f"unknown news subcommand {sub!r}\n{_USAGE}", file=sys.stderr)
    return 2
