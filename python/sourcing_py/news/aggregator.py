"""Segment 1 — news aggregator.

Two-level, history-aware aggregation of per-symbol RSS feeds:

  Pass A (feed): for each (symbol x feed) GET the RSS and parse it with feedparser. Each
    entry gives a headline, a link, and a publish date.
  Pass B (article): for feeds flagged `fetch_article`, GET the entry link (following
    redirects — Google News links redirect to the publisher) and extract the main body
    text with trafilatura. Feeds not so flagged store the RSS content directly.

History: every article is keyed by a stable `article_id` (hash of feed + guid/link). If a
row already exists we skip it entirely — no re-fetch, no re-parse — which is what makes the
overlapping multi-month windows Yahoo/Google return cheap to re-poll.

Concurrency: fetching is async (httpx.AsyncClient), so many article fetches — each a
seconds-long round trip — overlap instead of running one at a time. Politeness is enforced
*per domain*: request starts are spaced to at most `per_domain_rps` per host, while a global
`max_concurrency` semaphore caps total in-flight requests. Per-symbol feed failures are
logged for --retry-failed, mirroring the etrade collector. All DuckDB writes stay on the
calling (sync) thread; the event loop only does network + extraction.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import feedparser

from ..common import config, failures, http
from ..common.errors import FetchError
from . import db

_SCOPE = "news"  # state/news/failures.jsonl


def _article_id(feed: str, entry) -> str:
    key = entry.get("id") or entry.get("guid") or entry.get("link") or entry.get("title") or ""
    return hashlib.sha1(f"{feed}|{key}".encode()).hexdigest()


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc.removeprefix("www.")


def _published_at(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


# -- network (thin, monkeypatched in tests) -------------------------------------
async def _fetch_feed(client, url: str) -> str:
    _final, text = await http.aget_text(client, url)
    return text


async def _fetch_article(client, url: str) -> tuple[str, str]:
    """Return (resolved_url, html) for an article link, following redirects."""
    return await http.aget_text(client, url)


def extract_article(html: str) -> str | None:
    """Extract the main article text from raw HTML. Swappable extraction backend."""
    import trafilatura

    if not html:
        return None
    return trafilatura.extract(html, include_comments=False, include_tables=False) or None


class _DomainLimiter:
    """Space request starts to at most `rps` per domain, while allowing concurrency.

    A per-domain lock reserves the next start slot (spaced by 1/rps); the actual sleep to
    that slot happens outside the lock, so requests to the *same* host are throttled but
    requests to *different* hosts (and later requests to the same host) still overlap.
    """

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps if rps > 0 else 0.0
        self._next: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def wait(self, domain: str) -> None:
        if self._min_interval <= 0:
            return
        loop = asyncio.get_event_loop()
        async with self._locks[domain]:
            start = max(loop.time(), self._next[domain])
            self._next[domain] = start + self._min_interval
        delay = start - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


# -- orchestration --------------------------------------------------------------
def _target_symbols(*, symbols, fetch_all, etrade_db_path) -> list[str]:
    if symbols:
        return symbols
    if fetch_all:
        return [r["symbol"] for r in db.load_symbol_universe(etrade_db_path)]
    raise ValueError("specify --symbols A,B or --all")


def _base_row(aid, feed, feed_symbol, entry, published, parsed) -> dict:
    link = entry.get("link") or ""
    return {
        "article_id": aid,
        "feed": feed["name"],
        "feed_symbol": feed_symbol,
        "site": _domain(link),
        "url": link,
        "title": entry.get("title"),
        "summary": entry.get("summary"),
        "body_text": None,
        "published_at": published,
        "lang": entry.get("language") or parsed.feed.get("language"),
        "extract_status": "skipped",  # no deep fetch needed / requested
        "fetched_at": datetime.now(timezone.utc),
    }


async def _collect(targets, feeds, cfg, seen, cutoff, stats) -> list[dict]:
    """Fetch feeds + articles concurrently and return the rows to upsert."""
    limiter = _DomainLimiter(cfg["per_domain_rps"])
    sem = asyncio.Semaphore(cfg["max_concurrency"])
    timeout = cfg["request_timeout"]

    async with http.get_async_client(timeout=timeout, follow_redirects=True) as client:

        async def parse_feed(symbol, feed):
            url = feed["url_template"].format(symbol=symbol)
            try:
                await limiter.wait(_domain(url) or "feed")
                async with sem:
                    raw = await _fetch_feed(client, url)
            except FetchError as exc:
                stats["failed_feeds"] += 1
                failures.record_symbol(_SCOPE, symbol, f"feed {feed['name']}: {exc}")
                return []
            parsed = feedparser.parse(raw)
            feed_symbol = symbol if feed.get("tag", "symbol") == "symbol" else None
            # This loop has no awaits, so seen-dedup is atomic vs. other coroutines: the
            # same article surfaced by two symbols' feeds is added once, skipped after.
            items = []
            for entry in parsed.entries:
                aid = _article_id(feed["name"], entry)
                if aid in seen:
                    stats["skipped_seen"] += 1
                    continue
                published = _published_at(entry)
                if cutoff and published and published < cutoff:
                    continue  # outside the lookback window
                seen.add(aid)
                items.append((aid, feed, feed_symbol, entry, published, parsed))
            return items

        feed_items = await asyncio.gather(*(parse_feed(s, f) for s in targets for f in feeds))
        entries = [it for sub in feed_items for it in sub]

        async def fetch_article(item):
            aid, feed, feed_symbol, entry, published, parsed = item
            row = _base_row(aid, feed, feed_symbol, entry, published, parsed)
            link = entry.get("link") or ""
            if feed.get("fetch_article") and link:
                try:
                    await limiter.wait(_domain(link) or "article")
                    async with sem:
                        resolved, html = await _fetch_article(client, link)
                    body = await asyncio.to_thread(extract_article, html)
                    row["url"] = resolved
                    row["site"] = _domain(resolved)
                    row["body_text"] = body
                    row["extract_status"] = "ok" if body else "failed"
                except FetchError:
                    # Keep the feed's title/summary so sentiment can still run; mark failed
                    # so we don't hammer a dead link every run.
                    row["extract_status"] = "failed"
            return row

        return list(await asyncio.gather(*(fetch_article(it) for it in entries)))


def aggregate(
    *,
    symbols: list[str] | None = None,
    fetch_all: bool = False,
    days: int | None = None,
    resume: bool = True,
    feeds: list[dict] | None = None,
    db_path=None,
    etrade_db_path=None,
) -> dict:
    """Fetch + store new articles for the target symbols. Returns a CLI summary.

    `feeds` overrides the configured feed list for this call (e.g. Google-only) without
    touching global config; each entry is a {name, url_template, tag, fetch_article} dict.
    """
    cfg = config.news_config()
    feeds = feeds if feeds is not None else cfg["feeds"]
    lookback = days if days is not None else cfg["lookback_days"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback) if lookback else None

    targets = _target_symbols(symbols=symbols, fetch_all=fetch_all, etrade_db_path=etrade_db_path)
    stats = {"skipped_seen": 0, "failed_feeds": 0}

    with db.connect(db_path) as con:
        # History: preload known ids so an already-processed article is skipped without a
        # fetch. resume=False forces a full re-fetch/re-extract of even known articles.
        seen: set[str] = set()
        if resume:
            seen = {r[0] for r in con.execute("SELECT article_id FROM news_articles").fetchall()}

        # Network + extraction run on the event loop; DuckDB writes stay here on the main
        # thread. asyncio.run builds a fresh loop per call (one per chunk from the driver).
        rows = asyncio.run(_collect(targets, feeds, cfg, seen, cutoff, stats))

        # Set-based ON CONFLICT errors on a duplicate PK within one INSERT; seen-dedup makes
        # aids unique across the run, but guard anyway (last-wins) before the upsert.
        deduped = list({r["article_id"]: r for r in rows}.values())
        if deduped:
            db.bulk_upsert(
                con, "news_articles",
                ["article_id", "feed", "feed_symbol", "site", "url", "title", "summary",
                 "body_text", "published_at", "lang", "extract_status", "fetched_at"],
                "article_id",
                "url = excluded.url, site = excluded.site, body_text = excluded.body_text, "
                "extract_status = excluded.extract_status, fetched_at = excluded.fetched_at",
                deduped,
            )
        total = con.execute("SELECT count(*) FROM news_articles").fetchone()[0]

    return {
        "targets": len(targets),
        "new_articles": len(deduped),
        "skipped_seen": stats["skipped_seen"],
        "failed_feeds": stats["failed_feeds"],
        "total_articles": total,
    }
