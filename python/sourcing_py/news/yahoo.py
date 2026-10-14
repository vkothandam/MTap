"""Yahoo Finance topic scraper — the curated stock-market-news stream.

Yahoo's topic page (https://finance.yahoo.com/topic/stock-market-news/, pages /2/../6/) is
the broadest clean market-news list available without a paid feed. It only holds ~6 pages
(~165 articles, about two days); asking for page 7+ redirects to page 1, and older stories
drop off, so the scraper has to run often enough to see each article before it expires.

Two levels, like the RSS aggregator:

  Listing: each page's `section.main` holds one `li.list-item` per story. A story's first
    finance.yahoo.com `/article(s)/<slug>.html` link is the article; items without one
    are ads (Seeking Alpha, Taboola, ...). The listing only shows a relative age ("21h
    ago", floored to the unit). Yahoo computes it from the same datePublished, so it is
    kept as a cross-check and as the approximate time when the page has no exact one.
  Article: the page's JSON-LD NewsArticle carries datePublished / dateModified to the
    second (UTC), the author, the syndicating provider and the publisher's original url;
    `time[datetime]` is the fallback for the timestamp. The full body is server-rendered
    in `[data-testid="article-body"]` ("read more" is only a CSS collapse) and goes
    through trafilatura. trafilatura's own date is day-only, so it is never used for time.

Rows land in `news_articles` (feed 'yahoo-topic'), so `news analyze` scores them like any
other article. Raw HTML is kept gzipped under `raw_dir` for evaluating the parser later;
it is disposable. Yahoo answers the default mtap User-Agent with 429, so the scraper sends
a browser UA (config [news.yahoo]).
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

from selectolax.parser import HTMLParser, Node

from ..common import config, http, storage
from ..common.errors import ConfigError, FetchError
from . import db
from .aggregator import _DomainLimiter, body_status, extract_article

FEED = "yahoo-topic"
SITE = "finance.yahoo.com"

# Yahoo-hosted articles: /markets/stocks/articles/<slug>.html, /technology/article/<slug>.html,
# ... (query strings like ?.tsrc=fin-notif are dropped by the match).
ARTICLE_RE = re.compile(r"^https://finance\.yahoo\.com/(?:[^?#]+/)?articles?/[^/?#]+\.html")
_QUOTE_RE = re.compile(r"/quote/([^/?#]+)")
_LD_TYPES = {"NewsArticle", "Article", "ReportageNewsArticle", "AnalysisNewsArticle"}
_AGE_RE = re.compile(r"(\d+)\s*([mhd])\w* ago")
_AGE_UNIT = {"m": 60, "h": 3600, "d": 86400}

_COLUMNS = [
    "article_id", "feed", "feed_symbol", "site", "url", "title", "summary", "body_text", "body_status",
    "published_at", "lang", "extract_status", "fetched_at", "modified_at", "timestamp_source",
    "author", "provider", "original_url", "section", "tickers", "content_uuid",
    "listing_page", "listing_position", "first_listed_at", "raw_path", "listing_age",
]
# A re-fetch (--no-resume, or retrying a failed article) refreshes the content but keeps
# where and when the article was first seen on the listing.
_KEEP_FIRST = {"listing_page", "listing_position", "first_listed_at", "listing_age", "content_uuid"}
_UPDATE = ", ".join(
    f"{c} = coalesce({c}, excluded.{c})" if c in _KEEP_FIRST else f"{c} = excluded.{c}"
    for c in _COLUMNS if c != "article_id"
)


def article_id(url: str) -> str:
    return hashlib.sha1(f"{FEED}|{url}".encode()).hexdigest()


def page_url(topic_url: str, page: int) -> str:
    base = topic_url.rstrip("/")
    return f"{base}/" if page == 1 else f"{base}/{page}/"


# -- parsing (pure) -------------------------------------------------------------
def age_bounds(text: str | None) -> tuple[int, int] | None:
    """'21h ago' -> (75600, 79200): the story is at least 21h and under 22h old."""
    m = _AGE_RE.fullmatch((text or "").strip())
    if m is None:
        return None
    n, unit = int(m.group(1)), _AGE_UNIT[m.group(2)]
    return n * unit, (n + 1) * unit


def _quote_symbols(node: Node | None) -> list[str]:
    """Symbols from /quote/<SYM>/ links, url-decoded (%5EGSPC -> ^GSPC), in page order."""
    if node is None:
        return []
    out: list[str] = []
    for a in node.css('a[href*="/quote/"]'):
        m = _QUOTE_RE.search(a.attributes.get("href") or "")
        if m and (sym := unquote(m.group(1))) not in out:
            out.append(sym)
    return out


def parse_listing(html: str, page: int) -> list[dict]:
    """The articles on one topic page, in display order, ads skipped."""
    tree = HTMLParser(html)
    root = tree.css_first("section.main") or tree.css_first("#nimbus-app main")
    if root is None:
        return []
    items: list[dict] = []
    seen: set[str] = set()
    for li in root.css("li.list-item"):
        url = next(
            (m.group(0) for a in li.css("a[href]")
             if (m := ARTICLE_RE.match(a.attributes.get("href") or ""))),
            None,
        )
        if url is None or url in seen:
            continue
        seen.add(url)
        yga: dict = {}
        if (tagged := li.css_first("a[data-yga]")) is not None:
            try:
                yga = json.loads(tagged.attributes.get("data-yga") or "{}")
            except json.JSONDecodeError:
                pass
        heading = li.css_first("h3")
        age = li.css_first(".published-date")
        items.append({
            "url": url,
            "title": heading.text(strip=True) if heading else yga.get("yLinkText"),
            "listing_publisher": yga.get("yContentPartner"),
            "content_uuid": yga.get("yDestinationContentUUID"),
            "listing_tickers": _quote_symbols(li),
            "listing_age": age.text(strip=True) if age is not None else None,
            "page": page,
            "position": len(items) + 1,
        })
    return items


def _utc(value) -> datetime | None:
    """ISO-8601 -> tz-aware UTC datetime (naive input is taken as UTC); None if unparseable."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _news_ld(tree: HTMLParser) -> dict:
    """The page's JSON-LD article object ({} if none), looking through lists and @graph."""
    for script in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.text())
        except json.JSONDecodeError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            obj = stack.pop(0)
            if not isinstance(obj, dict):
                continue
            types = obj.get("@type")
            if set(types if isinstance(types, list) else [types]) & _LD_TYPES:
                return obj
            stack.extend(obj.get("@graph") or [])
    return {}


def _names(value) -> str | None:
    """'name' of a schema.org Person/Organization, or of each in a list, joined with '; '."""
    items = value if isinstance(value, list) else [value]
    names = [
        (v.get("name") if isinstance(v, dict) else v) for v in items if v
    ]
    names = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    return "; ".join(dict.fromkeys(names)) or None


def _meta(tree: HTMLParser, key: str) -> str | None:
    node = tree.css_first(f'meta[property="{key}"]') or tree.css_first(f'meta[name="{key}"]')
    return (node.attributes.get("content") or None) if node else None


def parse_article(html: str, url: str) -> dict:
    """Title, exact timestamps, byline, provider, tickers and clean body of an article page."""
    tree = HTMLParser(html)
    ld = _news_ld(tree)

    published, source = _utc(ld.get("datePublished")), "ld+json"
    time_tag = tree.css_first("time[datetime]")
    if published is None and time_tag is not None:
        published, source = _utc(time_tag.attributes.get("datetime")), "time-tag"
    if published is None:
        source = None

    body_node = tree.css_first('[data-testid="article-body"]')
    body = extract_article(body_node.html) if body_node is not None else None
    if not body:
        body = extract_article(html)

    canonical = (ld.get("mainEntityOfPage") or {})
    canonical = canonical.get("@id") if isinstance(canonical, dict) else canonical
    html_node = tree.css_first("html")
    title = ld.get("headline") or _meta(tree, "og:title")
    if not title and (h1 := tree.css_first("h1")) is not None:
        title = h1.text(strip=True)
    free = ld.get("isAccessibleForFree")
    if isinstance(free, str):
        free = {"true": True, "false": False}.get(free.strip().lower())
    if not body:
        status = "failed"
    elif published is None:
        status = "no_timestamp"
    else:
        status = "ok"
    return {
        "title": title,
        "summary": ld.get("description") or _meta(tree, "description"),
        "body_text": body,
        "body_status": body_status(body, free if isinstance(free, bool) else None),
        "published_at": published,
        "modified_at": _utc(ld.get("dateModified")),
        "timestamp_source": source,
        "author": _names(ld.get("author")),
        "provider": _names(ld.get("provider")),
        "original_url": canonical if isinstance(canonical, str) and canonical != url else None,
        "section": ld.get("articleSection") if isinstance(ld.get("articleSection"), str) else None,
        "tickers": _quote_symbols(tree.css_first("section.ticker-list")),
        "lang": html_node.attributes.get("lang") if html_node is not None else None,
        "extract_status": status,
    }


# -- network (thin, monkeypatched in tests) -------------------------------------
async def _fetch(client, url: str) -> tuple[str, str]:
    """Return (final_url, html), following redirects."""
    return await http.aget_text(client, url)


# -- orchestration --------------------------------------------------------------
def _write_raw(path: Path, html: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(html)
    return str(path)


def _progress(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def _slug(url: str) -> str:
    return url.rsplit("/", 1)[-1].removesuffix(".html")


def _check_age(row: dict, run_at: datetime, stats: dict) -> None:
    """Cross-check the exact time against the listing's relative age, or, when the page
    has no exact time, fall back to the middle of the age's range (approximate)."""
    bounds = age_bounds(row.get("listing_age"))
    if bounds is None:
        return
    lo, hi = bounds
    published = row.get("published_at")
    if published is None:
        row["published_at"] = run_at - timedelta(seconds=(lo + hi) / 2)
        row["timestamp_source"] = "listing-age"
        return
    # 2 min of slack: the listing page was fetched a little before run_at's articles
    age = (run_at - published).total_seconds()
    if not lo - 120 <= age < hi + 120:
        stats["age_mismatch"].append(
            f"{row['url']}: listed '{row['listing_age']}' but published {published:%Y-%m-%d %H:%M}Z"
        )


async def _collect(ycfg, pages, seen, run_at, raw_day, stats) -> list[dict]:
    limiter = _DomainLimiter(ycfg["per_domain_rps"])
    sem = asyncio.Semaphore(ycfg["max_concurrency"])
    stamp = run_at.strftime("%Y%m%dT%H%M%SZ")
    headers = {"user-agent": ycfg["user_agent"], "accept-language": "en-US,en;q=0.9"}

    async with http.get_async_client(headers=headers, follow_redirects=True) as client:
        # Listing pages one at a time, in order; past the last page Yahoo redirects to page 1.
        listed: dict[str, dict] = {}
        for n in range(1, pages + 1):
            want = page_url(ycfg["topic_url"], n)
            await limiter.wait(SITE)
            try:
                final, html = await _fetch(client, want)
            except FetchError as exc:
                if n == 1:
                    raise
                stats["errors"].append(f"listing page {n}: {exc}")
                break
            if final.rstrip("/") != want.rstrip("/"):
                break  # past the last page
            stats["pages"] += 1
            if raw_day is not None:
                await asyncio.to_thread(
                    _write_raw, raw_day / f"listing-{stamp}-p{n}.html.gz", html
                )
            found = parse_listing(html, n)
            for item in found:
                listed.setdefault(item["url"], item)
            _progress(f"page {n}: {len(found)} article(s), {len(listed)} unique so far")
        stats["listed"] = len(listed)

        # Already-parsed articles are never fetched again: same url (article_id) or same
        # Yahoo content id, whether stored by an earlier run or listed twice in this one.
        todo = []
        for it in listed.values():
            uuid = it["content_uuid"]
            if article_id(it["url"]) in seen["ids"] or (uuid and uuid in seen["uuids"]):
                continue
            if uuid:
                seen["uuids"].add(uuid)
            todo.append(it)
        stats["skipped_seen"] = len(listed) - len(todo)
        _progress(f"{len(todo)} to fetch, {stats['skipped_seen']} already stored")
        done = 0

        def report(status: str, row: dict) -> None:
            nonlocal done
            done += 1
            when = row.get("published_at")
            when = f"{when:%Y-%m-%d %H:%M}Z" if when else "no timestamp    "
            _progress(f"[{done}/{len(todo)}] {status:<9} {when}  {row.get('provider') or '?'}: "
                      f"{(row.get('title') or row['url'])[:70]}")

        async def fetch_article(item: dict) -> dict | None:
            url = item["url"]
            row = {
                "article_id": article_id(url), "feed": FEED, "feed_symbol": None, "site": SITE,
                "url": url, "title": item["title"], "provider": item["listing_publisher"],
                "tickers": item["listing_tickers"], "content_uuid": item["content_uuid"],
                "listing_page": item["page"], "listing_position": item["position"],
                "first_listed_at": run_at, "listing_age": item["listing_age"],
                "extract_status": "failed", "body_status": "missing",
            }
            try:
                await limiter.wait(SITE)
                async with sem:
                    _final, html = await _fetch(client, url)
            except FetchError as exc:
                stats["errors"].append(f"{url}: {exc}")
                row["fetched_at"] = datetime.now(UTC)
                report("failed", row)
                return row
            row["fetched_at"] = datetime.now(UTC)
            parsed = await asyncio.to_thread(parse_article, html, url)
            # The same syndicated story under a second Yahoo url: drop it. No await between
            # the check and the add, so concurrent fetches can't both claim it.
            original = parsed["original_url"]
            if original and original in seen["originals"]:
                stats["duplicates"] += 1
                report("duplicate", {**row, **parsed})
                return None
            if original:
                seen["originals"].add(original)
            # listing values stand in where the article page has nothing
            row.update({k: v for k, v in parsed.items() if v not in (None, [])})
            row["extract_status"] = parsed["extract_status"]
            _check_age(row, run_at, stats)
            if raw_day is not None:
                row["raw_path"] = await asyncio.to_thread(
                    _write_raw, raw_day / f"{_slug(url)}.html.gz", html
                )
            report(row["extract_status"], row)
            return row

        rows = await asyncio.gather(*(fetch_article(it) for it in todo))
        return [r for r in rows if r is not None]


def scrape(
    *, pages: int | None = None, resume: bool = True, keep_raw: bool = True, db_path=None
) -> dict:
    """Scrape the topic listing + every new article into news_articles. Returns a summary."""
    ycfg = config.news_config()["yahoo"]
    pages = pages if pages is not None else ycfg["pages"]
    if pages < 1:
        raise ValueError("--pages must be >= 1")
    run_at = datetime.now(UTC)

    raw_day = None
    if keep_raw:
        raw_dir = Path(ycfg["raw_dir"])
        if storage.unmounted_volume(raw_dir):
            raise ConfigError(f"raw_dir {raw_dir} is under /Volumes but the share is not mounted")
        raw_day = raw_dir / f"date={run_at:%Y-%m-%d}"

    # Every parsed article ('ok' or 'no_timestamp') counts as done, paywalled teasers too; only fetch failures and
    # empty bodies are retried while still on the listing. The DB is not held open during
    # the fetch. --no-resume re-extracts everything (still deduped within the run).
    seen: dict[str, set[str]] = {"ids": set(), "uuids": set(), "originals": set()}
    if resume:
        with db.connect(db_path) as con:
            for aid, uuid, original in con.execute(
                "SELECT article_id, content_uuid, original_url FROM news_articles "
                "WHERE feed = ? AND extract_status IN ('ok', 'no_timestamp')",
                [FEED],
            ).fetchall():
                seen["ids"].add(aid)
                if uuid:
                    seen["uuids"].add(uuid)
                if original:
                    seen["originals"].add(original)

    stats = {"pages": 0, "listed": 0, "skipped_seen": 0, "duplicates": 0, "errors": [],
             "age_mismatch": []}
    rows = asyncio.run(_collect(ycfg, pages, seen, run_at, raw_day, stats))

    with db.connect(db_path) as con:
        db.bulk_upsert(con, "news_articles", _COLUMNS, "article_id", _UPDATE, rows)
        total = con.execute(
            "SELECT count(*) FROM news_articles WHERE feed = ?", [FEED]
        ).fetchone()[0]

    by_status = {s: sum(r["extract_status"] == s for r in rows)
                 for s in ("ok", "no_timestamp", "failed")}
    by_body = {s: sum(r["body_status"] == s for r in rows) for s in ("paywalled", "short")}
    return {
        "pages": stats["pages"],
        "listed": stats["listed"],
        "skipped_seen": stats["skipped_seen"],
        "duplicates": stats["duplicates"],
        "fetched": len(rows),
        **by_status,
        **by_body,
        "errors": stats["errors"],
        "age_mismatch": stats["age_mismatch"],
        "total_articles": total,
    }
