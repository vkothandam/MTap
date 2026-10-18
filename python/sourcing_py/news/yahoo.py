"""Yahoo Finance page scraper: the stock-market-news topic stream and the most-active
page's news stream.

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

Most-active (https://finance.yahoo.com/markets/stocks/most-active/) lists the day's 25
most-traded tickers; its news column is not in the HTML but an infinite scroll fed by a
GraphQL query (GetQSPLeafNewsStream). Each item there carries the exact publish time, a
premium flag and, for stories Yahoo hosts, the same finance.yahoo.com article page. One
query across all the page's tickers, as the page sends it, paged back to a cutoff (default
00:00 New York yesterday) or Yahoo's 200-item cap, whichever comes first; videos,
outlinks and stories Yahoo doesn't host (publisher-only, mostly paywalled) are dropped.

Both sources share one identity per url, so a story is fetched once; `listed_on` records
which pages showed it. Rows land in `news_articles` (feed 'yahoo-topic'), so `news analyze` scores them like any
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
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

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
_ET = ZoneInfo("America/New_York")

TOPIC, MOST_ACTIVE = "stock-market-news", "most-active"
SOURCES = (TOPIC, MOST_ACTIVE)

# The most-active page's news column: the query its infinite scroll sends (from the
# browser's devtools), trimmed to the fields used. Works without cookies.
_STREAM_QUERY = """query GetQSPLeafNewsStream($listInput: LightyearListInput!,
  $clientContext: ClientContext!, $start: Int!, $count: Int!) {
  lightyearList(list_input: $listInput cc: $clientContext start: $start count: $count) {
    main: contentItems {
      stream: assets { __typename asset { __typename
        ... on Story { id: uuid title isPremiumNews isLiveBlog
          contentAttributes { pubDate: publishTime isHosted canonicalUrl
            clickthroughUrl(cc: $clientContext) provider { displayName } }
          finance { stockTickers { symbol } } }
        ... on Video { id: uuid title contentAttributes { pubDate: publishTime } }
        ... on Outlink { uuid } } }
      pageInfo { totalCount start nextPage: hasNextPage count } } } }"""

_COLUMNS = [
    "article_id", "feed", "feed_symbol", "site", "url", "title", "summary", "body_text", "body_status",
    "published_at", "lang", "extract_status", "fetched_at", "modified_at", "timestamp_source",
    "author", "provider", "original_url", "section", "tickers", "content_uuid",
    "listing_page", "listing_position", "first_listed_at", "raw_path", "listing_age", "listed_on",
    "stream_published_at",
]
# A re-fetch (--no-resume, or retrying a failed article) refreshes the content but keeps
# where and when the article was first seen on the listing.
_KEEP_FIRST = {"listing_page", "listing_position", "first_listed_at", "listing_age", "content_uuid",
               "stream_published_at"}
# listed_on only grows: a re-fetch adds this run's pages to the ones already recorded.
_MERGE_LISTED = "list_distinct(list_concat(coalesce({0}, []::VARCHAR[]), coalesce({1}, []::VARCHAR[])))"
_UPDATE = ", ".join(
    f"{c} = coalesce({c}, excluded.{c})" if c in _KEEP_FIRST
    else f"{c} = {_MERGE_LISTED.format(c, 'excluded.' + c)}" if c == "listed_on"
    else f"{c} = excluded.{c}"
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
            "listing_published": None,
            "premium": False,
            "page": page,
            "position": len(items) + 1,
        })
    return items


def parse_most_active_tickers(html: str) -> list[str]:
    """The symbols in the most-active screener table, in table order."""
    out: list[str] = []
    for row in HTMLParser(html).css("table tbody tr"):
        symbols = _quote_symbols(row)
        if symbols and symbols[0] not in out:
            out.append(symbols[0])
    return out


def parse_stream(payload: dict) -> tuple[list[dict], list[tuple[str, datetime | None]], dict]:
    """One most-active stream response -> (stories, dropped, page).

    Stories are hosted Yahoo articles, shaped like parse_listing items. `dropped` holds
    (reason, published) for the rest: videos, outlinks, live blogs, and stories Yahoo
    doesn't host (publisher site only, mostly paywalled: Barron's, IBD). `page` is
    {"next": more pages?, "oldest": oldest publish time on this page}.
    """
    main = ((payload.get("data") or {}).get("lightyearList") or {}).get("main") or {}
    if isinstance(main, list):
        main = main[0] if main else {}
    items: list[dict] = []
    dropped: list[tuple[str, datetime | None]] = []
    times: list[datetime] = []
    for entry in main.get("stream") or []:
        asset = entry.get("asset") or {}
        attrs = asset.get("contentAttributes") or {}
        published = _utc(attrs.get("pubDate"))
        if published is not None:
            times.append(published)
        kind = asset.get("__typename") or "unknown"
        m = ARTICLE_RE.match(attrs.get("clickthroughUrl") or "")
        if kind != "Story":
            reason = kind.lower()
        elif asset.get("isLiveBlog"):
            reason = "live blog"
        elif not attrs.get("isHosted"):
            reason = "not hosted"
        elif m is None:
            reason = "not a yahoo article"
        else:
            reason = None
        if reason:
            dropped.append((reason, published))
            continue
        tickers = ((asset.get("finance") or {}).get("stockTickers")) or []
        items.append({
            "url": m.group(0),
            "title": asset.get("title"),
            "listing_publisher": (attrs.get("provider") or {}).get("displayName"),
            "content_uuid": asset.get("id"),
            "listing_tickers": list(dict.fromkeys(t["symbol"] for t in tickers if t.get("symbol"))),
            "listing_age": None,
            "listing_published": published,
            "premium": bool(asset.get("isPremiumNews")),
            "page": None,
            "position": None,
        })
    info = main.get("pageInfo") or {}
    return items, dropped, {"next": bool(info.get("nextPage")), "oldest": min(times, default=None)}


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


async def _stream_page(client, ycfg: dict, tickers: list[str], start: int, count: int) -> dict:
    """One page of the most-active news stream for the page's tickers (GraphQL POST)."""
    body = {
        "operationName": "GetQSPLeafNewsStream",
        "query": _STREAM_QUERY,
        "variables": {
            "clientContext": {"device": "desktop", "lang": "en-US", "region": "US", "site": "finance"},
            "start": start,
            "count": count,
            "listInput": {
                "assetTypes": ["story", "video"], "disableDedupe": False,
                "enableBlockedContent": False, "filterClientContext": False,
                "queryVariables": {"tickerSymbol": tickers},
                "slug": "list=finance-US-en-US-ticker-all",
            },
        },
    }
    headers = {
        "accept": "application/json", "origin": "https://finance.yahoo.com",
        "referer": ycfg["most_active_url"], "x-yahoo-cg-client-name": "finance",
    }
    _final, text = await http.aget_text(
        client, ycfg["stream_url"], method="POST", json=body, headers=headers
    )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FetchError(f"stream @{start}: not JSON") from exc
    if payload.get("errors") and not payload.get("data"):
        raise FetchError(f"stream @{start}: {payload['errors'][0].get('message')}")
    return payload


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


def default_since(now: datetime) -> datetime:
    """00:00 New York time yesterday, in UTC."""
    yesterday = now.astimezone(_ET).date() - timedelta(days=1)
    return datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=_ET).astimezone(UTC)


def _check_age(row: dict, run_at: datetime, stats: dict, listed_published: datetime | None = None) -> None:
    """Cross-check the exact time against what the listing said: the most-active stream's
    publish time, or the topic page's relative age. When the page has no exact time, fall
    back to the stream time, else the middle of the age's range (approximate)."""
    published = row.get("published_at")
    if listed_published is not None:
        if published is None:
            row["published_at"], row["timestamp_source"] = listed_published, "stream"
        elif abs((published - listed_published).total_seconds()) > 120:
            stats["age_mismatch"].append(
                f"{row['url']}: stream says {listed_published:%Y-%m-%d %H:%M}Z but page "
                f"{published:%Y-%m-%d %H:%M}Z"
            )
        return
    bounds = age_bounds(row.get("listing_age"))
    if bounds is None:
        return
    lo, hi = bounds
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


async def _list_topic(client, ycfg, limiter, pages, raw_day, stamp, stats) -> list[dict]:
    """The stock-market-news topic pages, in order; past the last page Yahoo redirects to page 1."""
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
            await asyncio.to_thread(_write_raw, raw_day / f"listing-{stamp}-p{n}.html.gz", html)
        found = parse_listing(html, n)
        for item in found:
            listed.setdefault(item["url"], item)
        _progress(f"page {n}: {len(found)} article(s), {len(listed)} unique so far")
    return list(listed.values())


async def _list_most_active(client, ycfg, limiter, since, raw_day, stamp, stats) -> list[dict]:
    """Stories in the most-active page's news column since `since`.

    The column is one stream across the page's tickers, filled by a GraphQL query as it
    scrolls; this pages through the same query. Yahoo stops it at `stream_max` items (a few
    hours on a busy day): stats["capped"] is then the oldest time reached.
    """
    await limiter.wait(SITE)
    try:
        _final, html = await _fetch(client, ycfg["most_active_url"])
    except FetchError as exc:
        stats["errors"].append(f"most-active page: {exc}")
        return []
    if raw_day is not None:
        await asyncio.to_thread(_write_raw, raw_day / f"most-active-{stamp}.html.gz", html)
    tickers = parse_most_active_tickers(html)
    if not tickers:
        stats["errors"].append("most-active page: no tickers in the screener table")
        return []
    _progress(f"most-active: {len(tickers)} tickers: {' '.join(tickers)}")

    host = urlparse(ycfg["stream_url"]).netloc
    found: list[dict] = []
    start, oldest = 0, None
    while start < ycfg["stream_max"]:
        await limiter.wait(host)
        try:
            payload = await _stream_page(client, ycfg, tickers, start, ycfg["stream_count"])
        except FetchError as exc:
            stats["errors"].append(f"most-active {exc}")
            break
        if raw_day is not None:
            await asyncio.to_thread(
                _write_raw, raw_day / f"stream-{stamp}-{start}.json.gz", json.dumps(payload)
            )
        items, dropped, page = parse_stream(payload)
        for item in items:
            if item["listing_published"] is None or item["listing_published"] >= since:
                found.append(item)
        for reason, published in dropped:
            if published is None or published >= since:
                stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
        start += ycfg["stream_count"]
        oldest = min(filter(None, (oldest, page["oldest"])), default=None)  # not strictly sorted
        _progress(f"most-active stream @{start}: {len(found)} stor(ies) since {since:%Y-%m-%d %H:%M}Z"
                  + (f", back to {oldest:%Y-%m-%d %H:%M}Z" if oldest else ""))
        if not page["next"] or (oldest is not None and oldest < since):
            break
    # Yahoo's cap reached before the cutoff (it may or may not still say there is a next
    # page): stories older than `oldest` are missed
    if start >= ycfg["stream_max"] and oldest is not None and oldest >= since:
        stats["capped"] = oldest
    return found


REPORT_FATES = ("stored", "new", "duplicate", "failed")


def run_report(listed: list[dict], sources) -> dict[str, dict[str, int]]:
    """The run's final tally per listing page, plus the urls both pages listed and the
    unique total: listed = stored (by an earlier run) + new + duplicate (the same story
    under another url or syndicated copy, dropped) + failed."""
    groups = {src: [it for it in listed if src in it["listed_by"]] for src in sources}
    if len(sources) > 1:
        groups["on both pages"] = [it for it in listed if len(it["listed_by"]) > 1]
    groups["unique"] = listed
    return {name: {"listed": len(items), **{f: sum(it["fate"] == f for it in items) for f in REPORT_FATES}}
            for name, items in groups.items()}


async def _collect(ycfg, sources, pages, since, seen, run_at, raw_day, stats) -> list[dict]:
    limiter = _DomainLimiter(ycfg["per_domain_rps"])
    sem = asyncio.Semaphore(ycfg["max_concurrency"])
    stamp = run_at.strftime("%Y%m%dT%H%M%SZ")
    headers = {"user-agent": ycfg["user_agent"], "accept-language": "en-US,en;q=0.9"}

    async with http.get_async_client(headers=headers, follow_redirects=True) as client:
        # One entry per url, remembering every page that listed it (first sighting wins
        # for the listing details; a later source fills what the first lacked).
        listed: dict[str, dict] = {}

        def add(items: list[dict], source: str) -> None:
            for item in items:
                cur = listed.setdefault(item["url"], {**item, "sources": [], "listed_by": []})
                if source not in cur["sources"]:
                    cur["sources"].append(source)
                    cur["listed_by"].append(source)
                for key in ("content_uuid", "listing_published", "premium"):
                    cur[key] = cur.get(key) or item.get(key)

        if TOPIC in sources:
            add(await _list_topic(client, ycfg, limiter, pages, raw_day, stamp, stats), TOPIC)
        if MOST_ACTIVE in sources:
            add(await _list_most_active(client, ycfg, limiter, since, raw_day, stamp, stats), MOST_ACTIVE)
        stats["listed"] = len(listed)
        stats["by_source"] = {src: sum(src in it["sources"] for it in listed.values()) for src in sources}

        # Already-parsed articles are never fetched again: same url (article_id) or same
        # Yahoo content id, whether stored by an earlier run or listed twice in this one.
        # A stored one only gets this run's pages added to its listed_on.
        todo = []
        claimed: dict[str, dict] = {}
        for it in listed.values():
            uuid = it["content_uuid"]
            aid = article_id(it["url"])
            if aid in seen["ids"] or (uuid and uuid in seen["uuids"]):
                if uuid in claimed:  # same story under a second url in this run
                    claimed[uuid]["sources"] = list(dict.fromkeys(claimed[uuid]["sources"] + it["sources"]))
                    it["fate"] = "duplicate"
                else:
                    stats["relisted"].append((aid, uuid, it["sources"], it["listing_published"]))
                    it["fate"] = "stored"
                continue
            it["fate"] = "new"
            if uuid:
                seen["uuids"].add(uuid)
                claimed[uuid] = it
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
                "listed_on": item["sources"], "stream_published_at": item["listing_published"],
                "extract_status": "failed", "body_status": "missing",
            }
            try:
                await limiter.wait(SITE)
                async with sem:
                    _final, html = await _fetch(client, url)
            except FetchError as exc:
                stats["errors"].append(f"{url}: {exc}")
                row["fetched_at"] = datetime.now(UTC)
                item["fate"] = "failed"
                report("failed", row)
                return row
            row["fetched_at"] = datetime.now(UTC)
            parsed = await asyncio.to_thread(parse_article, html, url)
            # The same syndicated story under a second Yahoo url: drop it. No await between
            # the check and the add, so concurrent fetches can't both claim it.
            original = parsed["original_url"]
            if original and original in seen["originals"]:
                stats["duplicates"] += 1
                item["fate"] = "duplicate"
                report("duplicate", {**row, **parsed})
                return None
            if original:
                seen["originals"].add(original)
            # listing values stand in where the article page has nothing
            row.update({k: v for k, v in parsed.items() if v not in (None, [])})
            row["extract_status"] = parsed["extract_status"]
            if item["premium"] and row.get("body_text"):
                row["body_status"] = "paywalled"  # Yahoo marks it premium: a teaser
            _check_age(row, run_at, stats, item["listing_published"])
            if raw_day is not None:
                row["raw_path"] = await asyncio.to_thread(
                    _write_raw, raw_day / f"{_slug(url)}.html.gz", html
                )
            report(row["extract_status"], row)
            return row

        rows = await asyncio.gather(*(fetch_article(it) for it in todo))
        stats["report"] = run_report(list(listed.values()), sources)
        return [r for r in rows if r is not None]


def scrape(
    *,
    sources: tuple[str, ...] | list[str] | None = None,
    pages: int | None = None,
    since: datetime | None = None,
    resume: bool = True,
    keep_raw: bool = True,
    db_path=None,
) -> dict:
    """Scrape the listings of `sources` (default both pages) + every new article into
    news_articles. `since` bounds the most-active stream (default 00:00 New York
    yesterday). Returns a summary."""
    ycfg = config.news_config()["yahoo"]
    sources = tuple(sources) if sources else SOURCES
    if unknown := [s for s in sources if s not in SOURCES]:
        raise ValueError(f"unknown --source {', '.join(unknown)} (choose from {', '.join(SOURCES)})")
    pages = pages if pages is not None else ycfg["pages"]
    if pages < 1:
        raise ValueError("--pages must be >= 1")
    run_at = datetime.now(UTC)
    since = since if since is not None else default_since(run_at)

    raw_day = None
    if keep_raw:
        raw_dir = Path(ycfg["raw_dir"])
        if storage.unmounted_volume(raw_dir):
            raise ConfigError(f"raw_dir {raw_dir} is under /Volumes but the share is not mounted")
        raw_day = raw_dir / f"date={run_at:%Y-%m-%d}"

    # Every parsed article ('ok' or 'no_timestamp') counts as done, paywalled teasers too;
    # only fetch failures and empty bodies are retried while still listed. The DB is not
    # held open during the fetch. --no-resume re-extracts everything (deduped within the run).
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
             "age_mismatch": [], "by_source": {}, "dropped": {}, "capped": None, "relisted": []}
    rows = asyncio.run(_collect(ycfg, sources, pages, since, seen, run_at, raw_day, stats))

    with db.connect(db_path) as con:
        db.bulk_upsert(con, "news_articles", _COLUMNS, "article_id", _UPDATE, rows)
        if stats["relisted"]:
            con.executemany(
                f"UPDATE news_articles SET listed_on = {_MERGE_LISTED.format('listed_on', '?::VARCHAR[]')}, "
                "stream_published_at = coalesce(stream_published_at, ?) "
                "WHERE feed = ? AND (article_id = ? OR content_uuid = ?)",
                [[srcs, when, FEED, aid, uuid] for aid, uuid, srcs, when in stats["relisted"]],
            )
        total = con.execute(
            "SELECT count(*) FROM news_articles WHERE feed = ?", [FEED]
        ).fetchone()[0]

    by_status = {s: sum(r["extract_status"] == s for r in rows)
                 for s in ("ok", "no_timestamp", "failed")}
    by_body = {s: sum(r["body_status"] == s for r in rows) for s in ("paywalled", "short")}
    return {
        "sources": sources,
        "report": stats["report"],
        "since": since,
        "pages": stats["pages"],
        "listed": stats["listed"],
        "by_source": stats["by_source"],
        "dropped": stats["dropped"],
        "capped": stats["capped"],
        "skipped_seen": stats["skipped_seen"],
        "duplicates": stats["duplicates"],
        "fetched": len(rows),
        **by_status,
        **by_body,
        "errors": stats["errors"],
        "age_mismatch": stats["age_mismatch"],
        "total_articles": total,
    }
