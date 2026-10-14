"""Yahoo topic scraper: listing + article parsing against saved pages, and the scrape loop
(pagination stop, resume, raw archive) with the network monkeypatched."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sourcing_py.news import db, yahoo

_FIXTURES = Path(__file__).parent / "fixtures" / "news"
# DuckDB hands TIMESTAMP values back naive; the session zone is UTC, so this is UTC.
_STORED = datetime(2026, 10, 11, 14, 0, 2, tzinfo=UTC).replace(tzinfo=None)
_ARTICLE_URL = (
    "https://finance.yahoo.com/markets/stocks/articles/dominos-pizza-releases-earnings-pre-140002271.html"
)


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text()


# -- listing ---------------------------------------------------------------------
def test_listing_scoped_to_main_section_skips_ads():
    items = yahoo.parse_listing(_fixture("yahoo_topic_listing.html"), page=1)
    urls = [it["url"] for it in items]
    assert len(urls) == 25 and len(set(urls)) == 25
    assert all(yahoo.ARTICLE_RE.match(u) for u in urls)
    # the footer's evergreen personal-finance link and ad items are not stories
    assert not any("personal-finance" in u for u in urls)
    assert not any("seekingalpha" in u for u in urls)
    # both the /articles/ and the /article/ forms are accepted
    assert any("/article/" in u for u in urls) and any("/articles/" in u for u in urls)

    first = items[0]
    assert first["url"].endswith("/binance-weighs-stock-access-markets-070000608.html")
    assert first["title"].startswith("Binance weighs stock access")
    assert first["listing_publisher"] == "Fortune"
    assert first["content_uuid"] == "29cb3cdb-780b-3280-8be5-a02f9ece0e80"
    assert first["listing_tickers"] == ["ANTH.PVT"]
    assert (first["page"], first["position"]) == (1, 1)
    assert first["listing_age"] == "21h ago"


def test_age_bounds_floor_to_the_unit():
    assert yahoo.age_bounds("21h ago") == (21 * 3600, 22 * 3600)
    assert yahoo.age_bounds("1d ago") == (86400, 2 * 86400)
    assert yahoo.age_bounds("5m ago") == (300, 360)
    assert yahoo.age_bounds("yesterday") is None and yahoo.age_bounds(None) is None


def test_listing_without_main_section_is_empty():
    assert yahoo.parse_listing("<html><body><p>consent wall</p></body></html>", page=1) == []


# -- article ---------------------------------------------------------------------
def test_article_exact_timestamp_and_metadata():
    a = yahoo.parse_article(_fixture("yahoo_article_barchart.html"), _ARTICLE_URL)
    assert a["published_at"] == datetime(2026, 10, 11, 14, 0, 2, tzinfo=UTC)
    assert a["modified_at"] == datetime(2026, 10, 11, 14, 0, 2, tzinfo=UTC)
    assert a["timestamp_source"] == "ld+json"
    assert a["provider"] == "Barchart"
    assert a["author"] == "Mark R. Hake, CFA"
    assert a["section"] == "Stocks"
    assert a["tickers"] == ["DPZ"]
    assert a["original_url"].startswith("https://www.barchart.com/story/news/6026132/")
    assert a["title"].startswith("Domino's Pizza Releases Earnings")
    assert a["lang"] == "en-US"
    assert a["extract_status"] == "ok" and a["body_status"] == "full"
    body = a["body_text"]
    assert body.startswith("Domino's Pizza (DPZ) will release earnings")
    assert "investors following these plays have made an attractive return" in body
    # scoped to the article body: the byline / read-time header is not in the text
    assert "min read" not in body


def test_article_time_tag_fallback_without_json_ld():
    html = _fixture("yahoo_article_barchart.html")
    start = html.index('<script type="application/ld+json"')
    html = html[:start] + html[html.index("</script>", start) + len("</script>"):]
    a = yahoo.parse_article(html, _ARTICLE_URL)
    assert a["published_at"] == datetime(2026, 10, 11, 14, 0, 2, tzinfo=UTC)
    assert a["timestamp_source"] == "time-tag"
    assert a["modified_at"] is None and a["provider"] is None


def test_article_without_any_timestamp_is_flagged():
    html = "<html><body><div data-testid='article-body'><p>" + "Stocks rose today. " * 60 + "</p></div></body></html>"
    a = yahoo.parse_article(html, _ARTICLE_URL)
    assert a["published_at"] is None and a["timestamp_source"] is None
    assert a["extract_status"] == "no_timestamp"


def test_paywalled_teaser_is_flagged():
    teaser = ("Update: Equities Mixed at Close. " * 20 + "Upgrade to read this MT Newswires article "
              "and get so much more. A Silver or Gold subscription plan is required to access "
              "premium news articles.")
    html = f"<html><body><div data-testid='article-body'><p>{teaser}</p></div></body></html>"
    a = yahoo.parse_article(html, _ARTICLE_URL)
    assert a["body_status"] == "paywalled" and a["extract_status"] == "no_timestamp"


def test_subscriber_only_flag_marks_paywalled():
    html = _fixture("yahoo_article_barchart.html").replace(
        '"datePublished"', '"isAccessibleForFree":false,"datePublished"', 1)
    a = yahoo.parse_article(html, _ARTICLE_URL)
    assert a["body_status"] == "paywalled" and a["extract_status"] == "ok"


def test_body_status_levels():
    from sourcing_py.news.aggregator import body_status
    assert body_status(None) == "missing" and body_status("") == "missing"
    assert body_status("word " * 40) == "short"
    assert body_status("word " * 400) == "full"
    assert body_status("word " * 400, accessible_for_free=False) == "paywalled"


# -- listing age vs exact time ---------------------------------------------------
_RUN = datetime(2026, 10, 12, 11, 30, tzinfo=UTC)


def test_listing_age_agrees_with_exact_time():
    stats = {"age_mismatch": []}
    row = {"url": "u", "listing_age": "21h ago", "published_at": datetime(2026, 10, 11, 14, 0, 2, tzinfo=UTC)}
    yahoo._check_age(row, _RUN, stats)
    assert stats["age_mismatch"] == [] and row["published_at"].second == 2


def test_listing_age_disagreeing_with_exact_time_is_reported():
    stats = {"age_mismatch": []}
    row = {"url": "u", "listing_age": "3h ago", "published_at": datetime(2026, 10, 11, 14, 0, 2, tzinfo=UTC)}
    yahoo._check_age(row, _RUN, stats)
    assert len(stats["age_mismatch"]) == 1 and "listed '3h ago'" in stats["age_mismatch"][0]


def test_listing_age_is_the_approximate_time_when_the_page_has_none():
    row = {"url": "u", "listing_age": "21h ago", "published_at": None, "timestamp_source": None}
    yahoo._check_age(row, _RUN, {"age_mismatch": []})
    assert row["published_at"] == _RUN - timedelta(hours=21.5)
    assert row["timestamp_source"] == "listing-age"


# -- scrape loop -----------------------------------------------------------------
def _yahoo_cfg(tmp_path):
    return {
        "db_path": tmp_path / "news.duckdb",
        "yahoo": {
            "topic_url": "https://finance.yahoo.com/topic/stock-market-news/",
            "pages": 6,
            "user_agent": "test",
            "per_domain_rps": 0,  # no pacing in tests
            "max_concurrency": 4,
            "raw_dir": tmp_path / "raw",
        },
    }


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    """Pages 1-2 serve the saved listing (page 2 repeats page 1's stories), page 3
    redirects to page 1; every article serves the Barchart fixture, each with its own
    original url (`originals` maps an article url to a shared one, to fake syndication)."""
    listing, article = _fixture("yahoo_topic_listing.html"), _fixture("yahoo_article_barchart.html")
    calls: list[str] = []
    originals: dict[str, str] = {}
    barchart = "https://www.barchart.com/story/news/6026132/"

    async def fake_fetch(client, url):
        calls.append(url)
        if "/topic/" in url:
            if url.rstrip("/").endswith(("/3", "/4", "/5", "/6")):
                return "https://finance.yahoo.com/topic/stock-market-news/", listing
            return url, listing
        original = originals.get(url, f"https://www.barchart.com/story/{yahoo._slug(url)}")
        return url, article.replace(barchart, original + "#")

    monkeypatch.setattr(yahoo.config, "news_config", lambda: _yahoo_cfg(tmp_path))
    monkeypatch.setattr(yahoo, "_fetch", fake_fetch)
    fake_fetch.calls, fake_fetch.originals = calls, originals
    return fake_fetch


def test_scrape_stores_articles_stops_paging_and_resumes(tmp_path, fake_site):
    dbp = tmp_path / "news.duckdb"
    summary = yahoo.scrape(db_path=dbp)
    assert summary["pages"] == 2  # page 3 redirected to page 1
    assert summary["listed"] == 25 and summary["fetched"] == 25 and summary["ok"] == 25
    assert summary["total_articles"] == 25 and summary["errors"] == []
    assert sum("/topic/" in u for u in fake_site.calls) == 3

    with db.connect(dbp) as con:
        row = con.execute(
            "SELECT feed, site, published_at, provider, tickers, listing_page, raw_path "
            "FROM news_articles WHERE url LIKE '%binance-weighs%'"
        ).fetchone()
    feed, site, published, provider, tickers, page, raw_path = row
    assert (feed, site, provider, tickers, page) == ("yahoo-topic", "finance.yahoo.com", "Barchart", ["DPZ"], 1)
    # stored as naive UTC, not shifted into the machine's local zone
    assert published == _STORED
    with db.connect(dbp) as con:
        assert con.execute(
            "SELECT listing_age FROM news_articles WHERE url LIKE '%binance-weighs%'"
        ).fetchone()[0] == "21h ago"
    with gzip.open(raw_path, "rt", encoding="utf-8") as fh:
        assert "datePublished" in fh.read()
    assert len(list((tmp_path / "raw").glob("date=*/listing-*-p*.html.gz"))) == 2

    fake_site.calls.clear()
    again = yahoo.scrape(db_path=dbp)
    assert again["fetched"] == 0 and again["skipped_seen"] == 25
    assert not any("/articles/" in u or "/article/" in u for u in fake_site.calls)


def test_scrape_retries_failed_articles_keeping_first_listing(tmp_path, fake_site, monkeypatch):
    from sourcing_py.common.errors import FetchError

    dbp = tmp_path / "news.duckdb"
    good = yahoo._fetch

    async def flaky(client, url):
        if "binance-weighs" in url:
            raise FetchError("GET failed (HTTP 503)", status_code=503)
        return await good(client, url)

    monkeypatch.setattr(yahoo, "_fetch", flaky)
    first = yahoo.scrape(db_path=dbp, keep_raw=False)
    assert first["failed"] == 1 and len(first["errors"]) == 1

    monkeypatch.setattr(yahoo, "_fetch", good)
    second = yahoo.scrape(db_path=dbp, keep_raw=False)
    assert second["fetched"] == 1 and second["ok"] == 1
    with db.connect(dbp) as con:
        status, title, listed = con.execute(
            "SELECT extract_status, title, first_listed_at FROM news_articles "
            "WHERE url LIKE '%binance-weighs%'"
        ).fetchone()
        first_seen = con.execute(
            "SELECT min(first_listed_at) FROM news_articles"
        ).fetchone()[0]
    assert status == "ok" and title.startswith("Domino's")  # the fixture article page
    assert listed == first_seen  # first sighting kept across the retry


def test_bulk_upsert_keeps_utc(tmp_path):
    utc = datetime(2026, 10, 11, 14, 0, 2, tzinfo=UTC)
    with db.connect(tmp_path / "news.duckdb") as con:
        db.bulk_upsert(con, "news_articles", ["article_id", "published_at"], "article_id",
                       "published_at = excluded.published_at", [{"article_id": "x", "published_at": utc}])
        con.execute("INSERT INTO news_articles (article_id, published_at) VALUES ('y', ?)", [utc])
        got = {r[0]: r[1] for r in con.execute("SELECT article_id, published_at FROM news_articles").fetchall()}
    assert got == {"x": _STORED, "y": _STORED}


def test_scrape_never_refetches_a_known_content_uuid(tmp_path, fake_site):
    """A story already stored under another url (same Yahoo content id) is not fetched."""
    dbp = tmp_path / "news.duckdb"
    with db.connect(dbp) as con:
        con.execute(
            "INSERT INTO news_articles (article_id, feed, url, content_uuid, extract_status) "
            "VALUES ('old', 'yahoo-topic', 'https://finance.yahoo.com/x/articles/old-1.html', "
            "'29cb3cdb-780b-3280-8be5-a02f9ece0e80', 'ok')"
        )
    summary = yahoo.scrape(db_path=dbp, keep_raw=False)
    assert summary["skipped_seen"] == 1 and summary["fetched"] == 24
    assert not any("binance-weighs" in u for u in fake_site.calls)


def test_scrape_drops_second_url_of_a_syndicated_story(tmp_path, fake_site):
    """Two Yahoo urls whose pages name the same publisher original: stored once, and a
    later run that lists it under yet another url doesn't store it either."""
    dbp = tmp_path / "news.duckdb"
    items = yahoo.parse_listing(_fixture("yahoo_topic_listing.html"), page=1)
    a, b = items[1]["url"], items[2]["url"]
    fake_site.originals.update({a: "https://www.reuters.com/same-story", b: "https://www.reuters.com/same-story"})
    summary = yahoo.scrape(db_path=dbp, keep_raw=False)
    assert summary["duplicates"] == 1 and summary["fetched"] == 24
    with db.connect(dbp) as con:
        n = con.execute(
            "SELECT count(*) FROM news_articles WHERE original_url LIKE 'https://www.reuters.com/same-story#%'"
        ).fetchone()[0]
    assert n == 1
