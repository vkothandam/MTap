"""News layer: aggregator (two-level, history-aware), matcher (verified attribution),
sentiment (FinBERT injected as a fake), and the deferred daily rollup.

All network + the FinBERT model are monkeypatched, so nothing here touches the wire or
loads torch. The symbol universe is seeded in-process (via an explicit `matcher=` or a tmp
E*TRADE DuckDB) rather than read from the real fundamentals store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from sourcing_py.news import aggregator, daily, db, sentiment
from sourcing_py.news.matcher import Matcher

_FIXTURES = Path(__file__).parent / "fixtures" / "news"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text()


_UNIVERSE = [
    {"symbol": "AAPL", "company_name": "Apple Inc."},
    {"symbol": "MANE", "company_name": "Mannatech Incorporated"},
    {"symbol": "MSFT", "company_name": "Microsoft Corporation"},
]


# -- matcher (the precision-critical part) --------------------------------------
def test_matcher_name_only_matches():
    m = Matcher(_UNIVERSE)
    hits = m.match("Apple Inc. announced a new product line today.", feed_symbol="AAPL")
    assert hits == [{"symbol": "AAPL", "match_method": "name"}]


def test_matcher_ticker_plus_name_is_strongest():
    m = Matcher(_UNIVERSE)
    hits = m.match("Apple shares rose. Apple Inc. (AAPL) reported earnings.", feed_symbol="AAPL")
    assert hits == [{"symbol": "AAPL", "match_method": "ticker+name"}]


def test_matcher_cashtag_without_name():
    m = Matcher(_UNIVERSE)
    hits = m.match("Traders piled into $AAPL after the open.", feed_symbol=None)
    assert hits == [{"symbol": "AAPL", "match_method": "cashtag"}]


def test_matcher_discards_bagmane_noise_for_mane():
    """A Google search for MANE returning a Bagmane article must be dropped: the substring
    'Bagmane' is not a word-boundary MANE token and the company name is absent."""
    m = Matcher(_UNIVERSE)
    hits = m.match("Bagmane Developers expands Bangalore campus.", feed_symbol="MANE")
    assert hits == []


def test_matcher_bare_ticker_token_without_name_rejected():
    m = Matcher(_UNIVERSE)
    # 'MANE' appears as a bare uppercase token but nothing corroborates it (no name/cashtag).
    hits = m.match("The perfume house MANE opened a plant.", feed_symbol="MANE")
    assert hits == []


# -- aggregator (two-level, history-aware) --------------------------------------
def _news_cfg(**over):
    cfg = {
        "db_path": Path("unused.duckdb"),
        "requests_per_sec": 1000.0,  # no real pacing in tests
        "per_domain_rps": 1000.0,
        "max_concurrency": 8,
        "finbert_model": "fake/finbert",
        "lookback_days": 30,
        "request_timeout": 30.0,
        "feeds": [{
            "name": "yahoo",
            "url_template": "https://feeds.example.com/rss?s={symbol}",
            "tag": "symbol",
            "fetch_article": True,
        }],
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def _isolate_state(tmp_path, monkeypatch):
    # keep ratelimit lock + failures log inside tmp, never the real state/ dir
    monkeypatch.setattr("sourcing_py.common.ratelimit.config.repo_root", lambda: tmp_path)
    monkeypatch.setattr("sourcing_py.common.failures.config.repo_root", lambda: tmp_path)


def test_aggregate_fetches_extracts_and_dedups(tmp_path, monkeypatch, _isolate_state):
    async def fake_feed(client, url):
        return _fixture("yahoo_aapl.xml")

    async def fake_article(client, url):
        # the redirect resolves to the publisher; extraction yields the body
        return "https://www.reuters.com/tech/apple", _fixture("article_apple.html")

    monkeypatch.setattr(aggregator.config, "news_config", _news_cfg)
    monkeypatch.setattr(aggregator, "_fetch_feed", fake_feed)
    monkeypatch.setattr(aggregator, "_fetch_article", fake_article)
    monkeypatch.setattr(aggregator, "extract_article", lambda html: "Apple Inc. unveiled its latest iPhone.")
    dbp = tmp_path / "news.duckdb"

    summary = aggregator.aggregate(symbols=["AAPL"], days=0, db_path=dbp)  # days=0 -> no date cutoff
    assert summary["new_articles"] == 2 and summary["skipped_seen"] == 0

    with db.connect(dbp) as con:
        rows = con.execute(
            "SELECT feed, feed_symbol, site, url, body_text, extract_status FROM news_articles "
            "ORDER BY title"
        ).fetchall()
    feed, feed_symbol, site, url, body, status = rows[0]  # 'Apple unveils...' sorts first
    assert feed == "yahoo" and feed_symbol == "AAPL"
    assert site == "reuters.com" and url == "https://www.reuters.com/tech/apple"
    assert body == "Apple Inc. unveiled its latest iPhone." and status == "ok"

    # re-run is idempotent: both articles are already seen, nothing new
    summary2 = aggregator.aggregate(symbols=["AAPL"], days=0, db_path=dbp)
    assert summary2["new_articles"] == 0 and summary2["skipped_seen"] == 2
    assert summary2["total_articles"] == 2


def test_aggregate_marks_failed_extraction(tmp_path, monkeypatch, _isolate_state):
    from sourcing_py.common.errors import FetchError

    async def fake_feed(client, url):
        return _fixture("yahoo_aapl.xml")

    async def boom(client, url):
        raise FetchError("GET failed (HTTP 404)", status_code=404)

    monkeypatch.setattr(aggregator.config, "news_config", _news_cfg)
    monkeypatch.setattr(aggregator, "_fetch_feed", fake_feed)
    monkeypatch.setattr(aggregator, "_fetch_article", boom)
    dbp = tmp_path / "news.duckdb"
    aggregator.aggregate(symbols=["AAPL"], days=0, db_path=dbp)
    with db.connect(dbp) as con:
        statuses = {r[0] for r in con.execute("SELECT DISTINCT extract_status FROM news_articles").fetchall()}
    # a dead link keeps the RSS title/summary but is marked failed (not retried each run)
    assert statuses == {"failed"}


# -- sentiment (FinBERT injected) + verified attribution ------------------------
def _seed_article(con, article_id, title, summary, body, feed_symbol="AAPL"):
    con.execute(
        "INSERT INTO news_articles (article_id, feed, feed_symbol, title, summary, body_text, "
        "published_at, extract_status, fetched_at) VALUES (?, 'yahoo', ?, ?, ?, ?, ?, 'ok', now())",
        [article_id, feed_symbol, title, summary, body, datetime(2026, 8, 18, tzinfo=timezone.utc)],
    )


def _fake_scorer(texts):
    # deterministic: 'earnings beat' -> positive; else neutral. Probs sum to 1.
    out = []
    for t in texts:
        if "beat" in t.lower():
            out.append({"label": "positive", "score_pos": 0.8, "score_neg": 0.1,
                        "score_neu": 0.1, "score_agg": 0.7})
        else:
            out.append({"label": "neutral", "score_pos": 0.2, "score_neg": 0.2,
                        "score_neu": 0.6, "score_agg": 0.0})
    return out


def test_analyze_scores_and_attributes(tmp_path, monkeypatch):
    monkeypatch.setattr(sentiment.config, "news_config", _news_cfg)
    dbp = tmp_path / "news.duckdb"
    with db.connect(dbp) as con:
        _seed_article(con, "a1", "Apple earnings beat", "", "Apple Inc. reported an earnings beat.")
        _seed_article(con, "a2", "Bagmane campus news", "", "Bagmane Developers expanded.", feed_symbol="MANE")

    summary = sentiment.analyze(
        db_path=dbp, scorer=_fake_scorer, matcher=Matcher(_UNIVERSE),
    )
    assert summary["scored"] == 2
    # only a1 corroborates a symbol; a2 (Bagmane / MANE) is discarded as noise
    assert summary["attributions"] == 1 and summary["articles_attributed"] == 1

    with db.connect(dbp) as con:
        sent = dict(con.execute("SELECT article_id, label FROM article_sentiment").fetchall())
        assert sent == {"a1": "positive", "a2": "neutral"}
        agg = con.execute("SELECT score_agg, model FROM article_sentiment WHERE article_id='a1'").fetchone()
        assert abs(agg[0] - 0.7) < 1e-9 and agg[1] == "fake/finbert"
        syms = con.execute("SELECT article_id, symbol, match_method FROM article_symbols").fetchall()
        assert syms == [("a1", "AAPL", "name")]

    # resume: both articles already scored -> nothing reprocessed
    summary2 = sentiment.analyze(db_path=dbp, scorer=_fake_scorer, matcher=Matcher(_UNIVERSE))
    assert summary2["scored"] == 0 and summary2["attributions"] == 0


# -- daily rollup (segment 3, deferred) -----------------------------------------
def test_build_daily_aggregates_by_symbol_and_date(tmp_path):
    dbp = tmp_path / "news.duckdb"
    with db.connect(dbp) as con:
        for aid, day in [("a1", 18), ("a2", 18), ("a3", 19)]:
            con.execute(
                "INSERT INTO news_articles (article_id, published_at) VALUES (?, ?)",
                [aid, datetime(2026, 8, day, tzinfo=timezone.utc)],
            )
            con.execute("INSERT INTO article_symbols (article_id, symbol) VALUES (?, 'AAPL')", [aid])
        con.executemany(
            "INSERT INTO article_sentiment (article_id, label, score_agg) VALUES (?, ?, ?)",
            [("a1", "positive", 0.6), ("a2", "negative", -0.2), ("a3", "positive", 0.4)],
        )

    daily.build_daily(db_path=dbp)
    with db.connect(dbp) as con:
        rows = con.execute(
            "SELECT date, n_articles, mean_score_agg, pos_count, neg_count "
            "FROM symbol_sentiment_daily WHERE symbol='AAPL' ORDER BY date"
        ).fetchall()
    d18, d19 = rows
    assert d18[1] == 2 and abs(d18[2] - 0.2) < 1e-9 and d18[3] == 1 and d18[4] == 1
    assert d19[1] == 1 and abs(d19[2] - 0.4) < 1e-9 and d19[3] == 1

    # idempotent recompute: re-running doesn't duplicate the (symbol, date) rows
    daily.build_daily(db_path=dbp)
    with db.connect(dbp) as con:
        n = con.execute("SELECT count(*) FROM symbol_sentiment_daily").fetchone()[0]
    assert n == 2
