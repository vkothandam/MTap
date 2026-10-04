"""Segment 3 (deferred / on-demand) — roll per-article sentiment up to per-symbol, per-date.

One set-based SQL statement joins the verified attributions (article_symbols) to the
per-article FinBERT scores (article_sentiment) and the article dates (news_articles),
grouping by (symbol, published date) into symbol_sentiment_daily. Not part of the routine
pipeline yet — run it explicitly when the daily feature is needed for the TFT model.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import db


def build_daily(*, fromdate: str | None = None, todate: str | None = None, db_path=None) -> dict:
    """Recompute symbol_sentiment_daily over the optional [fromdate, todate] date window."""
    now = datetime.now(timezone.utc)
    where = ["a.published_at IS NOT NULL"]
    args: list = []
    if fromdate:
        where.append("a.published_at::date >= ?")
        args.append(fromdate)
    if todate:
        where.append("a.published_at::date <= ?")
        args.append(todate)
    where_sql = " AND ".join(where)

    with db.connect(db_path) as con:
        con.execute(
            f"""
            INSERT INTO symbol_sentiment_daily
                (symbol, date, n_articles, mean_score_agg, pos_count, neg_count, neu_count, updated_at)
            SELECT
                s.symbol,
                a.published_at::date AS date,
                count(*) AS n_articles,
                avg(t.score_agg) AS mean_score_agg,
                count(*) FILTER (WHERE t.label = 'positive') AS pos_count,
                count(*) FILTER (WHERE t.label = 'negative') AS neg_count,
                count(*) FILTER (WHERE t.label = 'neutral')  AS neu_count,
                ? AS updated_at
            FROM article_symbols s
            JOIN article_sentiment t ON s.article_id = t.article_id
            JOIN news_articles a ON s.article_id = a.article_id
            WHERE {where_sql}
            GROUP BY s.symbol, a.published_at::date
            ON CONFLICT (symbol, date) DO UPDATE SET
                n_articles = excluded.n_articles,
                mean_score_agg = excluded.mean_score_agg,
                pos_count = excluded.pos_count,
                neg_count = excluded.neg_count,
                neu_count = excluded.neu_count,
                updated_at = excluded.updated_at
            """,
            [now, *args],
        )
        rows = con.execute("SELECT count(*) FROM symbol_sentiment_daily").fetchone()[0]

    return {"rows_total": rows}
