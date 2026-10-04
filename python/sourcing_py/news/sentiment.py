"""Segment 2 — sentiment analysis + verified symbol attribution.

For each article not yet scored:
  • FinBERT (ProsusAI/finbert, local, CPU) scores the article text, producing positive /
    negative / neutral probabilities, an argmax label, and an aggregate signed score
    (score_pos - score_neg).  -> article_sentiment
  • the verified matcher attributes it to affected symbols, corroborating each candidate
    against the text (see matcher.py).  -> article_symbols

The FinBERT model + tokenizer load lazily as a process-wide singleton (torch/transformers
are heavy). Missing deps raise ConfigError with an install hint. `--resume` scores only
articles lacking a sentiment row, so re-runs don't reprocess.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..common import config
from ..common.errors import ConfigError
from . import db
from .matcher import Matcher

_MODEL_CACHE: dict = {}


def _finbert_scorer(model_id: str):
    """Return a `score(texts) -> list[dict]` callable backed by a cached FinBERT model."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ConfigError(
            "FinBERT needs `transformers` and `torch`. Install the project deps "
            "(`uv sync` in python/) — they are core dependencies."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    def score(texts: list[str]) -> list[dict]:
        with torch.no_grad():
            enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
        rows = []
        for p in probs:
            by = {id2label.get(i, str(i)): p[i] for i in range(len(p))}
            pos, neg, neu = by.get("positive", 0.0), by.get("negative", 0.0), by.get("neutral", 0.0)
            label = max((("positive", pos), ("negative", neg), ("neutral", neu)), key=lambda kv: kv[1])[0]
            rows.append({
                "label": label, "score_pos": pos, "score_neg": neg,
                "score_neu": neu, "score_agg": pos - neg,
            })
        return rows

    _MODEL_CACHE[model_id] = score
    return score


def _match_text(row: dict) -> str:
    parts = [row.get("title") or "", row.get("body_text") or "", row.get("summary") or ""]
    return "\n".join(p for p in parts if p)


def _sentiment_text(row: dict) -> str:
    title = row.get("title") or ""
    body = row.get("body_text") or row.get("summary") or ""
    return f"{title}. {body}".strip()


def analyze(
    *,
    resume: bool = True,
    db_path=None,
    etrade_db_path=None,
    scorer=None,
    matcher: Matcher | None = None,
    batch_size: int = 16,
) -> dict:
    """Score unscored articles with FinBERT and store verified symbol attributions.

    `scorer` / `matcher` can be injected (tests pass a fake scorer so torch isn't loaded);
    otherwise FinBERT and the E*TRADE symbol universe are used.
    """
    cfg = config.news_config()
    now = datetime.now(timezone.utc)

    with db.connect(db_path) as con:
        query = (
            "SELECT a.article_id, a.title, a.summary, a.body_text, a.feed_symbol "
            "FROM news_articles a "
        )
        if resume:
            query += "LEFT JOIN article_sentiment s ON a.article_id = s.article_id WHERE s.article_id IS NULL"
        rows = [
            {"article_id": r[0], "title": r[1], "summary": r[2], "body_text": r[3], "feed_symbol": r[4]}
            for r in con.execute(query).fetchall()
        ]
        if not rows:
            return {"scored": 0, "attributions": 0, "articles_attributed": 0, "pending": 0}

        if scorer is None:
            scorer = _finbert_scorer(cfg["finbert_model"])
        if matcher is None:
            matcher = Matcher(db.load_symbol_universe(etrade_db_path))

        sentiment_rows: list[dict] = []
        symbol_rows: dict[tuple, dict] = {}  # (article_id, symbol) -> row (dedup within batch)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            scores = scorer([_sentiment_text(r) for r in chunk])
            for r, sc in zip(chunk, scores):
                sentiment_rows.append({
                    "article_id": r["article_id"], "label": sc["label"],
                    "score_pos": sc["score_pos"], "score_neg": sc["score_neg"],
                    "score_neu": sc["score_neu"], "score_agg": sc["score_agg"],
                    "model": cfg["finbert_model"], "scored_at": now,
                })
                for m in matcher.match(_match_text(r), feed_symbol=r["feed_symbol"]):
                    symbol_rows[(r["article_id"], m["symbol"])] = {
                        "article_id": r["article_id"], "symbol": m["symbol"],
                        "match_method": m["match_method"], "matched_at": now,
                    }

        db.bulk_upsert(
            con, "article_sentiment",
            ["article_id", "label", "score_pos", "score_neg", "score_neu", "score_agg",
             "model", "scored_at"],
            "article_id",
            "label = excluded.label, score_pos = excluded.score_pos, score_neg = excluded.score_neg, "
            "score_neu = excluded.score_neu, score_agg = excluded.score_agg, "
            "model = excluded.model, scored_at = excluded.scored_at",
            sentiment_rows,
        )
        db.bulk_upsert(
            con, "article_symbols",
            ["article_id", "symbol", "match_method", "matched_at"],
            "article_id, symbol",
            "match_method = excluded.match_method, matched_at = excluded.matched_at",
            list(symbol_rows.values()),
        )
        articles_attributed = len({k[0] for k in symbol_rows})

    return {
        "scored": len(sentiment_rows),
        "attributions": len(symbol_rows),
        "articles_attributed": articles_attributed,
        "pending": 0,
    }
