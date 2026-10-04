"""Load settings (secrets + destination) and the shared source registry.

Layering, highest priority first:
  1. environment variables (OUT_ROOT, MASSIVE_API_KEY, ...)
  2. shared/config/settings.toml   (gitignored — real secrets live here)
  3. built-in defaults
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .errors import ConfigError

# python/sourcing_py/common/config.py -> repo root is three parents up from python/
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY = _REPO_ROOT / "shared" / "config" / "sources.json"
_SETTINGS = _REPO_ROOT / "shared" / "config" / "settings.toml"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    runtime: str
    format: str  # "jsonl" | "parquet"
    dataset: str  # output folder / info-type label, e.g. "daily_summary"
    schema_path: Path
    description: str


def repo_root() -> Path:
    return _REPO_ROOT


@lru_cache(maxsize=1)
def settings() -> dict:
    """Parsed settings.toml, or {} if it doesn't exist yet."""
    if _SETTINGS.exists():
        return tomllib.loads(_SETTINGS.read_text())
    return {}


def out_root() -> Path:
    """Destination base dir. env OUT_ROOT > [destination].out_root > ./out."""
    env = os.environ.get("OUT_ROOT")
    raw = env or settings().get("destination", {}).get("out_root")
    if not raw:
        return _REPO_ROOT / "out"
    path = Path(raw)
    return path if path.is_absolute() else _REPO_ROOT / path


def massive_config() -> dict:
    """Massive.com base_url + api_key, with env overrides. Raises if key is missing."""
    section = settings().get("massive", {})
    base_url = os.environ.get("MASSIVE_BASE_URL") or section.get("base_url") or "https://api.massive.com"
    api_key = os.environ.get("MASSIVE_API_KEY") or section.get("api_key")
    if not api_key or api_key == "REPLACE_WITH_YOUR_KEY":
        raise ConfigError(
            "Massive API key not configured. Set [massive].api_key in "
            "shared/config/settings.toml or the MASSIVE_API_KEY env var."
        )
    rpm_raw = os.environ.get("MASSIVE_RATE_LIMIT_PER_MIN") or section.get("rate_limit_per_min") or 5
    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "rate_limit_per_min": int(rpm_raw),
    }


def etrade_config() -> dict:
    """E*TRADE wsod fundamentals API config, with env overrides.

    Auth (cookies + stk1/stk2) is NOT here — it comes from the browser session file
    written by the Node E*TRADE service (see sourcing_py/etrade/client.py).
    """
    section = settings().get("etrade", {})
    base_url = (
        os.environ.get("ETRADE_BASE_URL")
        or section.get("base_url")
        or "https://etrade.api.wsod.com/etrade-api/1.0"
    )
    stocks_per_sec = os.environ.get("ETRADE_STOCKS_PER_SEC") or section.get("stocks_per_sec") or 2
    # Concurrent fetch workers. The rate-limit throttle still caps aggregate request rate
    # (stocks_per_sec * calls/stock), so workers just supply enough parallelism to reach
    # that ceiling despite per-request network latency. Default 10; env/settings override.
    workers = os.environ.get("ETRADE_WORKERS") or section.get("workers") or 10
    db_raw = (
        os.environ.get("ETRADE_DB_PATH")
        or section.get("db_path")
        or str(_REPO_ROOT / "state" / "etrade" / "fundamentals.duckdb")
    )
    db_path = Path(db_raw)
    if not db_path.is_absolute():
        db_path = _REPO_ROOT / db_path
    session_raw = os.environ.get("SESSION_PATH") or section.get("session_path")
    session_path = Path(session_raw) if session_raw else _REPO_ROOT / "state" / "etrade" / "session.json"
    if not session_path.is_absolute():
        session_path = _REPO_ROOT / session_path
    # Sibling MBin repo's SQLite trading DB — the source of the sector/industry mapping
    # (Industries + Symbol_Industry tables). Default assumes MBin/ next to this repo.
    trading_raw = (
        os.environ.get("MBIN_TRADING_DB")
        or section.get("trading_db_path")
        or str(_REPO_ROOT.parent / "MBin" / "data" / "db" / "trading.db")
    )
    trading_db_path = Path(trading_raw)
    if not trading_db_path.is_absolute():
        trading_db_path = _REPO_ROOT / trading_db_path
    return {
        "base_url": base_url.rstrip("/"),
        "stocks_per_sec": float(stocks_per_sec),
        "workers": int(workers),
        "db_path": db_path,
        "session_path": session_path,
        "trading_db_path": trading_db_path,
    }


def news_config() -> dict:
    """News processing layer config (sourcing_py/news), with env overrides.

    Mirrors etrade_config()'s cascade (env > [news] section > default; relative paths
    resolved under the repo root). `feeds` is a list of feed descriptors, each:
        {name, url_template, tag, fetch_article}
    where url_template has a `{symbol}` placeholder and fetch_article marks feeds whose
    RSS entries only carry a headline + link (so the article body needs a second fetch).
    The symbol universe + company names are read from the E*TRADE DuckDB (etrade_config's
    db_path, ATTACHed read-only), so no separate symbol store is configured here.
    """
    section = settings().get("news", {})
    db_raw = (
        os.environ.get("NEWS_DB_PATH")
        or section.get("db_path")
        or str(_REPO_ROOT / "state" / "news" / "news.duckdb")
    )
    db_path = Path(db_raw)
    if not db_path.is_absolute():
        db_path = _REPO_ROOT / db_path
    requests_per_sec = (
        os.environ.get("NEWS_REQUESTS_PER_SEC") or section.get("requests_per_sec") or 2
    )
    # Concurrent fetching (aggregator): starts are spaced to at most per_domain_rps per
    # domain (polite per host, but many hosts run in parallel), capped by a global
    # in-flight ceiling of max_concurrency.
    per_domain_rps = (
        os.environ.get("NEWS_PER_DOMAIN_RPS") or section.get("per_domain_rps") or 10
    )
    max_concurrency = (
        os.environ.get("NEWS_MAX_CONCURRENCY") or section.get("max_concurrency") or 20
    )
    finbert_model = (
        os.environ.get("NEWS_FINBERT_MODEL") or section.get("finbert_model") or "ProsusAI/finbert"
    )
    lookback_days = os.environ.get("NEWS_LOOKBACK_DAYS") or section.get("lookback_days") or 30
    request_timeout = section.get("request_timeout") or 30
    # Built-in default feeds (Yahoo Finance + Google News, both needing a deep body fetch)
    # so the layer works before settings.toml is filled in; a [news.feeds] section overrides.
    feeds = section.get("feeds") or [
        {
            "name": "yahoo",
            "url_template": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            "tag": "symbol",
            "fetch_article": True,
        },
        {
            "name": "google",
            "url_template": "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
            "tag": "symbol",
            "fetch_article": True,
        },
    ]
    return {
        "db_path": db_path,
        "requests_per_sec": float(requests_per_sec),
        "per_domain_rps": float(per_domain_rps),
        "max_concurrency": int(max_concurrency),
        "finbert_model": finbert_model,
        "lookback_days": int(lookback_days),
        "request_timeout": float(request_timeout),
        "feeds": feeds,
    }


def load_source(name: str) -> SourceConfig:
    registry = json.loads(_REGISTRY.read_text())
    sources = registry.get("sources", {})
    if name not in sources:
        available = ", ".join(sorted(sources)) or "(none)"
        raise KeyError(f"Unknown source {name!r}. Registered: {available}")
    entry = sources[name]
    return SourceConfig(
        name=name,
        runtime=entry["runtime"],
        format=entry["format"],
        dataset=entry.get("dataset", name),
        schema_path=_REPO_ROOT / entry["schema"],
        description=entry.get("description", ""),
    )
