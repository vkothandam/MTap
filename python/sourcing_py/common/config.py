"""Load settings (secrets + destination) and the shared source registry.

Layering, highest priority first:
  1. environment variables (OUT_ROOT, MASSIVE_API_KEY, ...)
  2. the repo-root .env      (gitignored; fills only variables not already set)
  3. shared/config/settings.toml   (gitignored — real secrets live here)
  4. built-in defaults

Two storage roots (see .env.example):
  MTAP_DATA_ROOT   large write-once parquet (e.g. a NAS mount); out/ lives under it
  MTAP_STATE_ROOT  local small state: DuckDB stores, session, locks, logs, segments
Both default to the repo layout (./ and ./state), so an unconfigured checkout works
exactly as before.
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
_DOTENV = _REPO_ROOT / ".env"
DATA_ROOT_MARKER = ".mtap-data-root"


def _parse_dotenv(text: str) -> dict[str, str]:
    """KEY=VALUE lines; '#' comments, optional `export `, optional surrounding quotes."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def load_dotenv(path: Path = _DOTENV) -> None:
    """Fill os.environ from the repo .env without overriding what is already set.

    Node (common/dotenv.ts) and the schedule scripts read the same file, so one .env
    per machine configures every runtime.
    """
    if path.exists():
        for key, value in _parse_dotenv(path.read_text()).items():
            os.environ.setdefault(key, value)


load_dotenv()


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


def _root(env_var: str, key: str, default: Path) -> Path:
    raw = os.environ.get(env_var) or settings().get("storage", {}).get(key)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else _REPO_ROOT / path


def data_root() -> Path:
    """Large write-once data (NAS). env MTAP_DATA_ROOT > [storage].data_root > repo root."""
    return _root("MTAP_DATA_ROOT", "data_root", _REPO_ROOT)


def state_root() -> Path:
    """Small local state. env MTAP_STATE_ROOT > [storage].state_root > ./state."""
    return _root("MTAP_STATE_ROOT", "state_root", _REPO_ROOT / "state")


def state_path(raw: str | Path) -> Path:
    """Resolve a configured path. Absolute paths stand; a relative path under ./state/
    resolves under state_root(); any other relative path resolves from the repo root."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts[1:] if path.parts[:1] == (".",) else path.parts
    if parts[:1] == ("state",):
        return state_root().joinpath(*parts[1:])
    return _REPO_ROOT / path


def out_root() -> Path:
    """Destination base dir. env OUT_ROOT > [destination].out_root > ./out, where a
    relative path resolves under data_root() (the repo root unless MTAP_DATA_ROOT is set)."""
    raw = os.environ.get("OUT_ROOT") or settings().get("destination", {}).get("out_root") or "out"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else data_root() / path


def storage_config() -> dict:
    """[storage]: the two roots (above) plus NAS and local-buffer settings."""
    section = settings().get("storage", {})
    return {
        "data_root": data_root(),
        "state_root": state_root(),
        "nas_url": os.environ.get("MTAP_NAS_URL") or section.get("nas_url") or "",
        # Unuploaded feed segments beyond this size are an alert (the NAS has been down a while).
        "max_local_segments_gb": float(
            os.environ.get("MTAP_MAX_LOCAL_SEGMENTS_GB") or section.get("max_local_segments_gb") or 20
        ),
    }


def intraday_config() -> dict:
    """[intraday]: which feeds run, and how the storage consumer writes them.

    env > [intraday] > defaults. `feeds` is a comma list in env INTRADAY_FEEDS.
    """
    section = settings().get("intraday", {})
    feeds_env = os.environ.get("INTRADAY_FEEDS")
    feeds = (
        [f.strip() for f in feeds_env.split(",") if f.strip()]
        if feeds_env
        else section.get("feeds") or ["etrade_watchlist"]
    )
    return {
        "feeds": feeds,
        "primary_source": os.environ.get("INTRADAY_PRIMARY_SOURCE")
        or section.get("primary_source")
        or feeds[0],
        "redis_url": os.environ.get("REDIS_URL") or section.get("redis_url") or "redis://127.0.0.1:6379",
        # Rows buffered in memory before a flush to the open segment (each flush is fsynced,
        # then the stream entries are acked).
        "flush_rows": int(os.environ.get("INTRADAY_FLUSH_ROWS") or section.get("flush_rows") or 20000),
        "flush_s": float(os.environ.get("INTRADAY_FLUSH_S") or section.get("flush_s") or 5),
        # An open segment with no new rows for this long is closed and uploaded early
        # (end of day, collector stopped).
        "idle_close_s": float(os.environ.get("INTRADAY_IDLE_CLOSE_S") or section.get("idle_close_s") or 600),
        "upload_retry_s": float(section.get("upload_retry_s") or 300),
    }


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
    db_path = state_path(
        os.environ.get("ETRADE_DB_PATH") or section.get("db_path") or "state/etrade/fundamentals.duckdb"
    )
    session_path = state_path(
        os.environ.get("SESSION_PATH") or section.get("session_path") or "state/etrade/session.json"
    )
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
    db_path = state_path(
        os.environ.get("NEWS_DB_PATH") or section.get("db_path") or "state/news/news.duckdb"
    )
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


def edgar_config() -> dict:
    """SEC EDGAR (sourcing_py/etrade/edgar.py), with env overrides.

    SEC asks every client to send a User-Agent naming the requester and a contact email,
    and to stay under 10 requests/second; it blocks clients that don't.
    """
    section = settings().get("edgar", {})
    user_agent = (
        os.environ.get("EDGAR_USER_AGENT")
        or section.get("user_agent")
        or "MTap vijayan.kothandaraman@gmail.com"
    )
    requests_per_sec = os.environ.get("EDGAR_REQUESTS_PER_SEC") or section.get("requests_per_sec") or 8
    return {
        "user_agent": user_agent,
        "requests_per_sec": min(float(requests_per_sec), 10.0),
        "workers": int(os.environ.get("EDGAR_WORKERS") or section.get("workers") or 4),
        "tickers_url": section.get("tickers_url") or "https://www.sec.gov/files/company_tickers.json",
        "archives_url": (
            section.get("archives_url") or "https://www.sec.gov/Archives/edgar/data"
        ).rstrip("/"),
        "submissions_url": (
            section.get("submissions_url") or "https://data.sec.gov/submissions"
        ).rstrip("/"),
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
