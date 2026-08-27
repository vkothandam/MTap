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
    return {
        "base_url": base_url.rstrip("/"),
        "stocks_per_sec": float(stocks_per_sec),
        "workers": int(workers),
        "db_path": db_path,
        "session_path": session_path,
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
