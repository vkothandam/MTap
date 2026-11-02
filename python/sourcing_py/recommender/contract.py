"""Contract helpers: vendored schemas, their hashes, validation, ids and timestamps.

Protocol references (§n) are to `../MBin/docs/contracts/mtap-mbin-protocol.md`.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from ..common import config

PROTOCOL_VERSION = "1.0-draft"
PARTY = "mtap"
PRODUCER = "mtap/recommender"

# What this component emits and reads in phase 1 (§7, REVIEW M10). POSITIONS_SNAPSHOT and
# EXIT_RECOMMENDATION join in phase 2.
PRODUCES = ("RECOMMENDATION", "CONTRACT_STATUS", "RESYNC_REQUEST")
CONSUMES = ("TRADING_PARAMETERS", "RECOMMENDATION_OUTCOME", "CONTRACT_STATUS", "RESYNC_REQUEST")

# sha256 of each vendored schema file, as MBin's tools/validate.py prints it (§6.3). A
# change here means MBin changed a schema: re-copy the bytes, re-check the code, update.
PINNED = {
    "contract_status.v1": "3756924807e00085a576d5a9c3e82b7615ba49e87f4db5ff7040f2e715af7b51",
    "exit_recommendation.v1": "c85f3d691a4e69167aa9de712c69e1669b4d72441f770b4cc1519e7aa849f34c",
    "positions_snapshot.v1": "a035fbfa92d06a0bbb7c9e987aa9686349eb25a30de27720353c1b012f09120a",
    "recommendation.v1": "dc25da271843e95b97adb0cef6e9c94446649198750d2487ec7e97857a8f5d7d",
    "recommendation_outcome.v1": "78f9a6a567ed504e8183c3ef5ff463ba37f9e045230ff065633fac7c300c8451",
    "resync_request.v1": "4fe412271e2a495b465c5251be7f85225c97339d88f28443764e9d368e3fe529",
    "trading_parameters.v1": "7b8c0905c8cbcb509919a390891001c78210bda61da6e18fd296fb9be76508ba",
}


def schemas_dir() -> Path:
    return config.repo_root() / "schemas" / "contracts"


def mbin_schemas_dir() -> Path:
    """MBin's copy, when the sibling repo is checked out (dev Macs; not the containers)."""
    return config.repo_root().parent / "MBin" / "docs" / "contracts" / "schemas"


def schema_name(msg_type: str, version: int = 1) -> str:
    return f"{msg_type.lower()}.v{version}"


@cache
def schema_bytes(name: str) -> bytes:
    return (schemas_dir() / f"{name}.schema.json").read_bytes()


def schema_sha256(msg_type: str, version: int = 1) -> str:
    return hashlib.sha256(schema_bytes(schema_name(msg_type, version))).hexdigest()


@cache
def _validator(name: str) -> Draft7Validator:
    return Draft7Validator(json.loads(schema_bytes(name)))


def has_schema(msg_type: str, version: int) -> bool:
    return (schemas_dir() / f"{schema_name(msg_type, version)}.schema.json").exists()


def validate(msg: dict[str, Any]) -> str | None:
    """None when `msg` validates against the schema its type + schema_version name, else
    the first error as `path: message` (the CONTRACT_STATUS `detail`, never content)."""
    name = schema_name(str(msg.get("type")), int(msg.get("schema_version") or 0))
    errors = sorted(_validator(name).iter_errors(msg), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    err = errors[0]
    path = "/".join(str(p) for p in err.absolute_path) or "(root)"
    return f"{path}: {err.message}"[:300]


# -- timestamps -------------------------------------------------------------------------

def ts_utc(dt: datetime | None = None) -> str:
    """ISO-8601 UTC with milliseconds and a Z (§3)."""
    dt = (dt or datetime.now(UTC)).astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)  # 3.11+ reads the trailing Z


def epoch_ms(value: str) -> int:
    return int(parse_ts(value).timestamp() * 1000)


# -- ids (REVIEW item 9) ------------------------------------------------------------------

def rec_id(symbol: str, data_ts: str, strategy_id: str, target_pct: float | None,
           horizon: str | None) -> str:
    """`r-<YYYYMMDD>-<HHMMSS>-<symbol>-<4 hex>`, deterministic for the same decision: the hex
    is sha1(symbol + data_ts_utc + strategy + target + horizon), the time is data_ts_utc."""
    dt = parse_ts(data_ts).astimezone(UTC)
    digest = hashlib.sha1(
        f"{symbol}{data_ts}{strategy_id}{target_pct}{horizon}".encode()
    ).hexdigest()[:4]
    sym = re.sub(r"[^A-Za-z0-9._-]", "-", symbol)  # BRK.B stays; a slash would not fit the id
    return f"r-{dt:%Y%m%d}-{dt:%H%M%S}-{sym}-{digest}"


def rec_key(rid: str) -> str:
    return f"rec:{rid}"


# -- reference data (§8.1) ----------------------------------------------------------------

CAP_BANDS = ((200e9, "mega"), (10e9, "large"), (2e9, "mid"), (250e6, "small"), (0.0, "micro"))


def cap_band(market_cap: float | None) -> str | None:
    """mega ≥ $200B, large $10–200B, mid $2–10B, small $250M–2B, micro < $250M."""
    if market_cap is None:
        return None
    return next(band for floor, band in CAP_BANDS if market_cap >= floor)
