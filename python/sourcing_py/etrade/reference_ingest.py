"""Load the economy and shares files into the fundamentals DuckDB (`etrade ingest-reference`).

Reads (all under the out root, written by the massive_* sources):
    treasury_yields/treasury_yields-*.jsonl   every fetch is a vintage -> `treasury_yields`
                                              (the latest vintage of each date wins)
    inflation/inflation-*.jsonl               every fetch is a vintage -> `inflation_vintages`
                                              (all) and `inflation` (the FIRST vintage that
                                              carries each month: its first release, never
                                              the later revisions)
    shares/shares-<snap_date>.jsonl           quarter-end snapshots -> `share_snapshots`

Tables are created if missing and upserted, so re-running is idempotent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..common import config

YIELD_COLS = (
    "yield_1_month", "yield_3_month", "yield_6_month", "yield_1_year", "yield_2_year",
    "yield_3_year", "yield_5_year", "yield_7_year", "yield_10_year", "yield_20_year",
    "yield_30_year",
)
INFLATION_COLS = ("cpi", "cpi_core", "cpi_year_over_year", "pce", "pce_core", "pce_spending")
SHARE_COLS = ("market_cap", "weighted_shares_outstanding", "share_class_shares_outstanding", "close_at_snap")

_DDL = f"""
CREATE TABLE IF NOT EXISTS treasury_yields (
    date DATE PRIMARY KEY,
    {', '.join(f'{c} DOUBLE' for c in YIELD_COLS)},
    fetched_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS inflation (
    date DATE PRIMARY KEY,   -- first of the reference month; the first release
    {', '.join(f'{c} DOUBLE' for c in INFLATION_COLS)},
    fetched_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS inflation_vintages (
    date DATE,
    {', '.join(f'{c} DOUBLE' for c in INFLATION_COLS)},
    fetched_at TIMESTAMP,
    PRIMARY KEY (date, fetched_at)
);
CREATE TABLE IF NOT EXISTS share_snapshots (
    symbol VARCHAR,
    snap_date DATE,
    {', '.join(f'{c} DOUBLE' for c in SHARE_COLS)},
    PRIMARY KEY (symbol, snap_date)
);
"""


def _read(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _fetched_at(path: Path, rec: dict | None = None) -> datetime:
    """When the vintage was fetched: the record's _ingested_at, else the file's mtime (UTC, naive)."""
    raw = (rec or {}).get("_ingested_at")
    if raw:
        return datetime.fromisoformat(raw).astimezone(timezone.utc).replace(tzinfo=None)
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)


def _files(directory: Path, prefix: str) -> list[Path]:
    # file names carry the run stamp (UTC), so name order is fetch order
    return sorted(directory.glob(f"{prefix}-*.jsonl"))


def _upsert(con, table: str, cols: list[str], keys: list[str], rows: list[tuple]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in keys)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates}")
    con.executemany(sql, rows)


def ingest_reference(con, out_root: Path | None = None) -> dict:
    """Create the tables if missing and upsert everything found. Returns row counts."""
    root = out_root or config.out_root()
    con.execute(_DDL)

    # --- treasury yields: oldest file first, so the latest vintage overwrites ---
    latest: dict[str, tuple] = {}
    for path in _files(root / "treasury_yields", "treasury_yields"):
        for rec in _read(path):
            if rec.get("date"):
                latest[rec["date"]] = (rec["date"], *(rec.get(c) for c in YIELD_COLS), _fetched_at(path, rec))
    _upsert(con, "treasury_yields", ["date", *YIELD_COLS, "fetched_at"], ["date"], list(latest.values()))

    # --- inflation: all vintages, and the first vintage carrying each month ---
    vintages: dict[tuple, tuple] = {}
    first: dict[str, tuple] = {}
    for path in _files(root / "inflation", "inflation"):
        for rec in _read(path):
            if not rec.get("date"):
                continue
            fetched = _fetched_at(path, rec)
            row = (rec["date"], *(rec.get(c) for c in INFLATION_COLS), fetched)
            vintages[(rec["date"], fetched)] = row
            # a month counts once it has a CPI value (the API lists the month before CPI is out)
            if rec["date"] not in first and rec.get("cpi") is not None:
                first[rec["date"]] = row
    icols = ["date", *INFLATION_COLS, "fetched_at"]
    _upsert(con, "inflation_vintages", icols, ["date", "fetched_at"], list(vintages.values()))
    # first release wins: never overwrite an existing month, only add new ones
    existing = {str(r[0]) for r in con.execute("SELECT date FROM inflation").fetchall()}
    _upsert(con, "inflation", icols, ["date"], [r for d, r in first.items() if d not in existing])

    # --- share snapshots ---
    snaps: dict[tuple, tuple] = {}
    for path in _files(root / "shares", "shares"):
        for rec in _read(path):
            if rec.get("symbol") and rec.get("snap_date"):
                snaps[(rec["symbol"], rec["snap_date"])] = (
                    rec["symbol"], rec["snap_date"], *(rec.get(c) for c in SHARE_COLS))
    _upsert(con, "share_snapshots", ["symbol", "snap_date", *SHARE_COLS], ["symbol", "snap_date"],
            list(snaps.values()))

    return {
        "treasury_yields": len(latest),
        "inflation_months": len(first),
        "inflation_vintages": len(vintages),
        "share_snapshots": len(snaps),
    }
