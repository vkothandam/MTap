"""Populate the sector/industry dimensions and each symbol's industry from MBin.

The sector + industry classification (TRBC: 11 sectors, 59 industries) already lives in
the sibling MBin repo's SQLite trading DB (default ../MBin/data/db/trading.db), in two
tables:

  Industries(Id, Sector_Code, Sector_Name, Industry_Code, Industry_Name, Source_Url, ...)
  Symbol_Industry(Symbol, Industry_Id -> Industries.Id, Created_At)   -- one row/symbol

We copy that into this store's normalized dimensions — `sectors` and `industries` — and
record memberships in the `symbol_industry` join table. A symbol may belong to MANY
industries (and therefore many sectors, via industries.sector_code); the current MBin data
happens to be 1:1, but the join table supports the general case without a schema change.
Symbols absent from the source (e.g. ETFs/warrants) simply get no membership rows.

Read-only against the MBin DB. The join table is fully derived from MBin, so each run
rebuilds the memberships for our symbols (delete + re-insert inside one transaction) —
idempotent, and it also drops memberships MBin has removed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

from ..common import config
from ..common.errors import ConfigError
from . import db


def _read_source(trading_db: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Pull sectors, industries, and symbol→industry_code rows from MBin's SQLite DB.

    The mapping is resolved to the stable industry_code here (joining Symbol_Industry's
    integer Industry_Id through Industries) so this store never stores MBin's row ids.
    """
    if not trading_db.exists():
        raise ConfigError(
            f"MBin trading DB not found at {trading_db}. Set [etrade].trading_db_path or the "
            "MBIN_TRADING_DB env var to the sibling MBin repo's data/db/trading.db."
        )
    # Read-only URI so we never write to MBin's DB (it may be open by MBin itself).
    con = sqlite3.connect(f"file:{trading_db}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        industries_rows = con.execute(
            "SELECT DISTINCT Sector_Code, Sector_Name, Industry_Code, Industry_Name, Source_Url "
            "FROM Industries WHERE Industry_Code IS NOT NULL"
        ).fetchall()
        mapping_rows = con.execute(
            "SELECT si.Symbol AS symbol, i.Industry_Code AS industry_code "
            "FROM Symbol_Industry si JOIN Industries i ON si.Industry_Id = i.Id "
            "WHERE i.Industry_Code IS NOT NULL AND si.Symbol IS NOT NULL"
        ).fetchall()
    finally:
        con.close()

    sectors: dict[str, dict] = {}
    industries: list[dict] = []
    for r in industries_rows:
        code = str(r["Industry_Code"])
        sector_code = str(r["Sector_Code"]) if r["Sector_Code"] is not None else None
        if sector_code and sector_code not in sectors:
            sectors[sector_code] = {"sector_code": sector_code, "sector_name": r["Sector_Name"]}
        industries.append(
            {
                "industry_code": code,
                "industry_name": r["Industry_Name"],
                "sector_code": sector_code,
                "source_url": r["Source_Url"],
            }
        )
    # De-dup on the (symbol, industry_code) PAIR — a symbol may legitimately appear under
    # several industries, so we must NOT collapse to one row per symbol.
    mapping = {
        (str(r["symbol"]), str(r["industry_code"])): {
            "symbol": str(r["symbol"]), "industry_code": str(r["industry_code"])
        }
        for r in mapping_rows
    }
    return list(sectors.values()), industries, list(mapping.values())


def _upsert(con, table: str, columns: list[str], conflict: str, update: str, rows: list[dict]) -> None:
    """Set-based upsert via a registered Arrow batch (mirrors fundamentals._bulk_upsert)."""
    if not rows:
        return
    tbl = pa.table({c: [r.get(c) for r in rows] for c in columns})
    con.register("_dim_batch", tbl)
    try:
        cols = ", ".join(columns)
        con.execute(
            f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _dim_batch "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {update}"
        )
    finally:
        con.unregister("_dim_batch")


def map_industries(*, trading_db_path: Path | None = None, db_path=None) -> dict:
    """Copy MBin's sector/industry classification into this store and rebuild the
    symbol_industry memberships for symbols in our universe. Returns a CLI summary."""
    cfg = config.etrade_config()
    trading_db = Path(trading_db_path) if trading_db_path is not None else cfg["trading_db_path"]
    now = datetime.now(timezone.utc)

    sectors, industries, mapping = _read_source(trading_db)
    for s in sectors:
        s["updated_at"] = now
    for i in industries:
        i["updated_at"] = now

    with db.connect(db_path) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            _upsert(
                con, "sectors", ["sector_code", "sector_name", "updated_at"],
                "sector_code", "sector_name = excluded.sector_name, updated_at = excluded.updated_at",
                sectors,
            )
            _upsert(
                con, "industries",
                ["industry_code", "industry_name", "sector_code", "source_url", "updated_at"],
                "industry_code",
                "industry_name = excluded.industry_name, sector_code = excluded.sector_code, "
                "source_url = excluded.source_url, updated_at = excluded.updated_at",
                industries,
            )
            # Rebuild memberships for the symbols we track. The join table is fully derived
            # from MBin, so we register the full mapping, delete existing rows for those
            # symbols, then insert only pairs whose symbol exists in our `symbols` table
            # (referential integrity to our universe). A full rebuild also prunes memberships
            # MBin has since removed. Source symbols we don't track are counted as unmatched.
            matched_pairs = 0
            matched_symbols = 0
            if mapping:
                tbl = pa.table({
                    "symbol": [m["symbol"] for m in mapping],
                    "industry_code": [m["industry_code"] for m in mapping],
                })
                con.register("_ind_map", tbl)
                try:
                    con.execute(
                        "DELETE FROM symbol_industry WHERE symbol IN (SELECT symbol FROM _ind_map)"
                    )
                    con.execute(
                        "INSERT INTO symbol_industry (symbol, industry_code, mapped_at) "
                        "SELECT m.symbol, m.industry_code, ? FROM _ind_map m "
                        "WHERE m.symbol IN (SELECT symbol FROM symbols)",
                        [now],
                    )
                    matched_pairs = con.execute(
                        "SELECT count(*) FROM _ind_map m WHERE m.symbol IN (SELECT symbol FROM symbols)"
                    ).fetchone()[0]
                    matched_symbols = con.execute(
                        "SELECT count(DISTINCT m.symbol) FROM _ind_map m "
                        "WHERE m.symbol IN (SELECT symbol FROM symbols)"
                    ).fetchone()[0]
                finally:
                    con.unregister("_ind_map")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        symbols_with_industry = con.execute(
            "SELECT count(DISTINCT symbol) FROM symbol_industry"
        ).fetchone()[0]
        source_symbols = len({m["symbol"] for m in mapping})

    return {
        "trading_db": str(trading_db),
        "sectors": len(sectors),
        "industries": len(industries),
        "source_pairs": len(mapping),        # distinct (symbol, industry) memberships in MBin
        "source_symbols": source_symbols,    # distinct symbols in MBin's mapping
        "memberships": matched_pairs,        # pairs stored (symbol in our universe)
        "symbols_matched": matched_symbols,  # distinct our-symbols classified
        "symbols_unmatched": source_symbols - matched_symbols,  # in MBin, not in our table
        "symbols_with_industry": symbols_with_industry,
    }
