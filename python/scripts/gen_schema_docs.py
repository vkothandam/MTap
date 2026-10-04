#!/usr/bin/env python3
"""Generate docs/database-schema.md from the DuckDB DDL — the single source of truth.

Each store's schema lives in a module `_DDL` string (etrade/db.py, news/db.py). This
script builds each schema in an in-memory DuckDB, introspects its tables/columns/PKs, and
emits a mermaid ER diagram per store plus the (logical) relationships. Because the columns
are read back from the real DDL, the diagrams cannot drift from the schema.

Relationships (FKs) are NOT declared in the DDL — DuckDB treats them as comments — so the
edges are maintained by hand in RELATIONSHIPS below; add an entry when you add a table or a
cross-table reference.

Usage:
    uv run python scripts/gen_schema_docs.py           # (re)write docs/database-schema.md
    uv run python scripts/gen_schema_docs.py --check    # exit 1 if the doc is out of date (CI)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# Import the DDL straight from the store modules so this file has one source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sourcing_py.etrade import db as etrade_db
from sourcing_py.news import db as news_db

DOC_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "database-schema.md"

# --- Store definitions -------------------------------------------------------------------
# Each store: the DDL+migrations that define it, its default on-disk path, and its logical
# relationships. A relationship is (child_table, parent_table, cardinality, label): the edge
# reads parent-to-child, e.g. one `sectors` row to many `industries` rows.
STORES = [
    {
        "key": "etrade",
        "title": "E*TRADE fundamentals store",
        "default_path": "state/etrade/fundamentals.duckdb",
        "module": "python/sourcing_py/etrade/db.py",
        "ddl": etrade_db._DDL,
        "migrations": etrade_db._MIGRATIONS,
        "blurb": (
            "The concise, normalized system-of-record for the E*TRADE collector. "
            "`symbols` is the canonical ticker universe; the price series, fundamentals, "
            "filings, and TRBC sector/industry dimensions hang off it."
        ),
        # child, parent, cardinality (mermaid), label
        "relationships": [
            ("industries", "sectors", "||--o{", "sector_code"),
            ("symbol_industry", "symbols", "||--o{", "symbol"),
            ("symbol_industry", "industries", "||--o{", "industry_code"),
            ("daily_bars", "symbols", "||--o{", "symbol"),
            ("daily_bars", "trading_calendar", "||--o{", "date -> day_idx"),
            ("fundamentals", "symbols", "||--o{", "xid / symbol"),
            ("sec_filings", "symbols", "||--o{", "xid / symbol"),
            ("stock_splits", "symbols", "||--o{", "ticker -> symbol"),
        ],
    },
    {
        "key": "news",
        "title": "News processing store",
        "default_path": "state/news/news.duckdb",
        "module": "python/sourcing_py/news/db.py",
        "ddl": news_db._DDL,
        "migrations": news_db._MIGRATIONS,
        "blurb": (
            "Independent store for the news layer. It never owns symbols: "
            "`article_symbols.symbol` and `symbol_sentiment_daily.symbol` reference the "
            "E*TRADE `symbols` table, read READ-ONLY via `load_symbol_universe()` "
            "(see the cross-store note below)."
        ),
        "relationships": [
            ("article_symbols", "news_articles", "||--o{", "article_id"),
            ("article_sentiment", "news_articles", "||--||", "article_id"),
        ],
    },
]

# Cross-store references (no in-DB FK possible; documented, not drawn as one graph).
CROSS_STORE_NOTE = (
    "The news store's `article_symbols.symbol` and `symbol_sentiment_daily.symbol` columns "
    "point at `symbols.symbol` in the **E*TRADE fundamentals store**. There is no database "
    "foreign key across the two files; the news layer opens the E*TRADE DB read-only to load "
    "the symbol universe (`news/db.py::load_symbol_universe`)."
)


def _introspect(ddl: str, migrations: tuple[str, ...]) -> list[tuple[str, list[dict]]]:
    """Build the schema in memory and return [(table, [{name, type, pk}, ...]), ...]."""
    con = duckdb.connect(":memory:")
    try:
        con.execute(ddl)
        for stmt in migrations:
            con.execute(stmt)

        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()]

        # Primary-key columns per table (PK is the only constraint kind in this schema).
        pk_cols: dict[str, set[str]] = {}
        for tbl, cols in con.execute(
            "SELECT table_name, constraint_column_names FROM duckdb_constraints() "
            "WHERE constraint_type = 'PRIMARY KEY'"
        ).fetchall():
            pk_cols.setdefault(tbl, set()).update(cols)

        out = []
        for tbl in tables:
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                [tbl],
            ).fetchall()
            pks = pk_cols.get(tbl, set())
            out.append((tbl, [
                {"name": name, "type": dtype, "pk": name in pks}
                for name, dtype in cols
            ]))
        return out
    finally:
        con.close()


def _mermaid(schema: list[tuple[str, list[dict]]], relationships: list[tuple]) -> str:
    lines = ["```mermaid", "erDiagram"]
    for tbl, cols in schema:
        lines.append(f"    {tbl} {{")
        for c in cols:
            key = " PK" if c["pk"] else ""
            lines.append(f"        {c['type']} {c['name']}{key}")
        lines.append("    }")
    for child, parent, card, label in relationships:
        lines.append(f'    {parent} {card} {child} : "{label}"')
    lines.append("```")
    return "\n".join(lines)


def render() -> str:
    parts = [
        "# Database schema",
        "",
        "<!-- GENERATED FILE — do not edit by hand.",
        "     Regenerate with: uv run python scripts/gen_schema_docs.py",
        "     Source of truth: each store's `_DDL` in the module noted per section. -->",
        "",
        (
            "MTap keeps two independent [DuckDB](https://duckdb.org) stores. Both are created and "
            "migrated idempotently on open (`CREATE TABLE IF NOT EXISTS` + additive `ALTER`), so "
            "the diagrams below reflect a freshly-opened DB. Foreign keys are logical only (DuckDB "
            "stores them as comments); the edges are drawn from the relationships in the generator."
        ),
        "",
    ]
    for store in STORES:
        schema = _introspect(store["ddl"], store["migrations"])
        parts += [
            f"## {store['title']}",
            "",
            store["blurb"],
            "",
            f"- **Default path:** `{store['default_path']}`",
            f"- **DDL source:** [`{store['module']}`](../{store['module']})",
            "",
            _mermaid(schema, store["relationships"]),
            "",
        ]
    parts += [
        "## Cross-store reference",
        "",
        CROSS_STORE_NOTE,
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    check = "--check" in sys.argv[1:]
    content = render()
    if check:
        current = DOC_PATH.read_text() if DOC_PATH.exists() else ""
        if current != content:
            print(
                f"{DOC_PATH} is out of date. Run: uv run python scripts/gen_schema_docs.py",
                file=sys.stderr,
            )
            return 1
        print(f"{DOC_PATH.name} is up to date.")
        return 0
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(content)
    print(f"Wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
