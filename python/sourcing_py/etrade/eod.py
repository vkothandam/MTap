"""Phase 1a — ingest end-of-day (EOD) OHLCV bars into DuckDB.

Phase 1 built the `symbols` master (distinct tickers). Phase 1a loads the actual
per-day bars those files hold — the price time series that Phase 2 fundamentals
augment for TFT — into the `daily_bars` table. No auth needed; this runs today.

Loading is done with DuckDB's native `read_json_auto`, which streams a whole
daily_summary jsonl file in one statement (far faster than row-by-row Python). Each
file is upserted on (symbol, date), so re-loading a corrected file refreshes it and
re-running is idempotent.

A second pass attaches the pre-market / after-hours columns from the extended_hours
files (sourcing_py/massive_ext) for the same dates: an UPDATE on (symbol, date), so it
never creates a bar, and the upsert above doesn't list those columns, so reloading a
daily file keeps them.

After loading, the `symbols` master is refreshed from `daily_bars` so first_seen /
last_seen / days_seen reflect the full ingested history (Phase 1 seeded it from a
narrower recent window).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from ..massive_ext.extended import NEW_COLUMN_TYPES, NEW_COLUMNS
from . import db
from .symbols import _file_date, daily_summary_dir, select_files

# The jsonl keys we map into daily_bars. read_json_auto infers types; we cast the
# ones that need pinning (date, transactions) and rename the `_`-prefixed provenance.
_INSERT_FILE = """
INSERT INTO daily_bars
    (symbol, date, open, high, low, close, volume, vwap, transactions,
     window_start_ms, source, ingested_at, run_id, loaded_at)
SELECT ticker, date, open, high, low, close, volume, vwap, transactions,
       window_start_ms, _source, _ingested_at, _run_id, ?
FROM read_json_auto(?)
WHERE ticker IS NOT NULL
ON CONFLICT (symbol, date) DO UPDATE SET
    open = excluded.open, high = excluded.high, low = excluded.low,
    close = excluded.close, volume = excluded.volume, vwap = excluded.vwap,
    transactions = excluded.transactions, window_start_ms = excluded.window_start_ms,
    source = excluded.source, ingested_at = excluded.ingested_at,
    run_id = excluded.run_id, loaded_at = excluded.loaded_at
"""

# One extended_hours file onto the bars it matches. fetched_at is an ISO string with an
# offset; stored as naive UTC like ingested_at.
# The VWAP columns are COALESCEd: a file written before they existed reads them as NULL and
# must not wipe values a newer file loaded.
_UPDATE_EXTENDED = f"""
UPDATE daily_bars b SET
    pre_market_price = e.pre_market_price, pre_market_last = e.pre_market_last,
    pre_market_volume = e.pre_market_volume, regular_volume = e.regular_volume,
    after_hours_price = e.after_hours_price, after_hours_volume = e.after_hours_volume,
    {", ".join(f"{c} = coalesce(e.{c}, b.{c})" for c in NEW_COLUMNS)},
    ext_fetched_at = timezone('UTC', TRY_CAST(e.fetched_at AS TIMESTAMPTZ))
FROM read_json(?, format = 'newline_delimited', columns = {{
    'ticker': 'VARCHAR', 'date': 'DATE', 'pre_market_price': 'DOUBLE',
    'pre_market_last': 'DOUBLE', 'pre_market_volume': 'DOUBLE', 'regular_volume': 'DOUBLE',
    'after_hours_price': 'DOUBLE', 'after_hours_volume': 'DOUBLE', 'fetched_at': 'VARCHAR',
    {", ".join(f"'{c}': '{t}'" for c, t in NEW_COLUMN_TYPES.items())}}}) e
WHERE b.symbol = e.ticker AND b.date = e.date
"""

EXTENDED_DATASET = "extended_hours"

# Rebuild the symbols master's date fields from the fuller daily_bars history.
_REFRESH_SYMBOLS = """
INSERT INTO symbols (symbol, first_seen, last_seen, days_seen, updated_at)
SELECT symbol, min(date), max(date), count(DISTINCT date), ?
FROM daily_bars GROUP BY symbol
ON CONFLICT (symbol) DO UPDATE SET
    first_seen = least(symbols.first_seen, excluded.first_seen),
    last_seen  = greatest(symbols.last_seen, excluded.last_seen),
    days_seen  = greatest(symbols.days_seen, excluded.days_seen),
    updated_at = excluded.updated_at
"""


def _load_file(con, path: Path, now: datetime) -> int:
    """Upsert one daily_summary file; return the rows it contributed."""
    con.execute(_INSERT_FILE, [now, str(path)])
    return con.execute(
        "SELECT count(*) FROM read_json_auto(?) WHERE ticker IS NOT NULL", [str(path)]
    ).fetchone()[0]


def extended_files(daily_dir: Path, lo: date, hi: date) -> list[Path]:
    """The extended_hours files (a sibling of the daily_summary folder) dated lo..hi."""
    out = []
    for p in sorted((daily_dir.parent / EXTENDED_DATASET).glob(f"{EXTENDED_DATASET}-*.jsonl")):
        try:
            d = date.fromisoformat(p.name[len(EXTENDED_DATASET) + 1:-len(".jsonl")])
        except ValueError:
            continue
        if lo <= d <= hi and p.stat().st_size:
            out.append(p)
    return out


_ISSUE_TYPE = {"CS": "CS", "ADRC": "DR"}


def listing_by_ticker(rows: list[dict]) -> dict[str, dict]:
    """One tickers_cs row per ticker: the active listing if there is one, else the most
    recently delisted (a ticker can be listed twice when a symbol was recycled)."""
    best: dict[str, dict] = {}
    for r in rows:
        t = r.get("ticker")
        if not t:
            continue
        key = (bool(r.get("active")), r.get("delisted_utc") or "", r.get("last_updated_utc") or "")
        if t not in best or key > best[t][0]:
            best[t] = (key, r)
    return {t: v[1] for t, v in best.items()}


def recycled_tickers(rows: list[dict]) -> list[str]:
    """Tickers listed under more than one CIK in the tickers_cs file (symbol reuse)."""
    ciks: dict[str, set] = {}
    for r in rows:
        if r.get("ticker") and r.get("cik"):
            ciks.setdefault(r["ticker"], set()).add(r["cik"])
    return sorted(t for t, c in ciks.items() if len(c) > 1)


def fill_issue_types(con, rows: list[dict]) -> int:
    """Set symbols.issue_type where NULL from the tickers_cs type (CS -> CS, ADRC -> DR);
    delisted names never got one from E*TRADE. Returns the symbols filled."""
    import pyarrow as pa

    pairs = [(t, _ISSUE_TYPE[r.get("type") or "CS"]) for t, r in listing_by_ticker(rows).items()
             if (r.get("type") or "CS") in _ISSUE_TYPE]
    if not pairs:
        return 0
    con.register("_types", pa.table({"symbol": [p[0] for p in pairs], "issue_type": [p[1] for p in pairs]}))
    try:
        before = con.execute("SELECT count(*) FROM symbols WHERE issue_type IS NULL").fetchone()[0]
        con.execute("UPDATE symbols s SET issue_type = t.issue_type FROM _types t "
                    "WHERE s.symbol = t.symbol AND s.issue_type IS NULL")
        return before - con.execute("SELECT count(*) FROM symbols WHERE issue_type IS NULL").fetchone()[0]
    finally:
        con.unregister("_types")


def ingest_eod(
    *,
    days: int | None = None,
    fromdate: str | None = None,
    todate: str | None = None,
    directory: Path | None = None,
    db_path: Path | None = None,
    log_every: int = 25,
    extended_only: bool = False,
) -> dict:
    """Run Phase 1a. Loads all daily_summary files by default (days=None).

    Returns a summary dict. Files that fail to parse are collected and reported
    rather than aborting the whole run.
    """
    if extended_only:
        return _ingest_extended_only(fromdate, todate, directory, db_path)
    files = select_files(days=days, fromdate=fromdate, todate=todate, directory=directory)
    if not files:
        raise FileNotFoundError(
            f"no daily_summary-*.jsonl files found in {directory or daily_summary_dir()}"
        )

    now = datetime.now(timezone.utc)
    rows = 0
    failed: list[dict] = []
    with db.connect(db_path) as con:
        for i, path in enumerate(files, 1):
            try:
                rows += _load_file(con, path, now)
            except Exception as exc:  # noqa: BLE001 — per-file: log and keep going
                failed.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"{path.name}: FAILED — {type(exc).__name__}: {exc}")
                continue
            if i % log_every == 0 or i == len(files):
                print(f"  loaded {i}/{len(files)} files, {rows:,} bars so far")
        ext_files = extended_files(directory or daily_summary_dir(), _file_date(files[0]), _file_date(files[-1]))
        ext_rows = 0
        for path in ext_files:
            try:
                ext_rows += con.execute(_UPDATE_EXTENDED, [str(path)]).fetchone()[0]
            except Exception as exc:  # noqa: BLE001 — per-file: log and keep going
                failed.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"{path.name}: FAILED — {type(exc).__name__}: {exc}")
        con.execute(_REFRESH_SYMBOLS, [now])
        issue_filled, recycled = _tickers_pass(con)
        bars_total = con.execute("SELECT count(*) FROM daily_bars").fetchone()[0]
        symbols_total = con.execute("SELECT count(*) FROM symbols").fetchone()[0]

    return {
        "files": len(files),
        "failed_files": failed,
        "window": (str(_file_date(files[0])), str(_file_date(files[-1]))),
        "bars_loaded": rows,
        "extended_files": len(ext_files),
        "bars_extended": ext_rows,
        "bars_total": bars_total,
        "symbols_total": symbols_total,
        "issue_types_filled": issue_filled,
        "recycled_tickers": recycled,
    }


def _tickers_pass(con) -> tuple[int, list[str]]:
    """Fill NULL issue_type from the newest tickers_cs file; (filled, recycled tickers).
    No file (or an unreadable one) is not an error: the pass is skipped."""
    from ..massive_ext.extended import latest_tickers

    try:
        rows = latest_tickers()
    except FileNotFoundError:
        return 0, []
    return fill_issue_types(con, rows), recycled_tickers(rows)


def _ingest_extended_only(fromdate, todate, directory, db_path) -> dict:
    """Only the extended-hours UPDATE pass, over every extended_hours file in range (a
    VWAP backfill must not reload the daily bars)."""
    lo = date.fromisoformat(fromdate) if fromdate else date.min
    hi = date.fromisoformat(todate) if todate else date.max
    ext_files = extended_files(directory or daily_summary_dir(), lo, hi)
    failed: list[dict] = []
    ext_rows = 0
    with db.connect(db_path) as con:
        for path in ext_files:
            try:
                ext_rows += con.execute(_UPDATE_EXTENDED, [str(path)]).fetchone()[0]
            except Exception as exc:  # noqa: BLE001 — per-file: log and keep going
                failed.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"{path.name}: FAILED — {type(exc).__name__}: {exc}")
        bars_total = con.execute("SELECT count(*) FROM daily_bars").fetchone()[0]
        symbols_total = con.execute("SELECT count(*) FROM symbols").fetchone()[0]
    return {
        "files": 0, "failed_files": failed,
        "window": (lo.isoformat() if fromdate else "-", hi.isoformat() if todate else "-"),
        "bars_loaded": 0, "extended_files": len(ext_files), "bars_extended": ext_rows,
        "bars_total": bars_total, "symbols_total": symbols_total,
        "issue_types_filled": 0, "recycled_tickers": [],
    }
