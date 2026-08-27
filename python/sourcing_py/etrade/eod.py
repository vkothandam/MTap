"""Phase 1a — ingest end-of-day (EOD) OHLCV bars into DuckDB.

Phase 1 built the `symbols` master (distinct tickers). Phase 1a loads the actual
per-day bars those files hold — the price time series that Phase 2 fundamentals
augment for TFT — into the `daily_bars` table. No auth needed; this runs today.

Loading is done with DuckDB's native `read_json_auto`, which streams a whole
daily_summary jsonl file in one statement (far faster than row-by-row Python). Each
file is upserted on (symbol, date), so re-loading a corrected file refreshes it and
re-running is idempotent.

After loading, the `symbols` master is refreshed from `daily_bars` so first_seen /
last_seen / days_seen reflect the full ingested history (Phase 1 seeded it from a
narrower recent window).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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


def ingest_eod(
    *,
    days: int | None = None,
    fromdate: str | None = None,
    todate: str | None = None,
    directory: Path | None = None,
    db_path: Path | None = None,
    log_every: int = 25,
) -> dict:
    """Run Phase 1a. Loads all daily_summary files by default (days=None).

    Returns a summary dict. Files that fail to parse are collected and reported
    rather than aborting the whole run.
    """
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
        con.execute(_REFRESH_SYMBOLS, [now])
        bars_total = con.execute("SELECT count(*) FROM daily_bars").fetchone()[0]
        symbols_total = con.execute("SELECT count(*) FROM symbols").fetchone()[0]

    return {
        "files": len(files),
        "failed_files": failed,
        "window": (str(_file_date(files[0])), str(_file_date(files[-1]))),
        "bars_loaded": rows,
        "bars_total": bars_total,
        "symbols_total": symbols_total,
    }
