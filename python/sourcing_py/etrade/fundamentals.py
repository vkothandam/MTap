"""Phase 2 — fetch per-symbol fundamentals into the normalized DuckDB tables.

GATED on the E*TRADE browser session (see client.EtradeClient.require_session): the
live scrape cannot run until the Node puppeteer login has produced state/etrade/session.json.

Per symbol:
  1. resolve XID (symbol-lookup) and enrich the `symbols` row,
  2. fetch balance-sheet + income-statement (period q and a) and SEC filings (10-K, 10-Q),
  3. normalize into the long/tidy `fundamentals` + `sec_filings` tables.

Pacing, failure logging, and resume reuse the shared common/ utilities.

═══════════════════════════════════════════════════════════════════════════════
(RE)RUNNING THIS COLLECTOR — operator runbook
═══════════════════════════════════════════════════════════════════════════════
Full pipeline to (re)generate the DuckDB store (state/etrade/fundamentals.duckdb):

  1. build-symbols   seed the `symbols` master from out/daily_summary (no auth).
  2. ingest-eod      load `daily_bars` OHLCV from the same files (no auth).
  3. login           produce state/etrade/session.json (auth; see below).
  4. fetch --all     this module — fill `fundamentals` + `sec_filings` per symbol.

  cd python
  uv run sourcing-py etrade build-symbols --days 90
  uv run sourcing-py etrade ingest-eod
  uv run sourcing-py etrade fetch --all --resume        # add ETRADE_STOCKS_PER_SEC=4

AUTH — session.json is required and holds ONE of:
  • a bearer token  (PRIMARY / verified): seed manually as {"accessToken": "<tok>"}
    from a logged-in browser. Short-lived; on 401/403 the run aborts cleanly and you
    re-seed a fresh token and re-run with --resume.
  • a puppeteer session (fallback): `sourcing-py etrade login` runs the Node service,
    which logs in (SMS/OTP) and writes cookies + stk1/stk2 (+ any captured bearer).
    See login.py; the cookie/stk-only path is unverified for this API.

HOW THE DUCKDB TABLES GET POPULATED (per symbol, one atomic transaction):
  resolve XID (symbol-lookup) → _upsert_symbol_info updates the `symbols` row →
  4 statement calls (balance-sheet/income-statement × q/a) normalize to long rows and
  bulk-upsert into `fundamentals` → 2 filing calls (10-K/10-Q) bulk-upsert into
  `sec_filings` → the `symbols` row is stamped fundamentals_status='ok'. Writes are
  set-based Arrow upserts (see _bulk_upsert) — the periods a call returns UPSERT onto the
  PK, so re-running over time ACCUMULATES history (old periods are kept, not overwritten).

RESUME / IDEMPOTENCY (safe to stop and re-run any time):
  --resume skips any symbol with a committed outcome (fundamentals_status set, or — for
  rows fetched before that column existed — any fundamentals rows). Symbols with no
  fundamentals (ETFs/funds → 556, delisted/warrants → 400) are committed as
  fundamentals_status='none' so they are NOT re-hit each run. Genuine errors go to
  state/etrade/failures.jsonl; `fetch --retry-failed` reruns just those.

Reference full run (2026-08, 15,886 symbols): 6,801 ok / 9,014 no-data, ~5.48M
fundamentals rows + ~48k filings, single bearer token, ~8 symbols/sec.
═══════════════════════════════════════════════════════════════════════════════

The `_normalize_*` helpers below were finalized against real sampled responses
(GOOG/GOOGL, 2026-08) — see the fixtures under tests/fixtures/etrade/. They stay
defensive (unexpected shape yields zero rows rather than a crash) since the upstream
JSON is not contractually stable.

Statement shape (`balance-sheet` / `income-statement`):
  data.summaryData:  flat top-line items (a subset of the tree) — used only for the
                     positional period-date index, since child series omit `date`.
  data.<balanceSheet|incomeStatement>.data:  a recursive tree of sections; each node is
                     {name, data:[{value, code, date?, periodLength?, periodLengthUnits?}],
                      children:[...]}. We walk it and emit one long row per (code, period).
  data.thousandMultiplier:  scaling flag carried onto every row.

Filings shape (`sec-filings`):
  data.secFilings:  [{formType, dateFiled, htmlDocKey}] (may be empty — filings attach to
                    the primary share class, e.g. GOOGL not GOOG); data.totalRows is a count.
"""

from __future__ import annotations

import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
from typing import Any

import pyarrow as pa

from ..common import failures
from ..common.errors import ConfigError, FetchError
from . import db
from .client import EtradeClient

_SCOPE = "etrade"  # state/etrade/failures.jsonl and rate/etrade.lock
_STATEMENTS = (("balance_sheet", "balance_sheet"), ("income_statement", "income_statement"))
_PERIODS = ("q", "a")
_FORM_TYPES = ("10-K", "10-Q")
# statement -> the response key holding the line-item tree
_TREE_KEY = {"balance_sheet": "balanceSheet", "income_statement": "incomeStatement"}
# Statuses that mean "this instrument legitimately has no fundamentals", not a failure:
#   400 — symbol-lookup "Symbol not found" (delisted / warrant / unit / bad suffix)
#   404 — resource absent
#   556 — wsod "No <statement> data was returned for venueXid ..." (ETFs, funds, SPACs)
# We record these as fundamentals_status='none' so --resume skips them instead of
# re-hitting the API for them (and burning the session) on every run.
_NO_DATA_STATUS = {400, 404, 556}


class _NoFundamentals(Exception):
    """Signal that a symbol has no fundamentals to store (see _NO_DATA_STATUS).
    Carries the resolved symbolInfo when we got as far as a successful lookup, so the
    symbols row is still enriched with xid/company even though it has no statements."""

    def __init__(self, symbol: str, info: dict | None) -> None:
        super().__init__(symbol)
        self.symbol = symbol
        self.info = info


# -- normalizers ----------------------------------------------------------------
def _to_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _period_dates(data: dict) -> list[date | None]:
    """Positional period-end dates. summaryData series always carry `date`; deeper
    tree entries align to it by index."""
    summary = data.get("summaryData") or []
    series = summary[0].get("data", []) if summary else []
    return [_to_date(e.get("date")) for e in series]


def _walk_statement(node: dict, ref_dates: list[date | None], thousand: bool | None,
                    statement: str, period_type: str, rows: list[dict]) -> None:
    for idx, entry in enumerate(node.get("data") or []):
        code = entry.get("code")
        if not code:
            continue
        fiscal_end = _to_date(entry.get("date")) or (ref_dates[idx] if idx < len(ref_dates) else None)
        rows.append(
            {
                "statement": statement,
                "period_type": period_type,
                "fiscal_end": fiscal_end,
                "line_item": str(code),
                "value": _to_float(entry.get("value")),
                "raw_label": node.get("name"),
                "period_length": entry.get("periodLength"),
                "period_length_units": entry.get("periodLengthUnits"),
                "thousand_multiplier": thousand,
            }
        )
    for child in node.get("children") or []:
        _walk_statement(child, ref_dates, thousand, statement, period_type, rows)


def _normalize_statement(raw: dict, statement: str, period_type: str) -> list[dict]:
    """Flatten a statement response to long rows: one per (line_item code, period)."""
    data = raw.get("data") or {}
    tree = data.get(_TREE_KEY.get(statement, "")) or {}
    ref_dates = _period_dates(data)
    thousand = data.get("thousandMultiplier")
    rows: list[dict] = []
    for section in tree.get("data") or []:
        _walk_statement(section, ref_dates, thousand, statement, period_type, rows)
    return rows


def _normalize_filings(raw: dict, form_type: str) -> list[dict]:
    """Flatten a sec-filings response. Records are {formType, dateFiled, htmlDocKey}."""
    data = raw.get("data") or {}
    rows: list[dict] = []
    for f in data.get("secFilings") or []:
        rows.append(
            {
                "form_type": f.get("formType") or form_type,
                "date_filed": _to_date(f.get("dateFiled")),
                "html_doc_key": f.get("htmlDocKey"),
            }
        )
    return rows


# -- persistence ----------------------------------------------------------------
def _upsert_symbol_info(con, symbol: str, info: dict) -> int | None:
    xid = info.get("XID")
    if xid is None:
        return None
    con.execute(
        """
        INSERT INTO symbols (symbol, xid, issue_type, exchange, company_name,
                             auto_invest, has_fund_commentary, xid_resolved_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (symbol) DO UPDATE SET
            xid = excluded.xid, issue_type = excluded.issue_type,
            exchange = excluded.exchange, company_name = excluded.company_name,
            auto_invest = excluded.auto_invest,
            has_fund_commentary = excluded.has_fund_commentary,
            xid_resolved_at = excluded.xid_resolved_at, updated_at = excluded.updated_at
        """,
        [
            symbol,
            xid,
            info.get("issueType"),
            info.get("exchange"),
            info.get("companyName"),
            info.get("autoInvest"),
            info.get("hasFundCommentary"),
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        ],
    )
    return int(xid)


def _set_status(con, symbol: str, status: str) -> None:
    """Record the Phase 2 outcome for a symbol ('ok' | 'none'). Upserts so it works
    whether or not _upsert_symbol_info already created the row this transaction."""
    now = datetime.now(timezone.utc)
    con.execute(
        """
        INSERT INTO symbols (symbol, fundamentals_status, fundamentals_checked_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (symbol) DO UPDATE SET
            fundamentals_status = excluded.fundamentals_status,
            fundamentals_checked_at = excluded.fundamentals_checked_at,
            updated_at = excluded.updated_at
        """,
        [symbol, status, now, now],
    )


def _mark_no_fundamentals(con, symbol: str, info: dict | None) -> None:
    """Persist a 'none' outcome. Keeps xid/company if the lookup succeeded (556 case)."""
    if info:
        _upsert_symbol_info(con, symbol, info)
    _set_status(con, symbol, "none")


def _bulk_upsert(con, table: str, columns: list[str], conflict: str, update: str,
                 rows: list[dict]) -> None:
    """Set-based upsert of `rows` into `table` via a registered Arrow batch. This is an
    order of magnitude faster than row-by-row executemany for the ~1k rows/symbol we
    write (DuckDB's per-row prepared-statement + index-probe cost dominates otherwise)."""
    tbl = pa.table({c: [r[c] for r in rows] for c in columns})
    con.register("_upsert_batch", tbl)
    try:
        cols = ", ".join(columns)
        con.execute(
            f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _upsert_batch "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {update}"
        )
    finally:
        con.unregister("_upsert_batch")


def _store_fundamentals(con, xid: int, symbol: str, rows: list[dict]) -> int:
    """Store statement rows. Returns the count dropped for lack of a period-end date
    (a row with no fiscal_end can't be keyed and isn't a usable time-series point)."""
    now = datetime.now(timezone.utc)
    keep = [r for r in rows if r["fiscal_end"] is not None]
    if not keep:
        return len(rows)  # nothing storable (e.g. all periods lacked an end date)
    # De-dupe within the batch (last wins) — a set-based ON CONFLICT upsert errors if the
    # same PK appears twice in one INSERT, whereas executemany silently took the last.
    deduped = {(r["statement"], r["period_type"], r["fiscal_end"], r["line_item"]): r for r in keep}
    batch = [
        {"xid": xid, "symbol": symbol, "statement": r["statement"], "period_type": r["period_type"],
         "fiscal_end": r["fiscal_end"], "line_item": r["line_item"], "value": r["value"],
         "raw_label": r["raw_label"], "period_length": r["period_length"],
         "period_length_units": r["period_length_units"],
         "thousand_multiplier": r["thousand_multiplier"], "fetched_at": now}
        for r in deduped.values()
    ]
    _bulk_upsert(
        con, "fundamentals",
        ["xid", "symbol", "statement", "period_type", "fiscal_end", "line_item", "value",
         "raw_label", "period_length", "period_length_units", "thousand_multiplier", "fetched_at"],
        "xid, statement, period_type, fiscal_end, line_item",
        "value = excluded.value, raw_label = excluded.raw_label, "
        "period_length = excluded.period_length, period_length_units = excluded.period_length_units, "
        "thousand_multiplier = excluded.thousand_multiplier, fetched_at = excluded.fetched_at",
        batch,
    )
    return len(rows) - len(keep)


def _store_filings(con, xid: int, symbol: str, rows: list[dict]) -> int:
    """Store filing rows. Returns the count dropped for lack of an html_doc_key (its
    primary-key component — a filing without one can't be deduped/addressed)."""
    now = datetime.now(timezone.utc)
    keep = [r for r in rows if r["html_doc_key"]]
    if not keep:
        return len(rows)  # e.g. a share class with no filings (GOOG vs GOOGL) — not an error
    deduped = {(r["form_type"], r["html_doc_key"]): r for r in keep}  # last wins (see above)
    batch = [
        {"xid": xid, "symbol": symbol, "form_type": r["form_type"],
         "date_filed": r["date_filed"], "html_doc_key": r["html_doc_key"], "fetched_at": now}
        for r in deduped.values()
    ]
    _bulk_upsert(
        con, "sec_filings",
        ["xid", "symbol", "form_type", "date_filed", "html_doc_key", "fetched_at"],
        "xid, form_type, html_doc_key",
        "date_filed = excluded.date_filed, fetched_at = excluded.fetched_at",
        batch,
    )
    return len(rows) - len(keep)


def _fetch_payloads(client: EtradeClient, symbol: str) -> dict:
    """NETWORK ONLY — safe to run in a worker thread (touches no DB). Resolves the XID
    and pulls the statement + filing responses, returning the raw JSON for the main
    thread to normalize and store.

    Raises _NoFundamentals when the instrument legitimately has no data (400 lookup miss
    or a wsod 556 "no <statement> data" response); ConfigError (401/403, from the client)
    and other FetchErrors propagate for the caller to abort or log."""
    try:
        info = client.resolve_xid(symbol)
    except FetchError as exc:
        if exc.status_code in _NO_DATA_STATUS:  # e.g. 400 "Symbol not found"
            raise _NoFundamentals(symbol, None) from exc
        raise
    xid = info.get("XID")
    if xid is None:
        raise FetchError(f"no XID for {symbol}")
    xid = int(xid)
    statements: list[tuple] = []
    filings: list[tuple] = []
    try:
        for statement, _label in _STATEMENTS:
            for period in _PERIODS:  # fetch BOTH quarterly ('q') and annual ('a')
                call = client.balance_sheet if statement == "balance_sheet" else client.income_statement
                statements.append((statement, period, call(xid, period)))
        for form_type in _FORM_TYPES:
            filings.append((form_type, client.sec_filings(xid, form_type)))
    except FetchError as exc:
        if exc.status_code in _NO_DATA_STATUS:  # e.g. 556 "No Balance Sheet data ..."
            raise _NoFundamentals(symbol, info) from exc
        raise
    return {"info": info, "statements": statements, "filings": filings}


def _store_payloads(con, symbol: str, payloads: dict) -> int:
    """DB ONLY — runs on the main thread inside a per-symbol transaction. Normalizes the
    payloads _fetch_payloads returned and stores them. Returns fundamentals rows written.

    Raises _NoFundamentals if nothing storable came back (resolved but every statement was
    empty), so the caller records 'none' and --resume won't keep re-fetching it."""
    xid = _upsert_symbol_info(con, symbol, payloads["info"])
    if xid is None:
        raise FetchError(f"no XID for {symbol}")
    written = 0
    dropped = 0
    for statement, period, raw in payloads["statements"]:
        rows = _normalize_statement(raw, statement, period)
        dropped += _store_fundamentals(con, xid, symbol, rows)
        written += len(rows)
    for form_type, raw in payloads["filings"]:
        rows = _normalize_filings(raw, form_type)
        dropped += _store_filings(con, xid, symbol, rows)
    if written == 0:
        raise _NoFundamentals(symbol, payloads["info"])
    if dropped:
        # Never silent: a positive count means the upstream shape drifted from what the
        # normalizers expect (e.g. a period with no resolvable end date). Surface it.
        print(f"{symbol}: WARNING — dropped {dropped} row(s) with no usable key (shape drift?)")
    _set_status(con, symbol, "ok")
    return written


def _already_done(con, symbol: str) -> bool:
    """A symbol counts as done once Phase 2 has attempted it — either fetched ('ok') or
    found to have no data ('none'). Combined with the per-symbol transaction (all-or-
    nothing), --resume skips only symbols whose outcome fully committed."""
    row = con.execute("SELECT fundamentals_status FROM symbols WHERE symbol = ?", [symbol]).fetchone()
    if row and row[0] is not None:
        return True
    # Fallback for rows fetched before fundamentals_status existed (e.g. a prior GOOGL run).
    row = con.execute("SELECT count(*) FROM fundamentals WHERE symbol = ?", [symbol]).fetchone()
    return bool(row and row[0] > 0)


def _target_symbols(con, *, symbols: list[str] | None, fetch_all: bool, retry_failed: bool) -> list[str]:
    if retry_failed:
        return failures.load_symbols(_SCOPE)
    if symbols:
        return symbols
    if fetch_all:
        return [r[0] for r in con.execute("SELECT symbol FROM symbols ORDER BY symbol").fetchall()]
    raise ValueError("specify --symbols A,B, or --all, or --retry-failed")


def fetch(
    *,
    symbols: list[str] | None = None,
    fetch_all: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    db_path=None,
) -> dict:
    """Run Phase 2. Fails fast (ConfigError) if the browser session is absent.

    Symbols are fetched concurrently (a pool of `workers`, network-only in the worker
    threads) while ALL DuckDB writes stay on this thread, one atomic per-symbol
    transaction each. The shared rate-limit throttle caps aggregate request rate, so the
    workers only supply enough parallelism to reach it despite per-request latency."""
    client = EtradeClient()
    client.require_session()  # gate: aborts cleanly until puppeteer login has run
    workers = max(1, int(client.cfg.get("workers", 1) or 1))

    ok = 0
    skipped = 0
    processed = 0
    aborted = False
    still_failed: list[dict] = []
    with db.connect(db_path) as con:
        targets = _target_symbols(con, symbols=symbols, fetch_all=fetch_all, retry_failed=retry_failed)

        # --- main-thread outcome handlers (every DB access lives here) --------------
        def _commit_none(sym: str, info: dict | None) -> None:
            # Not a failure: ETF/fund/invalid symbol with no statements. Commit 'none' so
            # --resume skips it rather than re-hitting the API each run.
            con.execute("BEGIN TRANSACTION")
            _mark_no_fundamentals(con, sym, info)
            con.execute("COMMIT")

        def _record_failure(sym: str, exc: Exception) -> None:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"{sym}: FAILED — {msg}")
            if retry_failed:
                still_failed.append({"symbol": sym, "error": msg, "at": datetime.now(timezone.utc).isoformat()})
            else:
                failures.record_symbol(_SCOPE, sym, msg)

        # --- rolling-window submission: keep ~`workers` symbols in flight ----------
        pending = iter(targets)
        inflight: dict = {}  # future -> symbol

        def _submit_next(pool) -> bool:
            for sym in pending:  # skip already-done here (main thread) before spending a call
                if resume and _already_done(con, sym):
                    continue
                inflight[pool.submit(_fetch_payloads, client, sym)] = sym
                return True
            return False

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(workers):
                if not _submit_next(pool):
                    break
            while inflight and not aborted:
                done, _pending = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    symbol = inflight.pop(fut)
                    processed += 1
                    if processed % 50 == 0:
                        print(f"  progress: {processed} processed — {ok} fetched, {skipped} no-data", flush=True)
                    try:
                        payloads = fut.result()
                    except _NoFundamentals as exc:  # no-data detected during fetch (400/556)
                        _commit_none(exc.symbol, exc.info)
                        skipped += 1
                    except ConfigError as exc:  # expired token — fatal for every remaining symbol
                        print(f"\nstopping: {exc}", file=sys.stderr)
                        aborted = True
                        break
                    except Exception as exc:  # noqa: BLE001 — per-symbol failure: log, keep going
                        _record_failure(symbol, exc)
                    else:
                        # Store atomically; roll back so a symbol is never left half-written.
                        con.execute("BEGIN TRANSACTION")
                        try:
                            _store_payloads(con, symbol, payloads)
                            con.execute("COMMIT")
                            ok += 1
                        except _NoFundamentals as exc:  # resolved but every statement empty
                            con.execute("ROLLBACK")
                            _commit_none(exc.symbol, exc.info)
                            skipped += 1
                        except Exception as exc:  # noqa: BLE001
                            con.execute("ROLLBACK")
                            _record_failure(symbol, exc)
                    if not aborted:
                        _submit_next(pool)
                if aborted:
                    # Stop feeding the pool; un-started calls are cancelled, in-flight ones
                    # are simply discarded (their symbols re-fetch cleanly on --resume).
                    for f in inflight:
                        f.cancel()
                    break

    if retry_failed and not aborted:
        failures.rewrite_symbols(_SCOPE, still_failed)
    return {
        "targets": len(targets),
        "ok": ok,
        "skipped": skipped,
        "failed": len(failures.load_symbols(_SCOPE)),
        "aborted": aborted,
    }
