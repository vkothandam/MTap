"""SEC EDGAR filings (`data.sec.gov` submissions API) into `sec_edgar_tickers` and
`sec_edgar_filings`: the source of the exact financial-results release dates.

A company's earnings press release is furnished on an 8-K under item 2.02 ("Results of
Operations and Financial Condition"), normally filed with the release itself, and EDGAR
stamps its acceptance time to the second. That time says which session first reacts: a
release before the 16:00 close moves that day's session, one after it moves the next.
derive-features turns these filings into `results_announcements` and the earnings windows.

Kept per company: 8-K (with its item codes), 10-K, 10-Q, 20-F and 40-F, plus amendments,
filed on or after `since` (default: 400 days before the first bar, so the first bars have a
year-ago release to estimate from).

Acceptance times come from each release's EDGAR header (`<ACCEPTANCE-DATETIME>`, New York
time), one request per 8-K with item 2.02, fetched once. The submissions JSON also has an
`acceptanceDateTime`, but it can't be trusted: it is labelled UTC, and for some filings it
is UTC while for others it is the Eastern time plus twice the UTC offset (Apple's release
accepted 16:30:28 EDT reads 00:30:28Z; Microsoft's accepted 16:04:53 EDT reads 20:04:53Z),
with no field that tells the two apart.

Ticker -> CIK comes from SEC's company_tickers.json; SEC writes class shares with a dash
(BRK-B) where our symbols use a dot (BRK.B). Several tickers can share a CIK (GOOG/GOOGL).

Run:
    uv run sourcing-py etrade fetch-edgar [--symbols AAPL,MSFT] [--since 2023-07-01]
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, date, datetime, timedelta

import httpx
import pyarrow as pa

from ..common import config, http, ratelimit
from ..common.errors import FetchError
from ..utils import trading_calendar as tc
from . import db

_PROVIDER = "sec"
_FORMS = frozenset({"8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"})
_HISTORY_DAYS = 400
_TIME_BATCH = 500
_ACCEPTED = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
_FILING_COLS = ("cik", "accession", "form", "filing_date", "report_date", "items", "primary_doc", "fetched_at")


def release_sql(alias: str = "") -> str:
    """SQL filter for the 8-Ks that carry a results release (item 2.02)."""
    a = f"{alias}." if alias else ""
    return f"{a}form = '8-K' AND list_contains(string_split({a}items, ','), '2.02')"


def sec_ticker(symbol: str) -> str:
    """Our symbol as SEC writes it: BRK.B -> BRK-B."""
    return symbol.upper().replace(".", "-")


def _headers(cfg: dict) -> dict:
    # sent on every request, so an injected client gets them too
    return {"user-agent": cfg["user_agent"], "accept-encoding": "gzip, deflate"}


def _get(client: httpx.Client, cfg: dict, url: str):
    rps = cfg["requests_per_sec"]
    ratelimit.throttle(_PROVIDER, 1.0 / rps if rps else 0.0)
    return http.get_json(client, url, headers=_headers(cfg))


def _block(block: dict) -> list[dict]:
    """One columnar filings block (`filings.recent` or an older history file) as records."""
    forms = block.get("form") or []
    keys = ("accessionNumber", "filingDate", "reportDate", "items", "primaryDocument")
    cols = {k: block.get(k) or [None] * len(forms) for k in keys}
    return [{"form": f, **{k: cols[k][i] for k in keys}} for i, f in enumerate(forms)]


def _company(client: httpx.Client, cfg: dict, cik: int, since: date, fetched_at: datetime):
    """The kept filings of one CIK filed on or after `since`. Reads the `recent` block (at
    least a year, or 1,000 filings) plus the older history files that reach back to `since`.
    Returns (rows, extra_files)."""
    base = cfg["submissions_url"]
    doc = _get(client, cfg, f"{base}/CIK{cik:010d}.json")
    filings = doc.get("filings") or {}
    recs = _block(filings.get("recent") or {})
    extra = [f for f in filings.get("files") or []
             if f.get("filingTo") and date.fromisoformat(f["filingTo"]) >= since]
    for f in extra:
        recs += _block(_get(client, cfg, f"{base}/{f['name']}"))
    rows = []
    for r in recs:
        if r["form"] not in _FORMS or not r["filingDate"]:
            continue
        filed = date.fromisoformat(r["filingDate"])
        if filed < since:
            continue
        rows.append({
            "cik": cik,
            "accession": r["accessionNumber"],
            "form": r["form"],
            "filing_date": filed,
            "report_date": date.fromisoformat(r["reportDate"]) if r["reportDate"] else None,
            "items": r["items"] or None,
            "primary_doc": r["primaryDocument"] or None,
            "fetched_at": fetched_at,
        })
    return rows, len(extra)


def accepted_time(client: httpx.Client, cfg: dict, cik: int, accession: str) -> datetime | None:
    """A filing's acceptance time (naive, New York) from its EDGAR header page."""
    url = f"{cfg['archives_url']}/{cik}/{accession.replace('-', '')}/{accession}-index-headers.html"
    ratelimit.throttle(_PROVIDER, 1.0 / cfg["requests_per_sec"] if cfg["requests_per_sec"] else 0.0)
    _, text = http.get_text(client, url, headers=_headers(cfg))
    m = _ACCEPTED.search(text)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None  # noqa: DTZ007 (naive NY time)


def _pending(con, ciks: list[int]) -> list[tuple[int, str]]:
    """The release 8-Ks of `ciks` with no acceptance time yet, newest first."""
    con.register("_ciks", pa.table({"cik": pa.array(ciks, pa.int64())}))
    try:
        return con.execute(
            f"SELECT cik, accession FROM sec_edgar_filings WHERE {release_sql()} "
            "AND accepted_at IS NULL AND cik IN (SELECT cik FROM _ciks) ORDER BY filing_date DESC"
        ).fetchall()
    finally:
        con.unregister("_ciks")


def _fill_times(db_path, client: httpx.Client, cfg: dict, todo: list[tuple[int, str]]) -> tuple[int, int]:
    """Fetch the acceptance times of `todo` (cik, accession), saving every _TIME_BATCH so an
    interrupted run resumes. The DB is only opened to save. Returns (filled, failed)."""

    def one(job):
        try:
            return job, accepted_time(client, cfg, *job)
        except FetchError:
            return job, None

    filled = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg["workers"])) as pool:
        for start in range(0, len(todo), _TIME_BATCH):
            got = list(pool.map(one, todo[start:start + _TIME_BATCH]))
            ok = [(job, t) for job, t in got if t is not None]
            failed += len(got) - len(ok)
            filled += len(ok)
            if ok:
                times = pa.table({
                    "cik": pa.array([j[0] for j, _ in ok], pa.int64()),
                    "accession": [j[1] for j, _ in ok],
                    "accepted_at": pa.array([t for _, t in ok], pa.timestamp("us")),
                })
                with db.connect(db_path) as con:
                    con.register("_times", times)
                    con.execute(
                        "UPDATE sec_edgar_filings f SET accepted_at = t.accepted_at FROM _times t "
                        "WHERE f.cik = t.cik AND f.accession = t.accession"
                    )
                    con.unregister("_times")
            if len(todo) > _TIME_BATCH:
                print(f"  release times: {start + len(got):,}/{len(todo):,}", file=sys.stderr)
    return filled, failed


def fetch_edgar(
    *,
    symbols: list[str] | None = None,
    since: str | None = None,
    issue_types: tuple[str, ...] = ("CS", "DR"),
    db_path=None,
    client: httpx.Client | None = None,
) -> dict:
    """Refresh the EDGAR filings of `symbols` (default: every symbol of `issue_types`), then
    fetch the acceptance time of each results release that has none yet.

    A fetched company's rows from `since` on are replaced by what EDGAR lists now (keeping
    the acceptance times already fetched); older rows, and the rows of a company that fails
    to fetch, are kept. The DB is held only while reading the universe and saving, so other
    commands can use it during the (long) fetch.
    """
    cfg = config.edgar_config()
    fetched_at = datetime.now(UTC)
    with db.connect(db_path) as con:
        if since is None:
            first_bar = con.execute("SELECT min(date) FROM daily_bars").fetchone()[0]
            since_d = (first_bar or tc.ANCHOR) - timedelta(days=_HISTORY_DAYS)
        else:
            since_d = date.fromisoformat(since)
        if symbols:
            universe = [s.strip().upper() for s in symbols]
        else:
            universe = [r[0] for r in con.execute(
                "SELECT symbol FROM symbols WHERE list_contains(?::VARCHAR[], issue_type) ORDER BY 1",
                [list(issue_types)],
            ).fetchall()]

    with nullcontext(client) if client is not None else http.get_client() as c:
        tickers = _get(c, cfg, cfg["tickers_url"])
        by_ticker = {t["ticker"].upper(): t for t in tickers.values()}
        mapped = {s: by_ticker[sec_ticker(s)] for s in universe if sec_ticker(s) in by_ticker}
        ciks = sorted({int(t["cik_str"]) for t in mapped.values()})

        def one(cik: int):
            try:
                return cik, *_company(c, cfg, cik, since_d, fetched_at), None
            except Exception as exc:  # noqa: BLE001 - one bad company must not lose the rest
                return cik, [], 0, exc

        with ThreadPoolExecutor(max_workers=max(1, cfg["workers"])) as pool:
            results = list(pool.map(one, ciks))

        ok = [cik for cik, *_, err in results if err is None]
        failed = {cik: f"{type(err).__name__}: {err}" for cik, *_, err in results if err is not None}
        rows = list({(r["cik"], r["accession"]): r for _, rs, *_ in results for r in rs}.values())
        with db.connect(db_path) as con:
            _store(con, universe, mapped, ok, rows, fetched_at, since_d)
            todo = _pending(con, ok)
        times_filled, times_failed = _fill_times(db_path, c, cfg, todo)

    with db.connect(db_path) as con:
        releases, timed = con.execute(
            f"SELECT count(*), count(accepted_at) FROM sec_edgar_filings WHERE {release_sql()}"
        ).fetchone()

    return {
        "since": since_d.isoformat(),
        "symbols": len(universe),
        "symbols_mapped": len(mapped),
        "unmapped": sorted(set(universe) - set(mapped)),
        "ciks": len(ciks),
        "ciks_failed": failed,
        "extra_files": sum(r[2] for r in results),
        "filings_fetched": len(rows),
        "release_times_fetched": times_filled,
        "release_times_failed": times_failed,
        "releases_total": releases,
        "releases_timed": timed,
    }


def _store(con, universe: list[str], mapped: dict, ok: list[int], rows: list[dict], fetched_at,
           since: date) -> None:
    tick_tbl = pa.table({
        "symbol": list(mapped),
        "cik": pa.array([int(t["cik_str"]) for t in mapped.values()], pa.int64()),
        "sec_ticker": [t["ticker"] for t in mapped.values()],
        "title": [t.get("title") for t in mapped.values()],
        "fetched_at": [fetched_at] * len(mapped),
    })
    filing_tbl = pa.table({
        "cik": pa.array([r["cik"] for r in rows], pa.int64()),
        "accession": pa.array([r["accession"] for r in rows], pa.string()),
        "form": pa.array([r["form"] for r in rows], pa.string()),
        "filing_date": pa.array([r["filing_date"] for r in rows], pa.date32()),
        "report_date": pa.array([r["report_date"] for r in rows], pa.date32()),
        "items": pa.array([r["items"] for r in rows], pa.string()),
        "primary_doc": pa.array([r["primary_doc"] for r in rows], pa.string()),
        "fetched_at": pa.array([r["fetched_at"] for r in rows], pa.timestamp("us", tz="UTC")),
    })
    tables = {
        "_universe": pa.table({"symbol": universe}),
        "_tickers": tick_tbl,
        "_ok": pa.table({"cik": pa.array(ok, pa.int64())}),
        "_filings": filing_tbl,
    }
    cols = ", ".join(_FILING_COLS)
    con.execute("BEGIN TRANSACTION")
    try:
        for name, tbl in tables.items():
            con.register(name, tbl)
        con.execute("DELETE FROM sec_edgar_tickers WHERE symbol IN (SELECT symbol FROM _universe)")
        con.execute("INSERT INTO sec_edgar_tickers SELECT * FROM _tickers")
        con.execute(
            "DELETE FROM sec_edgar_filings f WHERE f.cik IN (SELECT cik FROM _ok) "
            "AND f.filing_date >= ? AND NOT EXISTS "
            "(SELECT 1 FROM _filings n WHERE n.cik = f.cik AND n.accession = f.accession)",
            [since],
        )
        con.execute(
            f"INSERT INTO sec_edgar_filings ({cols}) SELECT {cols} FROM _filings "
            "ON CONFLICT (cik, accession) DO UPDATE SET form = excluded.form, "
            "filing_date = excluded.filing_date, report_date = excluded.report_date, "
            "items = excluded.items, primary_doc = excluded.primary_doc, fetched_at = excluded.fetched_at"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        for name in tables:
            con.unregister(name)
