"""Market series for the intraday model: SPY from Massive, SPX daily from a file.

    sourcing-py market fetch --fromdate D1 --todate D2 [--spx-csv PATH]

writes, under OUT_ROOT/market/ (each file replaced whole):

    spy_daily.parquet       date, o, h, l, c, v, vw, n        (adjusted=true: splits only)
    spy_1m.parquet          t (epoch ms), o, h, l, c, v, vw, n (04:00-20:00 ET, like bars_1m)
    spy_dividends.parquet   ex_date, cash, pay_date, kind     (every SPY dividend Massive has)
    spx_daily.parquet       date, o, h, l, c                  (from the CSV, default
                                                               docs/transient/spx-history.csv)

SPY is not in bars_1m (not a CS) and I:SPX data is 403 on this plan, so SPX comes from the
file. Massive's adjusted=true does not adjust dividends; the feature code adds the dividend
back on ex-dates (docs/intraday-features.md §4). D1 before the 5-year edge returns 403.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from .common import config, http
from .common.errors import ConfigError, FetchError
from .intraday.cli import _parse
from .massive_ext.extended import NY

SPX_CSV = Path(__file__).resolve().parents[2] / "docs" / "transient" / "spx-history.csv"
_USAGE = "usage:\n  sourcing-py market fetch --fromdate D1 --todate D2 [--spx-csv PATH]"
_BAR = ("o", "h", "l", "c", "v", "vw", "n")


def _get_all(client: httpx.Client, cfg: dict, url: str, params: dict) -> list[dict]:
    """Every result of a paged Massive endpoint (next_url keeps its cursor; the key is merged in)."""
    out: list[dict] = []
    params = {**params, "apiKey": cfg["api_key"]}
    while url:
        page = http.get_json(client, url, params=params)
        out += page.get("results") or []
        nxt = page.get("next_url")
        url, params = (str(httpx.URL(nxt).copy_merge_params({"apiKey": cfg["api_key"]})), None) if nxt else ("", None)
    return out


def _aggs(client, cfg, span: str, d0: date, d1: date) -> list[dict]:
    return _get_all(client, cfg, f"{cfg['base_url']}/v2/aggs/ticker/SPY/range/1/{span}/{d0}/{d1}",
                    {"adjusted": "true", "sort": "asc", "limit": 50000})


def _bars(rows: list[dict]) -> dict:
    return {k: [r.get(k) for r in rows] for k in _BAR}


def spy_tables(client: httpx.Client, cfg: dict, d0: date, d1: date) -> dict[str, pa.Table]:
    day = _aggs(client, cfg, "day", d0, d1)
    minute = _aggs(client, cfg, "minute", d0, d1)
    divs = _get_all(client, cfg, f"{cfg['base_url']}/v3/reference/dividends",
                    {"ticker": "SPY", "limit": 1000, "sort": "ex_dividend_date", "order": "asc"})
    as_date = [datetime.fromtimestamp(r["t"] / 1000, NY).date() for r in day]
    num = {k: pa.float64() for k in ("o", "h", "l", "c", "v", "vw")} | {"n": pa.int64()}
    return {
        "spy_daily": pa.table({"date": pa.array(as_date, pa.date32()),
                               **{k: pa.array(v, num[k]) for k, v in _bars(day).items()}}),
        "spy_1m": pa.table({"t": pa.array([r["t"] for r in minute], pa.int64()),
                            **{k: pa.array(v, num[k]) for k, v in _bars(minute).items()}}),
        "spy_dividends": pa.table({
            "ex_date": pa.array([date.fromisoformat(r["ex_dividend_date"]) for r in divs], pa.date32()),
            "cash": pa.array([r.get("cash_amount") for r in divs], pa.float64()),
            "pay_date": pa.array([date.fromisoformat(r["pay_date"]) if r.get("pay_date") else None
                                  for r in divs], pa.date32()),
            "kind": pa.array([r.get("dividend_type") or "" for r in divs]),
        }),
    }


def spx_table(path: Path) -> pa.Table:
    """The SPX daily file (DT,Open,High,Low,Close,Volume; Volume is always 0, so dropped)."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    cols = {"o": "Open", "h": "High", "l": "Low", "c": "Close"}
    return pa.table({"date": pa.array([date.fromisoformat(r["DT"]) for r in rows], pa.date32()),
                     **{k: pa.array([float(r[c]) for r in rows], pa.float64()) for k, c in cols.items()}})


def write(tables: dict[str, pa.Table], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        tmp = out / f".{name}.parquet.tmp"
        pq.write_table(t, tmp, compression="zstd")
        os.replace(tmp, out / f"{name}.parquet")


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "fetch":
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        params = _parse(argv[1:])
        d0, d1 = date.fromisoformat(params["fromdate"]), date.fromisoformat(params["todate"])
        if d1 < d0:
            raise ValueError(f"todate {d1} is before fromdate {d0}")
        unknown = set(params) - {"fromdate", "todate", "spx-csv"}
        if unknown:
            raise ValueError(f"unknown option(s): {', '.join(sorted(unknown))}")
        cfg = config.massive_config()
        with http.get_client() as client:
            tables = spy_tables(client, cfg, d0, d1)
        tables["spx_daily"] = spx_table(Path(params.get("spx-csv") or SPX_CSV))
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2
    except (ConfigError, FetchError, FileNotFoundError) as exc:
        print(f"error: {str(exc).replace(cfg['api_key'], '***') if 'cfg' in locals() else exc}", file=sys.stderr)
        return 1
    out = config.out_root() / "market"
    write(tables, out)
    for name, t in tables.items():
        first, last = (t.column(0)[0], t.column(0)[-1]) if t.num_rows else ("-", "-")
        if name == "spy_1m" and t.num_rows:
            first, last = (datetime.fromtimestamp(x.as_py() / 1000, NY).date() for x in (first, last))
        print(f"{name}: {t.num_rows:,} rows, {first} .. {last}")
    print(f"wrote {out}")
    return 0
