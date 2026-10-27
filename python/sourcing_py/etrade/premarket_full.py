"""The 09:45 full pre-market (`sourcing-py etrade premarket full --date D`).

Massive 1-minute bars of day d for every panel symbol, 64 calls at a time (`[premarket]
full_workers`), summarised by massive_ext's own code (extended.summarize), so the definitions are
the training ones: bars by START time before the cutoff, `vw`-weighted VWAP, the last bar that
traded (v > 0) for the last price, high and low of the traded bars, and the volume.

Massive's feed is `feed_delay_min` behind, so the window is only whole once now - delay >= cutoff
(09:45 for 09:30). Before that the command refuses unless --force; with --force the records carry
`complete = false` and export-next leaves their group masked.

Writes premarket_live/premarket-D.jsonl (+ premarket-D.summary.json), atomically.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from ..common import config, http, schema, writer
from ..common.errors import ConfigError
from ..massive_ext import extended as ext
from .premarket import cutoff_utc

SOURCE = "massive"


class TooEarly(Exception):
    """Massive does not yet reach the cutoff."""


def cutoff_key(cutoff: str) -> str:
    key = cutoff.replace(":", "")
    if key not in ext.CUTOFFS:
        raise ConfigError(f"cutoff {cutoff} is not one of {sorted(ext.CUTOFFS)} (massive-ext's cut-offs)")
    return key


def ready_at(d: date, cutoff: str, feed_delay_min: int) -> datetime:
    return cutoff_utc(d, cutoff) + timedelta(minutes=feed_delay_min)


def record(symbol: str, d: date, cutoff: str, day: dict | None, fetched_at: str) -> dict:
    """One premarket-D.jsonl record from massive-ext's per-day summary (None: no bars)."""
    k = cutoff_key(cutoff)
    day = day or {}
    vol = day.get(f"pre_market_volume_{k}")
    return {
        "symbol": symbol, "date": d.isoformat(), "cutoff": cutoff,
        "pm_vwap": day.get(f"pre_market_vwap_{k}"), "pm_volume": vol if vol is not None else 0.0,
        "pm_last": day.get(f"pm_last_{k}"), "pm_last_t": day.get(f"pm_last_t_{k}"),
        "pm_high": day.get(f"pm_high_{k}"), "pm_low": day.get(f"pm_low_{k}"),
        "n_bars": int(day.get("minute_bars") or 0), "source": SOURCE, "fetched_at": fetched_at,
    }


def fetch_symbol(client, cfg: dict, symbol: str, d: date, cutoff: str) -> dict:
    fetched_at = datetime.now(UTC).isoformat()
    rows = ext.summarize(symbol, ext._pages(client, cfg, symbol, d, d), fetched_at)
    day = next((r for r in rows if r["date"] == d.isoformat()), None)
    return record(symbol, d, cutoff, day, fetched_at)


def mark_complete(records: list[dict], d: date, cutoff: str, now: datetime, feed_delay_min: int) -> str:
    """data_until = now - feed delay; complete = data_until >= cutoff (the ec5520b rule)."""
    until = now - timedelta(minutes=feed_delay_min)
    ok = until >= cutoff_utc(d, cutoff)
    for r in records:
        r["data_until"] = until.strftime("%Y-%m-%dT%H:%M:%SZ")
        r["complete"] = ok
    return until.strftime("%Y-%m-%dT%H:%M:%SZ")


def output_path(d: date) -> Path:
    return config.out_root() / "premarket_live" / f"premarket-{d.isoformat()}.jsonl"


def write_records(records: list[dict], d: date) -> Path:
    enriched = writer._with_provenance(records, "etrade_premarket", writer._run_id())
    path = output_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in enriched))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def panel_symbols(panel_next: Path | None = None) -> list[str]:
    from .precheck import read_panel_next

    syms, _ = read_panel_next(panel_next or config.out_root() / "tft" / "panel.next.parquet")
    return syms


def run_full(d: date, symbols: list[str], *, cutoff: str | None = None, force: bool = False,
             now: datetime | None = None, client=None, cfg: dict | None = None) -> dict:
    pcfg = config.premarket_config()
    cutoff = cutoff or pcfg["cutoff"]
    cutoff_key(cutoff)
    now = now or datetime.now(UTC)
    ready = ready_at(d, cutoff, pcfg["feed_delay_min"])
    if now < ready and not force:
        raise TooEarly(f"Massive reaches {cutoff} ET only at {ready.astimezone(ext.NY):%H:%M} ET "
                       f"({pcfg['feed_delay_min']} min feed delay); now is {now.astimezone(ext.NY):%H:%M} ET. "
                       "Use --force to run anyway (records are then complete = false)")
    cfg = dict(cfg or config.massive_config())
    cfg["rate_limit_per_min"] = 0  # a paid key; the 64 workers are the limit
    workers = pcfg["full_workers"]
    records: dict[str, dict] = {}
    errors: dict[str, str] = {}

    def one(c, s):
        try:
            return s, fetch_symbol(c, cfg, s, d, cutoff), None
        except Exception as exc:  # per-symbol failure: counted, the rest still run
            return s, None, f"{type(exc).__name__}: {exc}"

    def run(c):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for s, rec, err in pool.map(lambda s: one(c, s), symbols):
                if rec:
                    records[s] = rec
                else:
                    errors[s] = err

    if client is not None:
        run(client)
    else:
        with http.get_client(timeout=60, limits=httpx.Limits(max_connections=workers + 4)) as c:
            run(c)
    out = [records[s] for s in symbols if s in records]
    data_until = mark_complete(out, d, cutoff, now, pcfg["feed_delay_min"])
    schema.validate_records(out, schema.load_schema(config.repo_root() / "schemas" / "etrade_premarket.schema.json"))
    path = write_records(out, d)
    summary = {
        "date": d.isoformat(), "cutoff": cutoff, "symbols": len(symbols), "written": len(out),
        "with_bars": sum(1 for r in out if r["n_bars"]), "errors": len(errors),
        "complete": sum(1 for r in out if r["complete"]), "data_until": data_until,
        "forced": bool(force and now < ready), "path": str(path),
    }
    path.with_name(f"premarket-{d.isoformat()}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary
