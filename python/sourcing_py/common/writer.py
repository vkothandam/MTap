"""Output writer: injects provenance, applies naming conventions, emits jsonl/parquet.

Implements the rules in schemas/conventions.md so no source has to reinvent them.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from . import config


def _run_id() -> str:
    # Timestamp-based run id, UTC, filename-safe.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _with_provenance(records: list[dict], source: str, run_id: str) -> list[dict]:
    ingested_at = datetime.now(timezone.utc).isoformat()
    return [
        {**r, "_source": source, "_ingested_at": ingested_at, "_run_id": run_id}
        for r in records
    ]


def _output_path(dataset: str, fmt: str, stamp: str) -> Path:
    ext = "jsonl" if fmt == "jsonl" else "parquet"
    # Flat: one folder per info-type, the date (or run stamp) in the filename.
    return config.out_root() / dataset / f"{dataset}-{stamp}.{ext}"


def write(
    records: list[dict],
    *,
    source: str,
    dataset: str,
    fmt: str,
    dt: date | None = None,
) -> Path:
    """Write records for `source` under folder `dataset` in `fmt`. Returns the path.

    Filename carries the logical date (`dt`) when known — so re-running a day
    overwrites it idempotently — otherwise a run timestamp.
    """
    run_id = _run_id()
    stamp = dt.isoformat() if dt else run_id
    enriched = _with_provenance(records, source, run_id)
    path = _output_path(dataset, fmt, stamp)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        _write_jsonl(enriched, path)
    elif fmt == "parquet":
        _write_parquet(enriched, path)
    else:
        raise ValueError(f"Unsupported format {fmt!r}")
    return path


def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str))
            f.write("\n")


def _write_parquet(records: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="snappy")
