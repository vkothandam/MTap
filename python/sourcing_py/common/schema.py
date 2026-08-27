"""Schema loading and record conformance validation.

The same JSON Schema files under `schemas/` govern both the Python and Node stacks,
so conformance here is the Python half of the shared contract enforcement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

# Provenance fields the writer injects on every record (see schemas/conventions.md).
PROVENANCE_FIELDS = ("_source", "_ingested_at", "_run_id")


def load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(Path(schema_path).read_text())


def validate_records(records: list[dict], schema: dict[str, Any]) -> None:
    """Raise ValueError listing every validation failure, or return None if all pass.

    Provenance fields are allowed even when the schema sets additionalProperties=false.
    """
    schema = _allow_provenance(schema)
    validator = Draft7Validator(schema)
    errors: list[str] = []
    for i, record in enumerate(records):
        for err in validator.iter_errors(record):
            errors.append(f"record[{i}]: {err.message}")
    if errors:
        raise ValueError("Schema conformance failed:\n" + "\n".join(errors))


def _allow_provenance(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("additionalProperties") is False and "properties" in schema:
        schema = dict(schema)
        props = dict(schema["properties"])
        for field in PROVENANCE_FIELDS:
            props.setdefault(field, {"type": "string"})
        schema["properties"] = props
    return schema
