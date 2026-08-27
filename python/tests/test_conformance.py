"""Schema-conformance test — the Python half of shared contract enforcement.

Each source's extract() output must validate against its declared schema.
"""

from __future__ import annotations

from sourcing_py.common import config, schema
from sourcing_py.sources.example_api.source import ExampleApiSource


def test_example_api_records_conform_to_schema():
    source = ExampleApiSource()
    records = source.extract(source.fetch({}))
    declared = schema.load_schema(config.load_source(source.name).schema_path)
    # Raises ValueError with details if any record fails.
    schema.validate_records(records, declared)


def test_example_api_produces_records():
    source = ExampleApiSource()
    records = source.extract(source.fetch({}))
    assert len(records) > 0
    assert all("id" in r for r in records)
