"""Base class for a source. Subclasses implement `fetch` and `extract`.

Lifecycle: fetch() -> raw payload; extract() -> list of records; run() validates
against the schema and writes via the shared writer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

from . import config, schema, writer


class Source(ABC):
    #: Registry name; must match a key in shared/config/sources.json.
    name: str = ""

    #: If True, the CLI drops Saturdays/Sundays from a --fromdate/--todate range
    #: (no request made). For markets that don't trade on weekends.
    weekdays_only: bool = False

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a class-level `name`")
        self.config = config.load_source(self.name)

    @abstractmethod
    def fetch(self, params: dict[str, Any]) -> Any:
        """Retrieve raw data (call an API, download a page). Return the raw payload.

        `params` carries per-run inputs (e.g. {"date": "2024-08-15"}) from the CLI.
        """

    @abstractmethod
    def extract(self, raw: Any) -> list[dict]:
        """Turn the raw payload into a list of records conforming to the schema."""

    def partition_date(self, params: dict[str, Any]) -> date | None:
        """Logical date embedded in the output filename.

        Defaults to None (writer uses a run timestamp). Date-based sources should
        override to return the data's own date, so files are named by trading date.
        """
        return None

    def cadence_hint(self) -> str | None:
        """Optional one-line description of request pacing, printed before a run."""
        return None

    def preflight(self) -> None:
        """Validate config/credentials once before the run. Raise ConfigError to abort.

        Default is a no-op. Override to fail fast on missing keys instead of failing
        every day in a long range.
        """
        return None

    def run(
        self,
        *,
        params: dict[str, Any] | None = None,
        validate: bool = True,
        skip_empty: bool = True,
    ) -> Path | None:
        params = params or {}
        raw = self.fetch(params)
        records = self.extract(raw)
        if not records and skip_empty:
            return None  # e.g. a market holiday / weekend — nothing to write
        if validate:
            schema.validate_records(records, schema.load_schema(self.config.schema_path))
        return writer.write(
            records,
            source=self.name,
            dataset=self.config.dataset,
            fmt=self.config.format,
            dt=self.partition_date(params),
        )
