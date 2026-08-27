"""Example API source. Copy this as the template for a real one.

Replace fetch() with your real API call and extract() with your real field mapping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sourcing_py.common.source import Source


class ExampleApiSource(Source):
    name = "example_api"

    def fetch(self, params: Any) -> Any:
        # Real version:
        #   from sourcing_py.common import http
        #   with http.get_client() as client:
        #       return http.get_json(client, "https://api.example.com/things")
        return [
            {"id": 1, "name": "alpha", "score": 9.5, "active": True},
            {"id": 2, "name": "beta", "score": 3.2, "active": False},
        ]

    def extract(self, raw: Any) -> list[dict]:
        now = datetime.now(timezone.utc).isoformat()
        return [
            {
                "id": item["id"],
                "name": item["name"],
                "score": item.get("score"),
                "active": item.get("active"),
                "fetched_at": now,
            }
            for item in raw
        ]
