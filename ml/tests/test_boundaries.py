"""ML reads only the exported files: no sourcing code, no DuckDB stores, no MTap config."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    re.compile(r"^\s*(import|from)\s+sourcing_py\b", re.M),
    re.compile(r"duckdb\.connect"),
    re.compile(r"shared/config"),
    re.compile(r"[\"'](\./)?state/"),
    re.compile(r"MTAP_(DATA|STATE)_ROOT"),
]


def test_ml_stays_on_its_side():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "test_boundaries.py":
            continue
        text = path.read_text()
        offenders += [f"{path.relative_to(ROOT)}: {rx.pattern}" for rx in FORBIDDEN if rx.search(text)]
    assert offenders == []
