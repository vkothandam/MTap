"""Sourcing never depends on the ML side (ml/ is meant to move to its own repo)."""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "sourcing_py"
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(ml|tft_vwap|tft_intraday|forecast_dash)\b", re.M)


def test_sourcing_does_not_import_ml():
    offenders = [str(p.relative_to(PKG)) for p in PKG.rglob("*.py") if FORBIDDEN.search(p.read_text())]
    assert offenders == []
