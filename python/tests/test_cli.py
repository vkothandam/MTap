"""Date-range expansion logic in the CLI (offline)."""

from __future__ import annotations

from datetime import date

import pytest

from sourcing_py.__main__ import _date_range, _parse_params


def test_single_date():
    assert _date_range({"date": "2024-08-16"}) == [date(2024, 8, 16)]


def test_inclusive_range():
    got = _date_range({"fromdate": "2024-08-16", "todate": "2024-08-19"})
    assert got == [date(2024, 8, i) for i in (16, 17, 18, 19)]


def test_no_dates_yields_single_none():
    assert _date_range({}) == [None]


def test_weekdays_only_drops_weekend():
    # 2024-08-16 Fri, 17 Sat, 18 Sun, 19 Mon, 20 Tue
    got = _date_range({"fromdate": "2024-08-16", "todate": "2024-08-20"}, weekdays_only=True)
    assert got == [date(2024, 8, 16), date(2024, 8, 19), date(2024, 8, 20)]


def test_explicit_single_date_honored_even_on_weekend():
    # An explicit single day is never weekend-filtered.
    assert _date_range({"date": "2024-08-17"}, weekdays_only=True) == [date(2024, 8, 17)]


def test_todate_before_fromdate_errors():
    with pytest.raises(ValueError):
        _date_range({"fromdate": "2024-08-19", "todate": "2024-08-16"})


def test_parse_params():
    assert _parse_params(["--fromdate", "2024-08-16", "--todate", "2024-08-17"]) == {
        "fromdate": "2024-08-16",
        "todate": "2024-08-17",
    }
