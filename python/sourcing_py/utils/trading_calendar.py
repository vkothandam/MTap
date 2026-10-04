"""Exchange trading-calendar helpers — a generic, reusable trading-day index.

The one job here is: map real market sessions (NYSE by default) to a contiguous
integer `day_idx`. Calendar dates carry weekend/holiday gaps that would otherwise
look like missing timesteps to a sequence model; a session-only index removes them.

`day_idx` is 1-based from the first session on/after `ANCHOR` (2020-01-01), so
`day_idx == 1` is 2020-01-02 (2020-01-01 is a holiday). Because the anchor is fixed
and sessions are chronological, the index is stable and append-only: adding future
sessions never renumbers past ones.

Weekends AND market holidays (incl. one-off closures) are excluded because the
underlying exchange calendar simply does not list them as sessions. Backed by
`exchange_calendars` (the maintained successor to `trading_calendars`); the XNYS
calendar is authoritative for the NYSE. This module is the only place pandas leaks
in — callers get plain `datetime.date` values and ints.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals

ANCHOR: date = date(2020, 1, 1)  # day_idx == 1 at the first session on/after this date
DEFAULT_CALENDAR: str = "XNYS"  # NYSE


@lru_cache(maxsize=8)
def _calendar(name: str):
    return xcals.get_calendar(name)


def sessions(
    start: date = ANCHOR, end: date | None = None, *, calendar: str = DEFAULT_CALENDAR
) -> list[date]:
    """Sorted trading-day dates in [start, end] inclusive (end defaults to today).

    Weekends and market holidays are excluded because they are not sessions.
    """
    end = end or date.today()
    if end < start:
        return []
    idx = _calendar(calendar).sessions_in_range(str(start), str(end))
    return [ts.date() for ts in idx]


def trading_calendar(
    end: date | None = None, *, anchor: date = ANCHOR, calendar: str = DEFAULT_CALENDAR
) -> list[tuple[date, int]]:
    """[(session_date, day_idx)] for every session in [anchor, end] (end defaults today).

    day_idx starts at 1 for the first session >= anchor and increments by exactly 1
    per session.
    """
    return [(d, i) for i, d in enumerate(sessions(anchor, end, calendar=calendar), start=1)]


def is_trading_day(d: date, *, calendar: str = DEFAULT_CALENDAR) -> bool:
    """True if d is a trading session on the given calendar."""
    return bool(_calendar(calendar).is_session(str(d)))


def next_trading_day(d: date, *, calendar: str = DEFAULT_CALENDAR) -> date:
    """The session on/after d (d itself if it is a session). Used to snap filing dates
    that land on a weekend/holiday forward to the first tradable day."""
    return _calendar(calendar).date_to_session(str(d), direction="next").date()


def day_idx_for(
    d: date, *, anchor: date = ANCHOR, calendar: str = DEFAULT_CALENDAR
) -> int | None:
    """day_idx of session d, or None if d is not a session on/after anchor."""
    if d < anchor or not is_trading_day(d, calendar=calendar):
        return None
    # Count of sessions in [anchor, d] inclusive; d is a session so it's the last one.
    return len(sessions(anchor, d, calendar=calendar))
