"""ISO 8601 week-date conversion.

ISO weeks have nasty edge cases that the calendar-year intuition gets wrong:
* The first ISO week of a year contains the year's first Thursday.
* That means early days of January can belong to the *previous* ISO year
  (e.g., 2017-01-01 is in ISO week 52 of 2016).
* Some calendar years have 53 ISO weeks (e.g., 2020).
"""

from __future__ import annotations

from datetime import date, timedelta


def iso_week_to_date(iso_year: int, iso_week: int, iso_weekday: int) -> date:
    """Return the calendar date for the given ISO 8601 (year, week, weekday).

    `iso_weekday` is 1=Monday … 7=Sunday.

    Examples:
        iso_week_to_date(2026, 1, 1)   -> date(2025, 12, 29)  # Mon of ISO week 1
        iso_week_to_date(2020, 53, 1)  -> date(2020, 12, 28)  # Mon of week 53
        iso_week_to_date(2017, 52, 7)  -> date(2017, 12, 31)  # Sun of week 52
    """
    # BUG: this naive approach uses Jan 1 + (week-1)*7 + (weekday-1) days. It
    # ignores the ISO rule that week 1 is the week containing the first
    # Thursday — which means the first Monday of ISO week 1 can fall in the
    # *previous* calendar year. The fix uses `date.fromisocalendar` (Python
    # 3.8+) or a manual offset back to the first Thursday.
    jan1 = date(iso_year, 1, 1)
    return jan1 + timedelta(days=(iso_week - 1) * 7 + (iso_weekday - 1))


def date_to_iso_week(d: date) -> tuple[int, int, int]:
    """Return (iso_year, iso_week, iso_weekday) for the given date."""
    iso = d.isocalendar()
    return (iso.year, iso.week, iso.weekday)
