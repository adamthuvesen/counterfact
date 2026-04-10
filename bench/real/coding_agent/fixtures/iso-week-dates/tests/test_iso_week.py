"""Tests for ISO 8601 week conversions."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iso_week import date_to_iso_week, iso_week_to_date

# --- iso_week_to_date ------------------------------------------------------


def test_first_iso_week_can_start_in_previous_calendar_year() -> None:
    # ISO week 1 of 2026 starts Monday 2025-12-29 because Jan 1, 2026 is Thursday.
    assert iso_week_to_date(2026, 1, 1) == date(2025, 12, 29)
    assert iso_week_to_date(2026, 1, 4) == date(2026, 1, 1)


def test_last_iso_week_of_2020_is_53() -> None:
    # 2020 is one of the years with 53 ISO weeks; week 53 starts Monday 2020-12-28.
    assert iso_week_to_date(2020, 53, 1) == date(2020, 12, 28)
    assert iso_week_to_date(2020, 53, 7) == date(2021, 1, 3)


def test_late_calendar_year_belongs_to_next_iso_year() -> None:
    # 2017-01-01 (a Sunday) is in ISO week 52 of 2016.
    assert iso_week_to_date(2016, 52, 7) == date(2017, 1, 1)


def test_simple_midyear_date() -> None:
    # ISO week 24 of 2026 — well away from year boundaries; sanity case.
    assert iso_week_to_date(2026, 24, 3) == date(2026, 6, 10)


# --- date_to_iso_week (round-trip; this side already works) ----------------


def test_date_to_iso_week_round_trips() -> None:
    samples = [date(2025, 12, 29), date(2026, 6, 10), date(2020, 12, 28), date(2017, 1, 1)]
    for d in samples:
        y, w, wd = date_to_iso_week(d)
        assert iso_week_to_date(y, w, wd) == d, f"round-trip failed for {d}"
