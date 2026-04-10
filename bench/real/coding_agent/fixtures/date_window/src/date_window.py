"""Date-window membership for the date_window fixture."""

from __future__ import annotations


def in_any_window(target_date: str, windows: list[tuple[str, str]]) -> bool:
    """Return whether target_date falls inside any date window.

    BUGGY ON PURPOSE: this public-passing implementation treats the end
    boundary as exclusive, assumes windows are sorted, and relies on string
    comparison instead of validating calendar dates.
    """
    for start, end in windows:
        if target_date < start:
            return False
        if start <= target_date < end:
            return True
    return False
