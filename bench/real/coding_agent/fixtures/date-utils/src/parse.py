"""ISO date parsing with graceful failure modes."""

from __future__ import annotations

from datetime import date


def parse_date_or_none(s: str | None) -> date | None:
    """Parse YYYY-MM-DD into a date; return None on bad input.

    Examples:
        parse_date_or_none("2026-05-02")  -> date(2026, 5, 2)
        parse_date_or_none("not-a-date")  -> None
        parse_date_or_none(None)          -> None
        parse_date_or_none("")            -> None
    """
    # BUG: only catches ValueError. date.fromisoformat raises TypeError on
    # non-string inputs (None, ints) which leaks past this guard.
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None
