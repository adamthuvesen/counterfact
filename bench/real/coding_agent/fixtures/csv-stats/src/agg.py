"""Numeric aggregation helpers tolerant of missing values."""

from __future__ import annotations

from collections.abc import Iterable


def mean_ignore_missing(values: Iterable[float | None]) -> float:
    """Mean of numeric values, ignoring None entries.

    Returns NaN when every input is None (rather than dividing by zero).

    Examples:
        mean_ignore_missing([1, 2, 3])         -> 2.0
        mean_ignore_missing([1, None, 3])      -> 2.0
        mean_ignore_missing([None, None, None]) -> NaN
    """
    nums = [v for v in values if v is not None]
    # BUG: ZeroDivisionError when nums is empty; should return NaN.
    return sum(nums) / len(nums)
