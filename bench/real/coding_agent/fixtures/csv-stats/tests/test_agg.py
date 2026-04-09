"""Tests for mean_ignore_missing."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agg import mean_ignore_missing  # noqa: E402


def test_simple_mean() -> None:
    assert mean_ignore_missing([1, 2, 3]) == 2.0


def test_skips_none_entries() -> None:
    assert mean_ignore_missing([1, None, 3]) == 2.0


def test_all_missing_returns_nan() -> None:
    # When every input is None, mean should be NaN, not raise ZeroDivisionError.
    result = mean_ignore_missing([None, None, None])
    assert math.isnan(result)
