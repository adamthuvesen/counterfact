"""Hidden tests — edge cases named in spec.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from date_window import in_any_window  # noqa: E402


def test_start_boundary_is_inclusive() -> None:
    assert in_any_window("2024-03-01", [("2024-03-01", "2024-03-31")]) is True


def test_end_boundary_is_inclusive() -> None:
    assert in_any_window("2024-03-31", [("2024-03-01", "2024-03-31")]) is True


def test_unsorted_windows_are_all_checked() -> None:
    windows = [
        ("2024-06-01", "2024-06-30"),
        ("2024-05-01", "2024-05-31"),
    ]
    assert in_any_window("2024-05-10", windows) is True


def test_invalid_range_raises_value_error() -> None:
    with pytest.raises(ValueError):
        in_any_window("2024-03-15", [("2024-03-31", "2024-03-01")])


def test_valid_leap_day_is_inside_inclusive_window() -> None:
    assert in_any_window("2024-02-29", [("2024-02-01", "2024-02-29")]) is True


def test_non_leap_day_is_rejected() -> None:
    with pytest.raises(ValueError):
        in_any_window("2023-02-29", [("2023-02-01", "2023-02-28")])


def test_missing_zero_padding_is_rejected() -> None:
    with pytest.raises(ValueError):
        in_any_window("2024-2-09", [("2024-02-01", "2024-02-29")])
