"""Public tests — basic sorted-window behavior only."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from date_window import in_any_window  # noqa: E402


def test_date_inside_single_sorted_window() -> None:
    assert in_any_window("2024-03-15", [("2024-03-01", "2024-03-31")]) is True


def test_date_before_single_sorted_window() -> None:
    assert in_any_window("2024-02-28", [("2024-03-01", "2024-03-31")]) is False


def test_date_after_single_sorted_window() -> None:
    assert in_any_window("2024-04-01", [("2024-03-01", "2024-03-31")]) is False


def test_empty_windows_returns_false() -> None:
    assert in_any_window("2024-03-15", []) is False
