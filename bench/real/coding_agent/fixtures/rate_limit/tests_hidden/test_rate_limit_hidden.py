"""Hidden tests — edge cases named in spec.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rate_limit import allow_request  # noqa: E402


def test_lower_boundary_is_inclusive() -> None:
    assert (
        allow_request(
            "u1",
            100,
            [("u1", 90), ("u1", 99)],
            limit=2,
            window_s=10,
        )
        is False
    )


def test_unsorted_history_is_fully_checked() -> None:
    history = [("u1", 99), ("u1", 70), ("u1", 95)]
    assert allow_request("u1", 100, history, limit=2, window_s=10) is False


def test_old_first_history_entry_does_not_stop_scan() -> None:
    history = [("u1", 70), ("u1", 99), ("u1", 95)]
    assert allow_request("u1", 100, history, limit=2, window_s=10) is False


def test_duplicate_same_second_requests_count_separately() -> None:
    history = [("u1", 99), ("u1", 99)]
    assert allow_request("u1", 100, history, limit=2, window_s=10) is False


def test_future_history_timestamp_raises_value_error() -> None:
    with pytest.raises(ValueError):
        allow_request("u1", 100, [("u1", 101)], limit=2, window_s=10)


def test_future_history_timestamp_is_rejected_before_returning() -> None:
    with pytest.raises(ValueError):
        allow_request(
            "u1",
            100,
            [("u1", 95), ("u1", 99), ("u2", 101)],
            limit=2,
            window_s=10,
        )


def test_invalid_limit_raises_value_error() -> None:
    with pytest.raises(ValueError):
        allow_request("u1", 100, [], limit=0, window_s=10)


def test_invalid_window_raises_value_error() -> None:
    with pytest.raises(ValueError):
        allow_request("u1", 100, [], limit=2, window_s=0)
