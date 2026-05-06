"""Public tests — basic fixed-window behavior only."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rate_limit import allow_request  # noqa: E402


def test_allows_when_under_limit() -> None:
    assert (
        allow_request(
            "u1",
            100,
            [("u1", 95), ("u1", 99)],
            limit=3,
            window_s=10,
        )
        is True
    )


def test_rejects_when_limit_reached() -> None:
    assert (
        allow_request(
            "u1",
            100,
            [("u1", 95), ("u1", 99)],
            limit=2,
            window_s=10,
        )
        is False
    )


def test_ignores_other_users_and_old_requests() -> None:
    assert (
        allow_request(
            "u1",
            100,
            [("u2", 99), ("u1", 80)],
            limit=1,
            window_s=10,
        )
        is True
    )
