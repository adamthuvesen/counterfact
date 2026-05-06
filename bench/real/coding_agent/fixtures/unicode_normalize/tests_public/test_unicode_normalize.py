"""Public tests — common ASCII duplicate behavior only."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from unicode_normalize import dedupe_normalized  # noqa: E402


def test_exact_duplicate_is_removed() -> None:
    assert dedupe_normalized(["alpha", "beta", "alpha"]) == ["alpha", "beta"]


def test_first_occurrence_order_is_stable() -> None:
    assert dedupe_normalized(["beta", "alpha", "beta"]) == ["beta", "alpha"]


def test_surrounding_whitespace_is_cleaned() -> None:
    assert dedupe_normalized(["  alpha  ", "alpha"]) == ["alpha"]
