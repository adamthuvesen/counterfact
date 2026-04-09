"""Tests for is_phone_number."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from match import is_phone_number  # noqa: E402


def test_exact_phone_number_accepted() -> None:
    assert is_phone_number("555-123-4567") is True


def test_rejects_extra_prefix() -> None:
    # `re.match` already anchors at the start, so this currently passes.
    # It's here to make sure a fix that flips to `re.search` doesn't regress it.
    assert is_phone_number("x555-123-4567") is False


def test_rejects_extra_suffix() -> None:
    # The bug: `re.match` doesn't anchor at the end, so this returns True.
    assert is_phone_number("555-123-4567 ext 99") is False


def test_rejects_extra_digits_at_end() -> None:
    # Another end-anchor case the model has to think through.
    assert is_phone_number("555-123-45678") is False


def test_rejects_too_short() -> None:
    assert is_phone_number("555-123-456") is False


def test_rejects_empty_string() -> None:
    assert is_phone_number("") is False
