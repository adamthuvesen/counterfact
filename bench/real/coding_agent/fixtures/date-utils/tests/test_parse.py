"""Tests for parse_date_or_none."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parse import parse_date_or_none


def test_valid_iso_date() -> None:
    assert parse_date_or_none("2026-05-02") == date(2026, 5, 2)


def test_invalid_string_returns_none() -> None:
    assert parse_date_or_none("not-a-date") is None


def test_none_input_returns_none() -> None:
    # parse_date_or_none must handle None gracefully without raising TypeError.
    assert parse_date_or_none(None) is None
