"""Tests for normalize_name."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from normalize import normalize_name  # noqa: E402


def test_strip_and_lowercase() -> None:
    assert normalize_name("  Alice  ") == "alice"


def test_collapses_internal_whitespace_runs() -> None:
    assert normalize_name("Alice  Smith") == "alice smith"
    assert normalize_name("Alice\tSmith") == "alice smith"
