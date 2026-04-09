"""Public tests — exact-string duplicates only."""

from __future__ import annotations

import sys
from pathlib import Path

# Tests run from the fixture root; src/ holds the module under test.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dedupe import dedupe  # noqa: E402


def test_exact_duplicates_collapse_to_one() -> None:
    assert dedupe(["a", "a"]) == ["a"]


def test_no_duplicates_returns_input_unchanged() -> None:
    assert dedupe(["a", "b", "c"]) == ["a", "b", "c"]


def test_empty_input_returns_empty() -> None:
    assert dedupe([]) == []


def test_first_occurrence_is_preserved() -> None:
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
