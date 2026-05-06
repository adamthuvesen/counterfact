"""Public tests — basic final-release ranges only."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from version_range import satisfies  # noqa: E402


def test_version_inside_range() -> None:
    assert satisfies("1.4.0", [">=1.2.0", "<2.0.0"]) is True


def test_version_below_range() -> None:
    assert satisfies("1.1.9", [">=1.2.0", "<2.0.0"]) is False


def test_exact_match() -> None:
    assert satisfies("2.0.0", ["==2.0.0"]) is True


def test_empty_constraints_accept_valid_version() -> None:
    assert satisfies("0.1.0", []) is True
