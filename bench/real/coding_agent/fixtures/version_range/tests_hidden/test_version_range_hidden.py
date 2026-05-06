"""Hidden tests — edge cases named in spec.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from version_range import satisfies  # noqa: E402


def test_strict_lower_bound_excludes_equal_version() -> None:
    assert satisfies("1.2.0", [">1.2.0"]) is False


def test_strict_upper_bound_excludes_equal_version() -> None:
    assert satisfies("2.0.0", ["<2.0.0"]) is False


def test_prerelease_compares_below_final_release() -> None:
    assert satisfies("1.2.0-rc1", [">=1.2.0"]) is False


def test_prerelease_can_satisfy_lower_prerelease_bound() -> None:
    assert satisfies("1.2.0-rc2", [">1.2.0-rc1", "<1.2.0"]) is True


def test_dot_separated_numeric_prerelease_identifiers_compare_numerically() -> None:
    assert satisfies("1.2.0-rc.10", [">1.2.0-rc.2", "<1.2.0"]) is True


def test_shorter_prerelease_list_compares_lower_when_prefix_matches() -> None:
    assert satisfies("1.2.0-alpha", ["<1.2.0-alpha.1"]) is True


def test_numeric_prerelease_identifier_compares_lower_than_text_identifier() -> None:
    assert satisfies("1.2.0-1", ["<1.2.0-alpha"]) is True


def test_empty_prerelease_identifier_raises_value_error() -> None:
    with pytest.raises(ValueError):
        satisfies("1.2.0-rc..1", ["<1.2.0"])


def test_malformed_version_raises_value_error() -> None:
    with pytest.raises(ValueError):
        satisfies("1.2", [">=1.0.0"])


def test_malformed_constraint_raises_value_error() -> None:
    with pytest.raises(ValueError):
        satisfies("1.2.0", ["~1.0.0"])
