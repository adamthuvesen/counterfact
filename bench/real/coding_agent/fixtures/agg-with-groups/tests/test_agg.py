"""Tests for group_sum over Decimal/float mixes."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agg import group_sum


def test_decimal_only_group_preserves_decimal_precision() -> None:
    out = group_sum([("a", Decimal("0.1")), ("a", Decimal("0.2"))])
    assert out == {"a": Decimal("0.3")}
    assert isinstance(out["a"], Decimal)


def test_float_only_group_returns_float() -> None:
    out = group_sum([("b", 1.5), ("b", 2.5)])
    assert out["b"] == 4.0


def test_mixed_group_does_not_raise() -> None:
    # The bug surfaces here: Decimal + float raises TypeError.
    # Mixed groups should fall back to float arithmetic instead.
    out = group_sum([("c", Decimal("1.5")), ("c", 2.5)])
    assert out["c"] == 4.0


def test_groups_are_independent() -> None:
    out = group_sum(
        [
            ("x", Decimal("1")),
            ("y", 2.0),
            ("x", Decimal("3")),
            ("y", 4.0),
        ]
    )
    assert out["x"] == Decimal("4")
    assert out["y"] == 6.0
