"""Semantic-version range checks for the version_range fixture."""

from __future__ import annotations


def _parse(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"malformed version: {version!r}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def satisfies(version: str, constraints: list[str]) -> bool:
    """Return whether `version` satisfies all semantic-version constraints.

    BUGGY ON PURPOSE: this handles only final releases and treats all bounds
    as inclusive.
    """
    parsed = _parse(version)
    for constraint in constraints:
        if constraint.startswith(">="):
            if parsed < _parse(constraint[2:]):
                return False
        elif constraint.startswith(">"):
            if parsed < _parse(constraint[1:]):
                return False
        elif constraint.startswith("<="):
            if parsed > _parse(constraint[2:]):
                return False
        elif constraint.startswith("<"):
            if parsed > _parse(constraint[1:]):
                return False
        elif constraint.startswith("=="):
            if parsed != _parse(constraint[2:]):
                return False
        else:
            raise ValueError(f"malformed constraint: {constraint!r}")
    return True
