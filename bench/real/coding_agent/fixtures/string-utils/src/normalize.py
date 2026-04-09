"""Name normalization for case-insensitive comparison."""

from __future__ import annotations


def normalize_name(name: str) -> str:
    """Strip whitespace, lowercase, and collapse internal whitespace runs.

    Examples:
        normalize_name("  Alice  ")      -> "alice"
        normalize_name("Alice  Smith")    -> "alice smith"
        normalize_name("alice\\tsmith")    -> "alice smith"
    """
    # BUG: doesn't collapse internal whitespace runs.
    return name.strip().lower()
