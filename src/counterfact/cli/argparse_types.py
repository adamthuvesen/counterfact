"""argparse type converters."""

from __future__ import annotations

import argparse


def positive_int(raw: str) -> int:
    """argparse type for ints that must be >= 1 (bootstrap counts)."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1; got {value}")
    return value
