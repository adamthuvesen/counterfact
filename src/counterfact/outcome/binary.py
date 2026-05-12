"""Strict binary outcome helpers for causal/statistical paths."""

from __future__ import annotations

from counterfact.errors import UnsupportedOutcomeError
from counterfact.schema import Run


def binary_outcome_value(run: Run) -> bool:
    """Return `Outcome.value` for binary runs; reject other outcome kinds."""
    if run.outcome.kind != "binary":
        raise UnsupportedOutcomeError(
            f"v0 only supports binary outcomes; got kind={run.outcome.kind!r} on run {run.run_id!r}"
        )
    value = run.outcome.value
    if not isinstance(value, bool):
        raise UnsupportedOutcomeError(
            f"binary outcome on run {run.run_id!r} must be bool; got {type(value).__name__}"
        )
    return value


__all__ = ["binary_outcome_value"]
