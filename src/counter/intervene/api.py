"""Intervention API (`intervene` + identifiability dispatch).

Full implementation — back-door adjustment, bounded path, replay path,
identifiability labels — lands in tasks §8. This module currently only
enforces the v0 outcome-kind boundary check so the cross-capability tests
in trace-schema can run.
"""

from __future__ import annotations

from typing import Any

from counter.dag import DAG
from counter.errors import UnsupportedOutcomeError


def intervene(
    *,
    dag: DAG,
    model: Any,
    step: int,
    intervention: dict[str, Any],
) -> Any:
    """Run an intervention query and return a CausalEstimate.

    v0 boundary check: the model's outcome kind must be binary. Full body
    (identifiability dispatch, adjustment, sensitivity bounds) lands in §8.
    """
    outcome_kind = getattr(model, "outcome_kind", None)
    if outcome_kind is not None and outcome_kind != "binary":
        raise UnsupportedOutcomeError(
            f"v0 only supports binary outcomes; got model.outcome_kind={outcome_kind!r}"
        )
    raise NotImplementedError(
        "intervene body lands in causal-engine §8 (identifiability dispatch)"
    )
