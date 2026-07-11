"""Single-source the degenerate (single-outcome-class) refusal path.

`counterfact` must surface the degenerate case as `unidentified` with a
concrete `next_step`, never silently fit a one-class outcome model. Both the
`counterfact demo` CLI and the `counterfact explain` report depend on this,
so the construction lives here rather than being duplicated in each caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from counterfact.baselines import pass_rate_by_arm
from counterfact.intervene.estimate import CausalEstimate, InterventionQuery
from counterfact.intervene.support import build_broaden_arm_support_estimate, missing_arms_for
from counterfact.outcome.binary import binary_outcome_value
from counterfact.schema import Run


def outcome_classes(runs: Iterable[Run]) -> set[bool]:
    """Distinct boolean outcome values across `runs` (binary outcomes only)."""
    return {binary_outcome_value(run) for run in runs}


def degenerate_estimate(
    runs: list[Run],
    *,
    decision_type: str,
    intervention_kind: str,
    target: Any | None = None,
) -> CausalEstimate:
    """Return the canonical `unidentified` refusal for a single-class corpus.

    Caller must guarantee `len(outcome_classes(runs)) == 1`.
    """
    classes = outcome_classes(runs)
    if len(classes) != 1:
        raise ValueError("degenerate estimate requires exactly one outcome class")
    observed = next(iter(classes))

    table = pass_rate_by_arm(runs, decision_type)
    observed_arm_names = [row.arm for row in table.rows]

    return build_broaden_arm_support_estimate(
        InterventionQuery(
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
            step=None,
        ),
        reason=(
            "real corpus is causally degenerate: every trace has "
            f"Outcome.value={observed}; no outcome variation exists for an outcome "
            "model or back-door adjustment to leverage"
        ),
        observed_rows=table.rows,
        missing_arms=missing_arms_for(decision_type, intervention_kind, observed_arm_names),
        missing_strata=[f"Outcome.value={not observed}"],
        warnings=["fit_outcome_model is intentionally skipped for single-class real corpora"],
        human_text=(
            "Collect or construct traces with both pass and fail outcomes before "
            "estimating decision-level effects on the real corpus."
        ),
        arm_name_for_suggestion=str(target) if target is not None else None,
        payload_arm_name="outcome",
    )
