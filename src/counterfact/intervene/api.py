"""Intervention API + identifiability dispatch.

For each (decision_type, intervention_kind) pair the taxonomy declares an
identifiability stance. `intervene` dispatches on that stance:

* `requires-randomized-support`: g-formula via the fitted outcome model —
  set the targeted feature to 1 in every training row, zero out sibling-arm
  features, predict, average. Returns `identified` with point + bootstrap CI.
* `requires-back-door-adjustment`: returns `bounded` and names the adjustment
  set. When observed support is missing, `bounds` is `None` and `next_step`
  explains how to broaden support.
* `always-replay`: returns `unidentified` with `next_step="replay"`.

The CausalEstimate result object is the public contract; full schema lives in
`counterfact.intervene.estimate`.
"""

from __future__ import annotations

from typing import Any

from counterfact.dag import DAG
from counterfact.errors import InvalidInterventionError, UnsupportedOutcomeError
from counterfact.intervene.dispatch import (
    _STANCE_HANDLERS,
    InterveneContext,
    handle_duplicate_step_refusal,
)
from counterfact.intervene.estimate import CausalEstimate, InterventionQuery
from counterfact.outcome.features import canonical_intervention_value, intervention_feature_family
from counterfact.outcome.model import OutcomeModel
from counterfact.taxonomy import identifiability_stance, is_valid_intervention
from counterfact.trace_localization import duplicate_decision_type_steps


def _decision_type_at_step(dag: DAG, step: int, *, decision_id: str | None = None) -> str:
    if dag.run is None:
        raise InvalidInterventionError("DAG has no associated trace")
    for s in dag.run.steps:
        if s.step_index == step:
            if decision_id is not None:
                for decision in s.decisions:
                    if decision.decision_id == decision_id:
                        return decision.decision_type
                raise InvalidInterventionError(
                    f"decision_id {decision_id!r} not found on step {step}"
                )
            if not s.decisions:
                raise InvalidInterventionError(f"step {step} has no decisions")
            if len(s.decisions) > 1:
                raise InvalidInterventionError(
                    f"step {step} has multiple decisions; the v0 step-scoped "
                    "intervene API cannot target it unambiguously"
                )
            return s.decisions[0].decision_type
    raise InvalidInterventionError(f"step {step} not found in trace")


def _resolve_intervention(intervention: dict[str, Any]) -> tuple[str, Any]:
    if not intervention or len(intervention) != 1:
        raise InvalidInterventionError(
            "intervention must be a single-key dict like {'tool_choice': 'inspect_file'}"
        )
    kind, value = next(iter(intervention.items()))
    return kind, value


def intervene(
    *,
    dag: DAG,
    model: OutcomeModel,
    step: int,
    intervention: dict[str, Any],
    decision_id: str | None = None,
) -> CausalEstimate:
    """Run an intervention query and return a CausalEstimate."""
    if model.outcome_kind != "binary":
        raise UnsupportedOutcomeError(
            f"v0 only supports binary outcomes; got model.outcome_kind={model.outcome_kind!r}"
        )

    intervention_kind, target_value = _resolve_intervention(intervention)
    decision_type = _decision_type_at_step(dag, step, decision_id=decision_id)

    if not is_valid_intervention(decision_type, intervention_kind):
        raise InvalidInterventionError(
            f"intervention {intervention_kind!r} is not valid on decision type {decision_type!r}"
        )

    query = InterventionQuery(
        decision_type=decision_type,
        intervention_kind=intervention_kind,
        target=target_value,
        step=step,
    )
    feature_family = intervention_feature_family(decision_type, intervention_kind)
    target_arm_name = canonical_intervention_value(intervention_kind, target_value)

    stance = identifiability_stance(decision_type, intervention_kind)
    ctx = InterveneContext(
        dag=dag,
        model=model,
        step=step,
        query=query,
        intervention_kind=intervention_kind,
        target_value=target_value,
        feature_family=feature_family,
        target_arm_name=target_arm_name,
        decision_type=decision_type,
    )

    if stance == "always-replay":
        return _STANCE_HANDLERS[stance](ctx)

    if dag.run is not None and stance == "requires-randomized-support":
        duplicate_steps = duplicate_decision_type_steps(
            dag.run, except_step=step, decision_type=decision_type
        )
        if duplicate_steps:
            return handle_duplicate_step_refusal(ctx, duplicate_steps)

    return _STANCE_HANDLERS[stance](ctx)
