"""Decision-type taxonomy.

The reusable abstraction is not the DAG; it is the typed decision schema.
Each decision type declares:

* the set of intervention kinds it accepts
* an identifiability stance per intervention kind
* the decision types that may causally precede it (DAG parents)
* a feature-extraction contract used by the outcome model

This module is intentionally data-first: edits should look like edits to a
table, not edits to algorithm code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

from counterfact.errors import UnknownDecisionTypeError
from counterfact.schema import Decision, DecisionTypeLiteral, Run

IdentifiabilityStance = Literal[
    "requires-randomized-support",
    "requires-back-door-adjustment",
    "always-replay",
]

DECISION_TYPES: tuple[str, ...] = get_args(DecisionTypeLiteral)


@dataclass(frozen=True)
class DecisionSpec:
    parents: tuple[str, ...]
    action_field: str
    interventions: tuple[tuple[str, IdentifiabilityStance], ...] = ()
    default_intervention: str | None = None


_SPECS: dict[str, DecisionSpec] = {
    "plan_step": DecisionSpec(
        ("plan_step", "model_call", "tool_call", "memory_read"), "plan_action"
    ),
    "model_call": DecisionSpec(
        ("plan_step", "memory_read"),
        "model_name",
        (
            ("model_choice", "requires-randomized-support"),
            ("temperature", "requires-randomized-support"),
            ("prompt_template", "always-replay"),
            ("prompt_content", "always-replay"),
        ),
        "model_choice",
    ),
    "tool_call": DecisionSpec(
        ("plan_step", "model_call"),
        "tool_name",
        (("tool_choice", "requires-randomized-support"),),
        "tool_choice",
    ),
    "memory_read": DecisionSpec(
        ("plan_step",),
        "memory_action",
        (("memory_content", "requires-back-door-adjustment"),),
    ),
    "retry": DecisionSpec(
        ("tool_call", "model_call"),
        "retry_action",
        (("retry_policy", "requires-randomized-support"),),
        "retry_policy",
    ),
    "termination": DecisionSpec(("plan_step", "model_call", "tool_call"), "termination_action"),
}
if set(_SPECS) != set(DECISION_TYPES):
    raise RuntimeError("decision taxonomy must cover exactly the schema's decision types")


def _ensure_known(decision_type: str) -> None:
    if decision_type not in DECISION_TYPES:
        raise UnknownDecisionTypeError(
            f"unknown decision_type={decision_type!r}; known: {DECISION_TYPES}"
        )


def valid_interventions(decision_type: str) -> frozenset[str]:
    """Return the intervention kinds accepted on this decision type."""
    _ensure_known(decision_type)
    return frozenset(kind for kind, _ in _SPECS[decision_type].interventions)


def is_valid_intervention(decision_type: str, intervention_kind: str) -> bool:
    """True iff `intervention_kind` is a valid intervention on this decision type."""
    if decision_type not in DECISION_TYPES:
        return False
    return intervention_kind in valid_interventions(decision_type)


def identifiability_stance(decision_type: str, intervention_kind: str) -> IdentifiabilityStance:
    """Return the declared stance for the (decision_type, intervention_kind) pair."""
    _ensure_known(decision_type)
    stances = dict(_SPECS[decision_type].interventions)
    if intervention_kind not in stances:
        raise UnknownDecisionTypeError(
            f"no identifiability stance declared for {(decision_type, intervention_kind)!r}"
        )
    return stances[intervention_kind]


def parent_types(decision_type: str) -> tuple[str, ...]:
    """Return the decision types that may causally precede this type within a trace."""
    _ensure_known(decision_type)
    return _SPECS[decision_type].parents


def default_intervention_kind(decision_type: str) -> str:
    """Return the v0 default intervention kind for CLI/demo/explain surfaces."""
    _ensure_known(decision_type)
    kind = _SPECS[decision_type].default_intervention
    if kind is None:
        raise UnknownDecisionTypeError(
            f"no default intervention kind for decision_type={decision_type!r}"
        )
    return kind


def attribution_intervention_kind(decision_type: str) -> str | None:
    """Pick the canonical intervention kind for failure attribution.

    Prefer the first declared kind whose stance is `requires-randomized-support`;
    otherwise fall back to the first valid kind in sorted order.
    """
    kinds = sorted(valid_interventions(decision_type))
    for kind in kinds:
        try:
            if identifiability_stance(decision_type, kind) == "requires-randomized-support":
                return kind
        except UnknownDecisionTypeError:
            continue
    return kinds[0] if kinds else None


def first_observed_arm(runs: Iterable[Run], decision_type: str) -> str | None:
    """Return the first non-empty `chosen_action` for `decision_type` in corpus order."""
    from counterfact.analysis.timeline import ordered_steps

    for run in runs:
        for step in ordered_steps(run):
            for decision in step.decisions:
                if decision.decision_type == decision_type and decision.chosen_action:
                    return decision.chosen_action
    return None


def extract_features(decision: Decision, run: Run) -> dict[str, Any]:
    """Map a Decision (in the context of its Run) to a feature dict for the outcome model.

    Per-type feature extractors live in this module so the taxonomy stays the
    single source of truth for both feature key naming and per-type metadata.

    The `feature_key` field is the one-hot identifier the outcome model uses
    (`"<decision_type>::<chosen_action>"`). It is `None` when the decision has
    no `chosen_action` or its type has no declared interventions, in which
    case the row is not included in the design matrix.
    """
    dt = getattr(decision, "decision_type", None)
    if dt not in DECISION_TYPES:
        raise UnknownDecisionTypeError(f"cannot extract features for unknown decision_type={dt!r}")

    step_index = _step_index_for(decision, run)
    chosen = decision.chosen_action
    feature_key: str | None = None
    if chosen is not None and _SPECS[dt].interventions:
        feature_key = f"{dt}::{chosen}"

    base: dict[str, Any] = {
        "decision_type": dt,
        "step_index": step_index,
        "feature_key": feature_key,
    }
    base[_SPECS[dt].action_field] = chosen or "<none>"
    return base


def _step_index_for(decision: Decision | Any, run: Run) -> int:
    """Resolve which step a decision belongs to in the run."""
    target_id = getattr(decision, "decision_id", None)
    for step in run.steps:
        for d in step.decisions:
            if d.decision_id == target_id:
                return step.step_index
    return -1


__all__ = [
    "DECISION_TYPES",
    "DecisionTypeLiteral",
    "IdentifiabilityStance",
    "attribution_intervention_kind",
    "default_intervention_kind",
    "extract_features",
    "first_observed_arm",
    "identifiability_stance",
    "is_valid_intervention",
    "parent_types",
    "valid_interventions",
]
