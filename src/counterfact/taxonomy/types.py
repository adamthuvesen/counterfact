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
from typing import Any, Literal

from counterfact.errors import UnknownDecisionTypeError
from counterfact.schema import Decision, DecisionTypeLiteral, Run

IdentifiabilityStance = Literal[
    "requires-randomized-support",
    "requires-back-door-adjustment",
    "always-replay",
]

DECISION_TYPES: tuple[str, ...] = (
    "plan_step",
    "model_call",
    "tool_call",
    "memory_read",
    "retry",
    "termination",
)

# Per-type interventions. Keys are decision types; values are the intervention
# kinds valid for that type.
_VALID_INTERVENTIONS: dict[str, frozenset[str]] = {
    "plan_step": frozenset(),
    "model_call": frozenset({"model_choice", "prompt_template", "prompt_content", "temperature"}),
    "tool_call": frozenset({"tool_choice"}),
    "memory_read": frozenset({"memory_content"}),
    "retry": frozenset({"retry_policy"}),
    "termination": frozenset(),
}

# Per (decision_type, intervention_kind) identifiability stance. Stances drive
# the dispatch in `intervene` (§8): randomized-support paths to back-door
# adjustment, back-door-adjustment paths to bounded estimates, always-replay
# paths to `unidentified` with `next_step="replay"`.
_IDENTIFIABILITY_STANCE: dict[tuple[str, str], IdentifiabilityStance] = {
    ("model_call", "model_choice"): "requires-randomized-support",
    ("model_call", "temperature"): "requires-randomized-support",
    ("model_call", "prompt_template"): "always-replay",
    ("model_call", "prompt_content"): "always-replay",
    ("tool_call", "tool_choice"): "requires-randomized-support",
    ("memory_read", "memory_content"): "requires-back-door-adjustment",
    ("retry", "retry_policy"): "requires-randomized-support",
}

# DAG parent declarations. A decision of type `t` may have edges from any
# decision whose type is in `_PARENT_TYPES[t]` and whose step index is earlier.
_PARENT_TYPES: dict[str, tuple[str, ...]] = {
    "plan_step": ("plan_step", "model_call", "tool_call", "memory_read"),
    "model_call": ("plan_step", "memory_read"),
    "tool_call": ("plan_step", "model_call"),
    "memory_read": ("plan_step",),
    "retry": ("tool_call", "model_call"),
    "termination": ("plan_step", "model_call", "tool_call"),
}

# v0 default intervention kind per decision type for demo/explain surfaces.
_DEFAULT_INTERVENTION_KIND: dict[str, str] = {
    "model_call": "model_choice",
    "tool_call": "tool_choice",
    "retry": "retry_policy",
}

_ACTION_FEATURE_FIELD: dict[str, str] = {
    "tool_call": "tool_name",
    "model_call": "model_name",
    "retry": "retry_action",
    "memory_read": "memory_action",
    "plan_step": "plan_action",
    "termination": "termination_action",
}


def _ensure_known(decision_type: str) -> None:
    if decision_type not in DECISION_TYPES:
        raise UnknownDecisionTypeError(
            f"unknown decision_type={decision_type!r}; known: {DECISION_TYPES}"
        )


def valid_interventions(decision_type: str) -> frozenset[str]:
    """Return the intervention kinds accepted on this decision type."""
    _ensure_known(decision_type)
    return _VALID_INTERVENTIONS[decision_type]


def is_valid_intervention(decision_type: str, intervention_kind: str) -> bool:
    """True iff `intervention_kind` is a valid intervention on this decision type."""
    if decision_type not in DECISION_TYPES:
        return False
    return intervention_kind in _VALID_INTERVENTIONS[decision_type]


def identifiability_stance(decision_type: str, intervention_kind: str) -> IdentifiabilityStance:
    """Return the declared stance for the (decision_type, intervention_kind) pair."""
    _ensure_known(decision_type)
    key = (decision_type, intervention_kind)
    if key not in _IDENTIFIABILITY_STANCE:
        raise UnknownDecisionTypeError(f"no identifiability stance declared for {key!r}")
    return _IDENTIFIABILITY_STANCE[key]


def parent_types(decision_type: str) -> tuple[str, ...]:
    """Return the decision types that may causally precede this type within a trace."""
    _ensure_known(decision_type)
    return _PARENT_TYPES.get(decision_type, ())


def default_intervention_kind(decision_type: str) -> str:
    """Return the v0 default intervention kind for CLI/demo/explain surfaces."""
    _ensure_known(decision_type)
    try:
        return _DEFAULT_INTERVENTION_KIND[decision_type]
    except KeyError as exc:
        raise UnknownDecisionTypeError(
            f"no default intervention kind for decision_type={decision_type!r}"
        ) from exc


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
    for run in runs:
        for step in run.steps:
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
    if chosen is not None and _VALID_INTERVENTIONS[dt]:
        feature_key = f"{dt}::{chosen}"

    base: dict[str, Any] = {
        "decision_type": dt,
        "step_index": step_index,
        "feature_key": feature_key,
    }
    base[_ACTION_FEATURE_FIELD[dt]] = chosen or "<none>"
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
