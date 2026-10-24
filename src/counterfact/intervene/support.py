"""Shared helpers for intervention estimates and arm tables."""

from __future__ import annotations

from typing import Any, Literal

from counterfact.baselines import PassRateRow
from counterfact.intervene.estimate import (
    CausalEstimate,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
)
from counterfact.intervene.suggest import known_arms, suggest_harness_command
from counterfact.outcome.model import OutcomeModel
from counterfact.stats import wilson_ci

IDENTIFIED_TIGHT_CI_WIDTH = 0.10


def arm_table_from_model(model: OutcomeModel, feature_family: str) -> list[PassRateRow]:
    """Per-arm rows derived from the model's one-hot feature matrix."""
    if not model.feature_index:
        return []

    prefix = f"{feature_family}::"
    rows: list[PassRateRow] = []
    for key, idx in sorted(model.feature_index.items()):
        if not key.startswith(prefix):
            continue
        arm = key[len(prefix) :]
        col = model.train_X[:, idx]
        n = int(col.sum())
        if n == 0:
            continue
        pass_count = int(((col > 0).astype(int) * model.train_y).sum())
        pass_rate = pass_count / n if n else 0.0
        ci_low, ci_high = wilson_ci(pass_count, n)
        rows.append(
            PassRateRow(
                arm=arm,
                n=n,
                pass_count=pass_count,
                pass_rate=pass_rate,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return rows


def arm_rows_as_payload(rows: list[PassRateRow]) -> list[dict[str, Any]]:
    """Serialize arm rows for `NextStep.payload` (dict shape for CLI/tests)."""
    return [row.model_dump() for row in rows]


def missing_arms_for(decision_type: str, intervention_kind: str, observed: list[str]) -> list[str]:
    """Canonical-arm-set minus observed arms. Empty when no canonical set exists."""
    canonical = known_arms(decision_type, intervention_kind)
    if not canonical:
        return []
    obs = set(observed)
    return [arm for arm in canonical if arm not in obs]


def replay_inputs_for(decision_type: str, intervention_kind: str) -> list[str]:
    """Names of upstream inputs that replay would have to reproduce."""
    if decision_type == "model_call" and intervention_kind in {"prompt_content", "prompt_template"}:
        return ["prompt_template", "latent_state_at_step"]
    if decision_type == "memory_read" and intervention_kind == "memory_content":
        return ["memory_state_at_step"]
    return ["upstream_inputs_at_step"]


REPLAY_NOTE = (
    "v0 does not ship replay infrastructure; this next step is upstream of the bench harness."
)


def build_broaden_arm_support_estimate(
    query: InterventionQuery,
    *,
    reason: str,
    observed_rows: list[PassRateRow],
    missing_arms: list[str],
    missing_strata: list[str],
    identifiability: IdentifiabilityStatus = IdentifiabilityStatus.UNIDENTIFIED,
    warnings: list[str] | None = None,
    human_text: str,
    suggestion_action: Literal["broaden_arm_support", "add_arm_randomization"] = (
        "broaden_arm_support"
    ),
    arm_name_for_suggestion: str | None = None,
    payload_arm_name: str | None = None,
) -> CausalEstimate:
    """Build a `broaden_arm_support` (or related) next-step estimate."""
    observed_payload = arm_rows_as_payload(observed_rows)
    suggestion = suggest_harness_command(
        decision_type=query.decision_type,
        intervention_kind=query.intervention_kind,
        action=suggestion_action,
        arm_name=arm_name_for_suggestion,
    )
    payload: dict[str, Any] = {
        "arm_name": payload_arm_name if payload_arm_name is not None else query.decision_type,
        "missing_strata": missing_strata,
        "observed_arms": observed_payload,
        "missing_arms": missing_arms,
    }
    if suggestion is not None:
        payload["suggested_command"] = suggestion
    return CausalEstimate(
        query=query,
        identifiability=identifiability,
        reason=reason,
        warnings=warnings or [],
        next_step=NextStep(
            action=suggestion_action,
            payload=payload,
            human_text=human_text,
        ),
    )
