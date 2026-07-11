"""Shared helpers for intervention estimates and arm tables."""

from __future__ import annotations

import math
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
from counterfact.stats import Z_95, wilson_ci

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


def power_from_arm_table(
    arm_table: list[PassRateRow],
    target_arm: str | None,
    target_ci_width: float,
    *,
    current_n: int,
    current_ci_width: float,
) -> tuple[int, str]:
    """Estimate required n + label which method produced it."""
    observable = [r for r in arm_table if r.n >= 1]
    if len(observable) >= 2:
        ranked = sorted(observable, key=lambda r: r.n, reverse=True)
        target_row = next((r for r in observable if r.arm == target_arm), None)
        if target_row is None:
            target_row = ranked[0]
            sibling_row = ranked[1]
        else:
            sibling_row = next((r for r in ranked if r.arm != target_row.arm), ranked[0])
        n_a = target_row.n
        n_b = sibling_row.n
        p_a = target_row.pass_rate
        p_b = sibling_row.pass_rate
        n_total = sum(r.n for r in arm_table)
        if n_total <= 0:
            n_total = n_a + n_b
        f_a = n_a / n_total
        f_b = n_b / n_total
        var_per_n = p_a * (1 - p_a) / f_a + p_b * (1 - p_b) / f_b
        if var_per_n <= 0:
            return max(current_n + 1, current_n), "binomial_wald_two_arm"
        n_required = math.ceil(var_per_n * (2.0 * Z_95 / target_ci_width) ** 2)
        return max(int(n_required), current_n + 1), "binomial_wald_two_arm"

    scale = (current_ci_width / target_ci_width) ** 2
    n_required = max(round(current_n * scale), current_n + 1)
    return int(n_required), "inline_scaling"


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
