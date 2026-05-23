"""Stance-dispatched intervention handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from counterfact.dag import DAG
from counterfact.errors import InvalidInterventionError
from counterfact.intervene.estimate import (
    CausalEstimate,
    DistributionSummary,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
    SensitivityBounds,
)
from counterfact.intervene.suggest import suggest_harness_command
from counterfact.intervene.support import (
    IDENTIFIED_TIGHT_CI_WIDTH,
    REPLAY_NOTE,
    arm_rows_as_payload,
    arm_table_from_model,
    build_broaden_arm_support_estimate,
    missing_arms_for,
    power_from_arm_table,
    replay_inputs_for,
)
from counterfact.outcome.features import canonical_intervention_value
from counterfact.outcome.model import OutcomeModel
from counterfact.taxonomy import IdentifiabilityStance


@dataclass(frozen=True)
class InterveneContext:
    dag: DAG
    model: OutcomeModel
    step: int
    query: InterventionQuery
    intervention_kind: str
    target_value: Any
    feature_family: str
    target_arm_name: str
    decision_type: str


class _NoSupport(Exception):
    """Internal: raised when the target arm has zero training support."""


def _bootstrap_predict(model: OutcomeModel, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boot_coefs = model.bootstrap_coefs
    boot_intercepts = model.bootstrap_intercepts
    z = X @ model.coefficients + model.intercept
    point = 1.0 / (1.0 + np.exp(-z))
    z_boot = X @ boot_coefs.T + boot_intercepts[None, :]
    boot = 1.0 / (1.0 + np.exp(-z_boot))
    return point, boot


def _adjust_g_formula(
    model: OutcomeModel,
    feature_family: str,
    intervention_kind: str,
    target_action: Any,
) -> DistributionSummary:
    feat_index = model.feature_index
    target_key = (
        f"{feature_family}::{canonical_intervention_value(intervention_kind, target_action)}"
    )
    if target_key not in feat_index:
        raise _NoSupport(target_key)

    sibling_keys = [k for k in feat_index if k.startswith(f"{feature_family}::")]
    target_idx = feat_index[target_key]
    sibling_idx = [feat_index[k] for k in sibling_keys]

    X = model.train_X.copy()
    X[:, sibling_idx] = 0.0
    X[:, target_idx] = 1.0

    point_per_run, boot_per_run = _bootstrap_predict(model, X)
    point = float(point_per_run.mean())
    boot_means = boot_per_run.mean(axis=0)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return DistributionSummary(
        point=point,
        ci_low=ci_lo,
        ci_high=ci_hi,
        n_bootstrap=len(boot_means),
    )


def _e_value_from_probs(point: float, baseline: float = 0.5) -> float:
    from counterfact.sensitivity import e_value as _ev

    p = max(min(point, 1 - 1e-12), 1e-12)
    b = max(min(baseline, 1 - 1e-12), 1e-12)
    return _ev(p / b)


def handle_always_replay(ctx: InterveneContext) -> CausalEstimate:
    q = ctx.query
    return CausalEstimate(
        query=q,
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason=(
            f"{ctx.decision_type}.{ctx.intervention_kind} is treated as always-replay; "
            f"the prompt/content is high-dim and not identifiable from "
            f"observational traces alone."
        ),
        assumptions=[
            f"{ctx.intervention_kind} on {ctx.decision_type} marked always-replay in taxonomy"
        ],
        warnings=[
            "latent prompt quality and LLM completion noise are unblocked; "
            "no identifying assumptions hold here without replay infrastructure."
        ],
        next_step=NextStep(
            action="replay_required",
            payload={
                "intervention_target": ctx.intervention_kind,
                "replay_inputs_required": replay_inputs_for(
                    ctx.decision_type, ctx.intervention_kind
                ),
                "note": REPLAY_NOTE,
            },
            human_text=(
                f"{ctx.intervention_kind} is high-dimensional; only deterministic "
                "replay can answer this query."
            ),
        ),
    )


def handle_duplicate_step_refusal(
    ctx: InterveneContext, duplicate_steps: list[int]
) -> CausalEstimate:
    q = ctx.query
    return CausalEstimate(
        query=q,
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason=(
            f"step {ctx.step} targets decision_type={ctx.decision_type!r}, but the "
            f"focal trace also has it at step(s) {duplicate_steps}; the v0 "
            "step-scoped intervene API cannot disambiguate which occurrence "
            "to intervene on, and the corpus-wide g-formula would answer "
            "'set everywhere' instead"
        ),
        assumptions=[
            "step-scoped intervene requires the targeted decision_type to "
            "appear at most once per trace under the v0 outcome-model schema"
        ],
        warnings=[
            "an estimate here would silently broaden a step-local query to a "
            "trace-wide one; promoting requires step-aware features"
        ],
        next_step=NextStep(
            action="add_arm_randomization",
            payload={
                "arm_name": ctx.decision_type,
                "current_policy": (
                    "outcome model uses a single (decision_type, action) "
                    "one-hot per trace; no step index in features"
                ),
                "duplicate_steps": duplicate_steps,
                "localization_limit": (
                    "v0 outcome features are keyed by decision_type/action, "
                    "not by decision_id or step"
                ),
            },
            human_text=(
                f"{ctx.decision_type!r} occurs at multiple steps in this trace; "
                "v0 cannot answer step-local without step-aware features."
            ),
        ),
    )


def handle_back_door_bounded(ctx: InterveneContext) -> CausalEstimate:
    q = ctx.query
    observed_rows = arm_table_from_model(ctx.model, ctx.feature_family)
    observed_arm_names = [r.arm for r in observed_rows]
    suggestion = suggest_harness_command(
        decision_type=ctx.decision_type,
        intervention_kind=ctx.intervention_kind,
        action="broaden_arm_support",
        arm_name=str(ctx.target_value) if ctx.target_value is not None else None,
    )
    payload: dict[str, Any] = {
        "arm_name": ctx.decision_type,
        "missing_strata": [
            "back-door adjustment requires randomized support over "
            f"{ctx.decision_type} jointly with its parents; v0 returns a bound."
        ],
        "observed_arms": arm_rows_as_payload(observed_rows),
        "missing_arms": missing_arms_for(
            ctx.decision_type, ctx.intervention_kind, observed_arm_names
        ),
    }
    if suggestion is not None:
        payload["suggested_command"] = suggestion
    return CausalEstimate(
        query=q,
        identifiability=IdentifiabilityStatus.BOUNDED,
        adjustment_set=[ctx.decision_type],
        assumptions=[
            f"back-door adjustment on {ctx.decision_type} via taxonomy parents",
            "E-value not computed: bounded path has no observed marginal "
            "P(success | do(...)) to compare against a baseline",
        ],
        bounds=None,
        warnings=[
            "bounded path: confounders may exist; v0 does not ship a "
            "sensitivity measure here — broaden arm support to promote to identified."
        ],
        next_step=NextStep(
            action="broaden_arm_support",
            payload=payload,
            human_text=(
                f"{ctx.decision_type} would identify under back-door adjustment if "
                "the corpus had randomized joint support over its parents; "
                "v0 ships the E-value bound instead."
            ),
        ),
    )


def handle_no_support(ctx: InterveneContext, exc: _NoSupport) -> CausalEstimate:
    observed_rows = arm_table_from_model(ctx.model, ctx.feature_family)
    observed_arm_names = [r.arm for r in observed_rows]
    canonical_missing = missing_arms_for(
        ctx.decision_type, ctx.intervention_kind, observed_arm_names
    )
    if ctx.target_arm_name and ctx.target_arm_name not in canonical_missing:
        canonical_missing = [ctx.target_arm_name, *canonical_missing]
    return build_broaden_arm_support_estimate(
        ctx.query,
        reason=f"no observed support for {exc}; randomization missed this arm",
        observed_rows=observed_rows,
        missing_arms=canonical_missing,
        missing_strata=[str(exc)],
        warnings=[
            "Increasing epsilon or extending the corpus could create support "
            "for this arm and promote the result to identified."
        ],
        human_text=(
            f"Arm {ctx.target_value!r} on {ctx.decision_type!r} has zero observed "
            "support; raise ε or run more traces with a deliberate sweep."
        ),
        arm_name_for_suggestion=ctx.target_arm_name,
    )


def handle_identified(ctx: InterveneContext, delta: DistributionSummary) -> CausalEstimate:
    q = ctx.query
    ev = _e_value_from_probs(delta.point, baseline=0.5)
    ci_width = delta.ci_high - delta.ci_low
    if ci_width <= IDENTIFIED_TIGHT_CI_WIDTH:
        next_step = NextStep(
            action="none",
            payload={},
            human_text=(
                f"CI width {ci_width:.3f} ≤ {IDENTIFIED_TIGHT_CI_WIDTH:.2f}; "
                "no further action required."
            ),
        )
    else:
        train_n = ctx.model.train_n
        if train_n <= 0:
            train_n = int(ctx.model.train_X.shape[0]) if ctx.model.train_X is not None else 0
        if train_n <= 0:
            raise InvalidInterventionError(
                "outcome model is missing train_n / train_X; cannot estimate power "
                "without the true corpus size"
            )
        current_n = train_n
        arm_table = arm_table_from_model(ctx.model, ctx.feature_family)
        estimated_required_n, power_method = power_from_arm_table(
            arm_table,
            ctx.target_arm_name,
            IDENTIFIED_TIGHT_CI_WIDTH,
            current_n=current_n,
            current_ci_width=ci_width,
        )
        suggestion = suggest_harness_command(
            decision_type=ctx.decision_type,
            intervention_kind=ctx.intervention_kind,
            action="increase_n",
            arm_name=str(ctx.target_value) if ctx.target_value is not None else None,
            estimated_required_n=estimated_required_n,
        )
        payload = {
            "current_n": current_n,
            "estimated_required_n": estimated_required_n,
            "target_ci_width": IDENTIFIED_TIGHT_CI_WIDTH,
            "power_method": power_method,
            "arm_breakdown": arm_rows_as_payload(arm_table),
        }
        if suggestion is not None:
            payload["suggested_command"] = suggestion
        next_step = NextStep(
            action="increase_n",
            payload=payload,
            human_text=(
                f"CI width {ci_width:.3f} > {IDENTIFIED_TIGHT_CI_WIDTH:.2f}; "
                f"~{estimated_required_n} traces would tighten it."
            ),
        )
    return CausalEstimate(
        query=q,
        identifiability=IdentifiabilityStatus.IDENTIFIED,
        estimand=f"E[Y | do({ctx.decision_type}={ctx.target_value})]",
        adjustment_set=sorted(ctx.model.feature_index.keys()),
        outcome_delta=delta,
        bounds=SensitivityBounds(e_value=ev, technique="e_value"),
        assumptions=[
            "back-door adjustment via empirical training distribution (g-formula)",
            "uniform / propensity-logged randomization at this decision type",
        ],
        next_step=next_step,
    )


def handle_randomized_support(ctx: InterveneContext) -> CausalEstimate:
    try:
        delta = _adjust_g_formula(
            ctx.model, ctx.feature_family, ctx.intervention_kind, ctx.target_value
        )
    except _NoSupport as exc:
        return handle_no_support(ctx, exc)
    return handle_identified(ctx, delta)


_STANCE_HANDLERS: dict[IdentifiabilityStance, Callable[[InterveneContext], CausalEstimate]] = {
    "always-replay": handle_always_replay,
    "requires-back-door-adjustment": handle_back_door_bounded,
    "requires-randomized-support": handle_randomized_support,
}
