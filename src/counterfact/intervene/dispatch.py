"""Stance-dispatched intervention handlers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from counterfact.dag import DAG
from counterfact.intervene.estimate import (
    CausalEstimate,
    DistributionSummary,
    EstimatorDiagnostics,
    IdentifiabilityStatus,
    IdentificationBounds,
    InterventionQuery,
    NextStep,
)
from counterfact.intervene.estimators import (
    TreatmentData,
    adjustment_set,
    aipw_contrast,
    manski_bounds,
    propensity_summary,
    randomization_gaps,
    treatment_data,
)
from counterfact.intervene.suggest import suggest_harness_command
from counterfact.intervene.support import (
    IDENTIFIED_TIGHT_CI_WIDTH,
    REPLAY_NOTE,
    arm_rows_as_payload,
    arm_table_from_model,
    build_broaden_arm_support_estimate,
    missing_arms_for,
    replay_inputs_for,
)
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
    baseline_arm_name: str | None
    decision_type: str


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
            f"{ctx.decision_type} jointly with its parents; no numeric bound is available."
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
            "no sensitivity bound computed: memory content is not logged as a "
            "randomized arm, so the corpus has no contrast to bound",
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
                "broaden arm support before estimating a point effect."
            ),
        ),
    )


def handle_no_support(ctx: InterveneContext, missing: list[str]) -> CausalEstimate:
    observed_rows = arm_table_from_model(ctx.model, ctx.feature_family)
    observed_arm_names = [r.arm for r in observed_rows]
    canonical_missing = missing_arms_for(
        ctx.decision_type, ctx.intervention_kind, observed_arm_names
    )
    canonical_missing = [*missing, *(arm for arm in canonical_missing if arm not in missing)]
    strata = [f"{ctx.feature_family}::{arm}" for arm in missing]
    return build_broaden_arm_support_estimate(
        ctx.query,
        reason=f"no observed support for {', '.join(strata)}; randomization missed this arm",
        observed_rows=observed_rows,
        missing_arms=canonical_missing,
        missing_strata=strata,
        warnings=[
            "Increasing epsilon or extending the corpus could create support "
            "for this arm and promote the result to identified."
        ],
        human_text=(
            f"Arm(s) {', '.join(repr(arm) for arm in missing)} on {ctx.decision_type!r} "
            "have zero observed support; raise ε or run more traces with a deliberate sweep."
        ),
        arm_name_for_suggestion=ctx.target_arm_name,
    )


def handle_single_class_rows(
    ctx: InterveneContext, data: TreatmentData, observed: int
) -> CausalEstimate:
    """The corpus has both outcomes, but every run that reaches this decision has one."""
    observed_rows = arm_table_from_model(ctx.model, ctx.feature_family)
    return build_broaden_arm_support_estimate(
        ctx.query,
        reason=(
            f"all {len(data.rows)} runs that reach {ctx.decision_type!r} have "
            f"outcome={bool(observed)}; no outcome variation exists to estimate a contrast"
        ),
        observed_rows=observed_rows,
        missing_arms=[],
        missing_strata=[f"Outcome.value={not observed}"],
        human_text=(
            f"Collect runs that reach {ctx.decision_type!r} with both pass and fail "
            "outcomes before estimating this contrast."
        ),
        arm_name_for_suggestion=ctx.target_arm_name,
    )


def handle_not_randomized(
    ctx: InterveneContext, data: TreatmentData, baseline: str, gaps: list[str]
) -> CausalEstimate:
    lower, upper = manski_bounds(data, ctx.target_arm_name, baseline)
    policies = sorted({str(r.decision.policy) for r in data.rows})
    return CausalEstimate(
        query=ctx.query,
        identifiability=IdentifiabilityStatus.BOUNDED,
        estimand=_contrast_estimand(ctx, baseline),
        reason=(
            "the corpus does not log randomized assignment for this contrast: " + "; ".join(gaps)
        ),
        bounds=IdentificationBounds(
            lower=lower,
            upper=upper,
            note=(
                f"Manski bounds over {len(data.rows)} runs; they hold under any "
                f"confounding, which is why they span width {upper - lower:.2f}"
            ),
        ),
        assumptions=[
            "none beyond consistency: the bounds use only the observed joint "
            "distribution of arm and outcome"
        ],
        warnings=[
            "a point estimate here would assume no unmeasured confounding, which "
            "unrandomized traces cannot support"
        ],
        next_step=NextStep(
            action="add_arm_randomization",
            payload={
                "arm_name": ctx.decision_type,
                "current_policy": ", ".join(policies),
                "randomization_gaps": gaps,
            },
            human_text=(
                f"Randomize {ctx.decision_type!r} with logged propensities over "
                f"{ctx.target_arm_name!r} and {baseline!r} to identify this contrast."
            ),
        ),
    )


def handle_identified(ctx: InterveneContext, data: TreatmentData, baseline: str) -> CausalEstimate:
    target = ctx.target_arm_name
    contrast = aipw_contrast(data, target, baseline, seed=ctx.model.seed)
    delta = DistributionSummary(
        point=contrast.delta, ci_low=contrast.delta_ci[0], ci_high=contrast.delta_ci[1]
    )
    min_p_target, ess_target = propensity_summary(data, target)
    min_p_baseline, ess_baseline = propensity_summary(data, baseline)
    n_used = len(data.rows)

    warnings: list[str] = []
    if data.n_excluded:
        warnings.append(
            f"{data.n_excluded} runs excluded: they never reach a {ctx.decision_type} "
            "decision, so the contrast is the effect among runs that do"
        )
    half_width = (delta.ci_high - delta.ci_low) / 2
    disagreement = abs(contrast.g_formula_delta - contrast.ipw_delta)
    if disagreement > half_width:
        warnings.append(
            f"g-formula ({contrast.g_formula_delta:+.3f}) and IPW ({contrast.ipw_delta:+.3f}) "
            "disagree by more than the AIPW sampling half-width; the outcome model is "
            "likely misspecified. AIPW stays consistent because the propensities are logged."
        )

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
        arm_table = arm_table_from_model(ctx.model, ctx.feature_family)
        # The influence-function CI width shrinks as 1/sqrt(n), so scaling the
        # observed width sizes the experiment for this estimator, adjustment
        # and propensities included.
        estimated_required_n = max(
            math.ceil(n_used * (ci_width / IDENTIFIED_TIGHT_CI_WIDTH) ** 2), n_used + 1
        )
        suggestion = suggest_harness_command(
            decision_type=ctx.decision_type,
            intervention_kind=ctx.intervention_kind,
            action="increase_n",
            arm_name=str(ctx.target_value) if ctx.target_value is not None else None,
            estimated_required_n=estimated_required_n,
        )
        payload: dict[str, Any] = {
            "current_n": n_used,
            "estimated_required_n": estimated_required_n,
            "target_ci_width": IDENTIFIED_TIGHT_CI_WIDTH,
            "power_method": "aipw_influence_scaling",
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
        query=ctx.query,
        identifiability=IdentifiabilityStatus.IDENTIFIED,
        estimand=_contrast_estimand(ctx, baseline),
        adjustment_set=adjustment_set(data),
        outcome_delta=delta,
        diagnostics=EstimatorDiagnostics(
            n_used=n_used,
            n_excluded=data.n_excluded,
            n_target=data.arm_count(target),
            n_baseline=data.arm_count(baseline),
            n_folds=contrast.n_folds,
            g_formula_delta=contrast.g_formula_delta,
            ipw_delta=contrast.ipw_delta,
            min_propensity_target=min_p_target,
            min_propensity_baseline=min_p_baseline,
            effective_n_target=ess_target,
            effective_n_baseline=ess_baseline,
        ),
        assumptions=[
            f"logged propensities are the true assignment probabilities (present on "
            f"all {n_used} analyzed runs)",
            "positivity: both arms had nonzero assignment probability on every analyzed run",
            "consistency and no interference: a run's outcome depends only on its own arm",
            "adjustment set holds only decisions logged before the treatment, so no "
            "mediator is conditioned on",
        ],
        warnings=warnings,
        next_step=next_step,
    )


def handle_randomized_support(ctx: InterveneContext) -> CausalEstimate:
    baseline = ctx.baseline_arm_name
    if baseline is None:
        raise ValueError("randomized-support contrasts require a resolved baseline arm")
    data = treatment_data(ctx.model.runs, ctx.decision_type, ctx.feature_family)
    missing = [arm for arm in (ctx.target_arm_name, baseline) if data.arm_count(arm) == 0]
    if missing:
        return handle_no_support(ctx, missing)
    analyzed_classes = {r.outcome for r in data.rows}
    if len(analyzed_classes) < 2:
        return handle_single_class_rows(ctx, data, analyzed_classes.pop())
    gaps = randomization_gaps(data, (ctx.target_arm_name, baseline))
    if gaps:
        return handle_not_randomized(ctx, data, baseline, gaps)
    return handle_identified(ctx, data, baseline)


def _contrast_estimand(ctx: InterveneContext, baseline: str) -> str:
    family = ctx.feature_family
    return f"E[Y | do({family}={ctx.target_arm_name})] - E[Y | do({family}={baseline})]"


_STANCE_HANDLERS: dict[IdentifiabilityStance, Callable[[InterveneContext], CausalEstimate]] = {
    "always-replay": handle_always_replay,
    "requires-back-door-adjustment": handle_back_door_bounded,
    "requires-randomized-support": handle_randomized_support,
}
