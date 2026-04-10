"""Intervention API + identifiability dispatch.

For each (decision_type, intervention_kind) pair the taxonomy declares an
identifiability stance. `intervene` dispatches on that stance:

* `requires-randomized-support`: g-formula via the fitted outcome model —
  set the targeted feature to 1 in every training row, zero out sibling-arm
  features, predict, average. Returns `identified` with point + bootstrap CI.
* `requires-back-door-adjustment`: returns `bounded` with sentinel bounds —
  the back-door adjustment set is named in `assumptions`. The full numerical
  bound lands alongside §9 sensitivity work.
* `always-replay`: returns `unidentified` with `next_step="replay"`.

The CausalEstimate result object is the public contract; full schema lives in
`counter.intervene.estimate`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from counter.dag import DAG
from counter.errors import InvalidInterventionError, UnsupportedOutcomeError
from counter.intervene.estimate import (
    CausalEstimate,
    DistributionSummary,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
    SensitivityBounds,
)

# Width below which an `identified` estimate is considered "tight enough" — no
# action is needed and `next_step.action="none"`. Above this, intervene
# emits `increase_n` with a binomial-Wald n estimate.
_IDENTIFIED_TIGHT_CI_WIDTH = 0.10


def _bootstrap_predict(
    model: Any, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute predictions per bootstrap draw. Returns (point_per_run, boot_per_run)."""
    boot_coefs = model.bootstrap_coefs  # (B, n_feat)
    boot_intercepts = model.bootstrap_intercepts  # (B,)
    z = X @ model.coefficients + model.intercept
    point = 1.0 / (1.0 + np.exp(-z))
    z_boot = X @ boot_coefs.T + boot_intercepts[None, :]
    boot = 1.0 / (1.0 + np.exp(-z_boot))  # (n_runs, B)
    return point, boot


def _adjust_g_formula(
    model: Any, decision_type: str, target_action: str
) -> DistributionSummary:
    """g-formula: set target one-hot to 1, zero out sibling arms in every training row.

    The marginal `P(Y | do(decision_type=target_action))` is the average of the
    per-row predicted probabilities under the modified feature matrix.
    """
    feat_index = model.feature_index
    target_key = f"{decision_type}::{target_action}"
    if target_key not in feat_index:
        # The target arm has zero observed support in the training corpus.
        # That is an unidentified situation in v0.
        raise _NoSupport(target_key)

    sibling_keys = [k for k in feat_index if k.startswith(f"{decision_type}::")]
    target_idx = feat_index[target_key]
    sibling_idx = [feat_index[k] for k in sibling_keys]

    X = model.train_X.copy()
    X[:, sibling_idx] = 0.0
    X[:, target_idx] = 1.0

    point_per_run, boot_per_run = _bootstrap_predict(model, X)
    point = float(point_per_run.mean())
    boot_means = boot_per_run.mean(axis=0)  # (B,)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return DistributionSummary(
        point=point,
        ci_low=ci_lo,
        ci_high=ci_hi,
        n_bootstrap=len(boot_means),
    )


class _NoSupport(Exception):
    """Internal: raised when the target arm has zero training support."""


def _e_value_from_probs(point: float, baseline: float = 0.5) -> float:
    """Compute the E-value for a marginal P(success) vs a baseline.

    Risk ratio = clamped(point) / clamped(baseline). Delegated to the canonical
    implementation in `counter.sensitivity` so the test in §9.2 can pin the
    formula directly.
    """
    from counter.sensitivity import e_value as _ev

    p = max(min(point, 1 - 1e-12), 1e-12)
    b = max(min(baseline, 1 - 1e-12), 1e-12)
    return _ev(p / b)


def _decision_type_at_step(dag: DAG, step: int) -> str:
    if dag.run is None:
        raise InvalidInterventionError("DAG has no associated trace")
    for s in dag.run.steps:
        if s.step_index == step:
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
    [(kind, value)] = list(intervention.items())
    return kind, value


def intervene(
    *,
    dag: DAG,
    model: Any,
    step: int,
    intervention: dict[str, Any],
) -> CausalEstimate:
    """Run an intervention query and return a CausalEstimate."""
    from counter.taxonomy import (
        identifiability_stance,
        is_valid_intervention,
    )

    outcome_kind = getattr(model, "outcome_kind", None)
    if outcome_kind is not None and outcome_kind != "binary":
        raise UnsupportedOutcomeError(
            f"v0 only supports binary outcomes; got model.outcome_kind={outcome_kind!r}"
        )

    intervention_kind, target_value = _resolve_intervention(intervention)
    decision_type = _decision_type_at_step(dag, step)

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

    stance = identifiability_stance(decision_type, intervention_kind)

    if stance == "always-replay":
        return CausalEstimate(
            query=query,
            identifiability=IdentifiabilityStatus.UNIDENTIFIED,
            reason=(
                f"{decision_type}.{intervention_kind} is treated as always-replay; "
                f"the prompt/content is high-dim and not identifiable from "
                f"observational traces alone."
            ),
            assumptions=[
                f"{intervention_kind} on {decision_type} marked always-replay in taxonomy"
            ],
            warnings=[
                "latent prompt quality and LLM completion noise are unblocked; "
                "no identifying assumptions hold here without replay infrastructure."
            ],
            next_step=NextStep(
                action="replay_required",
                payload={"intervention_target": intervention_kind},
                human_text=(
                    f"{intervention_kind} is high-dimensional; only deterministic "
                    "replay can answer this query."
                ),
            ),
        )

    if stance == "requires-back-door-adjustment":
        # v0 returns a bounded estimate: we name the adjustment strategy in
        # assumptions and emit an E-value sentinel against a 0.5 baseline.
        ev = _e_value_from_probs(0.5, baseline=0.5)
        return CausalEstimate(
            query=query,
            identifiability=IdentifiabilityStatus.BOUNDED,
            adjustment_set=[decision_type],
            assumptions=[
                f"back-door adjustment on {decision_type} via taxonomy parents",
                "E-value computed against a 0.5 baseline",
            ],
            bounds=SensitivityBounds(e_value=ev, technique="e_value"),
            warnings=[
                "bounded path: confounders may exist; sensitivity bound holds the "
                "result against unmeasured strength of confounding."
            ],
            next_step=NextStep(
                action="broaden_arm_support",
                payload={
                    "arm_name": decision_type,
                    "missing_strata": [
                        "back-door adjustment requires randomized support over "
                        f"{decision_type} jointly with its parents; v0 returns a bound."
                    ],
                },
                human_text=(
                    f"{decision_type} would identify under back-door adjustment if "
                    "the corpus had randomized joint support over its parents; "
                    "v0 ships the E-value bound instead."
                ),
            ),
        )

    # stance == "requires-randomized-support" → identified path via g-formula.
    try:
        delta = _adjust_g_formula(model, decision_type, target_value)
    except _NoSupport as exc:
        return CausalEstimate(
            query=query,
            identifiability=IdentifiabilityStatus.UNIDENTIFIED,
            reason=f"no observed support for {exc}; randomization missed this arm",
            warnings=[
                "Increasing epsilon or extending the corpus could create support "
                "for this arm and promote the result to identified."
            ],
            next_step=NextStep(
                action="broaden_arm_support",
                payload={
                    "arm_name": decision_type,
                    "missing_strata": [str(exc)],
                },
                human_text=(
                    f"Arm {target_value!r} on {decision_type!r} has zero observed "
                    "support; raise ε or run more traces with a deliberate sweep."
                ),
            ),
        )

    ev = _e_value_from_probs(delta.point, baseline=0.5)
    ci_width = delta.ci_high - delta.ci_low
    if ci_width <= _IDENTIFIED_TIGHT_CI_WIDTH:
        next_step = NextStep(
            action="none",
            payload={},
            human_text=(
                f"CI width {ci_width:.3f} ≤ {_IDENTIFIED_TIGHT_CI_WIDTH:.2f}; "
                "no further action required."
            ),
        )
    else:
        # Estimate the additional traces needed to bring CI width below the
        # tight-enough threshold. Binomial-Wald scales as 1/√n, so:
        #     n_required ≈ current_n * (current_width / target_width)^2
        current_n = int(
            getattr(model, "train_n", 0)
            or getattr(getattr(model, "train_X", None), "shape", [0])[0]
            or delta.n_bootstrap
        )
        current_n = max(current_n, 1)
        scale = (ci_width / _IDENTIFIED_TIGHT_CI_WIDTH) ** 2
        estimated_required_n = max(round(current_n * scale), current_n + 1)
        next_step = NextStep(
            action="increase_n",
            payload={
                "current_n": current_n,
                "estimated_required_n": estimated_required_n,
                "target_ci_width": _IDENTIFIED_TIGHT_CI_WIDTH,
            },
            human_text=(
                f"CI width {ci_width:.3f} > {_IDENTIFIED_TIGHT_CI_WIDTH:.2f}; "
                f"~{estimated_required_n} traces would tighten it."
            ),
        )
    return CausalEstimate(
        query=query,
        identifiability=IdentifiabilityStatus.IDENTIFIED,
        estimand=f"E[Y | do({decision_type}={target_value})]",
        adjustment_set=sorted(model.feature_index.keys()),
        outcome_delta=delta,
        bounds=SensitivityBounds(e_value=ev, technique="e_value"),
        assumptions=[
            "back-door adjustment via empirical training distribution (g-formula)",
            "uniform / propensity-logged randomization at this decision type",
        ],
        next_step=next_step,
    )
