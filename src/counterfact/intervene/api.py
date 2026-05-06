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
`counterfact.intervene.estimate`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from counterfact.dag import DAG
from counterfact.errors import InvalidInterventionError, UnsupportedOutcomeError
from counterfact.intervene.estimate import (
    CausalEstimate,
    DistributionSummary,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
    SensitivityBounds,
)
from counterfact.intervene.suggest import known_arms, suggest_harness_command

# Width below which an `identified` estimate is considered "tight enough" — no
# action is needed and `next_step.action="none"`. Above this, intervene
# emits `increase_n` with a binomial-Wald n estimate.
_IDENTIFIED_TIGHT_CI_WIDTH = 0.10

# 95% z-score, kept in sync with `counterfact.power._Z_95` so the inline
# two-arm power computation matches `power_analysis` semantically.
_Z_95 = 1.959963984540054


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Mirrors baselines._wilson_ci."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    z = _Z_95
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _arm_table_for(model: Any, decision_type: str) -> list[dict[str, Any]]:
    """Per-arm rows derived from the model's one-hot feature matrix.

    Each row mirrors the shape of `counterfact.baselines.PassRateRow` so the
    payload contract is uniform whether the rows came from `pass_rate_by_arm`
    or from an already-fitted model. We work off the model rather than the
    corpus so `intervene` does not need access to the original Run list.
    """
    feature_index = getattr(model, "feature_index", {}) or {}
    train_X = getattr(model, "train_X", None)
    train_y = getattr(model, "train_y", None)
    if train_X is None or train_y is None or not feature_index:
        return []

    prefix = f"{decision_type}::"
    rows: list[dict[str, Any]] = []
    for key, idx in sorted(feature_index.items()):
        if not key.startswith(prefix):
            continue
        arm = key[len(prefix) :]
        col = train_X[:, idx]
        n = int(col.sum())
        if n == 0:
            continue
        pass_count = int(((col > 0).astype(int) * train_y).sum())
        pass_rate = pass_count / n if n else 0.0
        ci_low, ci_high = _wilson_ci(pass_count, n)
        rows.append(
            {
                "arm": arm,
                "n": n,
                "pass_count": pass_count,
                "pass_rate": pass_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows


def _missing_arms_for(decision_type: str, intervention_kind: str, observed: list[str]) -> list[str]:
    """Canonical-arm-set minus observed arms. Empty when no canonical set exists."""
    canonical = known_arms(decision_type, intervention_kind)
    if not canonical:
        return []
    obs = set(observed)
    return [arm for arm in canonical if arm not in obs]


def _power_from_arm_table(
    arm_table: list[dict[str, Any]],
    target_arm: Any,
    target_ci_width: float,
    *,
    current_n: int,
    current_ci_width: float,
) -> tuple[int, str]:
    """Estimate required n + label which method produced it.

    When two or more arms have observed support, run the same binomial-Wald
    formula `counterfact.power_analysis` runs (per-arm fractions, two-arm
    variance). When only one arm is observed, fall back to the inline scaling
    estimator that the codebase shipped before this change.
    """
    observable = [r for r in arm_table if r.get("n", 0) >= 1]
    if len(observable) >= 2:
        # Pair the target arm against the largest sibling. If the target arm
        # is itself missing, treat the two largest observed arms as the pair —
        # the math is symmetric and the user reads `arm_breakdown` to decide.
        ranked = sorted(observable, key=lambda r: r["n"], reverse=True)
        target_row = next((r for r in observable if r["arm"] == target_arm), None)
        if target_row is None:
            target_row = ranked[0]
            sibling_row = ranked[1]
        else:
            sibling_row = next((r for r in ranked if r["arm"] != target_row["arm"]), ranked[0])
        n_a = target_row["n"]
        n_b = sibling_row["n"]
        p_a = target_row["pass_rate"]
        p_b = sibling_row["pass_rate"]
        f_a = n_a / (n_a + n_b)
        f_b = n_b / (n_a + n_b)
        var_per_n = p_a * (1 - p_a) / f_a + p_b * (1 - p_b) / f_b
        if var_per_n <= 0:
            # Degenerate per-arm rates (0 or 1 on both arms): the binomial
            # formula collapses; the user needs outcome variation, not more n.
            return max(current_n + 1, current_n), "binomial_wald_two_arm"
        n_required = math.ceil(var_per_n * (2.0 * _Z_95 / target_ci_width) ** 2)
        return max(int(n_required), current_n + 1), "binomial_wald_two_arm"

    # Single-arm fallback: same scaling rule the engine used pre-change.
    scale = (current_ci_width / target_ci_width) ** 2
    n_required = max(round(current_n * scale), current_n + 1)
    return int(n_required), "inline_scaling"


def _replay_inputs_for(decision_type: str, intervention_kind: str) -> list[str]:
    """Names of upstream inputs that replay would have to reproduce."""
    if decision_type == "model_call" and intervention_kind in {"prompt_content", "prompt_template"}:
        return ["prompt_template", "latent_state_at_step"]
    if decision_type == "memory_read" and intervention_kind == "memory_content":
        return ["memory_state_at_step"]
    return ["upstream_inputs_at_step"]


_REPLAY_NOTE = (
    "v0 does not ship replay infrastructure; this next step is upstream of "
    "the bench harness."
)


def _bootstrap_predict(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    implementation in `counterfact.sensitivity` so the test in §9.2 can pin the
    formula directly.
    """
    from counterfact.sensitivity import e_value as _ev

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


def _duplicate_decision_steps(dag: DAG, step: int, decision_type: str) -> list[int]:
    """Other steps in `dag.run` whose decisions include `decision_type`.

    The g-formula adjustment over `model.train_X` is corpus-wide and treats
    a (decision_type, action) one-hot as a single feature, so it cannot
    disambiguate between repeated occurrences of the same decision_type in a
    single trace. Callers use this to surface an honest `unidentified`
    verdict instead of returning a corpus-wide estimate that pretends to be
    step-local.
    """
    if dag.run is None:
        return []
    return [
        s.step_index
        for s in dag.run.steps
        if s.step_index != step
        and any(d.decision_type == decision_type for d in s.decisions)
    ]


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
    model: Any,
    step: int,
    intervention: dict[str, Any],
) -> CausalEstimate:
    """Run an intervention query and return a CausalEstimate."""
    from counterfact.taxonomy import (
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

    # Honesty check: a step-scoped intervention on a decision type that
    # repeats elsewhere in the same trace is not licensed by the v0 g-formula
    # (which is corpus-wide on a single one-hot feature). Surface as
    # unidentified rather than answering the broader "set everywhere" query.
    duplicate_steps = _duplicate_decision_steps(dag, step, decision_type)
    if duplicate_steps:
        return CausalEstimate(
            query=query,
            identifiability=IdentifiabilityStatus.UNIDENTIFIED,
            reason=(
                f"step {step} targets decision_type={decision_type!r}, but the "
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
                    "arm_name": decision_type,
                    "current_policy": (
                        "outcome model uses a single (decision_type, action) "
                        "one-hot per trace; no step index in features"
                    ),
                    "duplicate_steps": duplicate_steps,
                },
                human_text=(
                    f"{decision_type!r} occurs at multiple steps in this trace; "
                    "v0 cannot answer step-local without step-aware features."
                ),
            ),
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
                payload={
                    "intervention_target": intervention_kind,
                    "replay_inputs_required": _replay_inputs_for(
                        decision_type, intervention_kind
                    ),
                    "note": _REPLAY_NOTE,
                },
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
        observed_arms = _arm_table_for(model, decision_type)
        observed_arm_names = [r["arm"] for r in observed_arms]
        suggestion = suggest_harness_command(
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            action="broaden_arm_support",
            arm_name=str(target_value) if target_value is not None else None,
        )
        payload: dict[str, Any] = {
            "arm_name": decision_type,
            "missing_strata": [
                "back-door adjustment requires randomized support over "
                f"{decision_type} jointly with its parents; v0 returns a bound."
            ],
            "observed_arms": observed_arms,
            "missing_arms": _missing_arms_for(
                decision_type, intervention_kind, observed_arm_names
            ),
        }
        if suggestion is not None:
            payload["suggested_command"] = suggestion
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
                payload=payload,
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
        observed_arms = _arm_table_for(model, decision_type)
        observed_arm_names = [r["arm"] for r in observed_arms]
        # The target arm is, by definition, not in observed_arms here.
        target_arm_name = str(target_value) if target_value is not None else None
        canonical_missing = _missing_arms_for(
            decision_type, intervention_kind, observed_arm_names
        )
        if target_arm_name and target_arm_name not in canonical_missing:
            canonical_missing = [target_arm_name, *canonical_missing]
        suggestion = suggest_harness_command(
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            action="broaden_arm_support",
            arm_name=target_arm_name,
        )
        payload = {
            "arm_name": decision_type,
            "missing_strata": [str(exc)],
            "observed_arms": observed_arms,
            "missing_arms": canonical_missing,
        }
        if suggestion is not None:
            payload["suggested_command"] = suggestion
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
                payload=payload,
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
        current_n = int(
            getattr(model, "train_n", 0)
            or getattr(getattr(model, "train_X", None), "shape", [0])[0]
            or delta.n_bootstrap
        )
        current_n = max(current_n, 1)
        arm_table = _arm_table_for(model, decision_type)
        estimated_required_n, power_method = _power_from_arm_table(
            arm_table,
            target_value,
            _IDENTIFIED_TIGHT_CI_WIDTH,
            current_n=current_n,
            current_ci_width=ci_width,
        )
        suggestion = suggest_harness_command(
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            action="increase_n",
            arm_name=str(target_value) if target_value is not None else None,
            estimated_required_n=estimated_required_n,
        )
        payload = {
            "current_n": current_n,
            "estimated_required_n": estimated_required_n,
            "target_ci_width": _IDENTIFIED_TIGHT_CI_WIDTH,
            "power_method": power_method,
            "arm_breakdown": arm_table,
        }
        if suggestion is not None:
            payload["suggested_command"] = suggestion
        next_step = NextStep(
            action="increase_n",
            payload=payload,
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
