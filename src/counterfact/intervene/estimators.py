"""Contrast estimators for one decision type across a trace corpus.

The treatment is the arm a run chose at its *first* decision of the targeted
type. Later decisions of the same type are downstream of the treatment, so
they act as mediators: the run stays in, and the estimand is the total effect
of the first choice. Dropping such runs would select on a post-treatment
variable. Covariates are the intervenable decisions logged in strictly earlier
steps than the first occurrence. Decisions in the same step may be
simultaneous with the treatment or downstream of it, so they are left out and
the adjustment set never contains a mediator or a collider. Runs that never reach the decision
are left out, which conditions only on a pre-treatment event.

Identification is checked against the data, not declared:

* every analyzed run must log the propensity of its chosen arm, and
* every run must show positive probability for both contrasted arms
  (positivity). A chosen arm proves its own mass. Mass on an arm the run did
  not choose must follow from the log: two valid actions and a chosen-arm
  propensity below 1, or a policy whose probabilities can be reconstructed.
  Membership in `valid_actions` alone does not prove it.

Only then is the cross-fitted AIPW estimate reported as identified. The
outcome-regression (g-formula) and Hajek IPW estimates ride along as
cross-checks. Without logged randomization, the only defensible answer for a
binary outcome is the Manski no-assumption interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from counterfact.outcome.binary import binary_outcome_value
from counterfact.outcome.features import decision_feature_values
from counterfact.outcome.model import fit_logistic
from counterfact.schema import Decision, Run
from counterfact.stats import Z_95

MAX_CROSS_FIT_FOLDS = 5


@dataclass(frozen=True)
class TreatmentRow:
    arm: str
    propensity: float | None
    decision: Decision
    covariates: tuple[str, ...]
    outcome: int


@dataclass(frozen=True)
class TreatmentData:
    rows: list[TreatmentRow]
    n_excluded: int

    def arm_count(self, arm: str) -> int:
        return sum(1 for r in self.rows if r.arm == arm)


@dataclass(frozen=True)
class ContrastEstimate:
    delta: float
    delta_ci: tuple[float, float]
    g_formula_delta: float
    ipw_delta: float
    n_folds: int


def treatment_data(runs: list[Run], decision_type: str, feature_family: str) -> TreatmentData:
    """One row per run that reaches a `decision_type` decision with an arm, keyed on its first."""
    rows: list[TreatmentRow] = []
    excluded = 0
    for run in runs:
        located = next(
            (
                (step.step_index, d)
                for step in run.steps
                for d in step.decisions
                if d.decision_type == decision_type
            ),
            None,
        )
        arm = logged_arm(located[1], feature_family) if located is not None else None
        if located is None or arm is None:
            excluded += 1
            continue
        treatment_step, decision = located
        covariates = tuple(
            f"{family}::{value}"
            for step in run.steps
            if step.step_index < treatment_step
            for earlier in step.decisions
            for family, value in decision_feature_values(earlier)
        )
        # The logged propensity is the probability of `chosen_action`, so it only
        # describes the treatment when the feature family is the action itself.
        propensity = decision.propensity if feature_family == decision_type else None
        rows.append(
            TreatmentRow(
                arm=arm,
                propensity=propensity,
                decision=decision,
                covariates=covariates,
                outcome=int(binary_outcome_value(run)),
            )
        )
    return TreatmentData(rows=rows, n_excluded=excluded)


def logged_arm(decision: Decision, feature_family: str) -> str | None:
    """The canonical arm `decision` chose within `feature_family`, if it logged one."""
    for family, value in decision_feature_values(decision):
        if family == feature_family:
            return value
    return None


def randomization_gaps(data: TreatmentData, arms: tuple[str, str]) -> list[str]:
    """Reasons the logged assignment does not identify `arms`; empty when it does."""
    gaps: list[str] = []
    unlogged = sum(1 for r in data.rows if r.propensity is None)
    if unlogged:
        gaps.append(f"{unlogged}/{len(data.rows)} runs do not log a propensity for this decision")
    for arm in arms:
        mass = [
            has_mass(r.decision, r.arm, r.propensity, arm)
            for r in data.rows
            if r.propensity is not None
        ]
        zero = mass.count(False)
        unknown = mass.count(None)
        if zero:
            gaps.append(f"{zero} runs give {arm!r} probability 0 (positivity violation)")
        if unknown:
            gaps.append(
                f"{unknown} runs chose another arm under a policy whose probability "
                f"for {arm!r} cannot be reconstructed from the log"
            )
    return gaps


def has_mass(decision: Decision, chosen: str, propensity: float | None, arm: str) -> bool | None:
    """Whether the logged policy gave `arm` positive probability; None when unknowable."""
    if chosen == arm:
        return True
    valid = decision.valid_actions or []
    if arm not in valid:
        return False
    if len(valid) == 2:
        # The other arm's probability is 1 - P(chosen).
        return propensity is not None and propensity < 1.0
    # Policy names emitted by this repo's harnesses: `uniform` (bench/synthetic)
    # and `epsilon_greedy` (bench/real/coding_agent/randomize.py). A named policy
    # counts only when the logged propensity matches what it implies; metadata
    # that contradicts itself proves nothing.
    params = decision.policy_params or {}
    n = len(valid)
    if decision.policy == "uniform" and _matches(propensity, 1.0 / n):
        return True
    if decision.policy == "epsilon_greedy" and "epsilon" in params and "greedy" in params:
        epsilon = float(params["epsilon"])
        chose_greedy = chosen == params["greedy"]
        expected = (1.0 - epsilon) + epsilon / n if chose_greedy else epsilon / n
        if _matches(propensity, expected):
            return epsilon > 0 or params["greedy"] == arm
    return None


def _matches(logged: float | None, expected: float) -> bool:
    return logged is not None and math.isclose(logged, expected, rel_tol=1e-9, abs_tol=1e-9)


def adjustment_set(data: TreatmentData) -> list[str]:
    return sorted({c for r in data.rows for c in r.covariates})


def manski_bounds(data: TreatmentData, target: str, baseline: str) -> tuple[float, float]:
    """No-assumption bounds on E[Y(target)] - E[Y(baseline)] for a binary outcome."""
    n = len(data.rows)

    def arm_bounds(arm: str) -> tuple[float, float]:
        joint = sum(r.outcome for r in data.rows if r.arm == arm) / n
        other = sum(1 for r in data.rows if r.arm != arm) / n
        return joint, joint + other

    t_lo, t_hi = arm_bounds(target)
    b_lo, b_hi = arm_bounds(baseline)
    return t_lo - b_hi, t_hi - b_lo


def propensity_summary(data: TreatmentData, arm: str) -> tuple[float, float]:
    """Return (minimum logged propensity, Kish effective sample size) for `arm`."""
    weights = np.array([1.0 / r.propensity for r in data.rows if r.arm == arm and r.propensity])
    propensities = 1.0 / weights
    ess = float(weights.sum() ** 2 / (weights**2).sum())
    return float(propensities.min()), ess


def aipw_contrast(
    data: TreatmentData, target: str, baseline: str, *, seed: int | None
) -> ContrastEstimate:
    """Cross-fitted AIPW contrast with influence-function 95% intervals.

    Caller must have checked `randomization_gaps(...)` is empty, so every row
    has a positive logged propensity.
    """
    arms = sorted({r.arm for r in data.rows})
    covariate_names = adjustment_set(data)
    columns = {name: i for i, name in enumerate([f"arm::{a}" for a in arms] + covariate_names)}
    n = len(data.rows)

    X = np.zeros((n, len(columns)), dtype=float)
    for i, row in enumerate(data.rows):
        X[i, columns[f"arm::{row.arm}"]] = 1.0
        for name in row.covariates:
            X[i, columns[name]] = 1.0
    y = np.array([r.outcome for r in data.rows], dtype=float)
    a = np.array([r.arm for r in data.rows])
    pi = np.array([r.propensity for r in data.rows], dtype=float)
    arm_columns = [columns[f"arm::{arm}"] for arm in arms]

    n_folds = min(MAX_CROSS_FIT_FOLDS, n)
    folds = np.random.default_rng(seed).permutation(n) % n_folds
    mu_target = np.zeros(n)
    mu_baseline = np.zeros(n)
    for k in range(n_folds):
        train, held = folds != k, folds == k
        mu_target[held] = _predict_under(
            X[held], X[train], y[train], arm_columns, columns[f"arm::{target}"]
        )
        mu_baseline[held] = _predict_under(
            X[held], X[train], y[train], arm_columns, columns[f"arm::{baseline}"]
        )

    is_target = (a == target).astype(float)
    is_baseline = (a == baseline).astype(float)
    psi_target = mu_target + is_target / pi * (y - mu_target)
    psi_baseline = mu_baseline + is_baseline / pi * (y - mu_baseline)
    psi_delta = psi_target - psi_baseline

    w_target = is_target / pi
    w_baseline = is_baseline / pi
    ipw = float((w_target * y).sum() / w_target.sum() - (w_baseline * y).sum() / w_baseline.sum())

    return ContrastEstimate(
        delta=float(psi_delta.mean()),
        delta_ci=_mean_ci(psi_delta),
        g_formula_delta=float((mu_target - mu_baseline).mean()),
        ipw_delta=ipw,
        n_folds=n_folds,
    )


def _predict_under(
    X_held: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    arm_columns: list[int],
    arm_column: int,
) -> np.ndarray:
    """Outcome-model prediction for held-out rows with the arm set to `arm_column`."""
    X_set = X_held.copy()
    X_set[:, arm_columns] = 0.0
    X_set[:, arm_column] = 1.0
    if len(set(y_train.tolist())) < 2:
        # A one-class training fold has no slope to learn; its mean is the fit.
        return np.full(len(X_held), float(y_train.mean()))
    result: np.ndarray = fit_logistic(X_train, y_train.astype(int)).predict_proba(X_set)[:, 1]
    return result


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    half_width = Z_95 * float(values.std(ddof=1)) / np.sqrt(len(values))
    mean = float(values.mean())
    return mean - half_width, mean + half_width
