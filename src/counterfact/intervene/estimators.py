"""Contrast estimators for one decision type across a trace corpus.

The treatment is the arm a run chose at its *first* decision of the targeted
type. Later decisions of the same type are downstream of the treatment, so
they act as mediators: the run stays in, and the estimand is the total effect
of the first choice. Dropping such runs would select on a post-treatment
variable. Covariates are the intervenable decisions logged strictly before the
first occurrence, so the adjustment set never contains a mediator or a
collider that the treatment itself caused. Runs that never reach the decision
are left out, which conditions only on a pre-treatment event.

Identification is checked against the data, not declared:

* every analyzed run must log the propensity of its chosen arm, and
* no run may have chosen another arm with propensity 1.0 or have listed
  `valid_actions` that exclude a contrasted arm (positivity).

Only then is the cross-fitted AIPW estimate reported as identified. The
outcome-regression (g-formula) and Hajek IPW estimates ride along as
cross-checks. Without logged randomization, the only defensible answer for a
binary outcome is the Manski no-assumption interval.
"""

from __future__ import annotations

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
        flat = [d for step in run.steps for d in step.decisions]
        position = next((i for i, d in enumerate(flat) if d.decision_type == decision_type), None)
        arm = logged_arm(flat[position], feature_family) if position is not None else None
        if position is None or arm is None:
            excluded += 1
            continue
        decision = flat[position]
        covariates = tuple(
            f"{family}::{value}"
            for earlier in flat[:position]
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
        forced_away = sum(1 for r in data.rows if r.arm != arm and r.propensity == 1.0)
        if forced_away:
            gaps.append(
                f"{forced_away} runs chose another arm with propensity 1.0, so "
                f"P({arm}) = 0 for them (positivity violation)"
            )
        excluded_by_menu = sum(
            1
            for r in data.rows
            if r.decision.valid_actions is not None and arm not in r.decision.valid_actions
        )
        if excluded_by_menu:
            gaps.append(f"{excluded_by_menu} runs log valid_actions that exclude {arm!r}")
    return gaps


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
