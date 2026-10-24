"""Off-policy evaluation of agent policies over logged decision sequences.

The estimand is V(π) - V(baseline): the change in success probability if the
agent had followed π at every decision π controls, with every other decision
left as logged. The baseline is the logging policy itself (V = mean outcome)
or a second policy.

Agent decisions react to what happened earlier. A retry depends on whether
the tests passed, and the tests passing depends on the model chosen earlier.
That is time-varying confounding, so a single-decision contrast cannot answer
the question. This module indexes time by controlled-decision occurrence k and
estimates V(π) three ways:

* sequential doubly robust (headline): Bang and Robins' iterated regression
  plus the importance-weighted residual at every controlled node. Logged
  propensities are exact, so it stays consistent when the outcome models are
  wrong.
* iterated conditional expectation (ICE) g-formula and self-normalized
  importance sampling (SNIS) as cross-checks.

With a terminal-only outcome, per-decision importance sampling equals
trajectory importance sampling, so it is not reported separately.

Identification is checked on the data. Every controlled decision in every run
must log a propensity and give π's action positive probability. Otherwise the
answer is Manski bounds. A policy that cannot be evaluated on some run (its
feature is missing, or two controlled decisions share a step) is unidentified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from counterfact.intervene.estimate import (
    DistributionSummary,
    IdentifiabilityStatus,
    IdentificationBounds,
    NextStep,
)
from counterfact.intervene.estimators import has_mass
from counterfact.outcome.binary import binary_outcome_value
from counterfact.outcome.features import canonical_feature_value, decision_feature_values
from counterfact.outcome.model import fit_logistic
from counterfact.policy.model import HistoryFeature, Policy
from counterfact.schema import Decision, Run
from counterfact.stats import Z_95

MAX_CROSS_FIT_FOLDS = 5


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyDiagnostics(_Strict):
    n_runs: int
    n_controlled_decisions: int
    n_matching_runs: int
    effective_n: float
    max_weight: float
    policy_value: float
    baseline_value: float
    snis_delta: float
    ice_delta: float
    # Descriptive, not causal: mean outcome of runs that already followed each
    # policy everywhere it decides. It is what filtering the logs would report.
    matched_runs_delta: float | None
    n_folds: int


class PolicyEstimate(_Strict):
    policy: str
    baseline: str
    identifiability: IdentifiabilityStatus
    estimand: str
    outcome_delta: DistributionSummary | None = None
    bounds: IdentificationBounds | None = None
    diagnostics: PolicyDiagnostics | None = None
    reason: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_step: NextStep


@dataclass(frozen=True)
class _Node:
    step: int
    decision: Decision
    history: tuple[str, ...]
    target: str
    chosen: str


@dataclass(frozen=True)
class _RunNodes:
    nodes: list[_Node]
    outcome: float


class _Unevaluable(Exception):
    """The policy cannot be evaluated on some run; the reason is the message."""


def evaluate_policy(
    runs: list[Run],
    policy: Policy,
    *,
    baseline: Policy | None = None,
    observation_keys: set[str] | None = None,
    seed: int | None = 0,
) -> PolicyEstimate:
    """Estimate V(policy) - V(baseline), where `baseline=None` means the logged policy.

    `observation_keys` names structured observation fields, such as
    `tests_passed`, that the outcome models may condition on in addition to
    the keys the policies' rules read. They only sharpen the outcome models:
    the DR estimate stays consistent without them because propensities are logged.
    """
    baseline_name = baseline.name if baseline is not None else "logged"
    estimand = f"V({policy.name}) - V({baseline_name})"
    if not runs:
        raise ValueError("evaluate_policy requires at least one run")
    outcomes = {binary_outcome_value(r) for r in runs}
    if len(outcomes) < 2:
        return _refusal(
            policy,
            baseline_name,
            estimand,
            reason=f"every run has outcome={outcomes.pop()}; no outcome variation to evaluate",
            missing=["outcome with both classes"],
        )

    try:
        keys = (observation_keys or set()) | policy.observation_keys()
        if baseline is not None:
            keys |= baseline.observation_keys()
        target = [_run_nodes(r, policy, keys) for r in runs]
        reference = [_run_nodes(r, baseline, keys) for r in runs] if baseline is not None else None
    except _Unevaluable as exc:
        return _refusal(policy, baseline_name, estimand, reason=str(exc), missing=[str(exc)])

    for name, nodes in ((policy.name, target), (baseline_name, reference)):
        if nodes is not None and not any(r.nodes for r in nodes):
            # A policy that never acts would trivially equal the log; in practice
            # it means the corpus has no evidence about the question asked.
            return _refusal(
                policy,
                baseline_name,
                estimand,
                reason=f"no run contains a decision that policy {name!r} controls",
                missing=[f"decisions controlled by {name!r}"],
            )

    gaps = _positivity_gaps(target) + (_positivity_gaps(reference) if reference else [])
    if gaps:
        return _bounded(policy, baseline_name, estimand, target, reference, gaps)

    folds = _folds(len(runs), seed)
    psi, ice, snis, weights = _sequential_dr(target, folds)
    y = np.array([r.outcome for r in target])
    matched = _matched_mean(target)
    if reference is not None:
        psi_b, ice_b, snis_b, _ = _sequential_dr(reference, folds)
        matched_b = _matched_mean(reference)
    else:
        psi_b, ice_b, snis_b = y, float(y.mean()), float(y.mean())
        matched_b = float(y.mean())

    contrast = psi - psi_b
    half = Z_95 * float(contrast.std(ddof=1)) / np.sqrt(len(contrast))
    point = float(contrast.mean())
    matching = int((weights > 0).sum())
    warnings = []
    if matching == 0:
        warnings.append(
            "no run followed the policy at every controlled decision; the estimate "
            "rests entirely on the outcome models' extrapolation"
        )
    return PolicyEstimate(
        policy=policy.name,
        baseline=baseline_name,
        identifiability=IdentifiabilityStatus.IDENTIFIED,
        estimand=estimand,
        outcome_delta=DistributionSummary(point=point, ci_low=point - half, ci_high=point + half),
        diagnostics=PolicyDiagnostics(
            n_runs=len(runs),
            n_controlled_decisions=sum(len(r.nodes) for r in target),
            n_matching_runs=matching,
            effective_n=_kish(weights),
            max_weight=float(weights.max()),
            policy_value=float(psi.mean()),
            baseline_value=float(psi_b.mean()),
            snis_delta=snis - snis_b,
            ice_delta=ice - ice_b,
            matched_runs_delta=(
                matched - matched_b if matched is not None and matched_b is not None else None
            ),
            n_folds=int(folds.max()) + 1,
        ),
        assumptions=[
            "logged propensities are the true probabilities of each controlled decision",
            "every controlled decision gave the policy's action positive probability",
            "decisions the policy does not control would be made as the agent logged them",
            "consistency and no interference between runs",
        ],
        warnings=warnings,
        next_step=NextStep(
            action="none",
            payload={},
            human_text="identified from logged randomization; read the CI width for precision.",
        ),
    )


def _history(run: Run, before_step: int, observation_keys: set[str]) -> tuple[str, ...]:
    features: list[str] = []
    for step in run.steps:
        if step.step_index >= before_step:
            continue
        for decision in step.decisions:
            features.extend(
                f"{family}::{value}" for family, value in decision_feature_values(decision)
            )
        for observation in step.observations:
            for key in observation_keys & set(observation.content):
                features.append(f"obs::{key}::{canonical_feature_value(observation.content[key])}")
    return tuple(features)


def _feature_value(run: Run, before_step: int, feature: HistoryFeature) -> str | None:
    value: str | None = None
    for step in sorted(run.steps, key=lambda st: st.step_index):
        if step.step_index >= before_step:
            continue
        if feature.source == "decision":
            for decision in step.decisions:
                if decision.decision_type == feature.key and decision.chosen_action is not None:
                    value = canonical_feature_value(decision.chosen_action)
        else:
            for observation in step.observations:
                if feature.key in observation.content:
                    value = canonical_feature_value(observation.content[feature.key])
    return value


def _run_nodes(run: Run, policy: Policy, keys: set[str]) -> _RunNodes:
    # Time is step_index, not list position: a run may list steps out of order
    # or split one index across several Step entries.
    by_step: dict[int, list[Decision]] = {}
    for step in run.steps:
        for decision in step.decisions:
            if policy.rule_for(decision.decision_type) is not None:
                by_step.setdefault(step.step_index, []).append(decision)
    nodes: list[_Node] = []
    for step_index in sorted(by_step):
        controlled = by_step[step_index]
        if len(controlled) > 1:
            raise _Unevaluable(
                f"run {run.run_id!r} has {len(controlled)} controlled decisions at step "
                f"{step_index}; they cannot be ordered"
            )
        for decision in controlled:
            if decision.chosen_action is None:
                raise _Unevaluable(f"decision {decision.decision_id!r} logs no chosen_action")
            rule = policy.rule_for(decision.decision_type)
            assert rule is not None
            if rule.action is not None:
                target = rule.action
            else:
                assert rule.when is not None and rule.cases is not None
                value = _feature_value(run, step_index, rule.when)
                if value is None or value not in rule.cases:
                    raise _Unevaluable(
                        f"policy {policy.name!r} is undefined at {decision.decision_id!r}: "
                        f"{rule.when.source} {rule.when.key!r} is {value!r} in the earlier history"
                    )
                target = rule.cases[value]
            nodes.append(
                _Node(
                    step=step_index,
                    decision=decision,
                    history=_history(run, step_index, keys),
                    target=target,
                    chosen=canonical_feature_value(decision.chosen_action),
                )
            )
    return _RunNodes(nodes=nodes, outcome=float(binary_outcome_value(run)))


def _positivity_gaps(runs: list[_RunNodes]) -> list[str]:
    """Support gaps on the nodes the policy can reach.

    Once a run's logged action departs from the policy, its later nodes are off
    the policy's path: no estimator weights them, so they need no support.
    """
    unlogged = zero = unknown = 0
    for run in runs:
        for node in run.nodes:
            if node.decision.propensity is None:
                unlogged += 1
            else:
                mass = has_mass(node.decision, node.chosen, node.decision.propensity, node.target)
                zero += mass is False
                unknown += mass is None
            if node.chosen != node.target:
                break
    gaps = []
    if unlogged:
        gaps.append(f"{unlogged} controlled decisions log no propensity")
    if zero:
        gaps.append(f"{zero} controlled decisions give the policy's action probability 0")
    if unknown:
        gaps.append(
            f"{unknown} controlled decisions log a policy whose probability for the "
            "policy's action cannot be reconstructed"
        )
    return gaps


def _matched_all(run: _RunNodes) -> bool:
    return all(node.chosen == node.target for node in run.nodes)


def _matched_mean(runs: list[_RunNodes]) -> float | None:
    outcomes = [r.outcome for r in runs if _matched_all(r)]
    return float(np.mean(outcomes)) if outcomes else None


def _manski(runs: list[_RunNodes]) -> tuple[float, float]:
    """No-assumption bounds on V(π) for a binary outcome."""
    n = len(runs)
    joint = sum(r.outcome for r in runs if _matched_all(r)) / n
    unmatched = sum(1 for r in runs if not _matched_all(r)) / n
    return joint, joint + unmatched


def _bounded(
    policy: Policy,
    baseline_name: str,
    estimand: str,
    target: list[_RunNodes],
    reference: list[_RunNodes] | None,
    gaps: list[str],
) -> PolicyEstimate:
    lo, hi = _manski(target)
    if reference is None:
        logged = float(np.mean([r.outcome for r in target]))
        lo_b = hi_b = logged
    else:
        lo_b, hi_b = _manski(reference)
    controlled = sorted({rule.decision_type for rule in policy.rules})
    return PolicyEstimate(
        policy=policy.name,
        baseline=baseline_name,
        identifiability=IdentifiabilityStatus.BOUNDED,
        estimand=estimand,
        bounds=IdentificationBounds(
            lower=lo - hi_b,
            upper=hi - lo_b,
            note=f"Manski bounds over {len(target)} runs; they hold under any confounding",
        ),
        reason="the log does not identify this policy: " + "; ".join(gaps),
        assumptions=["none beyond consistency"],
        next_step=NextStep(
            action="add_arm_randomization",
            payload={
                "arm_name": ", ".join(controlled),
                "current_policy": "logged without usable propensities",
                "randomization_gaps": gaps,
            },
            human_text=(
                f"Randomize {', '.join(controlled)} with logged propensities that give "
                "the policy's actions positive probability."
            ),
        ),
    )


def _refusal(
    policy: Policy, baseline_name: str, estimand: str, *, reason: str, missing: list[str]
) -> PolicyEstimate:
    return PolicyEstimate(
        policy=policy.name,
        baseline=baseline_name,
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        estimand=estimand,
        reason=reason,
        next_step=NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": ", ".join(sorted(r.decision_type for r in policy.rules)),
                "missing_strata": missing,
                "observed_arms": [],
                "missing_arms": [],
            },
            human_text="Collect traces on which this policy is defined and outcomes vary.",
        ),
    )


def _folds(n: int, seed: int | None) -> np.ndarray:
    n_folds = min(MAX_CROSS_FIT_FOLDS, n)
    folds: np.ndarray = np.random.default_rng(seed).permutation(n) % n_folds
    return folds


def _kish(weights: np.ndarray) -> float:
    total = float((weights**2).sum())
    return float(weights.sum() ** 2 / total) if total > 0 else 0.0


def _sequential_dr(
    runs: list[_RunNodes], folds: np.ndarray
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Return per-run DR scores, the ICE value, the SNIS value, and trajectory weights."""
    n = len(runs)
    psi = np.zeros(n)
    ice = np.zeros(n)
    for fold in range(int(folds.max()) + 1):
        train = [runs[i] for i in range(n) if folds[i] != fold]
        held = [i for i in range(n) if folds[i] == fold]
        models, fallback = _fit_ice(train)
        for i in held:
            psi[i], ice[i] = _dr_score(runs[i], models, fallback)
    weights = np.array([_trajectory_weight(r) for r in runs])
    y = np.array([r.outcome for r in runs])
    snis = float((weights * y).sum() / weights.sum()) if weights.sum() > 0 else float("nan")
    return psi, float(ice.mean()), snis, weights


def _trajectory_weight(run: _RunNodes) -> float:
    weight = 1.0
    for node in run.nodes:
        if node.chosen != node.target:
            return 0.0
        assert node.decision.propensity is not None
        weight /= node.decision.propensity
    return weight


@dataclass
class _QModel:
    columns: dict[str, int]
    model: object | None
    constant: float

    def predict(self, features: tuple[str, ...]) -> float:
        if self.model is None:
            return self.constant
        x = np.zeros((1, len(self.columns)))
        for name in features:
            if name in self.columns:
                x[0, self.columns[name]] = 1.0
        prob: float = self.model.predict_proba(x)[0, 1]  # type: ignore[attr-defined]
        return prob


def _row_features(node: _Node, action: str) -> tuple[str, ...]:
    kind = node.decision.decision_type
    return (*node.history, f"node::{kind}", f"action::{kind}::{action}")


def _fit_q(rows: list[tuple[tuple[str, ...], float]], fallback: float) -> _QModel:
    """Logistic Q on fractional targets, via duplicated rows weighted v and 1 - v."""
    if not rows:
        return _QModel({}, None, fallback)
    targets = np.array([v for _, v in rows])
    if np.allclose(targets, targets[0]):
        return _QModel({}, None, float(targets[0]))
    names = sorted({name for features, _ in rows for name in features})
    columns = {name: i for i, name in enumerate(names)}
    X = np.zeros((len(rows), len(columns)))
    for r, (features, _) in enumerate(rows):
        for name in features:
            X[r, columns[name]] = 1.0
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(rows), dtype=int), np.zeros(len(rows), dtype=int)])
    w2 = np.concatenate([targets, 1.0 - targets])
    return _QModel(columns, fit_logistic(X2, y2, sample_weight=w2), float(targets.mean()))


def _fit_ice(train: list[_RunNodes]) -> tuple[list[_QModel], float]:
    """Fit Q_k backward over controlled-decision occurrence k on the training runs.

    Also returns the training mean outcome, the only prediction available for a
    held-out node deeper than every training run. It must never be the held-out
    run's own outcome, or the nuisance would leak the answer.
    """
    depth = max((len(r.nodes) for r in train), default=0)
    fallback = float(np.mean([r.outcome for r in train])) if train else 0.5
    models: list[_QModel] = [_QModel({}, None, fallback)] * depth
    for k in range(depth - 1, -1, -1):
        rows = []
        for run in train:
            if len(run.nodes) <= k:
                continue
            node = run.nodes[k]
            rows.append((_row_features(node, node.chosen), _next_value(run, k, models)))
        models[k] = _fit_q(rows, fallback)
    return models, fallback


def _next_value(run: _RunNodes, k: int, models: list[_QModel]) -> float:
    """V_{k+1}: the next node's Q under the policy, or the outcome after the last node."""
    if k + 1 >= len(run.nodes):
        return run.outcome
    node = run.nodes[k + 1]
    return models[k + 1].predict(_row_features(node, node.target))


def _dr_score(run: _RunNodes, models: list[_QModel], fallback: float) -> tuple[float, float]:
    """Per-run sequential DR score and its ICE plug-in value."""
    if not run.nodes:
        return run.outcome, run.outcome

    def q(k: int, action: str) -> float:
        if k >= len(models):
            return fallback
        return models[k].predict(_row_features(run.nodes[k], action))

    first = q(0, run.nodes[0].target)
    score = first
    weight = 1.0
    for k, node in enumerate(run.nodes):
        if node.chosen != node.target:
            break
        assert node.decision.propensity is not None
        weight /= node.decision.propensity
        next_value = run.outcome if k + 1 == len(run.nodes) else q(k + 1, run.nodes[k + 1].target)
        score += weight * (next_value - q(k, node.chosen))
    return score, first
