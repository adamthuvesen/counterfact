"""Score policy evaluation on the sequential scenario over many seeds.

Each case evaluates V(policy) - V(logged) against the exact value from
`bench.synthetic.sequential.policy_value`. The estimators are the sequential
doubly robust estimate with its 95% CI, and its ICE g-formula and SNIS
cross-checks. One case drops the retry propensity, so the engine must refuse.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from bench.synthetic.sequential import policy_value, sample
from counterfact.policy import Policy, evaluate_policy
from counterfact.schema import Run

N_RUNS = 2000

_ON_FAIL = {
    "decision_type": "retry",
    "when": {"source": "observation", "key": "tests_passed"},
    "cases": {"true": "no_retry", "false": "retry_once"},
}


def _retry_on_fail(passed: bool) -> str:
    return "no_retry" if passed else "retry_once"


@dataclass(frozen=True)
class PolicyCase:
    name: str
    policy: Policy
    model_rule: str | None
    retry_rule: Callable[[bool], str] | None
    observation_keys: frozenset[str]
    log_retry_propensity: bool
    expected_label: str

    @property
    def true_delta(self) -> float:
        return policy_value(self.model_rule, self.retry_rule) - policy_value(None, None)


CASES: tuple[PolicyCase, ...] = (
    PolicyCase(
        "always_retry, outcome model blind to tests_passed",
        Policy(name="always_retry", rules=[{"decision_type": "retry", "action": "retry_once"}]),
        None,
        lambda passed: "retry_once",
        frozenset(),
        True,
        "identified",
    ),
    PolicyCase(
        "always_retry",
        Policy(name="always_retry", rules=[{"decision_type": "retry", "action": "retry_once"}]),
        None,
        lambda passed: "retry_once",
        frozenset({"tests_passed"}),
        True,
        "identified",
    ),
    PolicyCase(
        "never_retry",
        Policy(name="never_retry", rules=[{"decision_type": "retry", "action": "no_retry"}]),
        None,
        lambda passed: "no_retry",
        frozenset({"tests_passed"}),
        True,
        "identified",
    ),
    PolicyCase(
        "retry_on_fail",
        Policy(name="retry_on_fail", rules=[_ON_FAIL]),
        None,
        _retry_on_fail,
        frozenset({"tests_passed"}),
        True,
        "identified",
    ),
    PolicyCase(
        "sonnet_then_retry_on_fail",
        Policy(
            name="sonnet_then_retry_on_fail",
            rules=[{"decision_type": "model_call", "action": "sonnet"}, _ON_FAIL],
        ),
        "sonnet",
        _retry_on_fail,
        frozenset({"tests_passed"}),
        True,
        "identified",
    ),
    PolicyCase(
        "retry_on_fail, retry propensity not logged",
        Policy(name="retry_on_fail", rules=[_ON_FAIL]),
        None,
        _retry_on_fail,
        frozenset({"tests_passed"}),
        False,
        "bounded",
    ),
)


@dataclass(frozen=True)
class PolicySeedResult:
    case: str
    label: str
    dr: float | None
    ci_low: float | None
    ci_high: float | None
    ice: float | None
    snis: float | None
    matched: float | None
    effective_n: float | None


@dataclass(frozen=True)
class PolicySummary:
    case: str
    n_runs: int
    n_seeds: int
    true_delta: float
    expected_label: str
    label_accuracy: float
    dr_bias: float | None
    dr_rmse: float | None
    ci_coverage: float | None
    ice_bias: float | None
    snis_bias: float | None
    matched_bias: float | None
    mean_effective_n: float | None


def run_policy_seed(case_name: str, seed: int) -> PolicySeedResult:
    case = next(c for c in CASES if c.name == case_name)
    rng = random.Random(f"sequential:{case.log_retry_propensity}:{seed}")
    runs = [
        Run.model_validate(sample(rng, i, log_retry_propensity=case.log_retry_propensity))
        for i in range(N_RUNS)
    ]
    estimate = evaluate_policy(
        runs, case.policy, observation_keys=set(case.observation_keys), seed=seed
    )
    delta, diag = estimate.outcome_delta, estimate.diagnostics
    return PolicySeedResult(
        case=case.name,
        label=estimate.identifiability.value,
        dr=delta.point if delta else None,
        ci_low=delta.ci_low if delta else None,
        ci_high=delta.ci_high if delta else None,
        ice=diag.ice_delta if diag else None,
        snis=diag.snis_delta if diag else None,
        matched=diag.matched_runs_delta if diag else None,
        effective_n=diag.effective_n if diag else None,
    )


def _bias(values: list[float | None], truth: float) -> tuple[float | None, float | None]:
    errors = np.array([v - truth for v in values if v is not None])
    if errors.size == 0:
        return None, None
    return float(errors.mean()), float(math.sqrt((errors**2).mean()))


def run_policy_suite(seeds: int, *, workers: int | None = None) -> list[PolicySummary]:
    jobs = [(c.name, seed) for c in CASES for seed in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_policy_seed, *zip(*jobs, strict=True), chunksize=4))
    summaries = []
    for case in CASES:
        rows = [r for r in results if r.case == case.name]
        truth = case.true_delta
        identified = [r for r in rows if r.label == "identified"]
        dr_bias, dr_rmse = _bias([r.dr for r in identified], truth)
        covered = [
            r.ci_low <= truth <= r.ci_high  # type: ignore[operator]
            for r in identified
        ]
        ess = [r.effective_n for r in identified if r.effective_n is not None]
        summaries.append(
            PolicySummary(
                case=case.name,
                n_runs=N_RUNS,
                n_seeds=len(rows),
                true_delta=truth,
                expected_label=case.expected_label,
                label_accuracy=sum(r.label == case.expected_label for r in rows) / len(rows),
                dr_bias=dr_bias,
                dr_rmse=dr_rmse,
                ci_coverage=sum(covered) / len(covered) if covered else None,
                ice_bias=_bias([r.ice for r in identified], truth)[0],
                snis_bias=_bias([r.snis for r in identified], truth)[0],
                matched_bias=_bias([r.matched for r in identified], truth)[0],
                mean_effective_n=float(np.mean(ess)) if ess else None,
            )
        )
    return summaries


def _num(value: float | None, fmt: str = "+.3f") -> str:
    return "—" if value is None else format(value, fmt)


def policy_markdown(summaries: list[PolicySummary]) -> str:
    lines = [
        "| Policy case | Truth V(π) - V(logged) | Matched-runs bias | Engine label (expected) "
        "| DR bias (RMSE) | 95% CI coverage | ICE bias | SNIS bias | Mean ESS |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        dr = "—" if s.dr_bias is None else f"{_num(s.dr_bias)} ({_num(s.dr_rmse, '.3f')})"
        coverage = "—" if s.ci_coverage is None else f"{s.ci_coverage:.0%}"
        lines.append(
            f"| `{s.case}` | {_num(s.true_delta)} | {_num(s.matched_bias)} "
            f"| {s.label_accuracy:.0%} {s.expected_label} "
            f"| {dr} | {coverage} | {_num(s.ice_bias)} | {_num(s.snis_bias)} "
            f"| {_num(s.mean_effective_n, '.0f')} |"
        )
    return "\n".join(lines) + "\n"
