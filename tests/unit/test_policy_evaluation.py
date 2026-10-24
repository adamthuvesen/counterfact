"""Policy evaluation: identification labels, estimates against exact truths, and
agreement with the single-decision AIPW contrast."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from bench.synthetic.scenarios import by_name
from bench.synthetic.sequential import policy_value, sample
from counterfact import build_dag, fit_outcome_model, intervene
from counterfact.intervene import IdentifiabilityStatus
from counterfact.policy import Policy, evaluate_policy
from counterfact.schema import Decision, Outcome, Run, Step

ON_FAIL = {
    "decision_type": "retry",
    "when": {"source": "observation", "key": "tests_passed"},
    "cases": {"true": "no_retry", "false": "retry_once"},
}


def _retry_on_fail(passed: bool) -> str:
    return "no_retry" if passed else "retry_once"


def _sequential(n: int = 2000, *, log_retry_propensity: bool = True, seed: int = 0) -> list[Run]:
    rng = random.Random(seed)
    return [
        Run.model_validate(sample(rng, i, log_retry_propensity=log_retry_propensity))
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def sequential_runs() -> list[Run]:
    return _sequential()


def test_rule_must_be_a_constant_or_a_lookup() -> None:
    with pytest.raises(ValidationError):
        Policy(name="both", rules=[{"decision_type": "retry", "action": "no_retry", **ON_FAIL}])
    with pytest.raises(ValidationError):
        Policy(name="neither", rules=[{"decision_type": "retry"}])
    with pytest.raises(ValidationError):
        Policy(name="dup", rules=[ON_FAIL, {"decision_type": "retry", "action": "no_retry"}])


@pytest.mark.parametrize(
    ("rules", "model_rule", "retry_rule"),
    [
        ([{"decision_type": "retry", "action": "retry_once"}], None, lambda p: "retry_once"),
        ([{"decision_type": "model_call", "action": "sonnet"}, ON_FAIL], "sonnet", _retry_on_fail),
    ],
)
def test_sequential_dr_covers_the_exact_policy_value(
    sequential_runs: list[Run], rules: list[dict], model_rule, retry_rule
) -> None:
    estimate = evaluate_policy(
        sequential_runs, Policy(name="p", rules=rules), observation_keys={"tests_passed"}
    )
    truth = policy_value(model_rule, retry_rule) - policy_value(None, None)

    assert estimate.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert estimate.outcome_delta is not None
    assert estimate.outcome_delta.ci_low <= truth <= estimate.outcome_delta.ci_high


def test_filtering_the_logs_gets_never_retry_backwards(sequential_runs: list[Run]) -> None:
    """Runs that skipped the retry mostly passed first time, so they look good.
    The engine sees that and reports that never retrying hurts."""
    estimate = evaluate_policy(
        sequential_runs,
        Policy(name="never_retry", rules=[{"decision_type": "retry", "action": "no_retry"}]),
        observation_keys={"tests_passed"},
    )
    truth = policy_value(None, lambda p: "no_retry") - policy_value(None, None)

    assert truth < 0
    assert estimate.diagnostics is not None
    assert estimate.diagnostics.matched_runs_delta is not None
    assert estimate.diagnostics.matched_runs_delta > 0
    assert estimate.outcome_delta is not None and estimate.outcome_delta.ci_high < 0


def test_unlogged_retry_propensity_is_bounded_and_the_bounds_hold() -> None:
    runs = _sequential(log_retry_propensity=False)
    estimate = evaluate_policy(runs, Policy(name="retry_on_fail", rules=[ON_FAIL]))
    truth = policy_value(None, _retry_on_fail) - policy_value(None, None)

    assert estimate.identifiability == IdentifiabilityStatus.BOUNDED
    assert estimate.bounds is not None
    assert estimate.bounds.lower <= truth <= estimate.bounds.upper
    assert estimate.next_step.action == "add_arm_randomization"


def test_policy_undefined_on_the_logged_history_is_unidentified(
    sequential_runs: list[Run],
) -> None:
    rule = {**ON_FAIL, "when": {"source": "observation", "key": "lint_passed"}}
    estimate = evaluate_policy(sequential_runs, Policy(name="lint", rules=[rule]))

    assert estimate.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert estimate.reason is not None and "lint_passed" in estimate.reason


def test_two_controlled_decisions_in_one_step_are_unidentified() -> None:
    def run(i: int, passed: bool) -> Run:
        decisions = [
            Decision(
                decision_id=f"d-{i}-{kind}",
                decision_type=kind,
                chosen_action=action,
                policy="uniform",
                policy_params={},
                valid_actions=arms,
                propensity=0.5,
                context_features={},
            )
            for kind, action, arms in (
                ("model_call", "haiku", ["haiku", "sonnet"]),
                ("retry", "no_retry", ["no_retry", "retry_once"]),
            )
        ]
        return Run(
            schema_version="0.1.0",
            run_id=f"r{i}",
            steps=[Step(step_index=0, decisions=decisions)],
            outcome=Outcome(kind="binary", value=passed, verifier="t"),
        )

    policy = Policy(
        name="both",
        rules=[
            {"decision_type": "model_call", "action": "sonnet"},
            {"decision_type": "retry", "action": "retry_once"},
        ],
    )
    estimate = evaluate_policy([run(0, True), run(1, False)], policy)

    assert estimate.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert "cannot be ordered" in (estimate.reason or "")


def test_single_class_corpus_is_unidentified(sequential_runs: list[Run]) -> None:
    all_pass = [
        r.model_copy(update={"outcome": r.outcome.model_copy(update={"value": True})})
        for r in sequential_runs[:50]
    ]
    estimate = evaluate_policy(all_pass, Policy(name="p", rules=[ON_FAIL]))

    assert estimate.identifiability == IdentifiabilityStatus.UNIDENTIFIED


def test_single_decision_policy_contrast_agrees_with_the_aipw_contrast() -> None:
    """With one controlled decision, V(sonnet) - V(haiku) as a policy contrast is
    the same estimand as the engine's single-decision contrast."""
    scenario = by_name("logged_confounding")
    rng = random.Random("consistency")
    runs = [Run.model_validate(scenario.sample(rng, i, None)) for i in range(2000)]
    single = intervene(
        dag=build_dag(runs[0]),
        model=fit_outcome_model(runs, seed=0),
        step=1,
        intervention={"model_choice": "sonnet"},
        baseline="haiku",
    )
    policy = evaluate_policy(
        runs,
        Policy(name="sonnet", rules=[{"decision_type": "model_call", "action": "sonnet"}]),
        baseline=Policy(name="haiku", rules=[{"decision_type": "model_call", "action": "haiku"}]),
    )

    assert single.outcome_delta is not None and policy.outcome_delta is not None
    assert abs(single.outcome_delta.point - policy.outcome_delta.point) < 0.02
    assert abs(policy.outcome_delta.point - scenario.true_effect) < 0.08


def test_held_out_outcome_never_stands_in_for_its_own_outcome_model() -> None:
    """Run A retried and passed; run B never reached a retry and failed. With two
    folds, A's outcome model is fit on B alone, which has no retry to learn
    from, so A's prediction must fall back to B's mean outcome (0), not to A's
    own outcome. Then V(never_retry) - V(logged) = mean(0 - 1, 0 - 0) = -0.5.
    Leaking A's outcome would report a zero-width 0.0 instead."""
    retried = Run(
        schema_version="0.1.0",
        run_id="a",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-a-retry",
                        decision_type="retry",
                        chosen_action="retry_once",
                        policy="uniform",
                        policy_params={},
                        valid_actions=["no_retry", "retry_once"],
                        propensity=0.5,
                        context_features={},
                    )
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=True, verifier="t"),
    )
    no_retry_step = Run(
        schema_version="0.1.0",
        run_id="b",
        steps=[],
        outcome=Outcome(kind="binary", value=False, verifier="t"),
    )

    estimate = evaluate_policy(
        [retried, no_retry_step],
        Policy(name="never_retry", rules=[{"decision_type": "retry", "action": "no_retry"}]),
    )

    assert estimate.outcome_delta is not None
    assert estimate.outcome_delta.point == pytest.approx(-0.5)


def _decision(i: int, kind: str, action: str, arms: list[str], propensity: float) -> Decision:
    return Decision(
        decision_id=f"d-{i}-{kind}",
        decision_type=kind,
        chosen_action=action,
        policy="router",
        policy_params={},
        valid_actions=arms,
        propensity=propensity,
        context_features={},
    )


def test_unreachable_nodes_need_no_support() -> None:
    """Haiku runs never retry (propensity 1.0), sonnet runs retry at random.
    A sonnet-then-retry policy never reaches a haiku run's retry node, so the
    haiku branch's missing retry support must not block identification."""
    rng = random.Random(3)
    runs = []
    for i in range(400):
        model = rng.choice(["haiku", "sonnet"])
        if model == "haiku":
            retry, p_retry = "no_retry", 1.0
        else:
            retry = rng.choice(["no_retry", "retry_once"])
            p_retry = 0.5
        runs.append(
            Run(
                schema_version="0.1.0",
                run_id=f"r{i}",
                steps=[
                    Step(
                        step_index=0,
                        decisions=[_decision(i, "model_call", model, ["haiku", "sonnet"], 0.5)],
                    ),
                    Step(
                        step_index=1,
                        decisions=[
                            _decision(i, "retry", retry, ["no_retry", "retry_once"], p_retry)
                        ],
                    ),
                ],
                outcome=Outcome(kind="binary", value=rng.random() < 0.5, verifier="t"),
            )
        )
    policy = Policy(
        name="sonnet_retry",
        rules=[
            {"decision_type": "model_call", "action": "sonnet"},
            {"decision_type": "retry", "action": "retry_once"},
        ],
    )

    assert evaluate_policy(runs, policy).identifiability == IdentifiabilityStatus.IDENTIFIED


def test_controlled_decisions_are_ordered_by_step_index_not_list_position() -> None:
    """Listing the retry step before the model step must not change the estimate,
    and two controlled decisions sharing a step_index must be refused."""
    runs = _sequential(n=600)
    reversed_runs = [r.model_copy(update={"steps": list(reversed(r.steps))}) for r in runs]
    policy = Policy(name="p", rules=[{"decision_type": "model_call", "action": "sonnet"}, ON_FAIL])

    forward = evaluate_policy(runs, policy, observation_keys={"tests_passed"})
    backward = evaluate_policy(reversed_runs, policy, observation_keys={"tests_passed"})
    assert forward.outcome_delta is not None and backward.outcome_delta is not None
    assert backward.outcome_delta.point == pytest.approx(forward.outcome_delta.point)

    split = [
        r.model_copy(
            update={
                "steps": [
                    s.model_copy(update={"step_index": 0}) if s.step_index == 2 else s
                    for s in r.steps
                ]
            }
        )
        for r in runs
    ]
    refused = evaluate_policy(split, policy)
    assert refused.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert "cannot be ordered" in (refused.reason or "")
