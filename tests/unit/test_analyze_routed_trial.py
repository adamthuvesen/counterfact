"""The routed-trial analysis recovers the trial truth and refuses without propensities."""

from __future__ import annotations

import random

import pytest

from bench.real.analyze_routed_trial import analyze, to_markdown
from counterfact.intervene import IdentifiabilityStatus
from counterfact.schema import Decision, Outcome, Run, Step

# Hard fixtures pass less often; the router sends them to the large model.
_P_PASS = {
    ("hard", "large"): 0.6,
    ("hard", "small"): 0.15,
    ("easy", "large"): 0.95,
    ("easy", "small"): 0.8,
}
_ROUTE = {"hard": 0.85, "easy": 0.15}


def _model_call(i: int, attempt: int, model: str, design: str, p_large: float) -> Decision:
    return Decision(
        decision_id=f"d-{i}-model-{attempt}",
        decision_type="model_call",
        chosen_action=model,
        policy=design,
        policy_params={"p_large": p_large},
        valid_actions=["small", "large"],
        propensity=p_large if model == "large" else 1 - p_large,
        context_features={},
    )


def _run(i: int, rng: random.Random) -> Run:
    fixture = "hard" if i % 2 else "easy"
    design, p_large = ("trial", 0.5) if rng.random() < 0.5 else ("fixture_router", _ROUTE[fixture])
    model = "large" if rng.random() < p_large else "small"
    passed = rng.random() < _P_PASS[(fixture, model)]
    steps = [Step(step_index=0, decisions=[_model_call(i, 1, model, design, p_large)])]
    if not passed:
        # Like the harness, a failed run logs a second model_call for its retry.
        steps.append(Step(step_index=1, decisions=[_model_call(i, 2, model, design, p_large)]))
    return Run(
        schema_version="0.1.0",
        run_id=f"real-{fixture}-{i:06d}",
        steps=steps,
        outcome=Outcome(
            kind="binary",
            value=passed,
            verifier="pytest_hidden",
            metadata={"fixture_id": fixture},
        ),
    )


def test_engine_matches_trial_truth_and_naive_does_not() -> None:
    rng = random.Random(0)
    runs = [_run(i, rng) for i in range(4000)]
    # Put a retried run first: it must not become the focal trace.
    runs.sort(key=lambda r: len(r.steps), reverse=True)
    report = analyze(runs)
    truth = 0.5 * ((0.6 - 0.15) + (0.95 - 0.8))  # +0.30

    assert abs(report.trial_effect - truth) < 3 * report.trial_se
    engine = report.engine_routed
    assert engine.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert engine.outcome_delta is not None
    assert engine.outcome_delta.ci_low <= truth <= engine.outcome_delta.ci_high
    # The router sends large to hard fixtures, so the naive gap understates the effect.
    assert report.naive_routed < truth - 0.1
    refusal = report.engine_without_propensities
    assert refusal.identifiability == IdentifiabilityStatus.BOUNDED
    assert "Engine minus truth" in to_markdown(report)


def test_missing_arm_in_a_design_is_a_clear_error() -> None:
    rng = random.Random(1)
    runs = [
        r
        for r in (_run(i, rng) for i in range(200))
        if r.steps[0].decisions[0].chosen_action == "large"
    ]
    with pytest.raises(ValueError, match="no \\['small'\\] model_call arm"):
        analyze(runs)


def test_trial_truth_counts_runs_not_retry_decisions() -> None:
    """Large: pass, pass, fail+retry. Small: pass, fail+retry, fail+retry.
    Per run that is 2/3 - 1/3 = +1/3. Counting the retry model_calls as extra
    observations would give 2/4 - 1/5 = +0.30 instead."""

    def run(i: int, design: str, model: str, passed: bool) -> Run:
        steps = [Step(step_index=0, decisions=[_model_call(i, 1, model, design, 0.5)])]
        if not passed:
            steps.append(Step(step_index=1, decisions=[_model_call(i, 2, model, design, 0.5)]))
        return Run(
            schema_version="0.1.0",
            run_id=f"real-x-{i:06d}",
            steps=steps,
            outcome=Outcome(
                kind="binary", value=passed, verifier="pytest", metadata={"fixture_id": "x"}
            ),
        )

    trial = [
        ("large", True),
        ("large", True),
        ("large", False),
        ("small", True),
        ("small", False),
        ("small", False),
    ]
    routed = [("large", True), ("small", False), ("large", False), ("small", True)]
    runs = [run(i, "trial", m, p) for i, (m, p) in enumerate(trial)]
    runs += [run(100 + i, "fixture_router", m, p) for i, (m, p) in enumerate(routed)]

    assert analyze(runs).trial_effect == pytest.approx(1 / 3)


def test_single_class_routed_runs_get_the_honest_refusal() -> None:
    rng = random.Random(2)
    runs = [_run(i, rng) for i in range(400)]
    routed_all_pass = [
        r.model_copy(update={"outcome": r.outcome.model_copy(update={"value": True})})
        if r.steps[0].decisions[0].policy == "fixture_router"
        else r
        for r in runs
    ]

    report = analyze(routed_all_pass)

    assert report.engine_routed.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert report.engine_without_propensities.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert "unidentified" in to_markdown(report)
