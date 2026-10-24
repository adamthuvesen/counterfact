"""Benchmark scenarios: structural causal models the estimator can fail.

Every scenario asks the same question: the effect of `model_choice = sonnet`
against `haiku` on task success. That keeps results comparable across
scenarios. Each scenario breaks a different assumption that a naive or an
adjust-everything analysis depends on. All variables are discrete, so each
`true_effect` is an exact sum over the model, not a Monte Carlo estimate.

`sample(rng, run_index, force_model)` draws one native trace. A non-None
`force_model` simulates do(model_call = arm) and leaves every other mechanism
unchanged; the tests use it to check each `true_effect` by simulation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

TOOLS = ("run_tests", "inspect_file", "search_docs")
MODELS = ("haiku", "sonnet")
TARGET, BASELINE = "sonnet", "haiku"

_TOOL_LOGIT = {"run_tests": 0.6, "inspect_file": -0.1, "search_docs": -0.4}
_MODEL_LOGIT = {"haiku": -0.4, "sonnet": 0.6}
# Latent task difficulty. No scenario writes it to a trace.
_HARD_LOGIT = -1.4
P_HARD = 0.5

Sampler = Callable[[random.Random, int, str | None], dict[str, Any]]


@dataclass(frozen=True)
class Scenario:
    name: str
    breaks: str
    expected_label: str
    n: int
    true_effect: float
    sample: Sampler


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _decision(
    run_index: int,
    name: str,
    decision_type: str,
    action: str,
    *,
    propensity: float | None = None,
    policy: str = "uniform",
    valid: tuple[str, ...] = (),
) -> dict[str, Any]:
    logged = propensity is not None
    return {
        "decision_id": f"d-{run_index:06d}-{name}",
        "decision_type": decision_type,
        "chosen_action": action,
        "policy": policy if logged else None,
        "policy_params": {} if logged else None,
        "valid_actions": list(valid) if logged else None,
        "propensity": propensity,
        "context_features": {} if logged else None,
    }


def _trace(
    scenario: str, run_index: int, decisions: list[dict[str, Any]], passed: bool
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "run_id": f"{scenario}-{run_index:06d}",
        "steps": [
            {"step_index": i, "decisions": [d], "observations": [], "metadata": {}}
            for i, d in enumerate(decisions)
        ],
        "outcome": {
            "kind": "binary",
            "value": passed,
            "verifier": f"scenario:{scenario}",
            "metadata": {},
        },
        "metadata": {"agent_name": "synthetic_scenario", "notes": scenario, "extra": {}},
    }


def _tool(rng: random.Random, i: int) -> tuple[str, dict[str, Any]]:
    tool = rng.choice(TOOLS)
    return tool, _decision(i, "tool", "tool_call", tool, propensity=1 / 3, valid=TOOLS)


def _model(rng: random.Random, p_sonnet: float, force_model: str | None) -> tuple[str, float]:
    if force_model is not None:
        return force_model, 1.0
    model = TARGET if rng.random() < p_sonnet else BASELINE
    return model, p_sonnet if model == TARGET else 1.0 - p_sonnet


def _model_decision(i: int, model: str, propensity: float | None, *, policy: str) -> dict[str, Any]:
    return _decision(
        i, "model", "model_call", model, propensity=propensity, policy=policy, valid=MODELS
    )


# --- randomized: the textbook case every estimator should pass --------------


def _p_main(tool: str, model: str) -> float:
    return _sigmoid(_TOOL_LOGIT[tool] + _MODEL_LOGIT[model])


def _main_effect() -> float:
    return sum(_p_main(t, TARGET) - _p_main(t, BASELINE) for t in TOOLS) / len(TOOLS)


def _randomized(rng: random.Random, i: int, force_model: str | None) -> dict[str, Any]:
    tool, tool_decision = _tool(rng, i)
    model, p = _model(rng, 0.5, force_model)
    passed = rng.random() < _p_main(tool, model)
    return _trace(
        "randomized", i, [tool_decision, _model_decision(i, model, p, policy="uniform")], passed
    )


# --- logged confounding: the arm depends on an earlier logged decision ------

_P_SONNET_GIVEN_TOOL = {"run_tests": 0.8, "inspect_file": 0.2, "search_docs": 0.2}


def _confounded_sampler(name: str, p_success: Callable[[str, str], float]) -> Sampler:
    def sample(rng: random.Random, i: int, force_model: str | None) -> dict[str, Any]:
        tool, tool_decision = _tool(rng, i)
        model, p = _model(rng, _P_SONNET_GIVEN_TOOL[tool], force_model)
        passed = rng.random() < p_success(tool, model)
        return _trace(
            name, i, [tool_decision, _model_decision(i, model, p, policy="by_tool")], passed
        )

    return sample


# --- interaction: the outcome model's main-effects form is wrong ------------


def _p_interaction(tool: str, model: str) -> float:
    # sonnet helps a lot after running the tests and hurts after searching docs.
    shift = 0.0
    if model == TARGET and tool == "run_tests":
        shift = 2.5
    elif model == TARGET and tool != "run_tests":
        shift = -1.5
    return _sigmoid(_TOOL_LOGIT[tool] + 0.3 * _MODEL_LOGIT[model] + shift)


def _interaction_effect() -> float:
    return sum(_p_interaction(t, TARGET) - _p_interaction(t, BASELINE) for t in TOOLS) / 3


# --- unlogged confounder: hidden difficulty drives the arm and the outcome --

_P_SONNET_GIVEN_HARD = {True: 0.25, False: 0.75}


def _p_hidden(hard: bool, model: str) -> float:
    return _sigmoid(0.8 + _MODEL_LOGIT[model] + (_HARD_LOGIT if hard else 0.0))


def _hidden_effect() -> float:
    return sum(
        (P_HARD if hard else 1 - P_HARD) * (_p_hidden(hard, TARGET) - _p_hidden(hard, BASELINE))
        for hard in (True, False)
    )


def _hidden_sampler(name: str, *, log_propensity: bool) -> Sampler:
    def sample(rng: random.Random, i: int, force_model: str | None) -> dict[str, Any]:
        hard = rng.random() < P_HARD
        model, p = _model(rng, _P_SONNET_GIVEN_HARD[hard], force_model)
        passed = rng.random() < _p_hidden(hard, model)
        logged = p if log_propensity else None
        return _trace(name, i, [_model_decision(i, model, logged, policy="router")], passed)

    return sample


# --- post-treatment retry: a mediator logged after the treatment ------------

_P_RETRY_AFTER_FAIL = 0.9
_P_RETRY_AFTER_PASS = 0.1


def _p_first(hard: bool, model: str) -> float:
    return _sigmoid(_MODEL_LOGIT[model] + (_HARD_LOGIT if hard else 0.0))


def _p_second(hard: bool, model: str) -> float:
    return _sigmoid(-0.5 + 0.5 * _MODEL_LOGIT[model] + (_HARD_LOGIT if hard else 0.0))


def _post_treatment_effect() -> float:
    def mean(model: str) -> float:
        total = 0.0
        for hard in (True, False):
            p_first = _p_first(hard, model)
            rescued = _P_RETRY_AFTER_FAIL * _p_second(hard, model)
            total += (P_HARD if hard else 1 - P_HARD) * (p_first + (1 - p_first) * rescued)
        return total

    return mean(TARGET) - mean(BASELINE)


def _post_treatment(rng: random.Random, i: int, force_model: str | None) -> dict[str, Any]:
    hard = rng.random() < P_HARD
    model, p = _model(rng, 0.5, force_model)
    first_pass = rng.random() < _p_first(hard, model)
    p_retry = _P_RETRY_AFTER_PASS if first_pass else _P_RETRY_AFTER_FAIL
    retry = "retry_once" if rng.random() < p_retry else "no_retry"
    passed = first_pass or (retry == "retry_once" and rng.random() < _p_second(hard, model))
    # The retry choice reacts to the first attempt, which the model caused.
    retry_decision = _decision(i, "retry", "retry", retry)
    return _trace(
        "post_treatment_retry",
        i,
        [_model_decision(i, model, p, policy="uniform"), retry_decision],
        passed,
    )


# --- positivity violation: deterministic routing, honestly logged -----------


def _positivity(rng: random.Random, i: int, force_model: str | None) -> dict[str, Any]:
    tool, tool_decision = _tool(rng, i)
    model = force_model or (TARGET if tool == "run_tests" else BASELINE)
    passed = rng.random() < _p_main(tool, model)
    return _trace(
        "positivity_violation",
        i,
        [tool_decision, _model_decision(i, model, 1.0, policy="tool_router")],
        passed,
    )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "randomized",
        "nothing: uniform randomization",
        "identified",
        1000,
        _main_effect(),
        _randomized,
    ),
    Scenario(
        "small_n",
        "sample size: the randomized case with n=80",
        "identified",
        80,
        _main_effect(),
        _randomized,
    ),
    Scenario(
        "logged_confounding",
        "naive pass rates: the arm depends on the logged tool",
        "identified",
        1000,
        _main_effect(),
        _confounded_sampler("logged_confounding", _p_main),
    ),
    Scenario(
        "interaction",
        "outcome-model form: a tool x model interaction",
        "identified",
        1000,
        _interaction_effect(),
        _confounded_sampler("interaction", _p_interaction),
    ),
    Scenario(
        "unlogged_confounder_logged_propensity",
        "covariate adjustment: hidden difficulty drives the arm, propensity logged",
        "identified",
        1000,
        _hidden_effect(),
        _hidden_sampler("unlogged_confounder_logged_propensity", log_propensity=True),
    ),
    Scenario(
        "unlogged_confounder",
        "point identification: hidden difficulty, no propensity logged",
        "bounded",
        1000,
        _hidden_effect(),
        _hidden_sampler("unlogged_confounder", log_propensity=False),
    ),
    Scenario(
        "post_treatment_retry",
        "adjust-everything: a retry that reacts to the treatment",
        "identified",
        1000,
        _post_treatment_effect(),
        _post_treatment,
    ),
    Scenario(
        "positivity_violation",
        "positivity: the model is routed deterministically by tool",
        "bounded",
        1000,
        _main_effect(),
        _positivity,
    ),
)


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown scenario {name!r}; known: {[s.name for s in SCENARIOS]}")
