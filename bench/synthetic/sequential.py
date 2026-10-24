"""A sequential agent with a time-varying confounder, and exact policy values.

Each run:

1. `model_call`: haiku or sonnet, randomized 50/50 with the propensity logged.
2. `tool_call` `run_tests`: its observation logs `tests_passed`. Whether the
   first attempt passes depends on the model and on hidden task difficulty.
3. `retry`: the agent's logging policy retries more often after a failure.
   The retry choice depends on `tests_passed`, which the model choice caused,
   so `tests_passed` is a time-varying confounder of retry -> success.
4. A retried run gets a second attempt that can rescue a failure.

Difficulty is never logged. The success indicator is the run outcome.

Policies decide one or both decisions and leave the rest as logged. All
variables are discrete, so `policy_value` sums over the model exactly.
`log_retry_propensity=False` drops the retry propensity, which is what an
unrandomized production retry heuristic looks like.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

MODELS = ("haiku", "sonnet")
RETRIES = ("no_retry", "retry_once")
P_HARD = 0.5
# Logging policy: P(retry_once | tests_passed).
P_RETRY_GIVEN_PASSED = {True: 0.2, False: 0.7}

_MODEL_LOGIT = {"haiku": -0.6, "sonnet": 0.6}
_HARD_LOGIT = -1.5

# A deterministic rule over the observed history: (tests_passed) -> action.
# None for a decision means "as logged".
ModelRule = str | None
RetryRule = Callable[[bool], str] | None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def p_first_pass(hard: bool, model: str) -> float:
    return _sigmoid(0.4 + _MODEL_LOGIT[model] + (_HARD_LOGIT if hard else 0.0))


def p_rescue(hard: bool, model: str) -> float:
    return _sigmoid(-0.3 + 0.8 * _MODEL_LOGIT[model] + (_HARD_LOGIT if hard else 0.0))


def policy_value(model_rule: ModelRule, retry_rule: RetryRule) -> float:
    """Exact P(success) when the agent follows the rules and logs everything else."""
    total = 0.0
    for hard in (True, False):
        p_u = P_HARD if hard else 1 - P_HARD
        models = {model_rule: 1.0} if model_rule else {m: 0.5 for m in MODELS}
        for model, p_m in models.items():
            p_pass = p_first_pass(hard, model)
            # A run that passed first time succeeds whatever it does next.
            if retry_rule is None:
                p_retry_after_fail = P_RETRY_GIVEN_PASSED[False]
            else:
                p_retry_after_fail = 1.0 if retry_rule(False) == "retry_once" else 0.0
            p_success = p_pass + (1 - p_pass) * p_retry_after_fail * p_rescue(hard, model)
            total += p_u * p_m * p_success
    return total


def _decision(
    run_index: int,
    name: str,
    decision_type: str,
    action: str,
    *,
    propensity: float | None,
    valid: tuple[str, ...],
    policy: str,
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


def sample(
    rng: random.Random,
    run_index: int,
    *,
    log_retry_propensity: bool = True,
    model_rule: ModelRule = None,
    retry_rule: RetryRule = None,
) -> dict[str, Any]:
    """Draw one run. Non-None rules make the agent follow a policy (tests only)."""
    hard = rng.random() < P_HARD
    model = rng.choice(MODELS)
    model = model_rule or model
    passed = rng.random() < p_first_pass(hard, model)
    p_retry = P_RETRY_GIVEN_PASSED[passed]
    retry = "retry_once" if rng.random() < p_retry else "no_retry"
    retry = retry_rule(passed) if retry_rule else retry
    success = passed or (retry == "retry_once" and rng.random() < p_rescue(hard, model))
    retry_propensity = p_retry if retry == "retry_once" else 1.0 - p_retry
    steps = [
        [
            _decision(
                run_index,
                "model",
                "model_call",
                model,
                propensity=0.5,
                valid=MODELS,
                policy="uniform",
            )
        ],
        [
            {
                "decision_id": f"d-{run_index:06d}-tests",
                "decision_type": "tool_call",
                "chosen_action": "run_tests",
                "policy": None,
                "policy_params": None,
                "valid_actions": None,
                "propensity": None,
                "context_features": None,
            }
        ],
        [
            _decision(
                run_index,
                "retry",
                "retry",
                retry,
                propensity=retry_propensity if log_retry_propensity else None,
                valid=RETRIES,
                policy="by_test_result",
            )
        ],
    ]
    observations: list[list[dict[str, Any]]] = [
        [],
        [{"observation_id": f"o-{run_index:06d}-tests", "content": {"tests_passed": passed}}],
        [],
    ]
    return {
        "schema_version": "0.1.0",
        "run_id": f"sequential-{run_index:06d}",
        "steps": [
            {"step_index": i, "decisions": d, "observations": o, "metadata": {}}
            for i, (d, o) in enumerate(zip(steps, observations, strict=True))
        ],
        "outcome": {
            "kind": "binary",
            "value": success,
            "verifier": "scenario:sequential",
            "metadata": {},
        },
        "metadata": {"agent_name": "synthetic_scenario", "notes": "sequential", "extra": {}},
    }
