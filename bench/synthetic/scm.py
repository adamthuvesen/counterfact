"""Synthetic structural causal model for CounterBench (no LLM).

The SCM models a tiny coding agent across four decisions per run:
    plan_step → tool_call → model_call → retry → termination
              (tool_choice) (model_choice) (retry_policy)

Three decision types are randomized: tool_choice, model_choice, retry_policy.
The outcome is binary success, drawn from a logistic of the per-arm log-odds.

The headline intervention is `model_choice = sonnet` versus `model_choice = haiku`
on the marginal probability of success: the true effect is exposed as
`HEADLINE_TRUE_EFFECT` so the SCM-recovery acceptance test can compare against
ground truth (design.md D9, D10).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

TOOL_CHOICE_ARMS = ("run_tests", "inspect_file", "search_docs")
MODEL_CHOICE_ARMS = ("haiku", "sonnet")
RETRY_POLICY_ARMS = ("no_retry", "retry_once", "retry_twice")

# Per-arm log-odds contributions to P(success). These sum on the logit scale
# and are passed through a sigmoid to get the outcome probability.
_TOOL_LOGITS = {"run_tests": +0.6, "inspect_file": -0.1, "search_docs": -0.4}
_MODEL_LOGITS = {"haiku": -0.4, "sonnet": +0.6}  # sonnet better than haiku
_RETRY_LOGITS = {"no_retry": -0.2, "retry_once": +0.1, "retry_twice": +0.2}
_INTERCEPT = 0.0

# Confounded mode: model_choice's arm probability is conditioned on the run's
# earlier tool_choice, creating a textbook back-door path tool → model and
# tool → outcome. The outcome equation is unchanged; confounding lives only in
# the arm-assignment policy. P(haiku | tool) = 1 - P(sonnet | tool).
_P_SONNET_GIVEN_TOOL: dict[str, float] = {
    "run_tests": 0.8,
    "inspect_file": 0.2,
    "search_docs": 0.2,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _p_success(tool: str, model: str, retry: str) -> float:
    return _sigmoid(
        _INTERCEPT + _TOOL_LOGITS[tool] + _MODEL_LOGITS[model] + _RETRY_LOGITS[retry]
    )


def _marginal_p_success(model: str) -> float:
    """P(success | model) marginalizing uniformly over tool and retry arms."""
    total = 0.0
    n = len(TOOL_CHOICE_ARMS) * len(RETRY_POLICY_ARMS)
    for t in TOOL_CHOICE_ARMS:
        for r in RETRY_POLICY_ARMS:
            total += _p_success(t, model, r)
    return total / n


# Headline intervention: sonnet vs haiku marginal effect on P(success).
HEADLINE_TRUE_EFFECT: float = _marginal_p_success("sonnet") - _marginal_p_success("haiku")


def _p_model_given_tool(model: str, tool: str) -> float:
    p_sonnet = _P_SONNET_GIVEN_TOOL[tool]
    return p_sonnet if model == "sonnet" else 1.0 - p_sonnet


def _confounded_p_tool_given_model(tool: str, model: str) -> float:
    """P(tool | model) under the confounded distribution (Bayes' rule on the
    conditional propensities and uniform P(tool))."""
    p_t = 1.0 / len(TOOL_CHOICE_ARMS)
    p_m = sum(
        _p_model_given_tool(model, t) * (1.0 / len(TOOL_CHOICE_ARMS))
        for t in TOOL_CHOICE_ARMS
    )
    return _p_model_given_tool(model, tool) * p_t / p_m


def _confounded_naive_p_success(model: str) -> float:
    """Observed marginal P(success | model) under the confounded distribution.

    Weights P(success | t, model, r) by the confounded P(tool | model) and
    uniform P(retry).
    """
    total = 0.0
    n_r = len(RETRY_POLICY_ARMS)
    for t in TOOL_CHOICE_ARMS:
        p_t_given_m = _confounded_p_tool_given_model(t, model)
        for r in RETRY_POLICY_ARMS:
            total += p_t_given_m * (1.0 / n_r) * _p_success(t, model, r)
    return total


# Confounded showcase constants. The outcome equation is unchanged in
# confounded mode, so the do-calculus arm gap equals HEADLINE_TRUE_EFFECT.
# The naive arm gap differs because tool is a confounder for model -> outcome.
CONFOUNDED_NAIVE_HEADLINE: float = (
    _confounded_naive_p_success("sonnet") - _confounded_naive_p_success("haiku")
)
CONFOUNDED_DO_HEADLINE: float = HEADLINE_TRUE_EFFECT
CONFOUNDED_NAIVE_VS_CAUSAL_GAP: float = (
    CONFOUNDED_NAIVE_HEADLINE - CONFOUNDED_DO_HEADLINE
)


@dataclass
class SyntheticSCM:
    """A reproducible synthetic SCM. Construct with a seed; sample runs as dicts.

    `confound=True` activates the showcase mode where `model_choice` is biased
    by the run's earlier `tool_choice` (see `_P_SONNET_GIVEN_TOOL`). The
    outcome equation is unchanged in confounded mode; only the arm-assignment
    policy differs. Default `confound=False` reproduces existing behavior
    bit-for-bit.
    """

    seed: int = 42
    confound: bool = False

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample_outcome(self, tool: str, model: str, retry: str) -> bool:
        return self._rng.random() < _p_success(tool, model, retry)

    def uniform_choice(self, arms: tuple[str, ...]) -> tuple[str, float]:
        chosen = self._rng.choice(arms)
        propensity = 1.0 / len(arms)
        return chosen, propensity

    def _confounded_model_choice(self, tool: str) -> tuple[str, float]:
        p_sonnet = _P_SONNET_GIVEN_TOOL[tool]
        chosen = "sonnet" if self._rng.random() < p_sonnet else "haiku"
        propensity = p_sonnet if chosen == "sonnet" else 1.0 - p_sonnet
        return chosen, propensity

    def sample_run(self, run_index: int) -> dict:
        """Generate a single run as a serializable dict in the native trace format."""
        tool, p_tool = self.uniform_choice(TOOL_CHOICE_ARMS)
        if self.confound:
            model, p_model = self._confounded_model_choice(tool)
            model_policy = "confounded_by_tool"
            model_policy_params: dict = {
                f"p_sonnet_given_{t}": _P_SONNET_GIVEN_TOOL[t]
                for t in TOOL_CHOICE_ARMS
            }
            model_context_features: dict = {"tool_chosen": tool}
        else:
            model, p_model = self.uniform_choice(MODEL_CHOICE_ARMS)
            model_policy = "uniform"
            model_policy_params = {}
            model_context_features = {}
        retry, p_retry = self.uniform_choice(RETRY_POLICY_ARMS)
        success = self.sample_outcome(tool, model, retry)

        return {
            "schema_version": "0.1.0",
            "run_id": f"syn-{run_index:06d}",
            "steps": [
                {
                    "step_index": 0,
                    "decisions": [
                        {
                            "decision_id": f"d-{run_index:06d}-plan",
                            "decision_type": "plan_step",
                            "chosen_action": "begin",
                        }
                    ],
                    "observations": [],
                    "metadata": {},
                },
                {
                    "step_index": 1,
                    "decisions": [
                        {
                            "decision_id": f"d-{run_index:06d}-tool",
                            "decision_type": "tool_call",
                            "chosen_action": tool,
                            "policy": "uniform",
                            "policy_params": {},
                            "valid_actions": list(TOOL_CHOICE_ARMS),
                            "propensity": p_tool,
                            "context_features": {},
                        }
                    ],
                    "observations": [],
                    "metadata": {},
                },
                {
                    "step_index": 2,
                    "decisions": [
                        {
                            "decision_id": f"d-{run_index:06d}-model",
                            "decision_type": "model_call",
                            "chosen_action": model,
                            "policy": model_policy,
                            "policy_params": model_policy_params,
                            "valid_actions": list(MODEL_CHOICE_ARMS),
                            "propensity": p_model,
                            "context_features": model_context_features,
                        }
                    ],
                    "observations": [],
                    "metadata": {},
                },
                {
                    "step_index": 3,
                    "decisions": [
                        {
                            "decision_id": f"d-{run_index:06d}-retry",
                            "decision_type": "retry",
                            "chosen_action": retry,
                            "policy": "uniform",
                            "policy_params": {},
                            "valid_actions": list(RETRY_POLICY_ARMS),
                            "propensity": p_retry,
                            "context_features": {},
                        }
                    ],
                    "observations": [],
                    "metadata": {},
                },
                {
                    "step_index": 4,
                    "decisions": [
                        {
                            "decision_id": f"d-{run_index:06d}-term",
                            "decision_type": "termination",
                            "chosen_action": "stop",
                        }
                    ],
                    "observations": [],
                    "metadata": {},
                },
            ],
            "outcome": {
                "kind": "binary",
                "value": success,
                "verifier": "synthetic_scm",
                "metadata": {},
            },
            "metadata": {
                "agent_name": "synthetic_scm",
                "notes": "generated by bench.synthetic",
                "extra": {},
            },
        }
