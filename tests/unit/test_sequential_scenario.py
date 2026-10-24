"""The sequential scenario's exact policy values match simulated policies."""

from __future__ import annotations

import math
import random

import pytest

from bench.synthetic.sequential import policy_value, sample
from counterfact.schema import Run

N = 40_000


def _on_fail(passed: bool) -> str:
    return "no_retry" if passed else "retry_once"


@pytest.mark.parametrize(
    ("model_rule", "retry_rule"),
    [
        (None, None),
        (None, lambda passed: "retry_once"),
        (None, lambda passed: "no_retry"),
        (None, _on_fail),
        ("sonnet", _on_fail),
        ("haiku", None),
    ],
)
def test_policy_value_matches_simulation(model_rule, retry_rule) -> None:
    rng = random.Random(f"{model_rule}:{retry_rule}")
    hits = sum(
        sample(rng, i, model_rule=model_rule, retry_rule=retry_rule)["outcome"]["value"]
        for i in range(N)
    )
    tolerance = 4 * math.sqrt(0.25 / N)
    assert abs(hits / N - policy_value(model_rule, retry_rule)) < tolerance


def test_sampled_runs_are_valid_with_and_without_retry_propensity() -> None:
    rng = random.Random(0)
    for i in range(50):
        Run.model_validate(sample(rng, i))
        Run.model_validate(sample(rng, i, log_retry_propensity=False))
