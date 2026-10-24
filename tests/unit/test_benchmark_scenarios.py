"""Each scenario's exact `true_effect` must match a simulated intervention."""

from __future__ import annotations

import math
import random

import pytest

from bench.synthetic.scenarios import BASELINE, SCENARIOS, TARGET, Scenario
from counterfact.schema import Run

N_SIMULATED = 20_000


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_true_effect_matches_simulated_intervention(scenario: Scenario) -> None:
    rng = random.Random(f"truth:{scenario.name}")

    def pass_rate(arm: str) -> float:
        return (
            sum(scenario.sample(rng, i, arm)["outcome"]["value"] for i in range(N_SIMULATED))
            / N_SIMULATED
        )

    simulated = pass_rate(TARGET) - pass_rate(BASELINE)
    # Two independent Bernoulli means; 4 standard errors keeps this test stable.
    tolerance = 4 * math.sqrt(2 * 0.25 / N_SIMULATED)
    assert abs(simulated - scenario.true_effect) < tolerance


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_sampled_traces_are_valid_runs(scenario: Scenario) -> None:
    rng = random.Random(0)
    for i in range(50):
        Run.model_validate(scenario.sample(rng, i, None))
