"""The benchmark's headline claims, checked on a small number of seeds.

The full table lives in docs/benchmark.md and comes from 500 seeds. Here a
dozen seeds are enough to pin the claims that matter: the engine labels every
scenario correctly, stays close to the truth where it reports a number, and
the scenarios really do break the naive and adjust-everything estimators.
"""

from __future__ import annotations

import pytest

from bench.synthetic.suite import ScenarioSummary, run_suite

SEEDS = 12
# With n=1000 per corpus, engine RMSE is about 0.04, so the mean over 12 seeds
# has a standard error near 0.012; 0.04 is more than three of those.
BIAS_TOLERANCE = 0.04


@pytest.fixture(scope="module")
def summaries() -> dict[str, ScenarioSummary]:
    return {s.scenario: s for s in run_suite(SEEDS, workers=2)}


def test_engine_label_is_right_in_every_scenario(summaries: dict[str, ScenarioSummary]) -> None:
    wrong = {name: s.label_accuracy for name, s in summaries.items() if s.label_accuracy < 1.0}
    assert not wrong, f"engine mislabeled scenarios: {wrong}"


def test_engine_is_close_to_truth_where_it_reports_a_number(
    summaries: dict[str, ScenarioSummary],
) -> None:
    for name, s in summaries.items():
        if s.expected_label != "identified" or name == "small_n":
            continue
        assert s.engine_bias is not None
        assert abs(s.engine_bias) < BIAS_TOLERANCE, (name, s.engine_bias)


def test_bounded_scenarios_keep_the_truth_inside_the_bounds(
    summaries: dict[str, ScenarioSummary],
) -> None:
    for s in summaries.values():
        if s.expected_label == "bounded":
            assert s.bound_coverage == 1.0, (s.scenario, s.bound_coverage)


def test_scenarios_break_the_estimators_they_target(summaries: dict[str, ScenarioSummary]) -> None:
    def bias(value: float | None) -> float:
        assert value is not None
        return abs(value)

    assert bias(summaries["logged_confounding"].naive_bias) > 0.08
    assert bias(summaries["unlogged_confounder_logged_propensity"].adjust_all_bias) > 0.1
    assert bias(summaries["post_treatment_retry"].adjust_all_bias) > 0.06
    assert bias(summaries["positivity_violation"].naive_bias) > 0.1
