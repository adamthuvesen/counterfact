"""The policy benchmark's headline claims on a small number of seeds."""

from __future__ import annotations

import pytest

from bench.synthetic.policy_suite import PolicySummary, run_policy_suite


@pytest.fixture(scope="module")
def summaries() -> dict[str, PolicySummary]:
    return {s.case: s for s in run_policy_suite(8, workers=2)}


def test_engine_labels_every_policy_case_correctly(summaries: dict[str, PolicySummary]) -> None:
    assert {name: s.label_accuracy for name, s in summaries.items() if s.label_accuracy < 1} == {}


def test_sequential_dr_is_close_to_the_truth(summaries: dict[str, PolicySummary]) -> None:
    for name, s in summaries.items():
        if s.expected_label == "identified":
            # DR RMSE is at most about 0.017 here; 8 seeds give a standard error near 0.006.
            assert s.dr_bias is not None and abs(s.dr_bias) < 0.02, (name, s.dr_bias)


def test_filtering_the_logs_flips_the_sign_of_both_retry_policies(
    summaries: dict[str, PolicySummary],
) -> None:
    for name in ("always_retry", "never_retry"):
        s = summaries[name]
        assert s.matched_bias is not None
        assert (s.true_delta + s.matched_bias) * s.true_delta < 0, name
