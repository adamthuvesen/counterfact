"""Unit coverage for `counterfact.corpus_analyzer.analyze`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bench.synthetic import generate_traces
from counterfact import analyze_corpus
from counterfact.corpus_analyzer import (
    DEFAULT_THRESHOLDS,
    CorpusReadinessReport,
    RubricThresholds,
    analyze,
)
from counterfact.schema import Decision, Outcome, Run, Step


def test_default_thresholds_match_v0_spec() -> None:
    assert DEFAULT_THRESHOLDS.min_pass_rate == 0.3
    assert DEFAULT_THRESHOLDS.max_pass_rate == 0.7
    assert DEFAULT_THRESHOLDS.min_arms_per_decision_type == 2
    assert DEFAULT_THRESHOLDS.min_n_per_arm == 5
    assert DEFAULT_THRESHOLDS.min_identified_decision_types == 1
    assert DEFAULT_THRESHOLDS.require_model_arm_outcome_mix is True


def test_thresholds_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RubricThresholds(min_pass_rate=0.3, surprise=True)  # type: ignore[call-arg]


def test_empty_corpus_returns_promote_false_with_reason() -> None:
    report = analyze([])
    assert isinstance(report, CorpusReadinessReport)
    assert report.n_traces == 0
    assert report.promote is False
    outcome_balance = next(c for c in report.criteria if c.name == "outcome_balance")
    assert outcome_balance.passed is False
    assert "empty corpus" in outcome_balance.reason


def _all_pass_run(run_id: str, model_arm: str = "large") -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id=f"{run_id}-d0",
                        decision_type="model_call",
                        chosen_action=model_arm,
                    )
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=True, verifier="stub"),
    )


def _model_arm_run(run_id: str, model_arm: str, outcome: bool) -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id=f"{run_id}-model",
                        decision_type="model_call",
                        chosen_action=model_arm,
                    )
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=outcome, verifier="stub"),
    )


def test_single_class_corpus_marks_unfittable_outcome_model() -> None:
    runs = [_all_pass_run(f"r{i}") for i in range(5)]
    report = analyze(runs)
    assert report.identifiability_coverage.unfittable_outcome_model is True
    assert "identified" not in report.identifiability_coverage.reachable
    assert report.promote is False


def test_model_arm_outcome_mix_rejects_perfect_large_arm() -> None:
    runs = [
        *[_model_arm_run(f"large-pass-{i}", "large", True) for i in range(6)],
        _model_arm_run("small-pass", "small", True),
        *[_model_arm_run(f"small-fail-{i}", "small", False) for i in range(6)],
    ]

    report = analyze(runs)
    criterion = next(c for c in report.criteria if c.name == "model_arm_outcome_mix")

    assert criterion.passed is False
    assert "large=pass:6,fail:0" in criterion.reason
    assert report.promote is False


def test_model_arm_outcome_mix_passes_when_both_model_arms_are_mixed() -> None:
    runs = [
        *[_model_arm_run(f"large-pass-{i}", "large", True) for i in range(3)],
        *[_model_arm_run(f"large-fail-{i}", "large", False) for i in range(3)],
        *[_model_arm_run(f"small-pass-{i}", "small", True) for i in range(3)],
        *[_model_arm_run(f"small-fail-{i}", "small", False) for i in range(3)],
    ]

    report = analyze(runs)
    criterion = next(c for c in report.criteria if c.name == "model_arm_outcome_mix")

    assert criterion.passed is True
    assert criterion.reason == "model_arm_outcome_mix: ok"


def test_outcome_balance_failing_reason_is_pinned_format() -> None:
    """Pin the failure reason format character-for-character so any future
    change to the formatting is deliberate."""
    runs = [_all_pass_run(f"r{i}") for i in range(10)]
    report = analyze(runs)
    outcome_balance = next(c for c in report.criteria if c.name == "outcome_balance")
    assert outcome_balance.reason == (
        "outcome_balance: pass_rate=1.000 outside [0.300, 0.700]"
    )


def test_passing_criteria_reasons_end_with_ok() -> None:
    runs = [Run.model_validate(t) for t in generate_traces(n=120, seed=2)]
    report = analyze(runs)
    for c in report.criteria:
        if c.passed:
            assert c.reason.endswith(": ok"), (c.name, c.reason)


def test_custom_thresholds_can_reject_a_previously_promoted_corpus() -> None:
    runs = [Run.model_validate(t) for t in generate_traces(n=120, seed=3)]
    default_report = analyze(runs)
    assert default_report.promote is True

    strict = RubricThresholds(
        min_pass_rate=0.99,
        max_pass_rate=1.0,
        min_arms_per_decision_type=2,
        min_n_per_arm=5,
        min_identified_decision_types=1,
    )
    strict_report = analyze(runs, thresholds=strict)
    assert strict_report.promote is False
    outcome_balance = next(
        c for c in strict_report.criteria if c.name == "outcome_balance"
    )
    assert outcome_balance.passed is False


def test_thresholds_are_echoed_in_report() -> None:
    custom = RubricThresholds(
        min_pass_rate=0.4,
        max_pass_rate=0.6,
        min_arms_per_decision_type=2,
        min_n_per_arm=5,
        min_identified_decision_types=1,
    )
    runs = [Run.model_validate(t) for t in generate_traces(n=40, seed=4)]
    report = analyze(runs, thresholds=custom)
    assert report.thresholds.min_pass_rate == 0.4
    assert report.thresholds.max_pass_rate == 0.6


def test_analyze_is_deterministic_for_same_input() -> None:
    runs = [Run.model_validate(t) for t in generate_traces(n=60, seed=5)]
    a = analyze(runs)
    b = analyze(runs)
    assert a == b


def test_top_level_analyze_corpus_alias_matches_module_function() -> None:
    runs = [Run.model_validate(t) for t in generate_traces(n=60, seed=6)]
    a = analyze_corpus(runs)
    b = analyze(runs)
    assert a == b
