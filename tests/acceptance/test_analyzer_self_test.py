"""Self-test anchors for the corpus-readiness analyzer.

These two tests pin the analyzer's discriminating power: `single_class_refusal` is the
canonical degenerate corpus and must never promote; the synthetic SCM is the
canonical mixed-outcome corpus and must always promote. If either anchor
breaks, the rubric or the analyzer is wrong, not the test.
"""

from __future__ import annotations

from bench.synthetic import generate_traces
from counterfact import analyze_corpus
from counterfact.schema import Run
from tests.fixtures.corpus_loaders import (
    load_single_class_refusal as _load_single_class_refusal,
)
from tests.fixtures.corpus_loaders import (
    load_smoke_mixed_outcome as _load_smoke_mixed_outcome,
)


def test_single_class_refusal_anchor_scores_unidentified_only() -> None:
    runs = _load_single_class_refusal()
    report = analyze_corpus(runs)
    assert report.promote is False
    assert report.identifiability_coverage.unfittable_outcome_model is True
    assert "identified" not in report.identifiability_coverage.reachable
    assert "bounded" not in report.identifiability_coverage.reachable

    outcome_balance = next(c for c in report.criteria if c.name == "outcome_balance")
    assert outcome_balance.passed is False
    assert "pass_rate=1.000" in outcome_balance.reason
    assert "[0.300, 0.700]" in outcome_balance.reason


def test_synthetic_anchor_promotes_with_identified_coverage() -> None:
    """The synthetic SCM does not produce memory_read decisions, so `bounded`
    is not reachable here. Reaching all three is a future-corpus goal."""
    runs = [Run.model_validate(t) for t in generate_traces(n=500, seed=42)]
    report = analyze_corpus(runs)
    assert report.promote is True
    assert "identified" in report.identifiability_coverage.reachable
    assert "unidentified" in report.identifiability_coverage.reachable
    assert all(c.passed for c in report.criteria), [
        (c.name, c.reason) for c in report.criteria if not c.passed
    ]


def test_smoke_mixed_outcome_anchor_promotes_with_mixed_model_arms() -> None:
    """`smoke_mixed_outcome` is the promoted real-trace demo corpus.

    It should pass the support-readiness checks while retaining both identified and
    unidentified branches for explain/demo coverage.
    """
    runs = _load_smoke_mixed_outcome()
    report = analyze_corpus(runs)
    assert report.promote is True
    model_mix = next(c for c in report.criteria if c.name == "model_arm_outcome_mix")
    assert model_mix.passed is True
    assert "identified" in report.identifiability_coverage.reachable
    assert "unidentified" in report.identifiability_coverage.reachable
    assert report.identifiability_coverage.unfittable_outcome_model is False


def test_synthetic_smoke_60_traces_runs_quickly() -> None:
    """Fast failure signal: a 60-trace synthetic subset should still promote
    on the default rubric (pass rate is intrinsic to the SCM, not n)."""
    runs = [Run.model_validate(t) for t in generate_traces(n=60, seed=1)]
    report = analyze_corpus(runs)
    assert report.promote is True
    assert "identified" in report.identifiability_coverage.reachable
