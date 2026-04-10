"""Self-test anchors for the corpus-readiness analyzer.

These two tests pin the analyzer's discriminating power: `runs_v1` is the
canonical degenerate corpus and must never promote; the synthetic SCM is the
canonical mixed-outcome corpus and must always promote. If either anchor
breaks, the rubric or the analyzer is wrong, not the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.synthetic import generate_traces
from counterfact import analyze_corpus
from counterfact.schema import Run

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_V1_DIR = REPO_ROOT / "bench" / "real" / "runs_v1"


def _load_runs_v1() -> list[Run]:
    if not RUNS_V1_DIR.exists():
        pytest.skip(f"runs_v1 corpus absent at {RUNS_V1_DIR}")
    return [
        Run.model_validate_json(p.read_text())
        for p in sorted(RUNS_V1_DIR.glob("*.json"))
    ]


def test_runs_v1_anchor_scores_unidentified_only() -> None:
    runs = _load_runs_v1()
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
    is not reachable here. Reaching all three is a `runs_v2` goal."""
    runs = [Run.model_validate(t) for t in generate_traces(n=500, seed=42)]
    report = analyze_corpus(runs)
    assert report.promote is True
    assert "identified" in report.identifiability_coverage.reachable
    assert "unidentified" in report.identifiability_coverage.reachable
    assert all(c.passed for c in report.criteria), [
        (c.name, c.reason) for c in report.criteria if not c.passed
    ]


def test_synthetic_smoke_60_traces_runs_quickly() -> None:
    """Fast failure signal: a 60-trace synthetic subset should still promote
    on the default rubric (pass rate is intrinsic to the SCM, not n)."""
    runs = [Run.model_validate(t) for t in generate_traces(n=60, seed=1)]
    report = analyze_corpus(runs)
    assert report.promote is True
    assert "identified" in report.identifiability_coverage.reachable
