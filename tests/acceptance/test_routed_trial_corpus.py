"""On the committed real-agent routed trial, the engine agrees with the randomized estimate."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.real.analyze_routed_trial import RoutedTrialReport, analyze
from counterfact.intervene import IdentifiabilityStatus
from counterfact.schema import Run
from counterfact.stats import Z_95

CORPUS = Path(__file__).resolve().parents[2] / "bench" / "real" / "routed_trial"


@pytest.fixture(scope="module")
def report() -> RoutedTrialReport:
    runs = [Run.model_validate_json(p.read_text()) for p in sorted(CORPUS.glob("real-*.json"))]
    assert len(runs) == 800
    return analyze(runs)


def test_engine_interval_contains_the_trial_estimate(report: RoutedTrialReport) -> None:
    estimate = report.engine_routed
    assert estimate.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert estimate.outcome_delta is not None
    assert estimate.outcome_delta.ci_low <= report.trial_effect <= estimate.outcome_delta.ci_high


def test_naive_gap_falls_outside_the_trial_interval(report: RoutedTrialReport) -> None:
    assert report.naive_routed < report.trial_effect - Z_95 * report.trial_se


def test_engine_refuses_without_propensities(report: RoutedTrialReport) -> None:
    assert report.engine_without_propensities.identifiability == IdentifiabilityStatus.BOUNDED
