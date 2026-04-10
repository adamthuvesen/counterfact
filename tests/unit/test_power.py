"""Acceptance tests for power_analysis (binomial-Wald CI width)."""

from __future__ import annotations

import math

from counter.power import power_analysis
from counter.schema import Decision, Outcome, Run, Step


def _arm_run(*, run_id: str, decision_type: str, action: str, outcome_pass: bool) -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id=f"{run_id}-d0",
                        decision_type=decision_type,
                        chosen_action=action,
                    )
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=outcome_pass, verifier="stub"),
    )


def _balanced_corpus(*, n_per_arm: int, p_a: float, p_b: float) -> list[Run]:
    """Build a 2-arm corpus with deterministic per-arm pass rates."""
    runs: list[Run] = []
    n_pass_a = round(n_per_arm * p_a)
    n_pass_b = round(n_per_arm * p_b)
    for i in range(n_per_arm):
        runs.append(
            _arm_run(
                run_id=f"a-{i}",
                decision_type="retry",
                action="no_retry",
                outcome_pass=i < n_pass_a,
            )
        )
        runs.append(
            _arm_run(
                run_id=f"b-{i}",
                decision_type="retry",
                action="retry_once",
                outcome_pass=i < n_pass_b,
            )
        )
    return runs


def test_required_n_is_larger_when_target_is_tighter() -> None:
    corpus = _balanced_corpus(n_per_arm=30, p_a=0.4, p_b=0.6)
    loose = power_analysis(
        corpus,
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.20,
    )
    tight = power_analysis(
        corpus,
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.05,
    )
    assert tight.estimated_required_n is not None
    assert loose.estimated_required_n is not None
    assert tight.estimated_required_n > loose.estimated_required_n


def test_required_n_exceeds_current_when_ci_too_wide() -> None:
    """Per spec scenario: when target < current_ci_width, required_n > current_n."""
    corpus = _balanced_corpus(n_per_arm=30, p_a=0.4, p_b=0.6)
    rep = power_analysis(
        corpus,
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.05,
    )
    assert rep.current_ci_width is not None
    assert rep.current_ci_width > rep.target_ci_width
    assert rep.estimated_required_n is not None
    assert rep.estimated_required_n > rep.current_n


def test_missing_arm_returns_none_and_warns() -> None:
    """Single-arm corpus → estimated_required_n is None, warning names the missing arm."""
    corpus = [
        _arm_run(run_id=f"r-{i}", decision_type="retry", action="retry_once", outcome_pass=True)
        for i in range(20)
    ]
    rep = power_analysis(
        corpus,
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.10,
    )
    assert rep.estimated_required_n is None
    assert any("no_retry" in w for w in rep.warnings)


def test_assumptions_list_is_non_empty_for_successful_estimate() -> None:
    corpus = _balanced_corpus(n_per_arm=30, p_a=0.4, p_b=0.6)
    rep = power_analysis(
        corpus,
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.10,
    )
    assert rep.estimated_required_n is not None
    assert len(rep.assumptions) >= 1
    assert any("binomial" in a.lower() for a in rep.assumptions)


def test_degenerate_pass_rates_yield_finite_recommendation() -> None:
    """When both arms are 0/0 or 1/1, variance is 0 and any n works."""
    corpus = _balanced_corpus(n_per_arm=30, p_a=1.0, p_b=1.0)
    rep = power_analysis(
        corpus,
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.10,
    )
    assert rep.estimated_required_n == rep.current_n
    assert rep.current_ci_width == 0.0


def test_doubling_n_roughly_halves_predicted_ci_width() -> None:
    """Sanity check that the formula scales as 1/√n.

    With a 2-arm balanced corpus and fixed per-arm pass rates, doubling the
    corpus size (and thereby per-arm n) should shrink the 95% CI width by
    roughly √2.
    """
    p_a, p_b = 0.4, 0.6
    rep_30 = power_analysis(
        _balanced_corpus(n_per_arm=30, p_a=p_a, p_b=p_b),
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.05,
    )
    rep_60 = power_analysis(
        _balanced_corpus(n_per_arm=60, p_a=p_a, p_b=p_b),
        decision_type="retry",
        arms=("no_retry", "retry_once"),
        target_ci_width=0.05,
    )
    assert rep_30.current_ci_width is not None
    assert rep_60.current_ci_width is not None
    # √2 ≈ 1.414; tolerance is loose because rounding to integer trace counts
    # introduces noise in p_a / p_b at small n.
    assert rep_30.current_ci_width / rep_60.current_ci_width == \
        math.isclose(math.sqrt(2), rep_30.current_ci_width / rep_60.current_ci_width, rel_tol=0.05) \
        or 1.30 < rep_30.current_ci_width / rep_60.current_ci_width < 1.55
