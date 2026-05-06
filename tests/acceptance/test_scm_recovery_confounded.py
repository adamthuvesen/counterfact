"""Confounded-SCM engine recovery acceptance test (tasks §3).

The confounded SCM biases `model_choice` by the run's earlier `tool_choice`,
creating a textbook back-door confounding scenario where the naive marginal
arm gap overstates what the corpus supports. The engine's g-formula
adjustment via the outcome model should recover the do-calculus arm gap
(equal to `HEADLINE_TRUE_EFFECT` because the outcome equation is unchanged
in confounded mode) within the project's standard ±0.05 tolerance.

If this test fails, the canonical demo's headline claim is at risk.
"""

from __future__ import annotations

import pytest

from bench.synthetic import (
    CONFOUNDED_DO_HEADLINE,
    CONFOUNDED_NAIVE_HEADLINE,
    HEADLINE_TRUE_EFFECT,
    generate_traces,
)
from counterfact import build_dag, fit_outcome_model, intervene, pass_rate_by_arm
from counterfact.intervene import IdentifiabilityStatus
from counterfact.schema import Run


@pytest.fixture(scope="module")
def confounded_corpus() -> tuple[list[Run], object]:
    runs = [
        Run.model_validate(t) for t in generate_traces(n=1000, seed=42, confound=True)
    ]
    model = fit_outcome_model(runs, n_bootstrap=50, seed=42)
    return runs, model


def _arm_gap(model: object, runs: list[Run]) -> float:
    dag = build_dag(runs[0])
    e_sonnet = intervene(
        dag=dag, model=model, step=2, intervention={"model_choice": "sonnet"}
    )
    e_haiku = intervene(
        dag=dag, model=model, step=2, intervention={"model_choice": "haiku"}
    )
    assert e_sonnet.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert e_haiku.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert e_sonnet.outcome_delta is not None and e_haiku.outcome_delta is not None
    return e_sonnet.outcome_delta.point - e_haiku.outcome_delta.point


def test_engine_recovers_do_calculus_arm_gap_within_tolerance(
    confounded_corpus: tuple[list[Run], object],
) -> None:
    """Engine's g-formula recovers the do-calculus arm gap within ±0.05."""
    runs, model = confounded_corpus
    engine_gap = _arm_gap(model, runs)
    assert abs(engine_gap - CONFOUNDED_DO_HEADLINE) <= 0.05, (
        f"engine gap {engine_gap:+.4f} deviates from do-calculus truth "
        f"{CONFOUNDED_DO_HEADLINE:+.4f} by more than 0.05"
    )
    # Pin the relationship that lets the demo cite a fixed do-calculus number:
    # the do-calculus headline equals HEADLINE_TRUE_EFFECT exactly because the
    # outcome equation is unchanged in confounded mode.
    assert abs(CONFOUNDED_DO_HEADLINE - HEADLINE_TRUE_EFFECT) < 1e-9


def test_engine_beats_naive_on_confounded_corpus(
    confounded_corpus: tuple[list[Run], object],
) -> None:
    """The engine's deviation from the do-calculus truth is strictly smaller
    than the descriptive baseline's deviation. This is the showcase claim:
    the marginal table overstates what the corpus supports; the causal
    estimate changes the conclusion."""
    runs, model = confounded_corpus

    table = pass_rate_by_arm(runs, "model_call")
    rates = {row.arm: row.pass_rate for row in table.rows}
    naive_gap = rates["sonnet"] - rates["haiku"]
    engine_gap = _arm_gap(model, runs)

    naive_dev = abs(naive_gap - CONFOUNDED_DO_HEADLINE)
    engine_dev = abs(engine_gap - CONFOUNDED_DO_HEADLINE)

    assert engine_dev < naive_dev, (
        f"engine deviation {engine_dev:.4f} is not strictly smaller than "
        f"naive deviation {naive_dev:.4f} — the showcase headline does not hold"
    )

    # Sanity: confounding actually showed up in the sample (naive gap is far
    # from do-calculus truth and close-ish to the analytic confounded headline).
    assert abs(naive_gap - CONFOUNDED_NAIVE_HEADLINE) < 0.10
