"""SCM-recovery acceptance test.

Generates a synthetic corpus with a known headline-intervention effect E,
fits the outcome model, and asserts the recovered estimate matches E within
the project tolerance of 0.05.

If this test fails, the causal pipeline is not recovering the known synthetic
truth and downstream causal claims are suspect.
"""

from __future__ import annotations

import pytest

from bench.synthetic import HEADLINE_TRUE_EFFECT, generate_traces
from counterfact import build_dag, fit_outcome_model, intervene
from counterfact.intervene import CausalEstimate, IdentifiabilityStatus
from counterfact.schema import Run


def _generate_runs(n: int, seed: int) -> list[Run]:
    return [Run.model_validate(t) for t in generate_traces(n=n, seed=seed)]


@pytest.fixture(scope="module")
def fitted_corpus() -> tuple[list[Run], object]:
    runs = _generate_runs(n=500, seed=42)
    model = fit_outcome_model(runs, seed=42)
    return runs, model


def _headline(model: object, runs: list[Run]) -> CausalEstimate:
    """Contrast `model_choice = sonnet` against `haiku` at the model_call step."""
    # Any run has the same step structure for the synthetic SCM; use the first.
    dag = build_dag(runs[0])
    return intervene(
        dag=dag, model=model, step=2, intervention={"model_choice": "sonnet"}, baseline="haiku"
    )


def test_scm_recovery__within_tolerance_on_default_seed(
    fitted_corpus: tuple[list[Run], object],
) -> None:
    """WHEN the SCM-recovery acceptance test is executed with the default seed
    THEN the estimated headline effect is within ±0.05 of the known true E."""
    runs, model = fitted_corpus

    estimate = _headline(model, runs)

    assert estimate.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert estimate.outcome_delta is not None
    diff = abs(estimate.outcome_delta.point - HEADLINE_TRUE_EFFECT)
    assert diff <= 0.05, (
        f"SCM recovery failed: estimated={estimate.outcome_delta.point:.4f}, "
        f"true={HEADLINE_TRUE_EFFECT:.4f}, |diff|={diff:.4f} > 0.05"
    )


def test_scm_recovery__ci_covers_true_effect(
    fitted_corpus: tuple[list[Run], object],
) -> None:
    """WHEN the SCM-recovery acceptance test computes the 95% CI for the headline effect
    THEN the interval contains E."""
    runs, model = fitted_corpus

    delta = _headline(model, runs).outcome_delta

    assert delta is not None
    assert delta.ci_low <= HEADLINE_TRUE_EFFECT <= delta.ci_high, (
        f"95% CI [{delta.ci_low:.4f}, {delta.ci_high:.4f}] does not cover "
        f"true E={HEADLINE_TRUE_EFFECT:.4f}"
    )
