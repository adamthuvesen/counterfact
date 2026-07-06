"""SCM-recovery acceptance test.

Generates a synthetic corpus with a known headline-intervention effect E,
fits the outcome model, and asserts the recovered estimate matches E within
the project tolerance of 0.05.

If this test fails, the causal pipeline is not recovering the known synthetic
truth and downstream causal claims are suspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.synthetic import HEADLINE_TRUE_EFFECT, generate_traces
from bench.synthetic.scm import MODEL_CHOICE_ARMS
from counterfact import build_dag, fit_outcome_model, intervene
from counterfact.intervene import CausalEstimate, IdentifiabilityStatus
from counterfact.schema import Run


def _generate_runs(n: int, seed: int) -> list[Run]:
    return [Run.model_validate(t) for t in generate_traces(n=n, seed=seed)]


@pytest.fixture(scope="module")
def fitted_corpus() -> tuple[list[Run], object]:
    runs = _generate_runs(n=500, seed=42)
    model = fit_outcome_model(runs, n_bootstrap=200, seed=42)
    return runs, model


def _intervention_p(model: object, runs: list[Run], action: str) -> CausalEstimate:
    """Run intervene for `model_choice = action` at the model_call step."""
    # Any run has the same step structure for the synthetic SCM; use the first.
    dag = build_dag(runs[0])
    return intervene(dag=dag, model=model, step=2, intervention={"model_choice": action})


def test_scm_recovery__within_tolerance_on_default_seed(
    fitted_corpus: tuple[list[Run], object],
) -> None:
    """WHEN the SCM-recovery acceptance test is executed with the default seed
    THEN the estimated headline effect is within ±0.05 of the known true E."""
    runs, model = fitted_corpus

    sonnet = _intervention_p(model, runs, "sonnet")
    haiku = _intervention_p(model, runs, "haiku")

    assert sonnet.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert haiku.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert sonnet.outcome_delta is not None and haiku.outcome_delta is not None

    estimated_effect = sonnet.outcome_delta.point - haiku.outcome_delta.point
    diff = abs(estimated_effect - HEADLINE_TRUE_EFFECT)
    assert diff <= 0.05, (
        f"SCM recovery failed: estimated={estimated_effect:.4f}, "
        f"true={HEADLINE_TRUE_EFFECT:.4f}, |diff|={diff:.4f} > 0.05"
    )


def test_scm_recovery__bootstrap_ci_covers_true_effect(
    fitted_corpus: tuple[list[Run], object],
) -> None:
    """WHEN the SCM-recovery acceptance test computes the 95% bootstrap CI for the headline effect
    THEN the interval contains E."""
    _, model = fitted_corpus

    # Compute the per-bootstrap effect estimate by re-running the g-formula
    # with each bootstrap coefficient draw.
    feat_index = model.feature_index
    target_keys = {a: f"model_call::{a}" for a in MODEL_CHOICE_ARMS}
    target_idx = {a: feat_index[k] for a, k in target_keys.items()}
    sibling_keys = [k for k in feat_index if k.startswith("model_call::")]
    sibling_idx = [feat_index[k] for k in sibling_keys]

    X = model.train_X.copy()

    def _marginal_p(coefs: np.ndarray, intercept: float, action: str) -> float:
        Xc = X.copy()
        Xc[:, sibling_idx] = 0.0
        Xc[:, target_idx[action]] = 1.0
        z = Xc @ coefs + intercept
        return float((1.0 / (1.0 + np.exp(-z))).mean())

    n_b = model.bootstrap_coefs.shape[0]
    effects = np.zeros(n_b)
    for b in range(n_b):
        coefs = model.bootstrap_coefs[b]
        intercept = float(model.bootstrap_intercepts[b])
        effects[b] = _marginal_p(coefs, intercept, "sonnet") - _marginal_p(
            coefs, intercept, "haiku"
        )

    ci_lo = float(np.percentile(effects, 2.5))
    ci_hi = float(np.percentile(effects, 97.5))
    assert ci_lo <= HEADLINE_TRUE_EFFECT <= ci_hi, (
        f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}] does not cover true E={HEADLINE_TRUE_EFFECT:.4f}"
    )
