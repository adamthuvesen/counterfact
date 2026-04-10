"""Tests for remaining causal-engine spec scenarios.

build_dag, SCM-recovery, and outcome-fit-on-binary scenarios are covered by
test_dag.py, tests/acceptance/test_scm_recovery.py, and test_trace_schema.py.
This module pins the rest: fit_outcome_model bootstrap CI, intervene paths,
attribute_failure, E-value, stacked uncertainty, and no-silent-L3.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from bench.synthetic import generate_traces
from counter import attribute_failure, build_dag, fit_outcome_model, intervene
from counter.errors import InvalidInterventionError, UnsupportedOutcomeError
from counter.intervene import IdentifiabilityStatus
from counter.schema import Decision, Outcome, Run, Step
from counter.sensitivity import e_value

# --- shared fixtures ---------------------------------------------------------


@pytest.fixture(scope="module")
def small_corpus() -> list[Run]:
    return [Run.model_validate(t) for t in generate_traces(n=120, seed=1)]


@pytest.fixture(scope="module")
def fitted(small_corpus: list[Run]) -> object:
    return fit_outcome_model(small_corpus, n_bootstrap=50, seed=1)


def _dag(run: Run):  # convenience
    return build_dag(run)


# --- fit_outcome_model spec scenarios ---------------------------------------


def test_fit_outcome_model__fits_on_synthetic_corpus_and_returns_a_model(
    fitted: object,
) -> None:
    """WHEN fit_outcome_model is called on synthetic traces with outcome="success"
    THEN the returned object has non-null .coefficients/.bootstrap_ci/.predict_proba."""
    assert fitted.coefficients is not None
    assert fitted.bootstrap_ci is not None
    assert callable(fitted.predict_proba)
    sample_X = fitted.train_X[:5]
    pred = fitted.predict_proba(sample_X)
    assert pred.shape == (5,)
    assert np.all((pred >= 0) & (pred <= 1))
    assert fitted.train_n == 120


def test_fit_outcome_model__bootstrap_cis_surround_point_estimate(fitted: object) -> None:
    """WHEN the model is fit with n_bootstrap=N
    THEN for every coefficient the point estimate is within the [low, high] of the bootstrap CI."""
    cis = fitted.bootstrap_ci
    for name, (lo, hi) in cis.items():
        i = fitted.feature_names.index(name)
        point = float(fitted.coefficients[i])
        assert lo <= point <= hi, (
            f"coefficient {name} point={point} outside [{lo}, {hi}]"
        )


# --- intervene spec scenarios ------------------------------------------------


def test_intervene__identified_result_has_outcome_delta_and_adjustment_set(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN intervene is called for a tool_choice query that has randomized support
    THEN identifiability="identified", outcome_delta is non-null, adjustment_set is non-empty."""
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,  # tool_call step in synthetic SCM
        intervention={"tool_choice": "run_tests"},
    )
    assert est.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert est.outcome_delta is not None
    assert len(est.adjustment_set) > 0


def test_intervene__bounded_result_has_bounds_and_assumptions(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN intervene is called for a memory_content query
    THEN identifiability="bounded", bounds is non-null, at least one assumption."""
    # The synthetic SCM has no memory_read steps, so we synthesize a small run
    # with a memory_read step at index 1.
    run = Run(
        schema_version="0.1.0",
        run_id="r-mem",
        steps=[
            Step(step_index=0, decisions=[Decision(decision_id="d0", decision_type="plan_step")]),
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="d1",
                        decision_type="memory_read",
                        chosen_action="recent_5",
                    )
                ],
            ),
        ],
        outcome=Outcome(kind="binary", value=False, verifier="pytest"),
    )
    est = intervene(
        dag=build_dag(run),
        model=fitted,
        step=1,
        intervention={"memory_content": "all"},
    )
    assert est.identifiability == IdentifiabilityStatus.BOUNDED
    assert est.bounds is not None
    assert len(est.assumptions) >= 1


def test_intervene__unidentified_result_has_reason_and_structured_next_step(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN intervene is called for a prompt_content query
    THEN identifiability="unidentified", reason is non-null, and next_step has
         action="replay_required" with intervention_target in payload."""
    # synthetic SCM has a model_call at step 2
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=2,
        intervention={"prompt_content": "be more careful"},
    )
    assert est.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert est.reason is not None and len(est.reason) > 0
    assert est.next_step.action == "replay_required"
    assert est.next_step.payload["intervention_target"] == "prompt_content"
    assert est.next_step.human_text


def test_intervene__invalid_intervention_for_decision_type_raises(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN intervene is called with intervention tool_choice on a model_call step
    THEN the system raises InvalidInterventionError."""
    with pytest.raises(InvalidInterventionError):
        intervene(
            dag=_dag(small_corpus[0]),
            model=fitted,
            step=2,  # model_call
            intervention={"tool_choice": "run_tests"},
        )


def test_intervene__increase_n_uses_training_trace_count(
    small_corpus: list[Run], fitted: object
) -> None:
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,
        intervention={"tool_choice": "run_tests"},
    )
    assert est.next_step.action == "increase_n"
    assert est.next_step.payload["current_n"] == len(small_corpus)
    assert est.next_step.payload["current_n"] != est.outcome_delta.n_bootstrap


def test_intervene__multi_decision_step_raises_clear_error(fitted: object) -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="multi",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-tool",
                        decision_type="tool_call",
                        chosen_action="run_tests",
                    ),
                    Decision(
                        decision_id="d-model",
                        decision_type="model_call",
                        chosen_action="haiku",
                    ),
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=True, verifier="stub"),
    )
    with pytest.raises(InvalidInterventionError, match="multiple decisions"):
        intervene(
            dag=build_dag(run),
            model=fitted,
            step=0,
            intervention={"model_choice": "sonnet"},
        )


# --- attribute_failure spec scenarios ----------------------------------------


def test_attribute_failure__top_k_returns_at_most_k(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN attribute_failure(...).top_k(5) is called
    THEN the result is a list of length <= 5."""
    attribution = attribute_failure(dag=_dag(small_corpus[0]), model=fitted)
    assert len(attribution.top_k(5)) <= 5


def test_attribute_failure__each_entry_carries_identifiability_label(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN any entry from top_k is inspected
    THEN it has identifiability set to one of identified, bounded, unidentified."""
    attribution = attribute_failure(dag=_dag(small_corpus[0]), model=fitted)
    for entry in attribution.top_k(10):
        assert entry.identifiability in {
            IdentifiabilityStatus.IDENTIFIED,
            IdentifiabilityStatus.BOUNDED,
            IdentifiabilityStatus.UNIDENTIFIED,
        }


def test_attribute_failure__empty_corpus_yields_empty_ranking(fitted: object) -> None:
    """WHEN attribute_failure is called on a trace with no decisions
    THEN top_k(5) returns []."""
    empty_run = Run(
        schema_version="0.1.0",
        run_id="empty",
        steps=[],
        outcome=Outcome(kind="binary", value=False, verifier="none"),
    )
    attribution = attribute_failure(dag=build_dag(empty_run), model=fitted)
    assert attribution.top_k(5) == []


def test_attribute_failure__multi_decision_step_is_unidentified(
    fitted: object,
) -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="multi-attr",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-tool",
                        decision_type="tool_call",
                        chosen_action="run_tests",
                    ),
                    Decision(
                        decision_id="d-model",
                        decision_type="model_call",
                        chosen_action="haiku",
                    ),
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=False, verifier="stub"),
    )
    entries = attribute_failure(dag=build_dag(run), model=fitted).top_k(10)
    assert entries
    assert {e.identifiability for e in entries} == {IdentifiabilityStatus.UNIDENTIFIED}
    assert all(e.influence == 0.0 for e in entries)


# --- E-value spec scenarios --------------------------------------------------


def test_evalue__present_on_identified_estimate(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN an intervene call returns an identified estimate
    THEN estimate.bounds.e_value is a finite float >= 1.0."""
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,
        intervention={"tool_choice": "run_tests"},
    )
    assert est.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert est.bounds is not None
    assert math.isfinite(est.bounds.e_value)
    assert est.bounds.e_value >= 1.0


def test_evalue__matches_reference_formula() -> None:
    """WHEN the E-value is computed for a known risk ratio RR > 1
    THEN the result equals RR + sqrt(RR * (RR - 1)) within numerical tolerance."""
    rr = 2.0
    expected = rr + math.sqrt(rr * (rr - 1))
    assert e_value(rr) == pytest.approx(expected, rel=1e-9)
    rr = 3.5
    expected = rr + math.sqrt(rr * (rr - 1))
    assert e_value(rr) == pytest.approx(expected, rel=1e-9)
    # symmetric: RR < 1 should equal E-value of 1/RR
    assert e_value(0.5) == pytest.approx(e_value(2.0), rel=1e-9)


# --- Stacked uncertainty / no-silent-L3 -------------------------------------


def test_stacked_uncertainty__bootstrap_ci_does_not_absorb_bounds(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN an estimate is bounded (or identified)
    THEN it has BOTH non-null outcome_delta AND non-null bounds, and they are not
    the same object."""
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,
        intervention={"tool_choice": "run_tests"},
    )
    assert est.outcome_delta is not None
    assert est.bounds is not None
    assert est.outcome_delta is not est.bounds


def test_stacked_uncertainty__stable_model_on_unidentified_query_stays_unidentified(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN intervene is called with a prompt_content query (always-replay)
    THEN identifiability remains unidentified regardless of model stability."""
    # Even though `fitted` has tight bootstrap CIs, the prompt_content query is
    # taxonomy-unidentified.
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=2,
        intervention={"prompt_content": "carefully consider edge cases"},
    )
    assert est.identifiability == IdentifiabilityStatus.UNIDENTIFIED


def test_no_silent_l3__identified_estimate_names_its_adjustment(
    small_corpus: list[Run], fitted: object
) -> None:
    """WHEN any identified estimate is inspected
    THEN estimate.assumptions is non-empty and at least one entry mentions 'adjustment'."""
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,
        intervention={"tool_choice": "run_tests"},
    )
    assert est.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert len(est.assumptions) > 0
    assert any("adjustment" in a for a in est.assumptions)


def test_no_silent_l3__bounded_estimate_names_its_sensitivity_technique(
    fitted: object,
) -> None:
    """WHEN any bounded estimate is inspected
    THEN estimate.assumptions references 'E-value' or the sensitivity technique."""
    run = Run(
        schema_version="0.1.0",
        run_id="r-mem-2",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-mem",
                        decision_type="memory_read",
                        chosen_action="recent_5",
                    )
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=False, verifier="pytest"),
    )
    est = intervene(
        dag=build_dag(run),
        model=fitted,
        step=0,
        intervention={"memory_content": "all"},
    )
    assert est.identifiability == IdentifiabilityStatus.BOUNDED
    assert any(
        "E-value" in a or "e_value" in a for a in est.assumptions
    ), f"assumptions did not name sensitivity technique: {est.assumptions}"


def test_unsupported_outcome__non_binary_model_rejected_at_intervene(
    small_corpus: list[Run],
) -> None:
    """An intervene call against a model whose outcome_kind != binary must raise."""

    class _ContinuousModel:
        outcome_kind = "continuous"
        feature_index: ClassVar[dict] = {}

    with pytest.raises(UnsupportedOutcomeError):
        intervene(
            dag=_dag(small_corpus[0]),
            model=_ContinuousModel(),
            step=1,
            intervention={"tool_choice": "run_tests"},
        )
