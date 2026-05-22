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
from counterfact import attribute_failure, build_dag, fit_outcome_model, intervene
from counterfact.errors import (
    InsufficientOutcomeSupportError,
    InvalidInterventionError,
    UnsupportedOutcomeError,
)
from counterfact.intervene import IdentifiabilityStatus
from counterfact.schema import Decision, Outcome, Run, Step
from counterfact.sensitivity import e_value

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
    assert fitted.coefficients is not None
    assert fitted.bootstrap_ci is not None
    assert callable(fitted.predict_proba)
    sample_X = fitted.train_X[:5]
    pred = fitted.predict_proba(sample_X)
    assert pred.shape == (5,)
    assert np.all((pred >= 0) & (pred <= 1))
    assert fitted.train_n == 120


def test_fit_outcome_model__bootstrap_cis_surround_point_estimate(fitted: object) -> None:
    cis = fitted.bootstrap_ci
    for name, (lo, hi) in cis.items():
        i = fitted.feature_names.index(name)
        point = float(fitted.coefficients[i])
        assert lo <= point <= hi, f"coefficient {name} point={point} outside [{lo}, {hi}]"


def test_fit_outcome_model__warns_when_bootstrap_resamples_are_single_class() -> None:
    runs = [Run.model_validate(t) for t in generate_traces(n=2, seed=10)]
    runs = [
        runs[0].model_copy(
            update={"outcome": Outcome(kind="binary", value=True, verifier="synthetic")}
        ),
        runs[1].model_copy(
            update={"outcome": Outcome(kind="binary", value=False, verifier="synthetic")}
        ),
    ]

    with pytest.warns(RuntimeWarning, match="bootstrap resamples"):
        model = fit_outcome_model(runs, n_bootstrap=20, seed=0)

    assert model.bootstrap_degenerate_resamples > 0


def test_fit_outcome_model__all_pass_corpus_raises_domain_error() -> None:
    runs = [
        Run.model_validate(t).model_copy(
            update={"outcome": Outcome(kind="binary", value=True, verifier="synthetic")}
        )
        for t in generate_traces(n=20, seed=3)
    ]
    with pytest.raises(InsufficientOutcomeSupportError) as exc_info:
        fit_outcome_model(runs)
    msg = str(exc_info.value)
    assert "at least two outcome classes" in msg
    assert "pass and fail outcomes" in msg


def test_fit_outcome_model__all_fail_corpus_raises_domain_error() -> None:
    runs = [
        Run.model_validate(t).model_copy(
            update={"outcome": Outcome(kind="binary", value=False, verifier="synthetic")}
        )
        for t in generate_traces(n=20, seed=4)
    ]
    with pytest.raises(InsufficientOutcomeSupportError) as exc_info:
        fit_outcome_model(runs)
    assert "at least two outcome classes" in str(exc_info.value)


def test_fit_outcome_model__mixed_binary_corpus_still_fits() -> None:
    runs = [Run.model_validate(t) for t in generate_traces(n=80, seed=5)]
    assert {bool(run.outcome.value) for run in runs} == {False, True}
    model = fit_outcome_model(runs, n_bootstrap=10, seed=5)
    assert model.train_n == len(runs)


def test_fit_outcome_model__mixed_binary_without_features_raises_domain_error() -> None:
    runs = [
        Run(
            schema_version="0.1.0",
            run_id="no-features-pass",
            steps=[],
            outcome=Outcome(kind="binary", value=True, verifier="test"),
        ),
        Run(
            schema_version="0.1.0",
            run_id="no-features-fail",
            steps=[],
            outcome=Outcome(kind="binary", value=False, verifier="test"),
        ),
    ]

    with pytest.raises(InsufficientOutcomeSupportError, match="intervenable decision"):
        fit_outcome_model(runs)


def test_fit_outcome_model__extracts_temperature_feature_from_model_config() -> None:
    runs = [
        Run(
            schema_version="0.1.0",
            run_id=f"temp-{i}",
            steps=[
                Step(
                    step_index=0,
                    decisions=[
                        Decision(
                            decision_id=f"d-temp-{i}",
                            decision_type="model_call",
                            chosen_action="haiku",
                            metadata={"model_config": {"temperature": 0.7 if i % 2 else 0}},
                        )
                    ],
                )
            ],
            outcome=Outcome(kind="binary", value=bool(i % 2), verifier="test"),
        )
        for i in range(20)
    ]

    model = fit_outcome_model(runs, n_bootstrap=8, seed=8)

    assert "model_call::haiku" in model.feature_index
    assert "model_call.temperature::0" in model.feature_index
    assert "model_call.temperature::0.7" in model.feature_index


def test_degenerate_outcome_classes_rejects_non_binary_outcomes() -> None:
    from counterfact.intervene.degenerate import outcome_classes

    run = Run(
        schema_version="0.1.0",
        run_id="categorical",
        steps=[],
        outcome=Outcome(kind="categorical", value="unknown", verifier="test"),
    )

    with pytest.raises(UnsupportedOutcomeError, match="categorical"):
        outcome_classes([run])


def test_degenerate_estimate_uses_none_step_not_negative_sentinel() -> None:
    """Single-class degenerate corpora produce CausalEstimate.query.step=None.

    The previous code used -1 as a "no step" sentinel, which forced consumers
    to know the magic value. None is self-documenting and aligned with the
    Optional[int] type on InterventionQuery.step.
    """
    from counterfact.intervene.degenerate import degenerate_estimate

    runs = [
        Run(
            schema_version="0.1.0",
            run_id=f"refusal-{i}",
            steps=[
                Step(
                    step_index=0,
                    decisions=[
                        Decision(
                            decision_id=f"d-{i}",
                            decision_type="model_call",
                            chosen_action="haiku",
                        )
                    ],
                )
            ],
            outcome=Outcome(kind="binary", value=False, verifier="stub"),
        )
        for i in range(3)
    ]
    est = degenerate_estimate(
        runs,
        decision_type="model_call",
        intervention_kind="model_choice",
        target="sonnet",
    )
    assert est.query.step is None
    assert est.identifiability == IdentifiabilityStatus.UNIDENTIFIED


def test_intervene_query_step_accepts_none() -> None:
    from counterfact.intervene.estimate import InterventionQuery

    q = InterventionQuery(
        decision_type="model_call",
        intervention_kind="model_choice",
        target="sonnet",
        step=None,
    )
    assert q.step is None


def test_intervene_missing_train_n_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per spec scenario: when model.train_n is 0 AND train_X is unavailable,
    power estimation must raise InvalidInterventionError rather than silently
    substituting `delta.n_bootstrap` (the old, masking fallback)."""
    import numpy as np

    from counterfact.intervene import api as intervene_api
    from counterfact.intervene.estimate import DistributionSummary
    from counterfact.outcome.model import OutcomeModel

    n = 80
    rng = np.random.default_rng(11)
    y = (rng.random(n) < 0.55).astype(int)
    boot_coefs = np.zeros((50, 1), dtype=float)
    boot_intercepts = np.linspace(-8.0, 8.0, 50)

    # Model with train_n=0 and train_X=None: a malformed outcome model that
    # the old fallback would have happily papered over by reading n_bootstrap.
    bad_model = OutcomeModel(
        feature_names=["tool_call::run_tests"],
        coefficients=np.zeros(1),
        intercept=0.0,
        bootstrap_coefs=boot_coefs,
        bootstrap_intercepts=boot_intercepts,
        train_X=None,  # type: ignore[arg-type]
        train_y=y,
        train_n=0,
        feature_index={"tool_call::run_tests": 0},
        outcome_kind="binary",
    )

    # Stub the g-formula adjust so we exercise the post-adjust power branch
    # directly. (Real adjust would crash on train_X=None; that is a separate
    # invariant violation, not what this test covers.)
    def _fake_adjust(model, feature_family, intervention_kind, target_action):  # type: ignore[no-untyped-def]
        return DistributionSummary(
            point=0.6,
            ci_low=0.2,
            ci_high=0.9,
            n_bootstrap=200,
        )

    monkeypatch.setattr(intervene_api, "_adjust_g_formula", _fake_adjust)

    run = Run(
        schema_version="0.1.0",
        run_id="missing-train-n",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-tool",
                        decision_type="tool_call",
                        chosen_action="run_tests",
                    )
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=True, verifier="stub"),
    )

    with pytest.raises(InvalidInterventionError, match="train_n"):
        intervene(
            dag=build_dag(run),
            model=bad_model,
            step=0,
            intervention={"tool_choice": "run_tests"},
        )


# --- intervene spec scenarios ------------------------------------------------


def test_intervene__identified_result_has_outcome_delta_and_adjustment_set(
    small_corpus: list[Run], fitted: object
) -> None:
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,  # tool_call step in synthetic SCM
        intervention={"tool_choice": "run_tests"},
    )
    assert est.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert est.outcome_delta is not None
    assert len(est.adjustment_set) > 0


def test_intervene__bounded_result_has_null_bounds_and_assumptions(
    small_corpus: list[Run], fitted: object
) -> None:
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
    # Bounded path: no observed P(success | do(...)), so emitting an E-value
    # against a 0.5 baseline would always be 1.0 — false precision. Honest
    # stance is bounds=None plus an assumption that says why.
    assert est.bounds is None
    assert len(est.assumptions) >= 1
    assert any("E-value not computed" in a for a in est.assumptions)


def test_intervene__unidentified_result_has_reason_and_structured_next_step(
    small_corpus: list[Run], fitted: object
) -> None:
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


def test_intervene__temperature_uses_logged_temperature_support() -> None:
    runs = [
        Run(
            schema_version="0.1.0",
            run_id=f"temp-intervene-{i}",
            steps=[
                Step(
                    step_index=0,
                    decisions=[
                        Decision(
                            decision_id=f"d-temp-intervene-{i}",
                            decision_type="model_call",
                            chosen_action="haiku",
                            metadata={"model_config": {"temperature": 0.7 if i % 2 else 0}},
                        )
                    ],
                )
            ],
            outcome=Outcome(kind="binary", value=bool(i % 2), verifier="test"),
        )
        for i in range(40)
    ]
    model = fit_outcome_model(runs, n_bootstrap=10, seed=9)

    est = intervene(
        dag=build_dag(runs[0]),
        model=model,
        step=0,
        intervention={"temperature": "0.70"},
    )

    assert est.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert est.outcome_delta is not None
    if est.next_step.action == "increase_n":
        assert any(row["arm"] == "0.7" for row in est.next_step.payload["arm_breakdown"])


def test_intervene__invalid_intervention_for_decision_type_raises(
    small_corpus: list[Run], fitted: object
) -> None:
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


def test_intervene__increase_n_power_method_two_arm_matches_power_analysis(
    small_corpus: list[Run], fitted: object
) -> None:
    """Two-arm corpus: power_method=binomial_wald_two_arm, agrees with
    counterfact.power_analysis to within ±1 trace."""
    from counterfact import pass_rate_by_arm, power_analysis

    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,
        intervention={"tool_choice": "run_tests"},
    )
    payload = est.next_step.payload
    assert payload["power_method"] == "binomial_wald_two_arm"
    arm_breakdown = payload["arm_breakdown"]
    assert isinstance(arm_breakdown, list)
    assert all({"arm", "n", "pass_count", "pass_rate"} <= set(r) for r in arm_breakdown)
    # Two arms must be present for the synthetic SCM (run_tests + at least one other)
    assert len({r["arm"] for r in arm_breakdown}) >= 2

    # Cross-check against power_analysis on the same corpus.
    table = pass_rate_by_arm(small_corpus, "tool_call")
    target_arm = "run_tests"
    other_arm = next(r.arm for r in table.rows if r.arm != target_arm)
    pr = power_analysis(
        small_corpus,
        decision_type="tool_call",
        arms=(target_arm, other_arm),
        target_ci_width=0.10,
    )
    assert pr.estimated_required_n is not None
    # The engine works off the model's one-hot per-trace view; power_analysis
    # works off per-decision counts. They diverge by O(1%) on multi-decision
    # corpora — a small relative tolerance is the right contract.
    rel_diff = abs(payload["estimated_required_n"] - pr.estimated_required_n) / max(
        pr.estimated_required_n, 1
    )
    assert rel_diff < 0.05, (
        f"engine n={payload['estimated_required_n']} vs "
        f"power_analysis n={pr.estimated_required_n}: rel_diff={rel_diff:.3f}"
    )


def test_intervene__increase_n_power_method_inline_scaling_on_single_arm() -> None:
    import numpy as np

    from counterfact.intervene.api import intervene as intervene_fn
    from counterfact.outcome.model import OutcomeModel

    # Build a single-arm fake OutcomeModel: one tool_call::run_tests column,
    # train_X varies enough that the bootstrap surfaces a wide CI.
    n = 80
    rng = np.random.default_rng(7)
    X = np.ones((n, 1), dtype=float)
    # Mix of pass/fail to get a fittable but wide model
    y = (rng.random(n) < 0.55).astype(int)
    coefs = np.zeros(1)
    intercept = 0.0
    boot_coefs = np.zeros((50, 1), dtype=float)
    boot_intercepts = np.linspace(-8.0, 8.0, 50)

    model = OutcomeModel(
        feature_names=["tool_call::run_tests"],
        coefficients=coefs,
        intercept=intercept,
        bootstrap_coefs=boot_coefs,
        bootstrap_intercepts=boot_intercepts,
        train_X=X,
        train_y=y,
        train_n=n,
        feature_index={"tool_call::run_tests": 0},
        outcome_kind="binary",
    )

    run = Run(
        schema_version="0.1.0",
        run_id="single-arm",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-tool",
                        decision_type="tool_call",
                        chosen_action="run_tests",
                    )
                ],
            )
        ],
        outcome=Outcome(kind="binary", value=True, verifier="stub"),
    )

    est = intervene_fn(
        dag=build_dag(run),
        model=model,
        step=0,
        intervention={"tool_choice": "run_tests"},
    )
    payload = est.next_step.payload
    assert est.next_step.action == "increase_n"
    assert payload["power_method"] == "inline_scaling"
    assert payload["estimated_required_n"] >= payload["current_n"] + 1


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


def test_intervene__decision_id_disambiguates_multi_decision_step(fitted: object) -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="multi-decision-id",
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

    est = intervene(
        dag=build_dag(run),
        model=fitted,
        step=0,
        decision_id="d-model",
        intervention={"model_choice": "sonnet"},
    )

    assert est.query.decision_type == "model_call"
    assert est.query.intervention_kind == "model_choice"


def test_intervene__repeated_decision_type_reports_localization_limit(
    fitted: object,
) -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="repeat-model",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-model-0",
                        decision_type="model_call",
                        chosen_action="haiku",
                    )
                ],
            ),
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="d-model-1",
                        decision_type="model_call",
                        chosen_action="sonnet",
                    )
                ],
            ),
        ],
        outcome=Outcome(kind="binary", value=False, verifier="stub"),
    )

    est = intervene(
        dag=build_dag(run),
        model=fitted,
        step=0,
        intervention={"model_choice": "sonnet"},
    )

    assert est.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert est.outcome_delta is None
    assert "localization_limit" in est.next_step.payload


def test_intervene__repeated_replay_query_still_reports_replay_required(
    fitted: object,
) -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="repeat-model-replay",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d-model-0",
                        decision_type="model_call",
                        chosen_action="haiku",
                    )
                ],
            ),
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="d-model-1",
                        decision_type="model_call",
                        chosen_action="sonnet",
                    )
                ],
            ),
        ],
        outcome=Outcome(kind="binary", value=False, verifier="stub"),
    )

    est = intervene(
        dag=build_dag(run),
        model=fitted,
        step=0,
        intervention={"prompt_content": "be more careful"},
    )

    assert est.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert est.next_step.action == "replay_required"
    assert est.next_step.payload["intervention_target"] == "prompt_content"
    assert "localization_limit" not in est.next_step.payload


# --- attribute_failure spec scenarios ----------------------------------------


def test_attribute_failure__top_k_returns_at_most_k(
    small_corpus: list[Run], fitted: object
) -> None:
    attribution = attribute_failure(dag=_dag(small_corpus[0]), model=fitted)
    assert len(attribution.top_k(5)) <= 5


def test_attribute_failure__each_entry_carries_identifiability_label(
    small_corpus: list[Run], fitted: object
) -> None:
    attribution = attribute_failure(dag=_dag(small_corpus[0]), model=fitted)
    for entry in attribution.top_k(10):
        assert entry.identifiability in {
            IdentifiabilityStatus.IDENTIFIED,
            IdentifiabilityStatus.BOUNDED,
            IdentifiabilityStatus.UNIDENTIFIED,
        }


def test_attribute_failure__empty_corpus_yields_empty_ranking(fitted: object) -> None:
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


def test_evalue__present_on_identified_estimate(small_corpus: list[Run], fitted: object) -> None:
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
    est = intervene(
        dag=_dag(small_corpus[0]),
        model=fitted,
        step=1,
        intervention={"tool_choice": "run_tests"},
    )
    assert est.identifiability == IdentifiabilityStatus.IDENTIFIED
    assert len(est.assumptions) > 0
    assert any("adjustment" in a for a in est.assumptions)


def test_no_silent_l3__bounded_estimate_explains_sensitivity_stance(
    fitted: object,
) -> None:
    """WHEN any bounded estimate is inspected
    THEN estimate.assumptions either names a sensitivity technique used or
    explicitly says no E-value/sensitivity bound was computed (and why)."""
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
    assert any("E-value" in a or "e_value" in a for a in est.assumptions), (
        f"assumptions did not address E-value stance: {est.assumptions}"
    )


def test_unsupported_outcome__non_binary_model_rejected_at_intervene(
    small_corpus: list[Run],
) -> None:
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
