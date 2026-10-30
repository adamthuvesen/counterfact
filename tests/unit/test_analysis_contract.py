"""Analysis boundaries preserve chronology, populations, and honest result states."""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from counterfact import AnalysisCorpus, build_dag, contrast, intervene
from counterfact.analysis.timeline import first_decision
from counterfact.cli.formatters import format_report
from counterfact.corpus_analyzer import analyze_corpus
from counterfact.errors import InvalidInterventionError
from counterfact.intervene import (
    CausalEstimate,
    DistributionSummary,
    IdentifiabilityStatus,
    IdentificationBounds,
    InterventionQuery,
    NextStep,
)
from counterfact.intervene.estimators import treatment_data
from counterfact.outcome.model import fit_logistic
from counterfact.policy import Policy, PolicyRule, evaluate_policy
from counterfact.schema import Decision, Outcome, Run, Step


def _runs(n: int = 40) -> list[Run]:
    return [
        Run(
            schema_version="0.1.0",
            run_id=f"r-{index}",
            steps=[
                Step(
                    step_index=0,
                    decisions=[
                        Decision(
                            decision_id=f"plan-{index}",
                            decision_type="plan_step",
                            chosen_action="inspect",
                        )
                    ],
                ),
                Step(
                    step_index=1,
                    decisions=[
                        Decision(
                            decision_id=f"tool-{index}",
                            decision_type="tool_call",
                            chosen_action="inspect" if index % 3 else "search",
                        )
                    ],
                ),
                Step(
                    step_index=10,
                    decisions=[
                        Decision(
                            decision_id=f"model-{index}",
                            decision_type="model_call",
                            chosen_action="large" if index % 2 else "small",
                            policy="uniform",
                            policy_params={},
                            valid_actions=["small", "large"],
                            propensity=0.5,
                            context_features={},
                            metadata={"model_config": {"temperature": 0.5}},
                        )
                    ],
                ),
            ],
            outcome=Outcome(kind="binary", value=index % 3 != 0, verifier="test"),
        )
        for index in range(n)
    ]


def _contrast(runs: list[Run] | AnalysisCorpus) -> CausalEstimate:
    return contrast(
        corpus=runs,
        decision_type="model_call",
        intervention={"model_choice": "large"},
        baseline="small",
        seed=7,
    )


def test_step_permutation_preserves_graph_treatment_and_estimate() -> None:
    runs = _runs()
    reversed_runs = [run.model_copy(update={"steps": list(reversed(run.steps))}) for run in runs]
    serialized = reversed_runs[0].model_dump_json()
    assert build_dag(runs[0]).edges == build_dag(reversed_runs[0]).edges
    original = treatment_data(runs, "model_call", "model_call")
    reordered = treatment_data(reversed_runs, "model_call", "model_call")
    assert original == reordered
    assert _contrast(runs).model_dump() == _contrast(reversed_runs).model_dump()
    assert reversed_runs[0].model_dump_json() == serialized


def test_corpus_snapshot_preserves_row_order_and_detaches_mutable_views() -> None:
    runs = _runs()
    corpus = AnalysisCorpus(runs, seed=7)
    expected = _contrast(corpus).model_dump()
    assert [run.run_id for run in corpus.runs] == [run.run_id for run in runs]
    runs[0].steps[2].decisions[0].chosen_action = "changed"
    view = corpus.runs
    view[1].outcome.value = False
    view[1].steps.clear()
    assert _contrast(corpus).model_dump() == expected
    with pytest.raises(FrozenInstanceError):
        corpus.facts[0].run_id = "changed"  # type: ignore[misc]
    config = corpus.facts[0].steps[2].decisions[0].metadata["model_config"]
    with pytest.raises(TypeError):
        config["temperature"] = 9  # type: ignore[index]


def test_duplicate_step_indexes_parse_but_analysis_rejects_them() -> None:
    run = _runs(1)[0]
    run.steps.append(Step(step_index=10))
    round_trip = Run.model_validate_json(run.model_dump_json())
    assert len(round_trip.steps) == 4
    with pytest.raises(ValueError, match=r"r-0.*duplicate step_index"):
        AnalysisCorpus([round_trip])
    with pytest.raises(ValueError, match=r"r-0.*duplicate step_index"):
        build_dag(round_trip)


def test_first_occurrence_tie_refuses_without_dropping_ambiguous_runs() -> None:
    runs = _runs()
    treatment = runs[0].steps[2]
    treatment.decisions.append(treatment.decisions[0].model_copy(update={"decision_id": "tie"}))
    with pytest.raises(ValueError, match=r"multiple first.*step 10"):
        first_decision(runs[0], "model_call")
    estimate = _contrast(runs)
    assert estimate.identifiability == "unidentified"
    assert "r-0" in (estimate.reason or "")
    assert estimate.outcome_delta is None


def test_later_occurrences_stay_in_total_effect_but_focal_query_refuses() -> None:
    runs = _runs()
    for run in runs:
        decision = run.steps[2].decisions[0]
        run.steps.append(
            Step(
                step_index=20,
                decisions=[
                    decision.model_copy(
                        update={
                            "decision_id": decision.decision_id + "-retry",
                            "chosen_action": "small",
                        }
                    )
                ],
            )
        )
    estimate = _contrast(runs)
    assert estimate.identifiability == "identified"
    assert estimate.query.step is None
    assert estimate.diagnostics is not None
    assert estimate.diagnostics.n_used == len(runs)
    assert estimate.diagnostics.n_excluded == 0
    assert all("model_call" not in name for name in estimate.adjustment_set)
    focal = intervene(
        dag=build_dag(runs[0]),
        model=AnalysisCorpus(runs),
        step=10,
        intervention={"model_choice": "large"},
        baseline="small",
    )
    assert focal.identifiability == "unidentified"


@pytest.mark.parametrize(
    "policy,params,actions",
    [
        ("epsilon_greedy", {"epsilon": 2, "greedy": "small"}, ["small", "large"]),
        ("epsilon_greedy", {"epsilon": 10**400, "greedy": "small"}, ["small", "large"]),
        ("epsilon_greedy", {"epsilon": "invalid", "greedy": "small"}, ["small", "large"]),
        ("epsilon_greedy", {"epsilon": float("nan"), "greedy": "small"}, ["small", "large"]),
        ("epsilon_greedy", {"epsilon": 0.5, "greedy": "absent"}, ["small", "large"]),
        ("epsilon_greedy", {"epsilon": 0.5}, ["small", "large"]),
        ("uniform", {}, ["small", "large", "third"]),
        ("uniform", {}, ["small", "large", "small"]),
    ],
)
def test_invalid_assignment_evidence_never_identifies_or_crashes(policy, params, actions) -> None:
    runs = _runs()
    for run in runs:
        decision = run.steps[2].decisions[0]
        decision.policy = policy
        decision.policy_params = params
        decision.valid_actions = actions
    assert _contrast(runs).identifiability == "bounded"
    policy_query = Policy(
        name="large", rules=[PolicyRule(decision_type="model_call", action="large")]
    )
    assert evaluate_policy(runs, policy_query).identifiability == "bounded"


def test_same_step_decisions_are_not_adjustment_features() -> None:
    runs = _runs()
    for run in runs:
        tool = run.steps.pop(1).decisions[0]
        run.steps[1].decisions.append(tool)
    estimate = _contrast(runs)
    assert estimate.identifiability == "identified"
    assert estimate.adjustment_set == []
    coverage = analyze_corpus(runs).identifiability_coverage
    assert "identified" in coverage.reachable


def test_analysis_accepts_single_class_without_a_descriptive_fit() -> None:
    runs = _runs()
    for run in runs:
        run.outcome.value = True
    assert _contrast(AnalysisCorpus(runs)).identifiability == "unidentified"


def test_mixed_imbalanced_corpus_guidance_does_not_claim_unfittable(tmp_path) -> None:
    runs = _runs()
    for index, run in enumerate(runs):
        run.outcome.value = index != 0
    report = analyze_corpus(runs)
    criterion = next(c for c in report.criteria if c.name == "outcome_balance")
    assert criterion.reason_code == "imbalanced"
    assert report.identifiability_coverage.unfittable_outcome_model is False
    rendered = format_report(report, tmp_path)
    assert "outcome model is unfittable" not in rendered
    assert "more balanced mix" in rendered


@pytest.mark.parametrize(
    "point,low,high",
    [
        (float("nan"), 0, 1),
        (0, float("-inf"), 1),
        (0, 1, -1),
        (2, 0, 1),
    ],
)
def test_sampling_interval_rejects_nonfinite_or_unordered_values(point, low, high) -> None:
    with pytest.raises(ValidationError):
        DistributionSummary(point=point, ci_low=low, ci_high=high)


def test_result_fields_and_nested_payloads_cannot_mutate_estimate_state() -> None:
    estimate = _contrast(_runs())
    before = estimate.model_dump_json()
    with pytest.raises(ValidationError):
        estimate.identifiability = "unidentified"  # type: ignore[misc,assignment]
    assert estimate.outcome_delta is not None
    with pytest.raises(ValidationError):
        estimate.outcome_delta.ci_low = float("nan")  # type: ignore[misc]
    payload = estimate.next_step.payload
    payload.clear()
    estimate.warnings.append("changed")
    assert estimate.model_dump_json() == before
    with pytest.raises(ValidationError):
        estimate.model_copy(update={"identifiability": "unidentified"})
    with pytest.raises(ValidationError):
        estimate.next_step.model_copy(update={"payload": {}})


def test_replay_query_detaches_nested_tuple_and_set_targets() -> None:
    labels = {"a", "b"}
    target = ({"content": "original", "labels": labels},)
    result = contrast(
        corpus=_runs(),
        decision_type="model_call",
        intervention={"prompt_content": target},
        baseline="original prompt",
    )
    original = result.model_dump()
    target[0]["content"] = "changed"
    labels.add("external")
    assert result.model_dump() == original
    view = result.query.target
    view[0]["content"] = "changed again"
    view[0]["labels"].add("public view")
    assert result.model_dump() == original


def test_analysis_facts_freeze_python_set_metadata() -> None:
    runs = _runs(1)
    labels = {"a", "b"}
    runs[0].steps[2].decisions[0].metadata["labels"] = labels
    corpus = AnalysisCorpus(runs)
    fact_labels = corpus.facts[0].steps[2].decisions[0].metadata["labels"]
    labels.add("external")
    assert fact_labels == frozenset({"a", "b"})
    assert isinstance(fact_labels, frozenset)
    with pytest.raises(AttributeError):
        fact_labels.add("mutation")  # type: ignore[attr-defined]


def test_label_contract_rejects_fake_identified_and_allows_bound_without_interval() -> None:
    query = InterventionQuery(
        decision_type="model_call",
        intervention_kind="model_choice",
        target="large",
        baseline="small",
        step=None,
    )
    next_step = NextStep(action="none", human_text="No action.")
    with pytest.raises(ValidationError, match="diagnostics"):
        CausalEstimate(
            query=query,
            identifiability=IdentifiabilityStatus.IDENTIFIED,
            next_step=next_step,
            outcome_delta=DistributionSummary(point=0.1, ci_low=0, ci_high=0.2),
        )
    assert (
        CausalEstimate(
            query=query, identifiability=IdentifiabilityStatus.BOUNDED, next_step=next_step
        ).bounds
        is None
    )
    with pytest.raises(ValidationError):
        IdentificationBounds(lower=0.5, upper=-0.5)


def test_payload_validates_types_and_detaches_constructor_input() -> None:
    missing_arms = ["large"]
    payload = {
        "arm_name": "model_call",
        "missing_strata": ["x"],
        "observed_arms": [],
        "missing_arms": missing_arms,
    }
    next_step = NextStep(
        action="broaden_arm_support", payload=payload, human_text="Collect support."
    )
    missing_arms.clear()
    assert next_step.payload["missing_arms"] == ["large"]
    with pytest.raises(ValidationError):
        next_step.model_copy(update={"payload": {**next_step.payload, "missing_arms": "large"}})


def test_logistic_fit_exposes_convergence_problems(monkeypatch) -> None:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    fit = LogisticRegression.fit

    def warn_and_fit(self, X, y, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
        return fit(self, X, y, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", warn_and_fit)
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        fit_logistic(np.eye(4), np.array([0, 1, 0, 1]))


def test_policy_interval_names_the_actual_estimator() -> None:
    policy = Policy(name="large", rules=[PolicyRule(decision_type="model_call", action="large")])
    result = evaluate_policy(_runs(), policy)
    assert result.outcome_delta is not None
    assert result.outcome_delta.ci_method == "sequential_dr_influence_function"


@pytest.mark.parametrize("target", [float("inf"), float("nan"), "-inf", "nan", "hot"])
def test_temperature_user_input_rejects_nonfinite_or_nonnumeric_values(target) -> None:
    with pytest.raises(InvalidInterventionError, match="finite numeric"):
        contrast(
            corpus=_runs(),
            decision_type="model_call",
            intervention={"temperature": target},
            baseline=0.5,
        )


@given(
    indexes=st.lists(st.integers(), min_size=3, max_size=3, unique=True),
    order=st.permutations((0, 1, 2)),
)
def test_numeric_chronology_is_invariant_to_serialized_step_permutation(indexes, order) -> None:
    run = _runs(1)[0]
    for step, index in zip(run.steps, sorted(indexes), strict=True):
        step.step_index = index
    permuted = run.model_copy(update={"steps": [run.steps[index] for index in order]})
    assert build_dag(run).edges == build_dag(permuted).edges
    assert treatment_data([run], "model_call", "model_call") == treatment_data(
        [permuted], "model_call", "model_call"
    )


@pytest.mark.parametrize("command", ["analyze", "intervene", "explain", "compare", "diagnose"])
def test_analysis_cli_rejects_ambiguous_chronology_without_partial_output(
    command, tmp_path, capsys
) -> None:
    from counterfact.cli import main

    runs = _runs(4)
    runs[0].steps.append(Step(step_index=10))
    paths = []
    for run in runs:
        path = tmp_path / f"{run.run_id}.json"
        path.write_text(run.model_dump_json())
        paths.append(str(path))
    args = {
        "analyze": ["analyze", "corpus", str(tmp_path)],
        "intervene": [
            "intervene",
            paths[0],
            "--runs-dir",
            str(tmp_path),
            "--decision-id",
            "model-0",
            "--set",
            "model_choice=large",
        ],
        "explain": ["explain", paths[0], "--runs-dir", str(tmp_path)],
        "compare": ["compare", paths[0], paths[1]],
        "diagnose": ["diagnose", paths[0], "--runs-dir", str(tmp_path)],
    }
    assert main(args[command]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "r-0" in captured.err
    assert "duplicate step_index" in captured.err
    assert "Traceback" not in captured.err
