"""Acceptance tests for the naive marginal estimator (`pass_rate_by_arm`)."""

from __future__ import annotations

from bench.synthetic import generate_traces
from counterfact import build_dag, fit_outcome_model, intervene
from counterfact.baselines import PassRateTable, pass_rate_by_arm
from counterfact.intervene import CausalEstimate, IdentifiabilityStatus
from counterfact.schema import Decision, Outcome, Run, Step


def _make_run(*, run_id: str, arms_for_decision: list[tuple[str, str]], outcome_pass: bool) -> Run:
    """Build a minimal Run with one Step per (decision_type, chosen_action) pair."""
    steps: list[Step] = []
    for i, (decision_type, action) in enumerate(arms_for_decision):
        steps.append(
            Step(
                step_index=i,
                decisions=[
                    Decision(
                        decision_id=f"{run_id}-d{i}",
                        decision_type=decision_type,
                        chosen_action=action,
                    )
                ],
            )
        )
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        steps=steps,
        outcome=Outcome(kind="binary", value=outcome_pass, verifier="stub"),
    )


def test_pass_rate_by_arm_returns_one_row_per_observed_arm() -> None:
    corpus = [
        _make_run(run_id="r0", arms_for_decision=[("model_call", "small")], outcome_pass=True),
        _make_run(run_id="r1", arms_for_decision=[("model_call", "large")], outcome_pass=True),
        _make_run(run_id="r2", arms_for_decision=[("model_call", "large")], outcome_pass=False),
    ]
    table = pass_rate_by_arm(corpus, "model_call")
    assert table.decision_type == "model_call"
    arms = sorted(r.arm for r in table.rows)
    assert arms == ["large", "small"]


def test_pass_rate_equals_pass_count_over_n() -> None:
    corpus = [
        _make_run(run_id="r0", arms_for_decision=[("retry", "no_retry")], outcome_pass=True),
        _make_run(run_id="r1", arms_for_decision=[("retry", "no_retry")], outcome_pass=False),
        _make_run(run_id="r2", arms_for_decision=[("retry", "retry_once")], outcome_pass=True),
        _make_run(run_id="r3", arms_for_decision=[("retry", "retry_once")], outcome_pass=True),
    ]
    table = pass_rate_by_arm(corpus, "retry")
    by_arm = {r.arm: r for r in table.rows}
    assert by_arm["no_retry"].n == 2
    assert by_arm["no_retry"].pass_count == 1
    assert by_arm["no_retry"].pass_rate == 0.5
    assert by_arm["retry_once"].n == 2
    assert by_arm["retry_once"].pass_count == 2
    assert by_arm["retry_once"].pass_rate == 1.0


def test_ci_brackets_pass_rate() -> None:
    corpus = [
        _make_run(
            run_id=f"r{i}",
            arms_for_decision=[("model_call", "large")],
            outcome_pass=(i % 3 != 0),
        )
        for i in range(30)
    ]
    table = pass_rate_by_arm(corpus, "model_call")
    [row] = table.rows
    assert row.ci_low <= row.pass_rate <= row.ci_high
    assert 0.0 <= row.ci_low <= 1.0
    assert 0.0 <= row.ci_high <= 1.0


def test_empty_corpus_returns_empty_table() -> None:
    table = pass_rate_by_arm([], "model_call")
    assert table.decision_type == "model_call"
    assert table.rows == []


def test_actionless_decision_is_skipped() -> None:
    corpus = [
        Run(
            schema_version="0.1.0",
            run_id="r0",
            steps=[
                Step(
                    step_index=0,
                    decisions=[
                        Decision(decision_id="r0-d0", decision_type="model_call"),
                        Decision(
                            decision_id="r0-d1",
                            decision_type="model_call",
                            chosen_action="large",
                        ),
                    ],
                )
            ],
            outcome=Outcome(kind="binary", value=True, verifier="stub"),
        )
    ]
    table = pass_rate_by_arm(corpus, "model_call")
    assert [row.arm for row in table.rows] == ["large"]


def test_all_actionless_matching_decisions_return_empty_table() -> None:
    corpus = [
        Run(
            schema_version="0.1.0",
            run_id="r0",
            steps=[
                Step(
                    step_index=0,
                    decisions=[
                        Decision(decision_id="r0-d0", decision_type="model_call"),
                    ],
                )
            ],
            outcome=Outcome(kind="binary", value=True, verifier="stub"),
        )
    ]
    table = pass_rate_by_arm(corpus, "model_call")
    assert table.rows == []


def test_decision_type_filter_excludes_other_types() -> None:
    run = _make_run(
        run_id="r",
        arms_for_decision=[("retry", "no_retry"), ("model_call", "large")],
        outcome_pass=True,
    )
    table = pass_rate_by_arm([run], "model_call")
    assert len(table.rows) == 1
    assert table.rows[0].arm == "large"
    assert table.rows[0].n == 1


def test_naive_baseline_and_causal_estimator_return_distinct_structures() -> None:
    """Behavioral pin on the descriptive-vs-causal distinction.

    `pass_rate_by_arm` is the naive marginal baseline: a tabular per-arm
    breakdown with raw rates and Wilson CIs. `intervene` is the causal
    estimator: a structured `CausalEstimate` with an identifiability label.
    Running both on the same corpus must produce different result types — if
    a future refactor accidentally aliases one to the other, this test fails.
    """
    corpus = [Run.model_validate(t) for t in generate_traces(n=120, seed=11)]

    table = pass_rate_by_arm(corpus, "model_call")
    model = fit_outcome_model(corpus, n_bootstrap=10, seed=11)
    estimate = intervene(
        dag=build_dag(corpus[0]),
        model=model,
        step=2,  # model_call step in synthetic SCM
        intervention={"model_choice": "sonnet"},
    )

    assert isinstance(table, PassRateTable)
    assert isinstance(estimate, CausalEstimate)
    assert type(table) is not type(estimate)

    # Descriptive: tabular rows of raw rates, no identifiability concept.
    assert hasattr(table, "rows")
    assert not hasattr(table, "identifiability")
    assert all(0.0 <= row.pass_rate <= 1.0 for row in table.rows)

    # Causal: identifiability is required and lives on a known enum.
    assert hasattr(estimate, "identifiability")
    assert isinstance(estimate.identifiability, IdentifiabilityStatus)
