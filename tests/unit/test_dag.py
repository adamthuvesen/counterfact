"""Tests for build_dag scenarios in causal-engine spec."""

from __future__ import annotations

import pytest

from counterfact import build_dag
from counterfact.errors import DAGCycleError
from counterfact.schema import Decision, Outcome, Run, Step


def _make_run(steps: list[Step]) -> Run:
    return Run(
        schema_version="0.1.0",
        run_id="r-dag",
        steps=steps,
        outcome=Outcome(kind="binary", value=True, verifier="pytest"),
    )


def test_build_dag__empty_trace_yields_empty_dag() -> None:
    dag = build_dag(_make_run([]))
    assert dag.nodes == []
    assert dag.edges == []


def test_build_dag__edges_respect_parent_declarations() -> None:
    plan_2 = Decision(
        decision_id="d-2-plan",
        decision_type="plan_step",
        chosen_action="investigate",
    )
    tool_3 = Decision(decision_id="d-3-tool", decision_type="tool_call", chosen_action="run_tests")
    run = _make_run(
        [
            Step(step_index=2, decisions=[plan_2]),
            Step(step_index=3, decisions=[tool_3]),
        ]
    )
    dag = build_dag(run)
    assert ("d-2-plan", "d-3-tool") in dag.edges


def test_build_dag__is_acyclic() -> None:
    plan = Decision(decision_id="d0", decision_type="plan_step")
    model = Decision(decision_id="d1", decision_type="model_call", chosen_action="claude-haiku")
    tool = Decision(decision_id="d2", decision_type="tool_call", chosen_action="run_tests")
    run = _make_run(
        [
            Step(step_index=0, decisions=[plan]),
            Step(step_index=1, decisions=[model]),
            Step(step_index=2, decisions=[tool]),
        ]
    )
    dag = build_dag(run)
    order = dag.topological_sort()
    assert len(order) == 3
    # parents must precede children in any topological order
    pos = {nid: i for i, nid in enumerate(order)}
    for parent, child in dag.edges:
        assert pos[parent] < pos[child]


def test_build_dag__cycle_raises() -> None:
    """A constructed DAG that would have a cycle raises DAGCycleError.

    This is internal-contract coverage: real traces should never produce a
    cycle (steps are temporally ordered) but the builder MUST refuse one if
    asked.
    """
    from counterfact.dag.graph import DAG

    a = Decision(decision_id="a", decision_type="plan_step")
    b = Decision(decision_id="b", decision_type="plan_step")
    with pytest.raises(DAGCycleError):
        DAG(nodes=[a, b], edges=[("a", "b"), ("b", "a")])


def test_build_dag__adjacency_lookups_are_constant_time() -> None:
    """parents_of/children_of read from precomputed adjacency dicts, not the edge list."""
    plan = Decision(decision_id="d0", decision_type="plan_step", chosen_action="investigate")
    tool = Decision(decision_id="d1", decision_type="tool_call", chosen_action="run_tests")
    run = _make_run(
        [
            Step(step_index=0, decisions=[plan]),
            Step(step_index=1, decisions=[tool]),
        ]
    )
    dag = build_dag(run)

    # Smoke-test the contract: lookups are backed by per-node adjacency dicts
    # so the implementation must store an entry for every node id, not scan
    # `dag.edges` linearly on each call.
    assert set(dag._adj.keys()) == {"d0", "d1"}
    assert set(dag._radj.keys()) == {"d0", "d1"}
    # The expected edge from plan -> tool is in adjacency.
    assert dag.children_of("d0") == ["d1"]
    assert dag.parents_of("d1") == ["d0"]
    # Appending to `dag.edges` after construction must not affect existing
    # lookups — that proves the adjacency dicts are independent of the edge
    # list rather than recomputed on each call.
    dag.edges.append(("d1", "d0"))
    assert dag.children_of("d1") == []
    assert dag.parents_of("d0") == []
