"""Unit coverage for `counterfact.runrecord_export` row derivations."""

from __future__ import annotations

from counterfact.runrecord_export import runs_to_runrecord_rows
from counterfact.schema import Decision, Metadata, Outcome, Run, Step


def _run(
    *,
    run_id: str,
    agent_name: str | None,
    decisions: list[Decision],
    outcome_value: bool = True,
) -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        metadata=Metadata(agent_name=agent_name),
        steps=[Step(step_index=0, decisions=decisions)] if decisions else [],
        outcome=Outcome(kind="binary", value=outcome_value, verifier="stub"),
    )


def test_model_id_uses_first_model_call_chosen_action() -> None:
    run = _run(
        run_id="r-with-model",
        agent_name="claude-agent-sdk",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="model_call",
                chosen_action="claude-sonnet-4-6",
            )
        ],
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["agent_id"] == "claude-agent-sdk"
    assert rows[0]["model_id"] == "claude-sonnet-4-6"


def test_model_id_is_none_when_no_model_call_exists() -> None:
    run = _run(
        run_id="r-no-model-call",
        agent_name="claude-agent-sdk",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="tool_call",
                chosen_action="bash",
            )
        ],
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["agent_id"] == "claude-agent-sdk"
    assert rows[0]["model_id"] is None


def test_model_id_picks_first_model_call_when_multiple_present() -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="r-multi",
        metadata=Metadata(agent_name="claude-agent-sdk"),
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d0",
                        decision_type="tool_call",
                        chosen_action="bash",
                    )
                ],
            ),
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="d1",
                        decision_type="model_call",
                        chosen_action="gpt-4o",
                    )
                ],
            ),
            Step(
                step_index=2,
                decisions=[
                    Decision(
                        decision_id="d2",
                        decision_type="model_call",
                        chosen_action="gpt-4o-mini",
                    )
                ],
            ),
        ],
        outcome=Outcome(kind="binary", value=True, verifier="stub"),
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["model_id"] == "gpt-4o"


def test_partial_credit_is_none_for_binary_outcomes() -> None:
    pass_run = _run(
        run_id="r-pass",
        agent_name="agent",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="model_call",
                chosen_action="m",
            )
        ],
        outcome_value=True,
    )
    fail_run = _run(
        run_id="r-fail",
        agent_name="agent",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="model_call",
                chosen_action="m",
            )
        ],
        outcome_value=False,
    )
    rows, _ = runs_to_runrecord_rows([pass_run, fail_run])
    # success still flips per outcome
    assert rows[0]["success"] is True
    assert rows[1]["success"] is False
    # partial_credit is None — we don't fake it as float(success)
    assert rows[0]["partial_credit"] is None
    assert rows[1]["partial_credit"] is None


def test_provider_totals_do_not_double_count_call_usage() -> None:
    run = _run(
        run_id="usage",
        agent_name="agent",
        decisions=[
            Decision(
                decision_id="d1",
                decision_type="model_call",
                chosen_action="model-a",
                metadata={"usage": {"input_tokens": 10, "output_tokens": 4}},
            ),
            Decision(
                decision_id="d2",
                decision_type="model_call",
                chosen_action="model-a",
                metadata={"usage": {"input_tokens": 20, "output_tokens": 6}},
            ),
        ],
    )
    run.metadata.extra.update(
        {
            "total_cost_usd": 0.5,
            "usage": {
                "input_tokens": 30,
                "output_tokens": 10,
            },
            "model_usage": {"model-a": {"inputTokens": 30, "outputTokens": 10}},
        }
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["tokens_in"] == 30
    assert rows[0]["tokens_out"] == 10
    assert rows[0]["tokens_in_by_model"] == [{"model_id": "model-a", "tokens": 30}]
    assert rows[0]["reported_run_total_cost_usd"] == 0.5
    assert rows[0]["reconstructed_per_task_cost_usd"] is None
    assert rows[0]["cost_provenance"] == "reported"


def test_unknown_tokens_and_outcome_remain_null() -> None:
    run = _run(run_id="unknown", agent_name="agent", decisions=[])
    run.outcome = Outcome(kind="categorical", value="unknown", verifier="missing")
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["tokens_in"] is None
    assert rows[0]["tokens_out"] is None
    assert rows[0]["success"] is None
    assert rows[0]["outcome_status"] == "ungraded"


def test_adapter_usage_and_committed_harness_cost_are_preserved() -> None:
    import json
    from pathlib import Path

    from counterfact.adapters.claude_agent_sdk import run_from_messages
    from counterfact.adapters.openai_agents import run_from_trace

    fixtures = Path(__file__).parents[1] / "fixtures/adapters"
    claude = run_from_messages(
        json.loads((fixtures / "claude_agent_sdk/minimal.jsonl").read_text().splitlines()[0])[
            "messages"
        ]
    )
    openai = run_from_trace(
        json.loads((fixtures / "openai_agents/minimal.json").read_text()), outcome=True
    )
    corpus = Path(__file__).parents[2] / "bench/real/smoke_mixed_outcome"
    harness = Run.model_validate_json(next(corpus.glob("*.json")).read_text())
    rows, _ = runs_to_runrecord_rows([claude, openai, harness])
    assert rows[0]["reported_run_total_cost_usd"] == 0.012
    assert rows[0]["tokens_in"] == 1500
    assert rows[1]["tokens_in"] == 200
    assert rows[1]["tokens_out"] == 50
    expected = sum(
        o.content["cost_usd"]
        for s in harness.steps
        for o in s.observations
        if "cost_usd" in o.content
    )
    assert rows[2]["reconstructed_per_task_cost_usd"] == expected
    assert rows[2]["reported_run_total_cost_usd"] is None
    assert rows[2]["tokens_in"] is None


def test_parquet_schema_is_stable_across_models_empty_and_missing_usage(tmp_path) -> None:
    import polars as pl

    from counterfact.runrecord_export import export_runrecord_parquet

    models = [
        Decision(decision_id="d1", decision_type="model_call", chosen_action="m1"),
        Decision(decision_id="d2", decision_type="model_call", chosen_action="m2"),
    ]
    run = _run(run_id="multi", agent_name="agent", decisions=models)
    first = tmp_path / "first.parquet"
    empty = tmp_path / "empty.parquet"
    export_runrecord_parquet([run], source_corpus=tmp_path, output_path=first)
    export_runrecord_parquet([], source_corpus=tmp_path, output_path=empty)
    assert pl.read_parquet(first).schema == pl.read_parquet(empty).schema
    assert pl.read_parquet(first)["tokens_in"].to_list() == [None]


def test_incomplete_call_costs_cannot_become_a_run_total() -> None:
    from counterfact.schema import Observation

    run = _run(
        run_id="partial",
        agent_name="agent",
        decisions=[
            Decision(decision_id="d1", decision_type="model_call", chosen_action="m1"),
            Decision(decision_id="d2", decision_type="model_call", chosen_action="m1"),
        ],
    )
    run.steps[0].observations.append(Observation(observation_id="o1", content={"cost_usd": 0.1}))
    rows, warnings = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] is None
    assert any("incomplete per-call cost" in warning for warning in warnings)


def test_model_usage_costs_and_tokens_are_totals_with_model_identity() -> None:
    run = _run(
        run_id="model-usage",
        agent_name="agent",
        decisions=[
            Decision(
                decision_id="d1",
                decision_type="model_call",
                chosen_action="arm",
                metadata={"model_id": "provider/model"},
            ),
        ],
    )
    run.metadata.extra["model_usage"] = {
        "provider/model": {"inputTokens": 12, "outputTokens": 3, "costUSD": 0.2},
    }
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["model_id"] == "arm"
    assert rows[0]["model_ids"] == ["provider/model"]
    assert rows[0]["tokens_in_by_model"] == [{"model_id": "provider/model", "tokens": 12}]
    assert rows[0]["reported_run_total_cost_usd"] == 0.2
    assert rows[0]["cost_sources"] == ["metadata.extra.model_usage"]
    run.metadata.agent_name = None
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["agent_id"] == "arm"


def test_export_receipt_failure_leaves_no_published_parquet(tmp_path, monkeypatch) -> None:
    import pytest

    from counterfact import runrecord_export
    from counterfact.corpus_io import CorpusIOError

    output = tmp_path / "runs.parquet"

    def fail_receipt(path, text):
        raise OSError("injected receipt failure")

    monkeypatch.setattr(runrecord_export, "atomic_write", fail_receipt)
    with pytest.raises(CorpusIOError, match="injected"):
        runrecord_export.export_runrecord_parquet([], source_corpus=tmp_path, output_path=output)
    assert not output.exists()
    assert not output.with_suffix(".parquet.receipt.json").exists()


def test_tool_cost_cannot_mask_missing_model_cost_in_parquet(tmp_path) -> None:
    import polars as pl

    from counterfact.runrecord_export import export_runrecord_parquet
    from counterfact.schema import Observation

    run = Run(
        schema_version="0.1.0",
        run_id="missing-model-cost",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="m1",
                        decision_type="model_call",
                        chosen_action="m1",
                    )
                ],
                observations=[Observation(observation_id="cost-m1", content={"cost_usd": 0.2})],
            ),
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="tool",
                        decision_type="tool_call",
                        chosen_action="paid-tool",
                    )
                ],
                observations=[Observation(observation_id="cost-tool", content={"cost_usd": 0.1})],
            ),
            Step(
                step_index=2,
                decisions=[
                    Decision(
                        decision_id="m2",
                        decision_type="model_call",
                        chosen_action="m2",
                    )
                ],
            ),
        ],
        outcome=Outcome(kind="binary", value=True, verifier="test"),
    )
    output = tmp_path / "missing.parquet"
    receipt = export_runrecord_parquet([run], source_corpus=tmp_path, output_path=output)
    assert pl.read_parquet(output)["reconstructed_per_task_cost_usd"].to_list() == [None]
    assert any("incomplete per-call cost" in warning for warning in receipt.warnings)


def test_ambiguous_step_cost_requires_explicit_model_call_identity() -> None:
    from counterfact.schema import Observation

    run = _run(
        run_id="mixed-cost",
        agent_name="agent",
        decisions=[
            Decision(decision_id="model", decision_type="model_call", chosen_action="m1"),
            Decision(decision_id="tool", decision_type="tool_call", chosen_action="paid-tool"),
        ],
    )
    run.steps[0].observations = [
        Observation(observation_id="o1", content={"cost_usd": 0.2, "call_id": "model-call"}),
        Observation(observation_id="o2", content={"cost_usd": 0.1, "call_id": "tool-call"}),
    ]
    rows, warnings = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] is None
    assert any("ambiguous" in warning for warning in warnings)
    run.steps[0].decisions[0].metadata["call_id"] = "model-call"
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] == 0.2 + 0.1


def test_duplicate_observation_call_id_does_not_double_count() -> None:
    from counterfact.schema import Observation

    run = _run(
        run_id="duplicate-call",
        agent_name="agent",
        decisions=[
            Decision(decision_id="m1", decision_type="model_call", chosen_action="m1"),
        ],
    )
    run.steps[0].observations = [
        Observation(
            observation_id="o1",
            content={"cost_usd": 0.2, "call_id": "same-call"},
        )
    ]
    run.steps.append(
        Step(
            step_index=1,
            decisions=[
                Decision(
                    decision_id="m2",
                    decision_type="model_call",
                    chosen_action="m1",
                )
            ],
            observations=[
                Observation(
                    observation_id="o2",
                    content={"cost_usd": 0.2, "call_id": "same-call"},
                )
            ],
        )
    )
    rows, warnings = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] is None
    assert any("ambiguous" in warning for warning in warnings)


def test_partial_provider_model_usage_does_not_become_a_run_total() -> None:
    run = _run(
        run_id="partial-models",
        agent_name="agent",
        decisions=[
            Decision(decision_id="m1", decision_type="model_call", chosen_action="m1"),
            Decision(decision_id="m2", decision_type="model_call", chosen_action="m2"),
        ],
    )
    run.metadata.extra["model_usage"] = {
        "m1": {"inputTokens": 12, "outputTokens": 3, "costUSD": 0.2},
    }
    rows, warnings = runs_to_runrecord_rows([run])
    assert rows[0]["tokens_in"] is None
    assert rows[0]["tokens_out"] is None
    assert rows[0]["reported_run_total_cost_usd"] is None
    assert rows[0]["tokens_in_by_model"] == [
        {"model_id": "m1", "tokens": 12},
        {"model_id": "m2", "tokens": None},
    ]
    assert any("model_usage omits known models" in warning for warning in warnings)
    run.metadata.extra.update(
        {
            "total_cost_usd": 0.7,
            "usage": {
                "input_tokens": 40,
                "output_tokens": 10,
            },
        }
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["tokens_in"] == 40
    assert rows[0]["tokens_out"] == 10
    assert rows[0]["reported_run_total_cost_usd"] == 0.7


def test_complete_model_coverage_preserves_tool_cost_and_unknown_tool_cost() -> None:
    import pytest

    from counterfact.schema import Observation

    run = Run(
        schema_version="0.1.0",
        run_id="observed-task-cost",
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="m1",
                        decision_type="model_call",
                        chosen_action="m1",
                    )
                ],
                observations=[Observation(observation_id="m1-cost", content={"cost_usd": 0.2})],
            ),
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="tool",
                        decision_type="tool_call",
                        chosen_action="paid-tool",
                    )
                ],
                observations=[Observation(observation_id="tool-cost", content={"cost_usd": 0.1})],
            ),
            Step(
                step_index=2,
                decisions=[
                    Decision(
                        decision_id="m2",
                        decision_type="model_call",
                        chosen_action="m2",
                    )
                ],
                observations=[Observation(observation_id="m2-cost", content={"cost_usd": 0.4})],
            ),
        ],
        outcome=Outcome(kind="binary", value=True, verifier="test"),
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] == pytest.approx(0.7)
    assert rows[0]["cost_sources"] == ["observations.content.cost_usd"]
    run.steps[1].observations[0].content["cost_usd"] = None
    rows, warnings = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] is None
    assert warnings
    run.steps[1].observations[0].content["cost_usd"] = 0.1
    run.steps[2].observations = []
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] is None


def test_duplicate_tool_call_cost_identity_makes_task_total_unknown() -> None:
    from counterfact.schema import Observation

    run = _run(
        run_id="duplicate-tool",
        agent_name="agent",
        decisions=[
            Decision(decision_id="m1", decision_type="model_call", chosen_action="m1"),
        ],
    )
    run.steps[0].observations = [
        Observation(observation_id="model-cost", content={"cost_usd": 0.2})
    ]
    run.steps.append(
        Step(
            step_index=1,
            observations=[
                Observation(
                    observation_id="tool-one", content={"cost_usd": 0.1, "call_id": "tool"}
                ),
                Observation(
                    observation_id="tool-two", content={"cost_usd": 0.1, "call_id": "tool"}
                ),
            ],
        )
    )
    rows, warnings = runs_to_runrecord_rows([run])
    assert rows[0]["reconstructed_per_task_cost_usd"] is None
    assert warnings
