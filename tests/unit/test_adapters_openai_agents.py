"""Unit coverage for `counterfact.adapters.openai_agents`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact.adapters._common import IngestError
from counterfact.adapters.openai_agents import (
    ingest_openai_agents,
    run_from_trace,
)
from counterfact.schema import Run

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures/adapters/openai_agents"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def test_minimal_trace_writes_run_with_required_outcome(tmp_path: Path) -> None:
    receipt = ingest_openai_agents(FIXTURE_DIR / "minimal.json", tmp_path, outcome=True)

    assert receipt.generated_count == 1
    run = Run.model_validate_json((tmp_path / "trace-ai-001.json").read_text())
    assert run.outcome.value is True
    assert run.outcome.verifier == "caller_supplied_outcome"


def test_function_span_becomes_tool_call_decision() -> None:
    run = run_from_trace(_load("minimal.json"), outcome=True)
    tool_calls = [d for step in run.steps for d in step.decisions if d.decision_type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].chosen_action == "web_search"
    assert tool_calls[0].metadata["input"] == {"query": "counterfact docs"}


def test_generation_span_becomes_model_call_decision() -> None:
    run = run_from_trace(_load("minimal.json"), outcome=True)
    model_calls = [
        d for step in run.steps for d in step.decisions if d.decision_type == "model_call"
    ]
    assert len(model_calls) == 1
    assert model_calls[0].chosen_action == "gpt-5-mini"
    assert model_calls[0].metadata["usage"] == {"prompt_tokens": 200, "completion_tokens": 50}


def test_steps_are_ordered_by_started_at() -> None:
    run = run_from_trace(_load("minimal.json"), outcome=True)
    decision_actions = [step.decisions[0].chosen_action for step in run.steps if step.decisions]
    # function span (web_search) starts before generation span (gpt-5-mini)
    assert decision_actions == ["web_search", "gpt-5-mini"]


def test_handoff_span_becomes_plan_step_decision() -> None:
    run = run_from_trace(_load("with_handoff.json"))
    handoffs = [d for step in run.steps for d in step.decisions if d.decision_type == "plan_step"]
    assert len(handoffs) == 1
    assert handoffs[0].chosen_action == "handoff:coder"
    assert handoffs[0].metadata["from_agent"] == "router"


def test_outcome_marker_span_drives_outcome_without_flag() -> None:
    run = run_from_trace(_load("with_handoff.json"))
    assert run.outcome.value is True
    assert run.outcome.verifier == "counterfact_outcome_marker"


def test_outcome_marker_rejects_string_boolean() -> None:
    trace = _load("with_handoff.json")
    marker = next(
        span for span in trace["spans"] if span["span_data"].get("name") == "counterfact.outcome"
    )
    marker["span_data"]["data"]["value"] = "false"

    with pytest.raises(IngestError, match="JSON boolean"):
        run_from_trace(trace)


def test_root_error_drives_fail_outcome_without_flag() -> None:
    run = run_from_trace(_load("root_error.json"))
    assert run.outcome.value is False
    assert run.outcome.verifier == "openai_agents_root_span_error"
    assert run.outcome.metadata["error"]["message"] == "tool timed out"


def test_missing_outcome_raises_ingest_error() -> None:
    with pytest.raises(IngestError, match="explicit outcome"):
        run_from_trace(_load("minimal.json"))


def test_decisions_carry_no_randomization_metadata() -> None:
    run = run_from_trace(_load("minimal.json"), outcome=True)
    for step in run.steps:
        for decision in step.decisions:
            assert decision.policy is None
            assert decision.propensity is None


def test_unknown_span_type_aborts() -> None:
    bad = _load("minimal.json")
    bad["spans"].append(
        {
            "object": "trace.span",
            "id": "span-bogus",
            "trace_id": "trace-ai-001",
            "parent_id": "span-root",
            "started_at": "2026-05-06T19:00:01.400Z",
            "ended_at": "2026-05-06T19:00:01.450Z",
            "span_data": {"type": "future_widget", "data": {}},
            "error": None,
        }
    )
    with pytest.raises(IngestError, match="future_widget"):
        run_from_trace(bad, outcome=True)


def test_unreachable_span_aborts() -> None:
    bad = _load("minimal.json")
    bad["spans"].append(
        {
            "object": "trace.span",
            "id": "span-orphan",
            "trace_id": "trace-ai-001",
            "parent_id": "span-missing",
            "started_at": "2026-05-06T19:00:01.400Z",
            "ended_at": "2026-05-06T19:00:01.450Z",
            "span_data": {"type": "function", "name": "lost_tool"},
            "error": None,
        }
    )

    with pytest.raises(IngestError, match=r"unreachable.*span-orphan"):
        run_from_trace(bad, outcome=True)


def test_duplicate_span_id_aborts() -> None:
    bad = _load("minimal.json")
    duplicate = dict(bad["spans"][1])
    duplicate["parent_id"] = "span-root"
    bad["spans"].append(duplicate)

    with pytest.raises(IngestError, match="duplicate span id"):
        run_from_trace(bad, outcome=True)


def test_metadata_extra_carries_root_name() -> None:
    run = run_from_trace(_load("minimal.json"), outcome=True)
    assert run.metadata.extra["source_format"] == "openai-agents"
    assert run.metadata.extra["root_name"] == "research_agent"


def test_ingest_writes_receipt_with_randomization_warning(tmp_path: Path) -> None:
    ingest_openai_agents(FIXTURE_DIR / "with_handoff.json", tmp_path)
    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert receipt["source_format"] == "openai-agents"
    assert any("randomization" in w for w in receipt["warnings"])
