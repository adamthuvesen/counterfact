"""Unit coverage for `counterfact.adapters.claude_agent_sdk`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact.adapters._common import IngestError
from counterfact.adapters.claude_agent_sdk import (
    ingest_claude_agent_sdk,
    run_from_messages,
)
from counterfact.schema import Run

FIXTURE = Path(__file__).parent.parent / "fixtures/adapters/claude_agent_sdk/minimal.jsonl"


def _pass_messages() -> list[dict]:
    return [
        {
            "__type__": "AssistantMessage",
            "model": "claude-sonnet-4-6",
            "session_id": "sess-pass-001",
            "content": [
                {
                    "__type__": "ToolUseBlock",
                    "id": "tu_1",
                    "name": "read_file",
                    "input": {"path": "src/foo.py"},
                }
            ],
        },
        {
            "__type__": "UserMessage",
            "session_id": "sess-pass-001",
            "parent_tool_use_id": "tu_1",
            "content": [],
            "tool_use_result": {"stdout": "ok\n", "exit_code": 0},
        },
        {
            "__type__": "AssistantMessage",
            "model": "claude-sonnet-4-6",
            "session_id": "sess-pass-001",
            "content": [{"__type__": "TextBlock", "text": "Looks good."}],
        },
        {
            "__type__": "ResultMessage",
            "subtype": "success",
            "session_id": "sess-pass-001",
            "duration_ms": 1234,
            "duration_api_ms": 980,
            "is_error": False,
            "num_turns": 2,
            "total_cost_usd": 0.01,
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "result": "Looks good.",
            "stop_reason": "end_turn",
        },
    ]


def test_run_id_comes_from_result_message_session_id() -> None:
    run = run_from_messages(_pass_messages())

    assert run.run_id == "sess-pass-001"


def test_tool_use_blocks_become_tool_call_decisions() -> None:
    run = run_from_messages(_pass_messages())

    tool_calls = [
        d
        for step in run.steps
        for d in step.decisions
        if d.decision_type == "tool_call"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].chosen_action == "read_file"
    assert tool_calls[0].metadata["input"] == {"path": "src/foo.py"}


def test_text_only_assistant_messages_become_model_call_decisions() -> None:
    run = run_from_messages(_pass_messages())

    model_calls = [
        d
        for step in run.steps
        for d in step.decisions
        if d.decision_type == "model_call"
    ]
    assert len(model_calls) == 1
    assert model_calls[0].chosen_action == "claude-sonnet-4-6"


def test_tool_use_result_attaches_to_same_step_as_tool_call() -> None:
    run = run_from_messages(_pass_messages())

    tool_call_step = next(
        step
        for step in run.steps
        if any(d.decision_type == "tool_call" for d in step.decisions)
    )
    assert len(tool_call_step.observations) == 1
    obs = tool_call_step.observations[0]
    assert obs.content["kind"] == "tool_use_result"
    assert obs.content["tool_use_id"] == "tu_1"


def test_result_message_drives_outcome() -> None:
    run = run_from_messages(_pass_messages())
    assert run.outcome.kind == "binary"
    assert run.outcome.value is True
    assert run.outcome.verifier == "claude_agent_sdk_result_message"


def test_is_error_true_produces_fail_outcome() -> None:
    msgs = _pass_messages()
    msgs[-1]["is_error"] = True
    msgs[-1]["subtype"] = "error"

    run = run_from_messages(msgs)

    assert run.outcome.value is False


def test_is_error_string_false_is_rejected() -> None:
    msgs = _pass_messages()
    msgs[-1]["is_error"] = "false"

    with pytest.raises(IngestError, match="JSON boolean"):
        run_from_messages(msgs)


def test_terminal_step_is_highest_index_and_marks_termination() -> None:
    run = run_from_messages(_pass_messages())

    last_step = run.steps[-1]
    assert last_step.step_index == max(s.step_index for s in run.steps)
    assert len(last_step.decisions) == 1
    assert last_step.decisions[0].decision_type == "termination"
    assert last_step.decisions[0].chosen_action == "success"


def test_metadata_carries_cost_and_usage() -> None:
    run = run_from_messages(_pass_messages())
    extra = run.metadata.extra
    assert extra["total_cost_usd"] == 0.01
    assert extra["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert extra["source_format"] == "claude-agent-sdk"


def test_fixture_ingest_writes_two_runs(tmp_path: Path) -> None:
    receipt = ingest_claude_agent_sdk(FIXTURE, tmp_path)

    assert receipt.source_format == "claude-agent-sdk"
    assert receipt.generated_count == 2
    assert any("randomization" in w for w in receipt.warnings)

    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert "ingest-receipt.json" in written
    assert "sess-pass-001.json" in written
    assert "sess-fail-002.json" in written

    fail_run = Run.model_validate_json((tmp_path / "sess-fail-002.json").read_text())
    assert fail_run.outcome.value is False
    pass_run = Run.model_validate_json((tmp_path / "sess-pass-001.json").read_text())
    assert pass_run.outcome.value is True


def test_decisions_have_no_randomization_metadata() -> None:
    run = run_from_messages(_pass_messages())

    for step in run.steps:
        for decision in step.decisions:
            assert decision.policy is None
            assert decision.policy_params is None
            assert decision.valid_actions is None
            assert decision.propensity is None
            assert decision.context_features is None


def test_empty_message_stream_is_rejected() -> None:
    with pytest.raises(IngestError, match="empty"):
        run_from_messages([])


def test_missing_session_id_is_rejected() -> None:
    with pytest.raises(IngestError, match="session_id"):
        run_from_messages(
            [
                {
                    "__type__": "AssistantMessage",
                    "model": "claude-sonnet-4-6",
                    "content": [{"__type__": "TextBlock", "text": "hi"}],
                },
                {
                    "__type__": "ResultMessage",
                    "subtype": "success",
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "is_error": False,
                    "num_turns": 1,
                },
            ]
        )


def test_receipt_warnings_appear_in_written_file(tmp_path: Path) -> None:
    ingest_claude_agent_sdk(FIXTURE, tmp_path)
    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert any("randomization" in w for w in receipt["warnings"])
