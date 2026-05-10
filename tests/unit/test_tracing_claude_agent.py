"""Coverage for `counterfact.tracing.ClaudeAgentTracer`."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from counterfact.adapters.claude_agent_sdk import (
    ingest_claude_agent_sdk,
    run_from_messages,
)
from counterfact.schema import Run
from counterfact.tracing import ClaudeAgentTracer


def _pass_messages() -> list[dict]:
    return [
        {
            "__type__": "AssistantMessage",
            "model": "claude-sonnet-4-6",
            "session_id": "sess-live-001",
            "content": [
                {
                    "__type__": "ToolUseBlock",
                    "id": "tu_x",
                    "name": "list_dir",
                    "input": {"path": "."},
                }
            ],
        },
        {
            "__type__": "UserMessage",
            "session_id": "sess-live-001",
            "parent_tool_use_id": "tu_x",
            "content": [],
            "tool_use_result": {"entries": ["a.txt"]},
        },
        {
            "__type__": "ResultMessage",
            "subtype": "success",
            "session_id": "sess-live-001",
            "duration_ms": 50,
            "duration_api_ms": 30,
            "is_error": False,
            "num_turns": 1,
            "stop_reason": "end_turn",
        },
    ]


def _run_async(coro):
    return asyncio.run(coro)


def test_tracer_writes_run_on_exit_with_dict_inputs(tmp_path: Path) -> None:
    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=tmp_path) as tracer:
            for msg in _pass_messages():
                tracer.observe(msg)

    _run_async(go())

    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert "sess-live-001.json" in written
    run = Run.model_validate_json((tmp_path / "sess-live-001.json").read_text())
    assert run.outcome.value is True


def test_tracer_output_matches_offline_ingest(tmp_path: Path) -> None:
    live_dir = tmp_path / "live"

    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=live_dir) as tracer:
            for msg in _pass_messages():
                tracer.observe(msg)

    _run_async(go())
    live_run = Run.model_validate_json((live_dir / "sess-live-001.json").read_text())
    offline_run = run_from_messages(_pass_messages())
    assert live_run.model_dump() == offline_run.model_dump()


def test_tracer_accepts_dataclass_messages(tmp_path: Path) -> None:
    @dataclass
    class TextBlock:
        text: str

    @dataclass
    class AssistantMessage:
        model: str
        session_id: str
        content: list[object]

    @dataclass
    class ResultMessage:
        subtype: str
        session_id: str
        duration_ms: int
        duration_api_ms: int
        is_error: bool
        num_turns: int

    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=tmp_path) as tracer:
            tracer.observe(
                AssistantMessage(
                    model="claude-sonnet-4-6",
                    session_id="sess-dc-1",
                    content=[TextBlock(text="hi")],
                )
            )
            tracer.observe(
                ResultMessage(
                    subtype="success",
                    session_id="sess-dc-1",
                    duration_ms=10,
                    duration_api_ms=5,
                    is_error=False,
                    num_turns=1,
                )
            )

    _run_async(go())

    run = Run.model_validate_json((tmp_path / "sess-dc-1.json").read_text())
    model_calls = [
        d for step in run.steps for d in step.decisions if d.decision_type == "model_call"
    ]
    assert len(model_calls) == 1
    assert model_calls[0].chosen_action == "claude-sonnet-4-6"


def test_tracer_writes_receipt_with_randomization_warning(tmp_path: Path) -> None:
    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=tmp_path) as tracer:
            for msg in _pass_messages():
                tracer.observe(msg)

    _run_async(go())

    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert any("randomization" in w for w in receipt["warnings"])


def test_tracer_with_no_messages_writes_nothing(tmp_path: Path) -> None:
    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=tmp_path):
            pass

    _run_async(go())
    assert list(tmp_path.glob("*.json")) == []


def test_tracer_rejects_non_serializable_object_with_clear_error(tmp_path: Path) -> None:
    class WeirdThing:
        pass

    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=tmp_path) as tracer:
            tracer.observe(WeirdThing())

    with pytest.raises((ImportError, TypeError)) as excinfo:
        _run_async(go())

    assert "claude_agent_sdk" in str(excinfo.value) or "serialize" in str(excinfo.value)


def _pass_messages_with_session(session_id: str) -> list[dict]:
    msgs = _pass_messages()
    for m in msgs:
        if "session_id" in m:
            m["session_id"] = session_id
        if m.get("__type__") == "UserMessage" and "parent_tool_use_id" in m:
            # session_id already covered above; nothing else to rewrite
            pass
    return msgs


def test_tracer_receipt_count_accumulates_across_sessions(tmp_path: Path) -> None:
    async def go() -> None:
        for i in range(5):
            sid = f"sess-cum-{i:03d}"
            async with ClaudeAgentTracer(output_dir=tmp_path) as tracer:
                for msg in _pass_messages_with_session(sid):
                    tracer.observe(msg)

    _run_async(go())

    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert receipt["generated_count"] == 5


def test_tracer_receipt_count_extends_existing_receipt(tmp_path: Path) -> None:
    (tmp_path / "ingest-receipt.json").write_text(
        json.dumps(
            {
                "source_format": "claude-agent-sdk",
                "source_file": "<live-tracer>",
                "mapping_file": "",
                "generated_count": 3,
                "warnings": [],
                "dropped_fields": [],
                "validation_errors": [],
            }
        )
    )

    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=tmp_path) as tracer:
            for msg in _pass_messages_with_session("sess-resume-001"):
                tracer.observe(msg)

    _run_async(go())

    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert receipt["generated_count"] == 4


def test_tracer_offline_round_trips_to_ingest_cli(tmp_path: Path) -> None:
    live_dir = tmp_path / "live"

    async def go() -> None:
        async with ClaudeAgentTracer(output_dir=live_dir) as tracer:
            for msg in _pass_messages():
                tracer.observe(msg)

    _run_async(go())
    live_text = (live_dir / "sess-live-001.json").read_text()

    jsonl_path = tmp_path / "src.jsonl"
    jsonl_path.write_text(json.dumps({"messages": _pass_messages()}) + "\n")
    offline_dir = tmp_path / "offline"
    ingest_claude_agent_sdk(jsonl_path, offline_dir)
    offline_text = (offline_dir / "sess-live-001.json").read_text()

    assert live_text == offline_text
