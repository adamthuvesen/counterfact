"""Adapter shape-drift hardening — unknown blocks/spans abort with a precise error."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact.adapters._common import IngestError, IngestReceipt, write_corpus
from counterfact.adapters.claude_agent_sdk import (
    ingest_claude_agent_sdk,
    run_from_messages,
)
from counterfact.schema import Outcome, Run


def _empty_run(run_id: str) -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        steps=[],
        outcome=Outcome(kind="binary", value=True, verifier="test"),
    )


def test_claude_unknown_block_aborts() -> None:
    messages = [
        {
            "__type__": "AssistantMessage",
            "model": "claude-sonnet-4-6",
            "session_id": "sess-1",
            "content": [{"__type__": "FutureWidgetBlock", "data": "???"}],
        },
        {
            "__type__": "ResultMessage",
            "subtype": "success",
            "session_id": "sess-1",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": False,
            "num_turns": 1,
        },
    ]

    with pytest.raises(IngestError) as excinfo:
        run_from_messages(messages)

    assert "FutureWidgetBlock" in str(excinfo.value)


def test_write_corpus_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    receipt = IngestReceipt(source_format="test", source_file="source", generated_count=1)

    with pytest.raises(IngestError, match="unsafe run_id"):
        write_corpus([_empty_run("../escaped")], tmp_path / "out", receipt)

    assert not (tmp_path / "escaped.json").exists()
    assert not (tmp_path / "out").exists()


def test_write_corpus_rejects_duplicate_run_ids_before_writing(tmp_path: Path) -> None:
    receipt = IngestReceipt(source_format="test", source_file="source", generated_count=2)
    out_dir = tmp_path / "out"

    with pytest.raises(IngestError, match="duplicate run_id"):
        write_corpus([_empty_run("same"), _empty_run("same")], out_dir, receipt)

    assert not out_dir.exists()


def test_claude_unknown_message_type_aborts() -> None:
    messages = [
        {"__type__": "MysteryMessage", "session_id": "sess-1"},
        {
            "__type__": "ResultMessage",
            "subtype": "success",
            "session_id": "sess-1",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": False,
            "num_turns": 0,
        },
    ]

    with pytest.raises(IngestError) as excinfo:
        run_from_messages(messages)

    assert "MysteryMessage" in str(excinfo.value)


def test_claude_unknown_block_jsonl_ingest_writes_no_partial_corpus(tmp_path: Path) -> None:
    bad_line = json.dumps(
        {
            "messages": [
                {
                    "__type__": "AssistantMessage",
                    "model": "claude-sonnet-4-6",
                    "session_id": "sess-bad",
                    "content": [{"__type__": "FutureWidgetBlock", "data": "x"}],
                },
                {
                    "__type__": "ResultMessage",
                    "subtype": "success",
                    "session_id": "sess-bad",
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "is_error": False,
                    "num_turns": 1,
                },
            ]
        }
    )
    source = tmp_path / "bad.jsonl"
    source.write_text(bad_line + "\n")

    out_dir = tmp_path / "out"
    with pytest.raises(IngestError) as excinfo:
        ingest_claude_agent_sdk(source, out_dir)

    assert "FutureWidgetBlock" in str(excinfo.value)
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []


def test_openai_unknown_span_type_aborts(tmp_path: Path) -> None:
    bad_trace = {
        "trace_id": "trace-bad",
        "spans": [
            {
                "object": "trace.span",
                "id": "span-root",
                "trace_id": "trace-bad",
                "parent_id": None,
                "started_at": "2026-05-06T20:00:00.000Z",
                "ended_at": "2026-05-06T20:00:00.500Z",
                "span_data": {
                    "type": "agent",
                    "name": "x",
                    "handoffs": [],
                    "tools": [],
                    "output_type": "string",
                },
                "error": None,
            },
            {
                "object": "trace.span",
                "id": "span-bogus",
                "trace_id": "trace-bad",
                "parent_id": "span-root",
                "started_at": "2026-05-06T20:00:00.100Z",
                "ended_at": "2026-05-06T20:00:00.300Z",
                "span_data": {"type": "future_widget", "data": {}},
                "error": None,
            },
        ],
    }
    src = tmp_path / "bad.json"
    src.write_text(json.dumps(bad_trace))
    out_dir = tmp_path / "out"

    from counterfact.adapters.openai_agents import ingest_openai_agents

    with pytest.raises(IngestError) as excinfo:
        ingest_openai_agents(src, out_dir, outcome=True)

    assert "future_widget" in str(excinfo.value)
    assert "span-bogus" in str(excinfo.value)
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []
