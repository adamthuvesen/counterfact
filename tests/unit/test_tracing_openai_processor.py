"""Coverage for `counterfact.tracing.CounterfactSpanProcessor`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from counterfact.adapters.openai_agents import run_from_trace
from counterfact.schema import Run
from counterfact.tracing import CounterfactSpanProcessor


@dataclass
class _FakeSpan:
    """Mimics the OpenAI Agents SDK Span.export() shape via duck-typed export()."""

    payload: dict[str, Any]

    def export(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass
class _FakeTrace:
    trace_id: str
    extras: dict[str, Any] = field(default_factory=dict)

    def export(self) -> dict[str, Any]:
        return {"id": self.trace_id, **self.extras}


def _spans_for_minimal(trace_id: str) -> list[_FakeSpan]:
    base = "2026-05-06T19:00:"
    return [
        _FakeSpan(
            {
                "object": "trace.span",
                "id": "span-root",
                "trace_id": trace_id,
                "parent_id": None,
                "started_at": f"{base}00.000Z",
                "ended_at": f"{base}01.500Z",
                "span_data": {
                    "type": "agent",
                    "name": "research_agent",
                    "handoffs": [],
                    "tools": ["web_search"],
                    "output_type": "string",
                },
                "error": None,
            }
        ),
        _FakeSpan(
            {
                "object": "trace.span",
                "id": "span-fn",
                "trace_id": trace_id,
                "parent_id": "span-root",
                "started_at": f"{base}00.100Z",
                "ended_at": f"{base}00.500Z",
                "span_data": {
                    "type": "function",
                    "name": "web_search",
                    "input": {"query": "x"},
                    "output": "y",
                    "mcp_data": None,
                },
                "error": None,
            }
        ),
        _FakeSpan(
            {
                "object": "trace.span",
                "id": "span-gen",
                "trace_id": trace_id,
                "parent_id": "span-root",
                "started_at": f"{base}00.600Z",
                "ended_at": f"{base}01.200Z",
                "span_data": {
                    "type": "generation",
                    "model": "gpt-5-mini",
                    "model_config": {"temperature": 0},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
                "error": None,
            }
        ),
    ]


def test_processor_writes_run_byte_equivalent_to_offline(tmp_path: Path) -> None:
    trace_id = "trace-proc-001"
    spans = _spans_for_minimal(trace_id)

    processor = CounterfactSpanProcessor(
        output_dir=tmp_path,
        outcome_provider=lambda payload: True,
    )
    processor.on_trace_start(_FakeTrace(trace_id=trace_id))
    for span in spans:
        processor.on_span_end(span)
    processor.on_trace_end(_FakeTrace(trace_id=trace_id))

    live_run = Run.model_validate_json((tmp_path / f"{trace_id}.json").read_text())
    offline_run = run_from_trace(
        {"trace_id": trace_id, "spans": [s.export() for s in spans]},
        outcome=True,
    )
    assert live_run.model_dump() == offline_run.model_dump()


def test_processor_uses_outcome_provider_callback(tmp_path: Path) -> None:
    trace_id = "trace-cb-001"
    captured_payloads: list[dict[str, Any]] = []

    def provider(trace_payload: dict[str, Any]) -> bool:
        captured_payloads.append(trace_payload)
        return False

    processor = CounterfactSpanProcessor(
        output_dir=tmp_path,
        outcome_provider=provider,
    )
    processor.on_trace_start(_FakeTrace(trace_id=trace_id))
    for span in _spans_for_minimal(trace_id):
        processor.on_span_end(span)
    processor.on_trace_end(_FakeTrace(trace_id=trace_id))

    assert captured_payloads, "outcome_provider was never called"
    run = Run.model_validate_json((tmp_path / f"{trace_id}.json").read_text())
    assert run.outcome.value is False


def test_processor_falls_back_to_unknown_when_no_outcome_derivable(
    tmp_path: Path,
) -> None:
    trace_id = "trace-unknown-001"
    processor = CounterfactSpanProcessor(
        output_dir=tmp_path,
        outcome_verifier="my_evaluator",
    )
    processor.on_trace_start(_FakeTrace(trace_id=trace_id))
    for span in _spans_for_minimal(trace_id):
        processor.on_span_end(span)
    processor.on_trace_end(_FakeTrace(trace_id=trace_id))

    run = Run.model_validate_json((tmp_path / f"{trace_id}.json").read_text())
    assert run.outcome.kind == "categorical"
    assert run.outcome.value == "unknown"
    assert run.outcome.verifier == "my_evaluator"

    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert any("no binary outcome" in w for w in receipt["warnings"])


def test_processor_with_no_spans_writes_nothing(tmp_path: Path) -> None:
    trace_id = "trace-empty"
    processor = CounterfactSpanProcessor(output_dir=tmp_path)
    processor.on_trace_start(_FakeTrace(trace_id=trace_id))
    processor.on_trace_end(_FakeTrace(trace_id=trace_id))

    assert list(tmp_path.glob("*.json")) == []


def test_processor_receipt_count_accumulates_across_traces(tmp_path: Path) -> None:
    processor = CounterfactSpanProcessor(
        output_dir=tmp_path,
        outcome_provider=lambda payload: True,
    )
    for i in range(5):
        trace_id = f"trace-cum-{i:03d}"
        processor.on_trace_start(_FakeTrace(trace_id=trace_id))
        for span in _spans_for_minimal(trace_id):
            processor.on_span_end(span)
        processor.on_trace_end(_FakeTrace(trace_id=trace_id))

    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert receipt["generated_count"] == 5


def test_processor_receipt_count_extends_existing_receipt(tmp_path: Path) -> None:
    # Pre-seed an existing receipt to simulate a resumed session.
    (tmp_path / "ingest-receipt.json").write_text(
        json.dumps(
            {
                "source_format": "openai-agents",
                "source_file": "<live-processor>",
                "mapping_file": "",
                "generated_count": 7,
                "warnings": [],
                "dropped_fields": [],
                "validation_errors": [],
            }
        )
    )

    processor = CounterfactSpanProcessor(
        output_dir=tmp_path,
        outcome_provider=lambda payload: True,
    )
    trace_id = "trace-resume-001"
    processor.on_trace_start(_FakeTrace(trace_id=trace_id))
    for span in _spans_for_minimal(trace_id):
        processor.on_span_end(span)
    processor.on_trace_end(_FakeTrace(trace_id=trace_id))

    receipt = json.loads((tmp_path / "ingest-receipt.json").read_text())
    assert receipt["generated_count"] == 8


def test_processor_shutdown_clears_buffers(tmp_path: Path) -> None:
    processor = CounterfactSpanProcessor(output_dir=tmp_path)
    processor.on_span_end(
        _FakeSpan(
            {
                "object": "trace.span",
                "id": "x",
                "trace_id": "t1",
                "parent_id": None,
                "started_at": "2026-05-06T19:00:00.000Z",
                "ended_at": "2026-05-06T19:00:00.100Z",
                "span_data": {
                    "type": "agent",
                    "name": "x",
                    "handoffs": [],
                    "tools": [],
                    "output_type": "string",
                },
                "error": None,
            }
        )
    )
    processor.shutdown()
    assert processor._spans_by_trace == {}
