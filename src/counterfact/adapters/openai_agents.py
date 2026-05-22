"""OpenAI Agents SDK → counterfact native Run adapter.

Maps an OpenAI Agents SDK trace export — a JSON document with a flat array
of `Span.export()` dicts plus a top-level `trace_id` — onto the native `Run`
schema. Mapping is documented in design D4 of the `add-agent-sdk-adapters`
change.

Inputs are JSON-serialized dicts. This module never imports the
`openai-agents` SDK itself; the live processor at
`counterfact.tracing.CounterfactSpanProcessor` handles SDK objects and
reuses `run_from_trace` here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from counterfact.adapters._common import (
    IngestError,
    IngestReceipt,
    randomization_warning,
    strict_bool,
    write_corpus,
)
from counterfact.schema import Decision, Metadata, Observation, Outcome, Run, Step
from counterfact.schema.models import SCHEMA_VERSION

SOURCE_FORMAT = "openai-agents"

_OUTCOME_MARKER_NAME = "counterfact.outcome"

_DECISION_SPAN_TYPES = frozenset({"function", "generation", "handoff"})
_OBSERVATION_SPAN_TYPES = frozenset(
    {"guardrail", "custom", "mcp_list_tools", "transcription", "speech"}
)
_CONTAINER_SPAN_TYPES = frozenset({"agent", "response"})
_KNOWN_SPAN_TYPES = _DECISION_SPAN_TYPES | _OBSERVATION_SPAN_TYPES | _CONTAINER_SPAN_TYPES


def _span_type(span: dict[str, Any]) -> str:
    span_data = span.get("span_data")
    if not isinstance(span_data, dict):
        raise IngestError(
            f"span {span.get('id')!r} has invalid span_data; expected object, "
            f"got {type(span_data).__name__}"
        )
    stype = span_data.get("type")
    if stype is None:
        raise IngestError(
            f"span {span.get('id')!r} has no span_data.type; cannot route to a decision/observation"
        )
    return str(stype)


def _span_id(span: dict[str, Any]) -> str:
    span_id = span.get("id")
    if not isinstance(span_id, str) or not span_id:
        raise IngestError(f"openai-agents span is missing a non-empty string id; got {span_id!r}")
    return span_id


def _parent_id(span: dict[str, Any]) -> str | None:
    parent = span.get("parent_id")
    if parent is not None and not isinstance(parent, str):
        raise IngestError(
            f"span_id={_span_id(span)!r} has invalid parent_id {parent!r}; expected string or null"
        )
    return parent


def _spans_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    raw_spans = trace.get("spans")
    if not isinstance(raw_spans, list):
        raise IngestError(
            "openai-agents trace field 'spans' must be a JSON array; "
            f"got {type(raw_spans).__name__}"
        )

    spans: list[dict[str, Any]] = []
    for idx, raw_span in enumerate(raw_spans, start=1):
        if not isinstance(raw_span, dict):
            raise IngestError(
                "openai-agents trace field 'spans' must contain JSON objects; "
                f"span {idx} is {type(raw_span).__name__}"
            )
        spans.append(raw_span)
    return spans


def _trace_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise IngestError(
            f"openai-agents trace is missing a non-empty string trace_id; got {value!r}"
        )
    return value


def _validate_span_types(spans: list[dict[str, Any]]) -> None:
    for span in spans:
        stype = _span_type(span)
        if stype not in _KNOWN_SPAN_TYPES:
            raise IngestError(
                f"unknown span_data.type {stype!r} on span_id={span.get('id')!r}; "
                f"supported: {sorted(_KNOWN_SPAN_TYPES)}"
            )


def _topological_order(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DFS the parent_id tree from the root, sorting siblings by started_at then span_id."""

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        span_id = _span_id(span)
        if span_id in seen_ids:
            duplicate_ids.add(span_id)
        seen_ids.add(span_id)
        by_parent.setdefault(_parent_id(span), []).append(span)
    if duplicate_ids:
        raise IngestError(
            "openai-agents trace has duplicate span id(s): " + ", ".join(sorted(duplicate_ids))
        )
    for siblings in by_parent.values():
        siblings.sort(key=lambda s: (s.get("started_at") or "", _span_id(s)))

    roots = by_parent.get(None, [])
    if len(roots) != 1:
        raise IngestError(
            "openai-agents trace must have exactly one root span "
            f"(parent_id is null); found {len(roots)}"
        )

    ordered: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = [roots[0]]
    while stack:
        span = stack.pop()
        ordered.append(span)
        children = list(by_parent.get(_span_id(span), []))
        # Reverse so DFS pops in sorted order.
        for child in reversed(children):
            stack.append(child)
    visited = {_span_id(span) for span in ordered}
    unreachable = sorted(seen_ids - visited)
    if unreachable:
        raise IngestError(
            "openai-agents trace contains span(s) unreachable from the root: "
            + ", ".join(unreachable)
        )
    return ordered


def _decision_from_span(span: dict[str, Any], *, run_id: str) -> Decision | None:
    stype = _span_type(span)
    sdata = span["span_data"]
    span_id = span["id"]
    decision_id = f"d-{run_id}-{span_id}"
    if stype == "function":
        return Decision(
            decision_id=decision_id,
            decision_type="tool_call",
            chosen_action=str(sdata.get("name", "")),
            metadata={
                "input": sdata.get("input"),
                "output": sdata.get("output"),
                "mcp_data": sdata.get("mcp_data"),
            },
        )
    if stype == "generation":
        model = sdata.get("model")
        if model is None:
            raise IngestError(
                f"generation span_id={span_id!r} has no model; cannot derive model_call"
            )
        return Decision(
            decision_id=decision_id,
            decision_type="model_call",
            chosen_action=str(model),
            metadata={
                "usage": sdata.get("usage"),
                "model_config": sdata.get("model_config"),
            },
        )
    if stype == "handoff":
        return Decision(
            decision_id=decision_id,
            decision_type="plan_step",
            chosen_action=f"handoff:{sdata.get('to_agent', '')}",
            metadata={"from_agent": sdata.get("from_agent")},
        )
    return None


def _observation_from_span(span: dict[str, Any], *, run_id: str) -> Observation | None:
    stype = _span_type(span)
    sdata = span["span_data"]
    span_id = span["id"]
    obs_id = f"o-{run_id}-{span_id}"
    if stype == "guardrail":
        return Observation(
            observation_id=obs_id,
            content={
                "kind": "guardrail",
                "name": sdata.get("name"),
                "triggered": sdata.get("triggered"),
            },
        )
    if stype == "custom":
        return Observation(
            observation_id=obs_id,
            content={
                "kind": "custom",
                "name": sdata.get("name"),
                "data": sdata.get("data"),
            },
        )
    if stype == "mcp_list_tools":
        return Observation(
            observation_id=obs_id,
            content={
                "kind": "mcp_list_tools",
                "server": sdata.get("server"),
                "result": sdata.get("result"),
            },
        )
    if stype in {"transcription", "speech"}:
        return Observation(
            observation_id=obs_id,
            content={"kind": stype, **{k: v for k, v in sdata.items() if k != "type"}},
        )
    return None


def _outcome_marker_value(spans: list[dict[str, Any]]) -> bool | None:
    for span in spans:
        if _span_type(span) != "custom":
            continue
        sdata = span["span_data"]
        if sdata.get("name") != _OUTCOME_MARKER_NAME:
            continue
        data = sdata.get("data") or {}
        if "value" in data:
            return strict_bool(
                data["value"],
                field_name=f"{_OUTCOME_MARKER_NAME}.data.value",
            )
    return None


def _root_span(spans: list[dict[str, Any]]) -> dict[str, Any]:
    roots = [s for s in spans if s.get("parent_id") is None]
    return roots[0]


def _outcome_for_trace(spans: list[dict[str, Any]], *, override: bool | str | None) -> Outcome:
    """Resolve outcome per design D4.

    Order: explicit override → marker span → root error → raise.
    """

    if override is True:
        return Outcome(kind="binary", value=True, verifier="caller_supplied_outcome")
    if override is False:
        return Outcome(kind="binary", value=False, verifier="caller_supplied_outcome")
    if isinstance(override, str):
        # Caller-supplied verifier label without a value implies pass; the CLI
        # surface only sets `override` to a string when the user typed
        # `--outcome <verifier>` for a custom verifier and did not specify
        # pass/fail. We refuse rather than guess.
        raise IngestError(
            f"outcome override {override!r} is a verifier label without a value; "
            "use --outcome pass or --outcome fail to set the binary outcome"
        )

    marker = _outcome_marker_value(spans)
    if marker is not None:
        return Outcome(
            kind="binary",
            value=marker,
            verifier="counterfact_outcome_marker",
        )

    root = _root_span(spans)
    if root.get("error"):
        return Outcome(
            kind="binary",
            value=False,
            verifier="openai_agents_root_span_error",
            metadata={"error": root.get("error")},
        )

    raise IngestError(
        "openai-agents trace has no explicit outcome: pass --outcome pass|fail, "
        f"include a CustomSpanData named {_OUTCOME_MARKER_NAME!r}, or surface "
        "an error on the root span"
    )


def run_from_trace(trace: dict[str, Any], *, outcome: bool | str | None = None) -> Run:
    """Convert one OpenAI Agents SDK trace export into a native Run.

    `trace` is the JSON document produced by the SDK's exporter. Two shapes are
    accepted: a top-level dict with `trace_id` and `spans`, or a top-level dict
    that is itself one span (treated as a single-trace export).
    """

    if not isinstance(trace, dict):
        raise IngestError(f"openai-agents trace must be a JSON object; got {type(trace).__name__}")
    spans: list[dict[str, Any]] = []
    trace_id: str | None = None
    if "spans" in trace:
        spans = _spans_from_trace(trace)
        trace_id = _trace_id(trace.get("trace_id") or trace.get("id"))
    elif trace.get("object") == "trace.span":
        # Degenerate single-span export — wrap as a one-span trace.
        spans = [trace]
        trace_id = _trace_id(trace.get("trace_id"))
    else:
        raise IngestError(
            "openai-agents trace must contain a 'spans' array or be a single 'trace.span' object"
        )
    if not spans:
        raise IngestError(f"openai-agents trace {trace_id!r} has no spans")

    _validate_span_types(spans)
    ordered = _topological_order(spans)

    steps: list[Step] = []
    last_decision_step: Step | None = None
    pending_observations: list[Observation] = []
    next_step_index = 0

    for span in ordered:
        if span.get("parent_id") is None:
            # Root span has no decision in itself. Its children carry the work.
            continue
        decision = _decision_from_span(span, run_id=trace_id)
        if decision is not None:
            step = Step(
                step_index=next_step_index,
                decisions=[decision],
                observations=pending_observations,
            )
            pending_observations = []
            steps.append(step)
            last_decision_step = step
            next_step_index += 1
            continue
        observation = _observation_from_span(span, run_id=trace_id)
        if observation is not None:
            if last_decision_step is not None:
                last_decision_step.observations.append(observation)
            else:
                pending_observations.append(observation)
            continue
        # Container spans (agent, response) — nothing to emit.

    if pending_observations:
        steps.append(
            Step(
                step_index=next_step_index,
                observations=pending_observations,
            )
        )
        next_step_index += 1

    resolved_outcome = _outcome_for_trace(spans, override=outcome)

    metadata_extra: dict[str, Any] = {"source_format": SOURCE_FORMAT}
    root = _root_span(spans)
    root_data = root.get("span_data") or {}
    if "name" in root_data:
        metadata_extra["root_name"] = root_data["name"]

    return Run(
        schema_version=SCHEMA_VERSION,
        run_id=str(trace_id),
        steps=steps,
        outcome=resolved_outcome,
        metadata=Metadata(agent_name="openai-agents-sdk", extra=metadata_extra),
    )


def ingest_openai_agents(
    source_path: Path,
    output_dir: Path,
    *,
    outcome: bool | str | None = None,
) -> IngestReceipt:
    """Read a JSON file and write one native Run per trace.

    The source file may be:
      - one JSON object with a `spans` array (single trace), or
      - a JSON array of such objects (corpus of traces).
    """

    payload = json.loads(source_path.read_text())
    traces: list[dict[str, Any]]
    if isinstance(payload, list):
        traces = payload
    elif isinstance(payload, dict):
        traces = [payload]
    else:
        raise IngestError(
            f"{source_path}: expected a JSON object or array; got {type(payload).__name__}"
        )

    runs: list[Run] = []
    warnings = [randomization_warning(SOURCE_FORMAT)]
    for idx, trace in enumerate(traces, start=1):
        try:
            run = run_from_trace(trace, outcome=outcome)
        except (IngestError, ValidationError, ValueError) as exc:
            raise IngestError(f"{source_path}: trace {idx}: {exc}") from exc
        runs.append(run)

    receipt = IngestReceipt(
        source_format=SOURCE_FORMAT,
        source_file=str(source_path),
        generated_count=len(runs),
        warnings=warnings,
    )
    write_corpus(runs, output_dir, receipt)
    return receipt


__all__ = [
    "SOURCE_FORMAT",
    "ingest_openai_agents",
    "run_from_trace",
]
