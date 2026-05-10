"""Claude Agent SDK → counterfact native Run adapter.

Maps the message-stream shape emitted by `claude_agent_sdk.query()` onto the
native `Run` schema. The mapping is documented in design D3 and D7 of the
`add-agent-sdk-adapters` change.

Inputs are JSON-serialized dicts — this module never imports `claude_agent_sdk`
itself. The live tracing wrapper at `counterfact.tracing.ClaudeAgentTracer`
handles SDK objects and reuses `run_from_messages` here.
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

SOURCE_FORMAT = "claude-agent-sdk"

_KNOWN_BLOCK_TYPES = frozenset({"TextBlock", "ToolUseBlock", "ToolResultBlock", "ThinkingBlock"})
_KNOWN_MESSAGE_TYPES = frozenset(
    {
        "AssistantMessage",
        "UserMessage",
        "SystemMessage",
        "ResultMessage",
        "StreamEvent",
        "RateLimitEvent",
    }
)


def _block_type(block: dict[str, Any]) -> str:
    btype = block.get("__type__") or block.get("type")
    if btype is None:
        raise IngestError(
            "content block missing '__type__' / 'type' tag; "
            f"cannot determine kind for block={block!r}"
        )
    return str(btype)


def _message_type(message: dict[str, Any]) -> str:
    mtype = message.get("__type__") or message.get("type")
    if mtype is None:
        raise IngestError(f"message missing '__type__' tag: {message!r}")
    return str(mtype)


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if content is None:
        return []
    if isinstance(content, str):
        return [{"__type__": "TextBlock", "text": content}]
    if not isinstance(content, list):
        raise IngestError(f"unexpected message.content type {type(content).__name__}: {content!r}")
    blocks: list[dict[str, Any]] = []
    for raw in content:
        if not isinstance(raw, dict):
            raise IngestError(f"unexpected content block (not dict): {raw!r}")
        blocks.append(raw)
    return blocks


def _tool_use_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in blocks if _block_type(b) == "ToolUseBlock"]


def _has_only_known_blocks(blocks: list[dict[str, Any]], *, source_index: int) -> None:
    for block in blocks:
        btype = _block_type(block)
        if btype not in _KNOWN_BLOCK_TYPES:
            raise IngestError(
                f"record {source_index}: unknown content block type {btype!r}; "
                f"supported: {sorted(_KNOWN_BLOCK_TYPES)}"
            )


def _text_and_thinking(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    text_parts = [b.get("text", "") for b in blocks if _block_type(b) == "TextBlock"]
    if text_parts:
        out["text"] = "\n".join(text_parts)
    thinking_parts = [b.get("thinking", "") for b in blocks if _block_type(b) == "ThinkingBlock"]
    if thinking_parts:
        out["thinking"] = "\n".join(thinking_parts)
    return out


def _result_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in reversed(messages):
        if _message_type(msg) == "ResultMessage":
            return msg
    return None


def _session_id(messages: list[dict[str, Any]]) -> str:
    result = _result_message(messages)
    if result and result.get("session_id"):
        return str(result["session_id"])
    for msg in messages:
        sid = msg.get("session_id")
        if sid:
            return str(sid)
    raise IngestError(
        "claude-agent-sdk message stream has no session_id; "
        "expected ResultMessage.session_id or a session_id field on any message"
    )


def _system_observation(message: dict[str, Any], obs_index: int, run_id: str) -> Observation:
    return Observation(
        observation_id=f"o-{run_id}-sys-{obs_index}",
        content={
            "kind": "system",
            "subtype": message.get("subtype"),
            "data": message.get("data", {}),
        },
    )


def _user_observation(message: dict[str, Any], obs_index: int, run_id: str) -> Observation:
    blocks = _content_blocks(message)
    _has_only_known_blocks(blocks, source_index=obs_index)
    if message.get("tool_use_result") is not None:
        return Observation(
            observation_id=f"o-{run_id}-tr-{obs_index}",
            content={
                "kind": "tool_use_result",
                "tool_use_id": message.get("parent_tool_use_id"),
                "tool_use_result": message["tool_use_result"],
            },
        )
    tool_result_blocks = [b for b in blocks if _block_type(b) == "ToolResultBlock"]
    if tool_result_blocks:
        block = tool_result_blocks[0]
        return Observation(
            observation_id=f"o-{run_id}-tr-{obs_index}",
            content={
                "kind": "tool_use_result",
                "tool_use_id": block.get("tool_use_id"),
                "tool_use_result": {
                    "content": block.get("content"),
                    "is_error": block.get("is_error"),
                },
            },
        )
    text_parts = [b.get("text", "") for b in blocks if _block_type(b) == "TextBlock"]
    return Observation(
        observation_id=f"o-{run_id}-user-{obs_index}",
        content={"kind": "user_text", "text": "\n".join(text_parts)},
    )


def _assistant_decisions(
    message: dict[str, Any],
    *,
    run_id: str,
    msg_index: int,
) -> list[Decision]:
    blocks = _content_blocks(message)
    _has_only_known_blocks(blocks, source_index=msg_index)
    tool_uses = _tool_use_blocks(blocks)
    decisions: list[Decision] = []
    if tool_uses:
        for tu in tool_uses:
            tu_id = tu.get("id") or f"tu-{msg_index}-{len(decisions)}"
            decisions.append(
                Decision(
                    decision_id=f"d-{run_id}-tu-{tu_id}",
                    decision_type="tool_call",
                    chosen_action=str(tu.get("name", "")),
                    metadata={"input": tu.get("input", {})},
                )
            )
        return decisions
    aux = _text_and_thinking(blocks)
    model = message.get("model")
    if model is None:
        raise IngestError(
            f"AssistantMessage at index {msg_index} has no model field "
            "and no tool_use blocks; cannot derive model_call decision"
        )
    metadata: dict[str, Any] = {}
    if aux:
        metadata.update(aux)
    if message.get("usage") is not None:
        metadata["usage"] = message["usage"]
    decision_id = f"d-{run_id}-msg-{message.get('message_id') or msg_index}"
    decisions.append(
        Decision(
            decision_id=decision_id,
            decision_type="model_call",
            chosen_action=str(model),
            metadata=metadata,
        )
    )
    return decisions


def _terminal_step(result: dict[str, Any], *, run_id: str, step_index: int) -> Step:
    is_error = strict_bool(result.get("is_error"), field_name="ResultMessage.is_error")
    return Step(
        step_index=step_index,
        decisions=[
            Decision(
                decision_id=f"d-{run_id}-term",
                decision_type="termination",
                chosen_action="error" if is_error else "success",
                metadata={
                    "stop_reason": result.get("stop_reason"),
                    "subtype": result.get("subtype"),
                    "num_turns": result.get("num_turns"),
                },
            )
        ],
    )


def _outcome(result: dict[str, Any] | None) -> Outcome:
    if result is None:
        raise IngestError(
            "claude-agent-sdk message stream has no ResultMessage; cannot derive Outcome"
        )
    is_error = strict_bool(result.get("is_error"), field_name="ResultMessage.is_error")
    return Outcome(
        kind="binary",
        value=not is_error,
        verifier="claude_agent_sdk_result_message",
        metadata={
            "stop_reason": result.get("stop_reason"),
            "subtype": result.get("subtype"),
        },
    )


def run_from_messages(messages: list[dict[str, Any]]) -> Run:
    """Convert a list of Claude Agent SDK message dicts into a native Run.

    Per design D7, each AssistantMessage starts a new step. Following
    UserMessages carrying tool_use_result attach as observations to the
    SAME step as the tool call. SystemMessages before the first assistant
    turn land on step 0. The final ResultMessage owns its own terminal
    step.
    """

    if not messages:
        raise IngestError("claude-agent-sdk message stream is empty")

    for idx, msg in enumerate(messages):
        if _message_type(msg) not in _KNOWN_MESSAGE_TYPES:
            raise IngestError(
                f"record {idx}: unknown message type {_message_type(msg)!r}; "
                f"supported: {sorted(_KNOWN_MESSAGE_TYPES)}"
            )

    result_msg = _result_message(messages)
    run_id = _session_id(messages)
    steps: list[Step] = []
    pending_pre_assistant_observations: list[Observation] = []
    obs_counter = 0

    current_step: Step | None = None
    next_step_index = 0

    for idx, msg in enumerate(messages):
        mtype = _message_type(msg)
        if mtype in {"StreamEvent", "RateLimitEvent"}:
            # Partial stream events are noise; the assistant + result messages
            # carry the canonical state. Skip them silently.
            continue
        if mtype == "ResultMessage":
            continue
        if mtype == "SystemMessage":
            obs = _system_observation(msg, obs_counter, run_id)
            obs_counter += 1
            if current_step is None:
                pending_pre_assistant_observations.append(obs)
            else:
                current_step.observations.append(obs)
            continue
        if mtype == "AssistantMessage":
            decisions = _assistant_decisions(msg, run_id=run_id, msg_index=idx)
            step = Step(
                step_index=next_step_index,
                decisions=decisions,
                observations=(pending_pre_assistant_observations if next_step_index == 0 else []),
            )
            pending_pre_assistant_observations = []
            steps.append(step)
            current_step = step
            next_step_index += 1
            continue
        if mtype == "UserMessage":
            obs = _user_observation(msg, obs_counter, run_id)
            obs_counter += 1
            if current_step is None:
                pending_pre_assistant_observations.append(obs)
            else:
                current_step.observations.append(obs)
            continue
        # Defensive: should be unreachable due to the up-front type check.
        raise IngestError(f"record {idx}: unhandled message type {mtype!r}")

    if pending_pre_assistant_observations:
        # No assistant turn ever fired. Park the observations on a synthetic step 0.
        steps.append(
            Step(
                step_index=next_step_index,
                observations=pending_pre_assistant_observations,
            )
        )
        next_step_index += 1

    if result_msg is None:
        raise IngestError(
            "claude-agent-sdk message stream has no ResultMessage; cannot derive Outcome"
        )
    steps.append(_terminal_step(result_msg, run_id=run_id, step_index=next_step_index))

    metadata_extra: dict[str, Any] = {"source_format": SOURCE_FORMAT}
    if result_msg is not None:
        for key in ("total_cost_usd", "usage", "model_usage", "duration_ms", "duration_api_ms"):
            value = result_msg.get(key)
            if value is not None:
                metadata_extra[key] = value

    return Run(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        steps=steps,
        outcome=_outcome(result_msg),
        metadata=Metadata(agent_name="claude-agent-sdk", extra=metadata_extra),
    )


def ingest_claude_agent_sdk(
    source_path: Path,
    output_dir: Path,
) -> IngestReceipt:
    """Read a JSONL file of Claude Agent SDK messages and write a native corpus.

    Each line is one message-stream session. Lines may either be:
      - a single JSON object representing the entire session as
        `{"messages": [...]}`, or
      - a single JSON list `[msg1, msg2, ...]`.

    A session is one `Run`. Multiple lines produce multiple Runs in the
    output directory.
    """

    text = source_path.read_text()
    runs: list[Run] = []
    warnings = [randomization_warning(SOURCE_FORMAT)]

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{source_path}: line {line_no}: invalid JSON: {exc}") from exc
        if isinstance(payload, dict) and "messages" in payload:
            messages = payload["messages"]
        elif isinstance(payload, list):
            messages = payload
        else:
            raise IngestError(
                f"{source_path}: line {line_no}: expected a JSON object with "
                "'messages' or a JSON array of message dicts; "
                f"got {type(payload).__name__}"
            )
        if not isinstance(messages, list):
            raise IngestError(f"{source_path}: line {line_no}: 'messages' must be a list")
        try:
            run = run_from_messages(messages)
        except (IngestError, ValidationError, ValueError) as exc:
            raise IngestError(f"{source_path}: line {line_no}: {exc}") from exc
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
    "ingest_claude_agent_sdk",
    "run_from_messages",
]
