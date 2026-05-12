"""`ClaudeAgentTracer` — async context manager that captures Claude Agent SDK messages.

Wraps `claude_agent_sdk.query()` (or any equivalent message stream) and writes a
single valid `Run` JSON file when the context exits. The mapping is shared with
`counterfact.adapters.claude_agent_sdk.run_from_messages`, so the live and
offline paths produce byte-identical output for identical input.

`claude_agent_sdk` itself is an OPTIONAL dependency. Importing
`ClaudeAgentTracer` works without the SDK installed; calling `observe(msg)`
with raw SDK dataclasses requires the SDK only if those dataclasses cannot be
JSON-dumped via `dataclasses.asdict`.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

from counterfact.adapters._common import (
    IngestReceipt,
    randomization_warning,
    read_existing_receipt_count,
    write_corpus,
)
from counterfact.adapters.claude_agent_sdk import SOURCE_FORMAT, run_from_messages


class ClaudeAgentTracer:
    """Async context manager that captures messages and emits a Run JSON on exit.

    Example:
        async with ClaudeAgentTracer(output_dir=Path("corpus/")) as tracer:
            async for msg in query(prompt="..."):
                tracer.observe(msg)
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._messages: list[dict[str, Any]] = []
        self._sdk_available = importlib.util.find_spec("claude_agent_sdk") is not None
        # Track traces written by this tracer instance; seeded from any
        # pre-existing receipt on first write so a restarted session keeps
        # extending the count rather than overwriting it with 1.
        self._cumulative_count = 0
        self._seeded_from_disk = False

    async def __aenter__(self) -> ClaudeAgentTracer:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Flush even on exception so partial sessions aren't silently lost.
        # If the message stream never produced a ResultMessage, run_from_messages
        # raises IngestError; let that bubble up.
        if not self._messages:
            return None
        run = run_from_messages(self._messages)
        if not self._seeded_from_disk:
            self._cumulative_count = read_existing_receipt_count(self.output_dir)
            self._seeded_from_disk = True
        self._cumulative_count += 1
        receipt = IngestReceipt(
            source_format=SOURCE_FORMAT,
            source_file="<live-tracer>",
            generated_count=self._cumulative_count,
            warnings=[randomization_warning(SOURCE_FORMAT)],
        )
        write_corpus([run], self.output_dir, receipt)
        return None

    def observe(self, message: Any) -> None:
        """Record one message. Accepts either a JSON-dumpable dict or an SDK dataclass."""

        self._messages.append(self._normalize(message))

    def _normalize(self, message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return _ensure_type_tag(message)
        if dataclasses.is_dataclass(message) and not isinstance(message, type):
            return _ensure_type_tag(_dataclass_to_dict(message))
        # Last resort: SDK might use plain classes (BaseModel etc).
        if hasattr(message, "model_dump"):
            payload: dict[str, Any] = message.model_dump()
            payload.setdefault("__type__", type(message).__name__)
            return payload
        if not self._sdk_available:
            raise ImportError(
                "ClaudeAgentTracer.observe received a non-dict message but "
                "`claude_agent_sdk` is not installed. Install the optional "
                "dependency with `uv pip install claude-agent-sdk` or pass "
                "JSON-dumpable dicts to `observe()`."
            )
        raise TypeError(
            f"ClaudeAgentTracer.observe cannot serialize message of type "
            f"{type(message).__name__}; pass a dict or a dataclass instance."
        )


def _dataclass_to_dict(message: Any) -> dict[str, Any]:
    payload = dataclasses.asdict(message)
    payload["__type__"] = type(message).__name__

    # `content` may itself be a list of dataclass instances (ContentBlock subclasses).
    # `dataclasses.asdict` already recurses, but it strips the type tag we need
    # to distinguish ToolUseBlock vs TextBlock. Re-add it from the original message.
    raw_content = getattr(message, "content", None)
    if isinstance(raw_content, list):
        tagged: list[Any] = []
        for original_block, dumped_block in zip(
            raw_content, payload.get("content", []), strict=False
        ):
            if isinstance(dumped_block, dict):
                dumped_block.setdefault("__type__", type(original_block).__name__)
            tagged.append(dumped_block)
        payload["content"] = tagged
    return payload


def _ensure_type_tag(payload: dict[str, Any]) -> dict[str, Any]:
    """Tolerate dicts that already have `__type__` or an equivalent `type` key."""

    if "__type__" not in payload and "type" in payload:
        # Don't mutate caller's dict in place.
        out = dict(payload)
        out["__type__"] = payload["type"]
        return out
    return payload


__all__ = ["ClaudeAgentTracer"]
