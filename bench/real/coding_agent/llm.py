"""LLM model-call layer.

The `LLMClient` protocol is what the agent loop talks to; concrete
implementations are `LiteLLMClient` (real, used at runtime) and any test
stub that conforms to the protocol.

Role mapping:
* `small`  → haiku-class model
* `large`  → sonnet-class model

Models are configured in one place (`ROLE_TO_MODEL`) so retiring a model
means editing one line.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol

ROLE_TO_MODEL: dict[str, str] = {
    "small": "claude-haiku-4-5-20251001",
    "large": "claude-sonnet-4-6",
}

DEFAULT_MAX_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT_S = 120
DEFAULT_NUM_RETRIES = 1


@dataclass
class LLMResponse:
    text: str
    cost_usd: float
    finish_reason: str | None = None


class CostUnknownError(RuntimeError):
    """Raised when a production LLM response cannot be priced safely."""


class LLMClient(Protocol):
    """Minimal interface the agent loop depends on."""

    def call(self, *, role: str, prompt: str) -> LLMResponse: ...


def extract_cost(resp: object) -> float:
    """Best-effort USD cost extraction from a litellm response.

    Strategy:
    1. Read `response_cost` if litellm populated it (varies by provider).
    2. Fall back to `litellm.completion_cost(completion_response=resp)` which
       computes cost from token usage and the published price table.
    3. Raise if both paths fail. Production budget accounting must fail closed.
    """
    raw = None
    if hasattr(resp, "get"):
        raw = resp.get("response_cost")  # type: ignore[union-attr]
    elif hasattr(resp, "response_cost"):
        raw = resp.response_cost
    # Distinguish "provider didn't populate cost" (None) from "provider says
    # the call was free" (0.0). Only the former should fall through to the
    # litellm price-table fallback; a real 0.0 must round-trip as 0.0.
    cost = float(raw) if raw is not None else None
    if cost is not None:
        if not math.isfinite(cost):
            raise CostUnknownError(f"non-finite response cost: {cost!r}")
        return cost
    try:
        import litellm  # type: ignore[import-not-found]

        fallback = float(litellm.completion_cost(completion_response=resp))
    except Exception as exc:
        raise CostUnknownError(
            "could not determine LLM response cost from provider response or "
            "litellm.completion_cost; refusing to treat the call as free"
        ) from exc
    if not math.isfinite(fallback) or fallback <= 0.0:
        # Production budget accounting fails closed: a zero / negative fallback
        # almost always means litellm has no price for this model, not that the
        # call was actually free. Mocked test clients never hit this path —
        # they construct LLMResponse(cost_usd=0.0) directly.
        raise CostUnknownError(
            "litellm.completion_cost returned non-positive fallback "
            f"({fallback!r}); refusing to treat the call as free"
        )
    return fallback


def _message_content_to_text(content: object) -> str:
    """Normalize provider message content to text for the patch extractor."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return str(content["text"])
    # Preserve the run as a controlled extraction failure rather than crashing
    # after a paid response. Keep a compact structured breadcrumb if possible.
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True)
    return ""


class LiteLLMClient:
    """Production client. Routes through litellm; cost is sourced from the
    provider response when available, falling back to litellm's published
    price table via `completion_cost`.

    Note: the actual API call is gated behind the first-run human approval
    marker in `runner.first_run_gate_check` — no LLM call happens until the
    operator creates `.counterfact/approved` after eyeballing a smoke corpus.
    """

    def __init__(self, role_to_model: dict[str, str] | None = None) -> None:
        self.role_to_model = role_to_model or ROLE_TO_MODEL

    def call(self, *, role: str, prompt: str) -> LLMResponse:
        # Lazy import: keep `litellm` out of the test-time import path so unit
        # tests using mocked clients don't pay the import cost.
        import litellm  # type: ignore[import-not-found]

        model = self.role_to_model[role]
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout=DEFAULT_REQUEST_TIMEOUT_S,
            num_retries=DEFAULT_NUM_RETRIES,
        )
        choice = resp["choices"][0]
        text = _message_content_to_text(choice["message"].get("content"))
        finish_reason = choice.get("finish_reason")
        cost = extract_cost(resp)
        return LLMResponse(text=text, cost_usd=cost, finish_reason=finish_reason)
