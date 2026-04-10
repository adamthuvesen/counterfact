"""LLM model-call layer.

The `LLMClient` protocol is what the agent loop talks to; concrete
implementations are `LiteLLMClient` (real, used at runtime) and any test
stub that conforms to the protocol.

Role mapping (design.md D16):
* `small`  → haiku-class model
* `large`  → sonnet-class model

Models are configured in one place (`ROLE_TO_MODEL`) so retiring a model
means editing one line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

ROLE_TO_MODEL: dict[str, str] = {
    "small": "claude-haiku-4-5-20251001",
    "large": "claude-sonnet-4-6",
}


@dataclass
class LLMResponse:
    text: str
    cost_usd: float


class LLMClient(Protocol):
    """Minimal interface the agent loop depends on."""

    def call(self, *, role: str, prompt: str) -> LLMResponse: ...


def extract_cost(resp: object) -> float:
    """Best-effort USD cost extraction from a litellm response.

    Strategy:
    1. Read `response_cost` if litellm populated it (varies by provider).
    2. Fall back to `litellm.completion_cost(completion_response=resp)` which
       computes cost from token usage and the published price table.
    3. Return 0.0 only if both paths failed — the caller can decide whether
       that's acceptable.
    """
    raw = None
    if hasattr(resp, "get"):
        raw = resp.get("response_cost")  # type: ignore[union-attr]
    elif hasattr(resp, "response_cost"):
        raw = resp.response_cost
    cost = float(raw or 0.0)
    if cost > 0.0:
        return cost
    try:
        import litellm  # type: ignore[import-not-found]

        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return 0.0


class LiteLLMClient:
    """Production client. Routes through litellm; cost is sourced from the
    provider response when available, falling back to litellm's published
    price table via `completion_cost`.

    Note: the actual API call is gated behind §12.3 HUMAN GATE — no LLM call
    happens until the human approves the smoke run.
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
            max_tokens=1024,
        )
        text = resp["choices"][0]["message"]["content"]
        cost = extract_cost(resp)
        return LLMResponse(text=text, cost_usd=cost)
