"""Pydantic models for the native counter trace format.

The schema is the contract between trace producers (CounterBench, external
adapters) and the rest of counter. Models are strict (`extra="forbid"`) and
support full JSON round-trip preservation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1.0"

DecisionTypeLiteral = Literal[
    "plan_step",
    "model_call",
    "tool_call",
    "memory_read",
    "retry",
    "termination",
]


class _Strict(BaseModel):
    """Base for all schema models — forbids unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Outcome(_Strict):
    """Tagged-union outcome.

    `kind` selects the value type. v0 only fits `kind="binary"` at runtime, but
    the schema accepts all three kinds at parse time so v1 can extend without a
    rewrite (see design.md D2).
    """

    kind: Literal["binary", "categorical", "continuous"]
    value: bool | str | float
    verifier: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _value_matches_kind(self) -> Outcome:
        expected: dict[str, type] = {
            "binary": bool,
            "categorical": str,
            "continuous": float,
        }
        want = expected[self.kind]
        if not isinstance(self.value, want):
            raise ValueError(
                f"Outcome.kind={self.kind!r} requires value of type {want.__name__}, "
                f"got {type(self.value).__name__}"
            )
        return self


class Decision(_Strict):
    """A single decision made by the agent at one step.

    Randomization metadata is optional and all-or-nothing: when a decision is
    randomized via a logged policy, the six fields below are recorded together.
    """

    decision_id: str
    decision_type: DecisionTypeLiteral
    chosen_action: str | None = None
    policy: str | None = None
    policy_params: dict[str, Any] | None = None
    valid_actions: list[str] | None = None
    propensity: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    context_features: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(_Strict):
    observation_id: str
    content: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Step(_Strict):
    step_index: int
    decisions: list[Decision] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Metadata(_Strict):
    agent_name: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Run(_Strict):
    schema_version: str
    run_id: str
    steps: list[Step] = Field(default_factory=list)
    outcome: Outcome
    metadata: Metadata = Field(default_factory=Metadata)
