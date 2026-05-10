"""Pydantic models for the native counterfact trace format.

The schema is the contract between trace producers (CounterBench, external
adapters) and the rest of counterfact. Models are strict (`extra="forbid"`) and
support full JSON round-trip preservation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

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
    rewrite — `categorical` and `continuous` traces parse cleanly today and
    fail at fit time with a clear `UnsupportedOutcomeError`.
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

    @model_validator(mode="after")
    def _randomization_metadata_is_complete(self) -> Decision:
        randomization_fields = {
            "policy": self.policy,
            "policy_params": self.policy_params,
            "valid_actions": self.valid_actions,
            "propensity": self.propensity,
            "context_features": self.context_features,
        }
        if not any(value is not None for value in randomization_fields.values()):
            return self

        missing = [name for name, value in randomization_fields.items() if value is None]
        if self.chosen_action is None:
            missing.append("chosen_action")
        if missing:
            raise ValueError(
                "randomized decisions must log complete metadata; missing: "
                + ", ".join(sorted(missing))
            )
        if self.valid_actions is not None and self.chosen_action not in self.valid_actions:
            raise ValueError(
                f"chosen_action={self.chosen_action!r} must be present in "
                f"valid_actions={self.valid_actions!r}"
            )
        return self


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

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(f"unrecognized schema_version={v!r}; supported versions: {supported}")
        return v

    @model_validator(mode="after")
    def _decision_ids_are_unique(self) -> Run:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for step in self.steps:
            for decision in step.decisions:
                if decision.decision_id in seen:
                    duplicates.add(decision.decision_id)
                seen.add(decision.decision_id)
        if duplicates:
            raise ValueError(
                "decision_id values must be unique within a run; duplicates: "
                + ", ".join(sorted(duplicates))
            )
        return self
