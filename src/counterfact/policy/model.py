"""Deterministic agent policies over logged decision types.

A policy sets the action for some decision types and leaves every other
decision as the agent logged it. A rule is either a constant action, or a
lookup from one feature of the earlier history to an action. History means
strictly earlier steps, the same boundary the single-decision engine uses.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from counterfact.schema import DecisionTypeLiteral


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryFeature(_Strict):
    """The latest earlier value of a decision's action or of an observation key."""

    source: Literal["decision", "observation"]
    key: str


class PolicyRule(_Strict):
    decision_type: DecisionTypeLiteral
    action: str | None = None
    when: HistoryFeature | None = None
    # Feature value (canonical string, e.g. "true") -> action.
    cases: dict[str, str] | None = None

    @model_validator(mode="after")
    def _one_form(self) -> PolicyRule:
        constant = self.action is not None
        lookup = self.when is not None and self.cases is not None
        if constant == lookup or (self.when is None) != (self.cases is None):
            raise ValueError("a rule sets either `action`, or both `when` and `cases`")
        return self


class Policy(_Strict):
    name: str
    rules: list[PolicyRule]

    @model_validator(mode="after")
    def _one_rule_per_type(self) -> Policy:
        types = [rule.decision_type for rule in self.rules]
        if not types:
            raise ValueError("a policy needs at least one rule")
        if len(types) != len(set(types)):
            raise ValueError(f"a policy has one rule per decision_type; got {types}")
        return self

    def rule_for(self, decision_type: str) -> PolicyRule | None:
        return next((r for r in self.rules if r.decision_type == decision_type), None)

    def observation_keys(self) -> set[str]:
        return {
            r.when.key for r in self.rules if r.when is not None and r.when.source == "observation"
        }
