"""Report shape returned by `corpus_analyzer.analyze`.

The shape is part of the public contract — consumers (CI, downstream tooling,
the demo) read `criteria` and `promote` directly. Reasons are stable strings
so callers can grep without parsing prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from counterfact.corpus_analyzer.rubric import RubricThresholds

IdentifiabilityName = Literal["identified", "bounded", "unidentified"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutcomeBalance(_Strict):
    pass_rate: float
    n_pass: int
    n_fail: int


class ArmSupportRow(_Strict):
    decision_type: str
    arm: str
    n: int
    pass_count: int
    pass_rate: float


class IdentifiabilityCoverage(_Strict):
    reachable: list[IdentifiabilityName] = Field(default_factory=list)
    unfittable_outcome_model: bool = False


class RubricCriterion(_Strict):
    name: Literal[
        "outcome_balance",
        "arm_support",
        "identifiability_coverage",
        "model_arm_outcome_mix",
    ]
    passed: bool
    reason: str


class CorpusReadinessReport(_Strict):
    n_traces: int
    outcome_balance: OutcomeBalance
    arm_support: list[ArmSupportRow] = Field(default_factory=list)
    identifiability_coverage: IdentifiabilityCoverage
    criteria: list[RubricCriterion] = Field(default_factory=list)
    promote: bool
    thresholds: RubricThresholds
