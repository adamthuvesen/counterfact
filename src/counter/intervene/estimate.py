"""CausalEstimate result schema for the intervene API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IdentifiabilityStatus(str, Enum):
    IDENTIFIED = "identified"
    BOUNDED = "bounded"
    UNIDENTIFIED = "unidentified"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class InterventionQuery(_Strict):
    decision_type: str
    intervention_kind: str
    target: Any
    step: int


class DistributionSummary(_Strict):
    """Summary of the predicted outcome distribution under the intervention.

    Per design.md D3 the bootstrap CI is *coefficient/prediction* uncertainty.
    Identifiability uncertainty is a separate concern (see SensitivityBounds).
    """

    point: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


class SensitivityBounds(_Strict):
    """Identifiability/confounding uncertainty. Distinct from prediction CIs."""

    e_value: float
    technique: Literal["e_value"] = "e_value"
    note: str | None = None


class CausalEstimate(_Strict):
    """The result object returned by `counter.intervene`."""

    query: InterventionQuery
    identifiability: IdentifiabilityStatus
    estimand: str | None = None
    reason: str | None = None
    adjustment_set: list[str] = Field(default_factory=list)
    outcome_delta: DistributionSummary | None = None
    bounds: SensitivityBounds | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_step: str | None = None
