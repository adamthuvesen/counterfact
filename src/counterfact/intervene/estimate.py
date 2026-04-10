"""CausalEstimate result schema for the intervene API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IdentifiabilityStatus(StrEnum):
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


# Required payload keys per action. Each action's payload must contain at
# least these keys; extra keys are tolerated. `none` accepts an empty payload.
# The full per-action schema (including optional keys like `suggested_command`)
# is documented in openspec/specs/causal-engine/spec.md under the
# `intervene returns a CausalEstimate with identifiability label` requirement.
_REQUIRED_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "increase_n": (
        "current_n",
        "estimated_required_n",
        "target_ci_width",
        "power_method",
        "arm_breakdown",
    ),
    "broaden_arm_support": (
        "arm_name",
        "missing_strata",
        "observed_arms",
        "missing_arms",
    ),
    "replay_required": (
        "intervention_target",
        "replay_inputs_required",
        "note",
    ),
    "add_arm_randomization": ("arm_name", "current_policy"),
    "none": (),
}


class NextStep(_Strict):
    """Structured guidance for what to do when an estimate is not actionable as-is.

    The validator enforces that each action's minimum payload keys are present
    so consumers can rely on the shape rather than string-matching `human_text`.
    See openspec/specs/causal-engine/spec.md for the full per-action contract,
    including optional keys (`suggested_command`) not enforced here.
    """

    action: Literal[
        "increase_n",
        "broaden_arm_support",
        "replay_required",
        "add_arm_randomization",
        "none",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)
    human_text: str

    @model_validator(mode="after")
    def _check_payload_contract(self) -> NextStep:
        required = _REQUIRED_PAYLOAD_KEYS[self.action]
        missing = [k for k in required if k not in self.payload]
        if missing:
            raise ValueError(
                f"NextStep(action={self.action!r}) is missing required payload "
                f"keys: {missing}"
            )
        return self


class CausalEstimate(_Strict):
    """The result object returned by `counterfact.intervene`."""

    query: InterventionQuery
    identifiability: IdentifiabilityStatus
    estimand: str | None = None
    reason: str | None = None
    adjustment_set: list[str] = Field(default_factory=list)
    outcome_delta: DistributionSummary | None = None
    bounds: SensitivityBounds | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_step: NextStep
