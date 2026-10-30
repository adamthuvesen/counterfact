"""CausalEstimate result schema for the intervene API."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from typing import Any, Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from counterfact.baselines import PassRateRow


class IdentifiabilityStatus(StrEnum):
    IDENTIFIED = "identified"
    BOUNDED = "bounded"
    UNIDENTIFIED = "unidentified"


class ResultModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
        revalidate_instances="always",
        frozen=True,
    )

    def model_post_init(self, context: Any) -> None:
        for name in type(self).model_fields:
            value = object.__getattribute__(self, name)
            if isinstance(value, list | dict | tuple | set | frozenset):
                object.__setattr__(self, name, deepcopy(value))

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name in type(self).model_fields and isinstance(
            value, list | dict | tuple | set | frozenset
        ):
            return deepcopy(value)
        return value

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Validate replacements rather than bypassing the public result contract."""
        values = self.model_dump(round_trip=True)
        values.update(update or {})
        return type(self).model_validate(values)


class InterventionQuery(ResultModel):
    decision_type: str
    intervention_kind: str
    target: Any
    # The arm the contrast is taken against. None only on refusals that never
    # reached a contrast (e.g. corpus-wide degenerate or replay-only queries).
    baseline: Any = None
    # `step` is None when no specific trace step applies (e.g. corpus-wide
    # degenerate refusal). Avoid sentinels like -1 — None is self-documenting.
    step: int | None


class DistributionSummary(ResultModel):
    """A point estimate with a 95% sampling interval.

    The interval is sampling uncertainty only, from the AIPW influence
    function. Identification uncertainty (what the data cannot rule out) is a
    separate concern, carried by `IdentificationBounds`. Never conflate them.
    """

    point: float
    ci_low: float
    ci_high: float
    ci_method: Literal["aipw_influence_function", "sequential_dr_influence_function"] = (
        "aipw_influence_function"
    )

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        if not self.ci_low <= self.point <= self.ci_high:
            raise ValueError("sampling interval must contain the point estimate in order")
        return self


class IdentificationBounds(ResultModel):
    """Bounds on the contrast that hold without any no-confounding assumption."""

    lower: float
    upper: float
    technique: Literal["manski"] = "manski"
    note: str | None = None

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("identification bounds must be ordered")
        return self


class EstimatorDiagnostics(ResultModel):
    """How the identified contrast was computed and how far to trust it."""

    estimator: Literal["aipw_cross_fit"] = "aipw_cross_fit"
    n_used: int = Field(gt=1)
    n_excluded: int = Field(ge=0)
    n_target: int = Field(gt=0)
    n_baseline: int = Field(gt=0)
    n_folds: int = Field(ge=2)
    g_formula_delta: float
    ipw_delta: float
    min_propensity_target: float = Field(gt=0, le=1)
    min_propensity_baseline: float = Field(gt=0, le=1)
    effective_n_target: float = Field(gt=0)
    effective_n_baseline: float = Field(gt=0)

    @model_validator(mode="after")
    def _counts(self) -> Self:
        if self.n_target + self.n_baseline > self.n_used or self.n_folds > self.n_used:
            raise ValueError("arm counts and fold count cannot exceed the analyzed population")
        return self


class SupportPayload(TypedDict, total=False):
    """Static-typing hint for the support-context shape inside `NextStep.payload`.

    `NextStep.payload` itself stays `dict[str, Any]` so we don't break the
    intervene contract; this TypedDict is a renderer-side narrowing aid for
    mypy and editors. Keys are the union of fields produced by the
    `broaden_arm_support` and `replay_required` actions.
    """

    observed_arms: list[str | dict[str, object]]
    missing_arms: list[str]
    missing_strata: list[str]
    localization_limit: object
    replay_inputs_required: list[str] | dict[str, object]
    suggested_command: str
    arm_name: str
    intervention_target: str
    note: str


class _Payload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)


class IncreaseNPayload(_Payload):
    current_n: int = Field(ge=0)
    estimated_required_n: int = Field(gt=0)
    target_ci_width: float = Field(gt=0)
    power_method: str
    arm_breakdown: list[PassRateRow]


class BroadenArmSupportPayload(_Payload):
    arm_name: str
    missing_strata: list[str]
    observed_arms: list[str | PassRateRow]
    missing_arms: list[str]


class ReplayRequiredPayload(_Payload):
    intervention_target: str
    replay_inputs_required: list[str] | dict[str, Any]
    note: str


class AddArmRandomizationPayload(_Payload):
    arm_name: str
    current_policy: str


_PAYLOAD_MODELS: dict[str, type[_Payload]] = {
    "increase_n": IncreaseNPayload,
    "broaden_arm_support": BroadenArmSupportPayload,
    "replay_required": ReplayRequiredPayload,
    "add_arm_randomization": AddArmRandomizationPayload,
    "none": _Payload,
}


class NextStep(ResultModel):
    """Structured guidance for what to do when an estimate is not actionable as-is.

    The validator enforces that each action's minimum payload keys are present
    so consumers can rely on the shape rather than string-matching `human_text`.
    Optional keys (e.g. `suggested_command`) are tolerated but not enforced
    here; each payload model declares its required fields.
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
        payload_model = _PAYLOAD_MODELS[self.action]
        missing = [
            name
            for name, field in payload_model.model_fields.items()
            if field.is_required() and name not in self.payload
        ]
        if missing:
            raise ValueError(
                f"NextStep(action={self.action!r}) is missing required payload keys: {missing}"
            )
        payload_model.model_validate(self.payload)
        return self


class CausalEstimate(ResultModel):
    """The result object returned by `counterfact.intervene`."""

    query: InterventionQuery
    identifiability: IdentifiabilityStatus
    estimand: str | None = None
    reason: str | None = None
    adjustment_set: list[str] = Field(default_factory=list)
    # E[Y | do(target)] - E[Y | do(baseline)]: the contrast the label refers to.
    outcome_delta: DistributionSummary | None = None
    bounds: IdentificationBounds | None = None
    diagnostics: EstimatorDiagnostics | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_step: NextStep

    @model_validator(mode="after")
    def _label_payload(self) -> Self:
        validate_estimate_payload(
            self.identifiability, self.outcome_delta, self.bounds, self.diagnostics
        )
        if self.identifiability == IdentifiabilityStatus.IDENTIFIED:
            from counterfact.analysis.features import canonical_intervention_value

            if self.query.baseline is None:
                raise ValueError("identified contrasts require a named baseline arm")
            if canonical_intervention_value(
                self.query.intervention_kind, self.query.target
            ) == canonical_intervention_value(self.query.intervention_kind, self.query.baseline):
                raise ValueError("identified contrasts require distinct target and baseline arms")
            if (
                self.outcome_delta is not None
                and self.outcome_delta.ci_method != "aipw_influence_function"
            ):
                raise ValueError("decision contrasts require the AIPW interval method")
        return self


def validate_estimate_payload(
    label: IdentifiabilityStatus,
    delta: DistributionSummary | None,
    bounds: IdentificationBounds | None,
    diagnostics: object | None,
) -> None:
    """Enforce the payload guarantees shared by decision and policy estimates."""
    if label == IdentifiabilityStatus.IDENTIFIED:
        if delta is None or bounds is not None or diagnostics is None:
            raise ValueError(
                "identified estimates require a point estimate, estimator diagnostics, "
                "and no identification bounds"
            )
    elif delta is not None or diagnostics is not None:
        raise ValueError(
            "only identified estimates may carry point estimates or estimator diagnostics"
        )
    elif label == IdentifiabilityStatus.UNIDENTIFIED and bounds is not None:
        raise ValueError("unidentified estimates cannot carry identification bounds")
