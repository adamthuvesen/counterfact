"""Promotion rubric for corpus readiness.

Single source of truth for the thresholds the analyzer enforces. The whole
point of pulling this into one Pydantic model is that bumping the bar is a
one-line PR with a visible diff. Threshold changes are product decisions; they
should never be hidden behind environment variables or yaml.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RubricThresholds(BaseModel):
    """Thresholds the analyzer scores a corpus against.

    Lives in code (not config or env) so threshold changes show up as a visible
    diff and require a deliberate PR. Bumping the bar is a product decision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_pass_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    max_pass_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    min_arms_per_decision_type: int = Field(default=2, ge=1)
    min_n_per_arm: int = Field(default=5, ge=1)
    min_identified_decision_types: int = Field(default=1, ge=0)
    require_model_arm_outcome_mix: bool = True


DEFAULT_THRESHOLDS = RubricThresholds()
