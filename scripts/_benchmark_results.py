"""Read the committed benchmark result contract used by derived artifacts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bench.synthetic.policy_suite import PolicySummary
from bench.synthetic.suite import ScenarioSummary


class BenchmarkResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seeds: int = Field(gt=0)
    command: str
    scenarios: list[ScenarioSummary]
    policies: list[PolicySummary]
