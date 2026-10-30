"""Validated analysis snapshots, independent of a descriptive model fit."""

from collections.abc import Iterable
from dataclasses import dataclass

from counterfact.analysis.facts import RunFact
from counterfact.analysis.features import decision_feature_values
from counterfact.analysis.outcomes import binary_outcome_value
from counterfact.analysis.timeline import ordered_steps
from counterfact.schema import Run
from counterfact.schema.validation import validate_runs


@dataclass(frozen=True, init=False)
class AnalysisCorpus:
    """An immutable trace snapshot that preserves the supplied population order."""

    _traces: tuple[str, ...]
    facts: tuple[RunFact, ...]
    seed: int | None
    outcome_kind: str = "binary"

    def __init__(self, runs: Iterable[Run], *, seed: int | None = 0) -> None:
        snapshots = validate_runs(runs)
        if not snapshots:
            raise ValueError("AnalysisCorpus requires at least one trace")
        for run in snapshots:
            ordered_steps(run)
            binary_outcome_value(run)
        object.__setattr__(self, "_traces", tuple(run.model_dump_json() for run in snapshots))
        object.__setattr__(self, "facts", tuple(RunFact.from_run(run) for run in snapshots))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "outcome_kind", "binary")

    @property
    def runs(self) -> list[Run]:
        """Return detached copies so callers cannot change the analysis facts."""
        return [Run.model_validate_json(trace) for trace in self._traces]

    @property
    def feature_index(self) -> dict[str, int]:
        names = sorted(
            {
                f"{family}::{value}"
                for run in self.facts
                for step in ordered_steps(run)
                for decision in step.decisions
                for family, value in decision_feature_values(decision)
            }
        )
        return {name: index for index, name in enumerate(names)}
