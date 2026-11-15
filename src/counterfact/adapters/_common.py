"""Shared adapter mapping helpers and imports from the corpus IO owner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from counterfact.corpus_io import CorpusIOError as IngestError
from counterfact.corpus_io import IngestReceipt, write_corpus
from counterfact.schema import Decision, Observation, Run, Step


class StepBuilder:
    """Collect adapter decisions and observations into ordered trace steps."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._current_step: Step | None = None
        self._pending_observations: list[Observation] = []
        self._next_step_index = 0

    @property
    def next_step_index(self) -> int:
        return self._next_step_index

    def add_decision_step(self, decisions: list[Decision]) -> Step:
        step = Step(
            step_index=self._next_step_index,
            decisions=decisions,
            observations=self._pending_observations,
        )
        self._pending_observations = []
        self.steps.append(step)
        self._current_step = step
        self._next_step_index += 1
        return step

    def add_observation(self, observation: Observation) -> None:
        if self._current_step is None:
            self._pending_observations.append(observation)
        else:
            self._current_step.observations.append(observation)

    def flush_pending_observations(self) -> Step | None:
        if not self._pending_observations:
            return None
        step = Step(
            step_index=self._next_step_index,
            observations=self._pending_observations,
        )
        self._pending_observations = []
        self.steps.append(step)
        self._current_step = step
        self._next_step_index += 1
        return step


def strict_bool(value: Any, *, field_name: str) -> bool:
    """Return a JSON boolean or reject ambiguous truthy/falsy values."""
    if isinstance(value, bool):
        return value
    raise IngestError(f"{field_name} must be a JSON boolean; got {type(value).__name__}")


def randomization_warning(source_format: str) -> str:
    """The corpus-level warning every SDK adapter must emit.

    Format-specific adapters never log randomization metadata (`policy`,
    `propensity`, etc.) because the upstream SDKs do not log it. Counterfact
    callers need to know that randomized-support claims will be unavailable
    on the resulting corpus before they reach for `intervene` and get an
    `unidentified` answer.
    """

    return (
        f"{source_format} traces do not log randomization metadata; "
        "randomized-support claims will be unavailable for this corpus. "
        "Use bench/real for randomized arms."
    )


def write_sdk_corpus(
    runs: list[Run],
    output_dir: Path,
    *,
    source_format: str,
    source_file: str,
    append: bool = False,
    warnings: Sequence[str] = (),
) -> IngestReceipt:
    """Publish SDK runs with the corpus-level `randomization_warning` first in the receipt."""
    receipt = IngestReceipt(
        source_format=source_format,
        source_file=source_file,
        generated_count=0,  # write_corpus counts the published runs
        warnings=[randomization_warning(source_format), *warnings],
    )
    return write_corpus(runs, output_dir, receipt, append=append)


def per_decision_randomization_warnings(run: Run, *, source_index: int) -> list[str]:
    """Per-record warnings for decisions with chosen_action but no randomization.

    Emitted by `generic-jsonl` so partially-mapped sources still flag missing
    metadata at the row level. SDK adapters use the corpus-level
    `randomization_warning` instead because every decision is in this state.
    """

    warnings: list[str] = []
    for step in run.steps:
        for decision in step.decisions:
            has_randomization = any(
                value is not None
                for value in (
                    decision.policy,
                    decision.policy_params,
                    decision.valid_actions,
                    decision.propensity,
                    decision.context_features,
                )
            )
            if decision.chosen_action is not None and not has_randomization:
                warnings.append(
                    f"record {source_index}: decision {decision.decision_id} "
                    "has chosen_action but no randomization metadata; "
                    "randomized-support claims may be unavailable"
                )
    return warnings


__all__ = [
    "IngestError",
    "IngestReceipt",
    "StepBuilder",
    "per_decision_randomization_warnings",
    "randomization_warning",
    "strict_bool",
    "write_corpus",
    "write_sdk_corpus",
]
