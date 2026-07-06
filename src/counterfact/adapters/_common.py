"""Shared helpers used by every ingest adapter.

The receipt model, the corpus-level randomization warning sentence, and the
run-writer all live here so per-adapter modules stay focused on the shape
mapping. Both `counterfact.ingest.ingest_generic_jsonl` and the SDK-specific
adapters under `counterfact.adapters` reuse these primitives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from counterfact.schema import Decision, Observation, Run, Step


class IngestReceipt(BaseModel):
    """Per-corpus receipt written alongside generated trace files.

    Adapters that do not consume an explicit user-supplied mapping file emit
    `mapping_file=""` so every ingest receipt has the same shape.
    """

    model_config = ConfigDict(extra="forbid")

    source_format: str
    source_file: str
    mapping_file: str = ""
    generated_count: int
    warnings: list[str] = Field(default_factory=list)
    dropped_fields: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class IngestError(RuntimeError):
    """Raised when an ingest adapter cannot produce a valid native corpus."""


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


def write_corpus(runs: list[Run], output_dir: Path, receipt: IngestReceipt) -> None:
    """Write each run as `<run_id>.json` plus a single `ingest-receipt.json`.

    Filenames are derived from `run_id`, so the adapter is responsible for
    making sure run_ids are unique across the corpus.
    """

    planned: list[tuple[Run, Path]] = []
    seen: set[str] = set()
    base = output_dir.resolve()
    for run in runs:
        if run.run_id in seen:
            raise IngestError(f"duplicate run_id in corpus: {run.run_id!r}")
        seen.add(run.run_id)
        if not run.run_id or run.run_id in {".", ".."} or "\\" in run.run_id:
            raise IngestError(f"unsafe run_id for output filename: {run.run_id!r}")
        run_id_path = Path(run.run_id)
        if run_id_path.is_absolute() or len(run_id_path.parts) != 1:
            raise IngestError(f"unsafe run_id for output filename: {run.run_id!r}")
        out_path = base / f"{run.run_id}.json"
        if base not in out_path.resolve().parents:
            raise IngestError(f"run_id escapes output directory: {run.run_id!r}")
        planned.append((run, out_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    for run, out_path in planned:
        out_path.write_text(run.model_dump_json(indent=2) + "\n")
    (output_dir / "ingest-receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")


def read_existing_receipt_count(output_dir: Path) -> int:
    """Return `generated_count` from any pre-existing receipt, else 0.

    Live tracers use this to seed their per-session counter so a new trace
    written to a directory that already has a receipt extends the count
    rather than overwriting it with 1. A missing or unparseable receipt is
    treated as a fresh start; we do not raise here because the on-disk
    state is informational, not authoritative.
    """

    receipt_path = output_dir / "ingest-receipt.json"
    if not receipt_path.exists():
        return 0
    try:
        payload = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    count = payload.get("generated_count")
    if isinstance(count, int) and count >= 0:
        return count
    return 0


__all__ = [
    "IngestError",
    "IngestReceipt",
    "StepBuilder",
    "per_decision_randomization_warnings",
    "randomization_warning",
    "read_existing_receipt_count",
    "strict_bool",
    "write_corpus",
]
