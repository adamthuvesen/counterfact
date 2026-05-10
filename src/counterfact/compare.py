"""Descriptive trace comparison utilities."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from counterfact._fmt import outcome_label as _outcome_label
from counterfact.diagnose import DiagnosisReport
from counterfact.schema import Run


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class DecisionDiff(_Strict):
    step: int | None
    decision_type: str
    left_decision_id: str | None = None
    right_decision_id: str | None = None
    left_chosen_action: str | None = None
    right_chosen_action: str | None = None


class StepDiff(_Strict):
    step: int
    left_decision_count: int
    right_decision_count: int
    left_observation_count: int
    right_observation_count: int


class TraceComparison(_Strict):
    left_run_id: str
    right_run_id: str
    left_outcome: str
    right_outcome: str
    left_step_count: int
    right_step_count: int
    decision_diffs: list[DecisionDiff] = Field(default_factory=list)
    step_diffs: list[StepDiff] = Field(default_factory=list)
    diagnosis: DiagnosisReport | None = None
    note: str = "descriptive trace diff only; causal claims require a corpus-backed diagnosis"


def _decision_rows(run: Run) -> dict[tuple[int, str, int], tuple[str, str | None]]:
    rows: dict[tuple[int, str, int], tuple[str, str | None]] = {}
    for step in run.steps:
        type_counts: dict[str, int] = {}
        for decision in step.decisions:
            offset = type_counts.get(decision.decision_type, 0)
            type_counts[decision.decision_type] = offset + 1
            rows[(step.step_index, decision.decision_type, offset)] = (
                decision.decision_id,
                decision.chosen_action,
            )
    return rows


def compare_traces(
    left: Run,
    right: Run,
    *,
    diagnosis: DiagnosisReport | None = None,
) -> TraceComparison:
    left_rows = _decision_rows(left)
    right_rows = _decision_rows(right)
    decision_diffs: list[DecisionDiff] = []
    for key in sorted(set(left_rows) | set(right_rows)):
        left_decision = left_rows.get(key)
        right_decision = right_rows.get(key)
        if left_decision == right_decision:
            continue
        step, decision_type, _offset = key
        decision_diffs.append(
            DecisionDiff(
                step=step,
                decision_type=decision_type,
                left_decision_id=left_decision[0] if left_decision else None,
                right_decision_id=right_decision[0] if right_decision else None,
                left_chosen_action=left_decision[1] if left_decision else None,
                right_chosen_action=right_decision[1] if right_decision else None,
            )
        )

    left_steps = {step.step_index: step for step in left.steps}
    right_steps = {step.step_index: step for step in right.steps}
    step_diffs: list[StepDiff] = []
    for step_index in sorted(set(left_steps) | set(right_steps)):
        left_step = left_steps.get(step_index)
        right_step = right_steps.get(step_index)
        diff = StepDiff(
            step=step_index,
            left_decision_count=len(left_step.decisions) if left_step else 0,
            right_decision_count=len(right_step.decisions) if right_step else 0,
            left_observation_count=len(left_step.observations) if left_step else 0,
            right_observation_count=len(right_step.observations) if right_step else 0,
        )
        if (
            diff.left_decision_count != diff.right_decision_count
            or diff.left_observation_count != diff.right_observation_count
        ):
            step_diffs.append(diff)

    return TraceComparison(
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        left_outcome=_outcome_label(left),
        right_outcome=_outcome_label(right),
        left_step_count=len(left.steps),
        right_step_count=len(right.steps),
        decision_diffs=decision_diffs,
        step_diffs=step_diffs,
        diagnosis=diagnosis,
    )


__all__ = ["DecisionDiff", "StepDiff", "TraceComparison", "compare_traces"]
