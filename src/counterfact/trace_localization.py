"""Trace-localization helpers for step-scoped intervention honesty."""

from __future__ import annotations

from counterfact.schema import Run


def duplicate_decision_type_steps(run: Run, *, except_step: int, decision_type: str) -> list[int]:
    """Other steps in `run` whose decisions include `decision_type`."""
    return [
        step.step_index
        for step in run.steps
        if step.step_index != except_step
        and any(d.decision_type == decision_type for d in step.decisions)
    ]


def decision_type_repeats_elsewhere(run: Run, step_index: int, decision_type: str) -> bool:
    """True iff `decision_type` appears in any step other than `step_index`."""
    return bool(
        duplicate_decision_type_steps(run, except_step=step_index, decision_type=decision_type)
    )
