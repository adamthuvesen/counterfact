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
