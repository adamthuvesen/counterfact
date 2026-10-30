"""The numeric step chronology used by every analysis consumer."""

from typing import overload

from counterfact.analysis.facts import DecisionFact, RunFact, StepFact
from counterfact.schema import Decision, Run, Step


class AmbiguousChronologyError(ValueError):
    """A trace has no unique numeric step chronology."""


@overload
def ordered_steps(run: Run) -> tuple[Step, ...]: ...


@overload
def ordered_steps(run: RunFact) -> tuple[StepFact, ...]: ...


def ordered_steps(run: Run | RunFact) -> tuple[Step | StepFact, ...]:
    """Read steps chronologically without changing the serialized trace."""
    indexes = [step.step_index for step in run.steps]
    if len(indexes) != len(set(indexes)):
        raise AmbiguousChronologyError(
            f"run {run.run_id!r} has duplicate step_index values; steps cannot be ordered"
        )
    return tuple(sorted(run.steps, key=lambda step: step.step_index))


@overload
def first_decision(run: Run, decision_type: str) -> tuple[int, Decision] | None: ...


@overload
def first_decision(run: RunFact, decision_type: str) -> tuple[int, DecisionFact] | None: ...


def first_decision(
    run: Run | RunFact, decision_type: str
) -> tuple[int, Decision | DecisionFact] | None:
    """Locate an unambiguous first occurrence, refusing same-step ties."""
    for step in ordered_steps(run):
        decisions = [d for d in step.decisions if d.decision_type == decision_type]
        if len(decisions) > 1:
            raise ValueError(
                f"run {run.run_id!r} has multiple first {decision_type!r} decisions "
                f"at step {step.step_index}; their order is ambiguous"
            )
        if decisions:
            return step.step_index, decisions[0]
    return None
