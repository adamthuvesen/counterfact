"""Immutable analysis projections of the serialized trace contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from counterfact.schema import Decision, Run, Step


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


def _mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze(item) for key, item in value.items()})


@dataclass(frozen=True)
class DecisionFact:
    decision_id: str
    decision_type: str
    chosen_action: str | None
    policy: str | None
    policy_params: Mapping[str, object] | None
    valid_actions: tuple[str, ...] | None
    propensity: float | None
    metadata: Mapping[str, object]

    @classmethod
    def from_decision(cls, decision: Decision) -> DecisionFact:
        return cls(
            decision.decision_id,
            decision.decision_type,
            decision.chosen_action,
            decision.policy,
            _mapping(decision.policy_params) if decision.policy_params is not None else None,
            tuple(decision.valid_actions) if decision.valid_actions is not None else None,
            decision.propensity,
            _mapping(decision.metadata),
        )


@dataclass(frozen=True)
class ObservationFact:
    content: Mapping[str, object]


@dataclass(frozen=True)
class StepFact:
    step_index: int
    decisions: tuple[DecisionFact, ...]
    observations: tuple[ObservationFact, ...]

    @classmethod
    def from_step(cls, step: Step) -> StepFact:
        return cls(
            step.step_index,
            tuple(DecisionFact.from_decision(decision) for decision in step.decisions),
            tuple(
                ObservationFact(_mapping(observation.content)) for observation in step.observations
            ),
        )


@dataclass(frozen=True)
class OutcomeFact:
    kind: str
    value: bool | str | float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class RunFact:
    run_id: str
    steps: tuple[StepFact, ...]
    outcome: OutcomeFact

    @classmethod
    def from_run(cls, run: Run) -> RunFact:
        return cls(
            run.run_id,
            tuple(StepFact.from_step(step) for step in run.steps),
            OutcomeFact(run.outcome.kind, run.outcome.value, _mapping(run.outcome.metadata)),
        )
