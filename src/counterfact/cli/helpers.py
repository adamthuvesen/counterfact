"""CLI intervention targeting helpers."""

from __future__ import annotations

import argparse
import sys

from counterfact.intervene.estimate import CausalEstimate
from counterfact.schema import Decision, Run, Step
from counterfact.taxonomy import default_intervention_kind, first_observed_arm


def first_step_for_decision_type(runs: list[Run], decision_type: str) -> tuple[Run, int]:
    for run in runs:
        for step in run.steps:
            if len(step.decisions) != 1:
                continue
            if step.decisions[0].decision_type == decision_type:
                return run, step.step_index
    raise ValueError(f"no single-decision step found for {decision_type!r}")


def first_arm(runs: list[Run], decision_type: str) -> str:
    arm = first_observed_arm(runs, decision_type)
    if arm is None:
        raise ValueError(f"no chosen_action found for {decision_type!r}")
    return arm


def intervention_kind(decision_type: str) -> str:
    return default_intervention_kind(decision_type)


def decision_by_id(run: Run, decision_id: str) -> tuple[Step, Decision] | None:
    for step in run.steps:
        for decision in step.decisions:
            if decision.decision_id == decision_id:
                return step, decision
    return None


def _validate_targeting_mode(args: argparse.Namespace) -> bool:
    if args.decision_id is not None and args.step is not None:
        print(
            "counterfact intervene: only one targeting mode is allowed: --decision-id or --step",
            file=sys.stderr,
        )
        return False
    if args.decision_id is None and args.step is None:
        print(
            "counterfact intervene: specify --decision-id or --step",
            file=sys.stderr,
        )
        return False
    return True


def _target_by_step(focal: Run, step_index: int) -> tuple[Step, Decision] | None:
    for step in focal.steps:
        if step.step_index != step_index:
            continue
        if not step.decisions:
            print(
                f"counterfact intervene: step {step_index} has no decisions",
                file=sys.stderr,
            )
            return None
        if len(step.decisions) > 1:
            ids = ", ".join(decision.decision_id for decision in step.decisions)
            print(
                f"counterfact intervene: step {step_index} has multiple decisions "
                f"({ids}); rerun with --decision-id",
                file=sys.stderr,
            )
            return None
        return step, step.decisions[0]
    print(f"counterfact intervene: step not found: {step_index}", file=sys.stderr)
    return None


def resolve_intervention_target(
    args: argparse.Namespace, focal: Run
) -> tuple[Step, Decision] | None:
    if not _validate_targeting_mode(args):
        return None
    if args.decision_id is not None:
        resolved = decision_by_id(focal, args.decision_id)
        if resolved is None:
            print(
                f"counterfact intervene: decision_id not found: {args.decision_id}",
                file=sys.stderr,
            )
        return resolved
    return _target_by_step(focal, args.step)


def parse_decision_edit(raw: str | None) -> tuple[str, str] | None:
    if raw is None or "=" not in raw:
        print(
            "counterfact intervene: --set expects key=value",
            file=sys.stderr,
        )
        return None
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        print(
            "counterfact intervene: --set expects non-empty key=value",
            file=sys.stderr,
        )
        return None
    return key, value


def add_cli_diagnostics(
    estimate: CausalEstimate,
    *,
    decision: Decision,
    step: Step,
    targeting_mode: str,
) -> CausalEstimate:
    payload = dict(estimate.next_step.payload)
    payload.update(
        {
            "targeting_mode": targeting_mode,
            "decision_id": decision.decision_id,
            "step": step.step_index,
            "decision_type": decision.decision_type,
        }
    )
    next_step = estimate.next_step.model_copy(update={"payload": payload})
    return estimate.model_copy(update={"next_step": next_step})
