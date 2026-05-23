"""Confounded synthetic demo showcase helpers."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import counterfact.cli as cli_module
from counterfact.cli.constants import DEMO_CONTRAST_TEMPLATE
from counterfact.schema import Run


def resolve_demo_target(
    *,
    args: argparse.Namespace,
    runs: list[Run],
    decision_type: str,
    first_arm_fn: Callable[[list[Run], str], str],
) -> str:
    if args.target is not None:
        return str(args.target)
    if args.confound and decision_type == "model_call":
        return "sonnet"
    return first_arm_fn(runs, decision_type)


def maybe_print_contrast(
    *,
    args: argparse.Namespace,
    decision_type: str,
    estimate,
    model,
    runs: list[Run],
    target: str,
    intervention_kind: str,
    first_step_fn,
    intervene_fn,
    build_dag_fn,
    pass_rate_by_arm_fn,
) -> None:
    if (
        not args.confound
        or decision_type != "model_call"
        or estimate.outcome_delta is None
        or model is None
    ):
        return
    table = pass_rate_by_arm_fn(runs, decision_type)
    rates = {row.arm: row.pass_rate for row in table.rows}
    sibling = "haiku" if target == "sonnet" else "sonnet"
    if target not in rates or sibling not in rates:
        return
    run_for_intervene, step = first_step_fn(runs, decision_type)
    sibling_estimate = intervene_fn(
        dag=build_dag_fn(run_for_intervene),
        model=model,
        step=step,
        intervention={intervention_kind: sibling},
    )
    if sibling_estimate.outcome_delta is None:
        return
    naive_gap = rates[target] - rates[sibling]
    causal_gap = estimate.outcome_delta.point - sibling_estimate.outcome_delta.point
    threshold = getattr(cli_module, "_DEMO_CONTRAST_THRESHOLD", cli_module.DEMO_CONTRAST_THRESHOLD)
    if abs(naive_gap - causal_gap) >= threshold:
        print(DEMO_CONTRAST_TEMPLATE.format(naive=naive_gap, causal=causal_gap))
