from __future__ import annotations

import argparse
import sys

import counterfact.cli as cli_module
from counterfact.cli.constants import DEMO_CONTRAST_TEMPLATE
from counterfact.cli.formatters import format_pass_rate_table
from counterfact.cli.helpers import first_step_for_decision_type
from counterfact.cli.loaders import demo_runs_dir, load_trace_dir, synthetic_runs
from counterfact.intervene.degenerate import degenerate_estimate, outcome_classes
from counterfact.intervene.estimate import CausalEstimate
from counterfact.outcome.model import OutcomeModel
from counterfact.schema import Run
from counterfact.taxonomy import default_intervention_kind, first_observed_arm


def _load_demo_runs(args: argparse.Namespace) -> tuple[list[Run], str] | None:
    try:
        if args.confound:
            runs = synthetic_runs(n=args.synthetic_n, seed=args.seed, confound=True)
            source = f"synthetic SCM (confounded, n={args.synthetic_n}, seed={args.seed})"
            return runs, source

        runs_dir, source = demo_runs_dir(args.runs_dir)
        loaded_runs = load_trace_dir(runs_dir, command="demo")
        if loaded_runs is None:
            return None
        if loaded_runs:
            return loaded_runs, source
        if not args.synthetic_fallback:
            print(
                "counterfact demo: no real traces found at "
                f"{runs_dir}; pass --confound for the synthetic showcase or "
                "--synthetic-fallback to opt into synthetic data.",
                file=sys.stderr,
            )
            return None
        runs = synthetic_runs(n=args.synthetic_n, seed=args.seed)
        source = f"synthetic SCM (n={args.synthetic_n}, seed={args.seed})"
        return runs, source
    except ImportError as exc:
        print(f"counterfact demo: {exc}", file=sys.stderr)
        return None


def _resolve_demo_target(args: argparse.Namespace, runs: list[Run], decision_type: str) -> str:
    if args.target is not None:
        return str(args.target)
    if args.confound and decision_type == "model_call":
        return "sonnet"
    arm = first_observed_arm(runs, decision_type)
    if arm is None:
        raise ValueError(f"no chosen_action found for {decision_type!r}")
    return arm


def _demo_estimate(
    *,
    args: argparse.Namespace,
    runs: list[Run],
    decision_type: str,
    intervention_kind: str,
    target: str,
) -> tuple[CausalEstimate, OutcomeModel | None]:
    from counterfact import fit_outcome_model, intervene
    from counterfact.dag import build_dag

    if len(outcome_classes(runs)) == 1:
        return (
            degenerate_estimate(
                runs,
                decision_type=decision_type,
                intervention_kind=intervention_kind,
                target=target,
            ),
            None,
        )

    run, step = first_step_for_decision_type(runs, decision_type)
    model = fit_outcome_model(runs, n_bootstrap=args.bootstrap, seed=args.seed)
    estimate = intervene(
        dag=build_dag(run),
        model=model,
        step=step,
        intervention={intervention_kind: target},
    )
    return estimate, model


def _print_contrast(
    *,
    args: argparse.Namespace,
    decision_type: str,
    estimate: CausalEstimate,
    model: OutcomeModel | None,
    runs: list[Run],
    target: str,
    intervention_kind: str,
) -> None:
    """Print the naive-vs-causal gap when the confounded showcase can compute one."""
    from counterfact import intervene, pass_rate_by_arm
    from counterfact.dag import build_dag

    if (
        not args.confound
        or decision_type != "model_call"
        or estimate.outcome_delta is None
        or model is None
    ):
        return
    table = pass_rate_by_arm(runs, decision_type)
    rates = {row.arm: row.pass_rate for row in table.rows}
    sibling = "haiku" if target == "sonnet" else "sonnet"
    if target not in rates or sibling not in rates:
        return
    run_for_intervene, step = first_step_for_decision_type(runs, decision_type)
    sibling_estimate = intervene(
        dag=build_dag(run_for_intervene),
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


def run(args: argparse.Namespace) -> int:
    from counterfact.outcome.binary import binary_outcome_value

    loaded = _load_demo_runs(args)
    if loaded is None:
        return 2
    runs, source = loaded

    decision_type = args.decision_type
    intervention_kind = default_intervention_kind(decision_type)
    target = _resolve_demo_target(args, runs, decision_type)

    pass_count = sum(1 for run in runs if binary_outcome_value(run))
    print("counterfact demo: naive vs honest")
    print(f"data: {source}")
    print(f"outcomes: {pass_count} pass / {len(runs) - pass_count} fail")
    print()
    print("\n".join(format_pass_rate_table(runs, decision_type)))
    print()

    estimate, model = _demo_estimate(
        args=args,
        runs=runs,
        decision_type=decision_type,
        intervention_kind=intervention_kind,
        target=target,
    )

    print(f"intervene({decision_type} -> {target})")
    print(f"identifiability: {estimate.identifiability.value}")
    if estimate.outcome_delta is not None:
        delta = estimate.outcome_delta
        print(f"outcome_delta: {delta.point:.3f} [{delta.ci_low:.3f}, {delta.ci_high:.3f}]")
    if estimate.reason:
        print(f"reason: {estimate.reason}")
    if estimate.warnings:
        print(f"warning: {estimate.warnings[0]}")
    print(f"next_step: {estimate.next_step.action} - {estimate.next_step.human_text}")
    suggested = estimate.next_step.payload.get("suggested_command")
    if suggested:
        print(f"suggested_command: {suggested}")

    _print_contrast(
        args=args,
        decision_type=decision_type,
        estimate=estimate,
        model=model,
        runs=runs,
        target=target,
        intervention_kind=intervention_kind,
    )
    return 0
