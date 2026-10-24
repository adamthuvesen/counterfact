from __future__ import annotations

import argparse
import sys

from counterfact.baselines import pass_rate_by_arm
from counterfact.cli.constants import DEMO_CONTRAST_TEMPLATE, DEMO_CONTRAST_THRESHOLD
from counterfact.cli.formatters import estimate_lines, format_pass_rate_table
from counterfact.cli.helpers import first_step_for_decision_type
from counterfact.cli.loaders import demo_runs_dir, load_trace_dir, synthetic_runs
from counterfact.intervene.degenerate import degenerate_estimate, outcome_classes
from counterfact.intervene.estimate import CausalEstimate
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


def _resolve_demo_arms(
    args: argparse.Namespace, runs: list[Run], decision_type: str
) -> tuple[str, str | None]:
    """Return (target, baseline); baseline is None when only one arm was observed."""
    observed = [row.arm for row in pass_rate_by_arm(runs, decision_type).rows]
    if args.target is not None:
        target = str(args.target)
    elif args.confound and decision_type == "model_call":
        target = "sonnet"
    else:
        arm = first_observed_arm(runs, decision_type)
        if arm is None:
            raise ValueError(f"no chosen_action found for {decision_type!r}")
        target = arm
    return target, next((arm for arm in observed if arm != target), None)


def _demo_estimate(
    *,
    args: argparse.Namespace,
    runs: list[Run],
    decision_type: str,
    intervention_kind: str,
    target: str,
    baseline: str | None,
) -> CausalEstimate:
    from counterfact import fit_outcome_model, intervene
    from counterfact.dag import build_dag

    if len(outcome_classes(runs)) == 1:
        return degenerate_estimate(
            runs,
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
        )

    if baseline is None:
        raise ValueError(
            f"only one observed {decision_type} arm ({target!r}); a contrast needs two"
        )
    run, step = first_step_for_decision_type(runs, decision_type)
    return intervene(
        dag=build_dag(run),
        model=fit_outcome_model(runs, seed=args.seed),
        step=step,
        intervention={intervention_kind: target},
        baseline=baseline,
    )


def _print_contrast(
    *,
    args: argparse.Namespace,
    decision_type: str,
    estimate: CausalEstimate,
    runs: list[Run],
    target: str,
    baseline: str | None,
) -> None:
    """Print the naive-vs-causal gap when the confounded showcase can compute one."""
    if not args.confound or estimate.outcome_delta is None or baseline is None:
        return
    rates = {row.arm: row.pass_rate for row in pass_rate_by_arm(runs, decision_type).rows}
    naive_gap = rates[target] - rates[baseline]
    causal_gap = estimate.outcome_delta.point
    if abs(naive_gap - causal_gap) >= DEMO_CONTRAST_THRESHOLD:
        print(DEMO_CONTRAST_TEMPLATE.format(naive=naive_gap, causal=causal_gap))


def run(args: argparse.Namespace) -> int:
    from counterfact.outcome.binary import binary_outcome_value

    loaded = _load_demo_runs(args)
    if loaded is None:
        return 2
    runs, source = loaded

    decision_type = args.decision_type
    intervention_kind = default_intervention_kind(decision_type)
    try:
        target, baseline = _resolve_demo_arms(args, runs, decision_type)
    except ValueError as exc:
        print(f"counterfact demo: {exc}", file=sys.stderr)
        return 2

    try:
        estimate = _demo_estimate(
            args=args,
            runs=runs,
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
            baseline=baseline,
        )
    except ValueError as exc:
        print(f"counterfact demo: {exc}", file=sys.stderr)
        return 2

    pass_count = sum(1 for run in runs if binary_outcome_value(run))
    print("counterfact demo: naive vs honest")
    print(f"data: {source}")
    print(f"outcomes: {pass_count} pass / {len(runs) - pass_count} fail")
    print()
    print("\n".join(format_pass_rate_table(runs, decision_type)))
    print()

    contrast = f"{baseline} -> {target}" if baseline is not None else f"-> {target}"
    print(f"intervene({decision_type}: {contrast})")
    print("\n".join(estimate_lines(estimate)))
    print(f"next_step: {estimate.next_step.action} - {estimate.next_step.human_text}")
    suggested = estimate.next_step.payload.get("suggested_command")
    if suggested:
        print(f"suggested_command: {suggested}")

    _print_contrast(
        args=args,
        decision_type=decision_type,
        estimate=estimate,
        runs=runs,
        target=target,
        baseline=baseline,
    )
    return 0
