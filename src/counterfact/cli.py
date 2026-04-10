"""`counterfact` CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from counterfact.intervene.estimate import (
    CausalEstimate,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
)
from counterfact.schema import Run


def _load_trace_dir(path: Path) -> list[Run]:
    if not path.exists():
        return []
    return [Run.model_validate_json(p.read_text()) for p in sorted(path.glob("*.json"))]


def _synthetic_runs(n: int, seed: int) -> list[Run]:
    from bench.synthetic import generate_traces

    return [Run.model_validate(trace) for trace in generate_traces(n=n, seed=seed)]


def _outcome_classes(runs: list[Run]) -> set[bool]:
    return {bool(run.outcome.value) for run in runs}


def _first_step_for_decision_type(runs: list[Run], decision_type: str) -> tuple[Run, int]:
    for run in runs:
        for step in run.steps:
            if len(step.decisions) != 1:
                continue
            if step.decisions[0].decision_type == decision_type:
                return run, step.step_index
    raise ValueError(f"no single-decision step found for {decision_type!r}")


def _first_arm(runs: list[Run], decision_type: str) -> str:
    for run in runs:
        for step in run.steps:
            for decision in step.decisions:
                if decision.decision_type == decision_type and decision.chosen_action:
                    return decision.chosen_action
    raise ValueError(f"no chosen_action found for {decision_type!r}")


def _intervention_kind(decision_type: str) -> str:
    return {
        "model_call": "model_choice",
        "tool_call": "tool_choice",
        "retry": "retry_policy",
    }[decision_type]


def _degenerate_estimate(
    runs: list[Run], *, decision_type: str, intervention_kind: str, target: Any
) -> CausalEstimate:
    from counterfact import pass_rate_by_arm
    from counterfact.intervene.suggest import known_arms, suggest_harness_command

    classes = _outcome_classes(runs)
    if len(classes) != 1:
        raise ValueError("degenerate estimate requires exactly one outcome class")
    observed = next(iter(classes))

    table = pass_rate_by_arm(runs, decision_type)
    observed_arms = [row.model_dump() for row in table.rows]
    observed_arm_names = [row.arm for row in table.rows]
    canonical = known_arms(decision_type, intervention_kind)
    missing_arms = [arm for arm in canonical if arm not in observed_arm_names]

    suggestion = suggest_harness_command(
        decision_type=decision_type,
        intervention_kind=intervention_kind,
        action="broaden_arm_support",
        arm_name=str(target) if target is not None else None,
    )

    payload: dict[str, Any] = {
        "arm_name": "outcome",
        "missing_strata": [f"Outcome.value={not observed}"],
        "observed_arms": observed_arms,
        "missing_arms": missing_arms,
    }
    if suggestion is not None:
        payload["suggested_command"] = suggestion

    return CausalEstimate(
        query=InterventionQuery(
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
            step=-1,
        ),
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason=(
            "real corpus is causally degenerate: every trace has "
            f"Outcome.value={observed}; no outcome variation exists for an outcome "
            "model or back-door adjustment to leverage"
        ),
        warnings=[
            "fit_outcome_model is intentionally skipped for single-class real corpora"
        ],
        next_step=NextStep(
            action="broaden_arm_support",
            payload=payload,
            human_text=(
                "Collect or construct traces with both pass and fail outcomes before "
                "estimating decision-level effects on the real corpus."
            ),
        ),
    )


def _format_pass_rate_table(runs: list[Run], decision_type: str) -> list[str]:
    from counterfact import pass_rate_by_arm

    table = pass_rate_by_arm(runs, decision_type)
    lines = [f"pass_rate_by_arm({decision_type})", "arm              n  pass  rate    95% CI"]
    if not table.rows:
        lines.append("(no observed arms)")
        return lines
    for row in table.rows:
        lines.append(
            f"{row.arm:<14} {row.n:>3} {row.pass_count:>5} "
            f"{row.pass_rate:>5.3f}  [{row.ci_low:>5.3f}, {row.ci_high:>5.3f}]"
        )
    return lines


def _demo(args: argparse.Namespace) -> int:
    from counterfact import fit_outcome_model, intervene
    from counterfact.dag import build_dag

    runs = _load_trace_dir(args.runs_dir)
    source = str(args.runs_dir)
    if not runs:
        runs = _synthetic_runs(n=args.synthetic_n, seed=args.seed)
        source = f"synthetic SCM (n={args.synthetic_n}, seed={args.seed})"

    decision_type = args.decision_type
    intervention_kind = _intervention_kind(decision_type)
    target = args.target or _first_arm(runs, decision_type)

    pass_count = sum(1 for run in runs if bool(run.outcome.value))
    print("counterfact demo: naive vs honest")
    print(f"data: {source}")
    print(f"outcomes: {pass_count} pass / {len(runs) - pass_count} fail")
    print()
    print("\n".join(_format_pass_rate_table(runs, decision_type)))
    print()

    if len(_outcome_classes(runs)) == 1:
        estimate = _degenerate_estimate(
            runs,
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
        )
    else:
        run, step = _first_step_for_decision_type(runs, decision_type)
        model = fit_outcome_model(runs, n_bootstrap=args.bootstrap, seed=args.seed)
        estimate = intervene(
            dag=build_dag(run),
            model=model,
            step=step,
            intervention={intervention_kind: target},
        )

    print(f"intervene({decision_type} -> {target})")
    print(f"identifiability: {estimate.identifiability.value}")
    if estimate.outcome_delta is not None:
        delta = estimate.outcome_delta
        print(
            "outcome_delta: "
            f"{delta.point:.3f} [{delta.ci_low:.3f}, {delta.ci_high:.3f}]"
        )
    if estimate.reason:
        print(f"reason: {estimate.reason}")
    if estimate.warnings:
        print(f"warning: {estimate.warnings[0]}")
    print(f"next_step: {estimate.next_step.action} - {estimate.next_step.human_text}")
    suggested = estimate.next_step.payload.get("suggested_command")
    if suggested:
        print(f"suggested_command: {suggested}")
    return 0


def _bench_synthetic(args: argparse.Namespace) -> int:
    from bench.synthetic.generate import generate_corpus

    out = generate_corpus(n=args.n, seed=args.seed, output_dir=args.output_dir)
    print(f"Wrote {args.n} synthetic traces to {out}")
    return 0


def _bench_real(args: argparse.Namespace) -> int:
    from bench.real.coding_agent.agent import AgentRunConfig
    from bench.real.coding_agent.runner import run_real_corpus

    config = AgentRunConfig(
        seed=args.seed,
        epsilon=args.epsilon,
        tool_greedy=args.tool_greedy,
        tool_epsilon=args.tool_epsilon,
        model_greedy=args.model_greedy,
        model_epsilon=args.model_epsilon,
        retry_greedy=args.retry_greedy,
        retry_epsilon=args.retry_epsilon,
    )
    fixture_ids = (
        tuple(s.strip() for s in args.fixtures.split(",") if s.strip())
        if args.fixtures
        else None
    )
    return run_real_corpus(
        n=args.n,
        budget_cap_usd=args.budget_cap,
        output_dir=args.output_dir,
        config=config,
        fixture_ids=fixture_ids,
        fixture_set=args.fixture_set,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="counterfact", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "demo",
        help="Print a local naive-vs-honest causal demo without LLM calls",
    )
    demo.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("bench/real/runs_v1"),
        help="Directory of committed real traces (default: bench/real/runs_v1)",
    )
    demo.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
        help="Decision type to summarize (default: model_call)",
    )
    demo.add_argument("--target", default=None, help="Optional intervention arm")
    demo.add_argument("--synthetic-n", type=int, default=500)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--bootstrap", type=int, default=200)
    demo.set_defaults(func=_demo)

    bench = sub.add_parser("bench", help="Generate CounterBench corpora")
    bench_sub = bench.add_subparsers(dest="bench_kind", required=True)

    syn = bench_sub.add_parser(
        "synthetic", help="Generate synthetic SCM traces (no LLM, deterministic)"
    )
    syn.add_argument("--n", type=int, required=True, help="Number of traces to generate")
    syn.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    syn.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench/synthetic/_out"),
        help="Where to write traces (default: bench/synthetic/_out)",
    )
    syn.set_defaults(func=_bench_synthetic)

    real = bench_sub.add_parser(
        "real", help="Generate real-agent traces (HUMAN GATE on first run)"
    )
    real.add_argument("--n", type=int, required=True)
    real.add_argument("--budget-cap", type=float, default=50.0)
    real.add_argument("--output-dir", type=Path, default=Path("bench/real/runs"))
    real.add_argument("--seed", type=int, default=0, help="Per-trace RNG seed (default: 0)")
    real.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Default ε used for any decision whose --*-epsilon is not set (default: 0.2)",
    )
    real.add_argument("--tool-greedy", type=str, default="inspect_file")
    real.add_argument("--tool-epsilon", type=float, default=None)
    real.add_argument(
        "--model-greedy",
        type=str,
        default="large",
        choices=["small", "large"],
        help="Greedy arm for model_choice (default: large)",
    )
    real.add_argument(
        "--model-epsilon",
        type=float,
        default=None,
        help="ε for model_choice; falls back to --epsilon when unset",
    )
    real.add_argument(
        "--retry-greedy",
        type=str,
        default="retry_once",
        choices=["no_retry", "retry_once"],
        help="Greedy arm for retry_policy (default: retry_once)",
    )
    real.add_argument(
        "--retry-epsilon",
        type=float,
        default=None,
        help="ε for retry_policy; falls back to --epsilon when unset",
    )
    real.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help=(
            "Comma-separated fixture ids to iterate over (e.g. 'csv_dedupe' "
            "or 'csv_dedupe,date_window'). Overrides --fixture-set."
        ),
    )
    real.add_argument(
        "--fixture-set",
        type=str,
        default=None,
        choices=["v0", "easy", "hidden_v1"],
        help=(
            "Named fixture-set shortcut. 'v0' is the original hard fixtures "
            "(default behavior), 'easy' is the original easy fixtures, "
            "'hidden_v1' is the public/hidden split fixture set."
        ),
    )
    real.set_defaults(func=_bench_real)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
