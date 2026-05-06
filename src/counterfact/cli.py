"""`counterfact` CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from counterfact.intervene.degenerate import (
    degenerate_estimate as _shared_degenerate_estimate,
)
from counterfact.intervene.degenerate import outcome_classes as _shared_outcome_classes
from counterfact.intervene.estimate import CausalEstimate
from counterfact.schema import Run


def _load_trace_dir(path: Path) -> list[Run]:
    if not path.exists():
        return []
    return [Run.model_validate_json(p.read_text()) for p in sorted(path.glob("*.json"))]


def _positive_int(raw: str) -> int:
    """argparse type for ints that must be >= 1.

    Used by every flag that drives bootstrap counts — passing 0 would have
    `intervene` percentile over an empty array and a negative value would
    crash NumPy at allocation.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1; got {value}")
    return value


def _synthetic_runs(n: int, seed: int, confound: bool = False) -> list[Run]:
    from bench.synthetic import generate_traces

    return [
        Run.model_validate(trace)
        for trace in generate_traces(n=n, seed=seed, confound=confound)
    ]


# Demo's naive-vs-causal contrast threshold and one-line template. Centralized
# so future tuning is one edit. The contrast line is sourced from printed
# numbers — no editorial copy beyond named values and this fixed template.
_DEMO_CONTRAST_THRESHOLD = 0.05
_DEMO_CONTRAST_TEMPLATE = (
    "naive_vs_causal_contrast: naive arm gap = {naive:+.3f}; "
    "causal arm gap (do-calculus, g-formula) = {causal:+.3f}; "
    "the marginal table overstates what the corpus supports — see "
    "DAG and assumptions."
)


def _outcome_classes(runs: list[Run]) -> set[bool]:
    return _shared_outcome_classes(runs)


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
    return _shared_degenerate_estimate(
        runs,
        decision_type=decision_type,
        intervention_kind=intervention_kind,
        target=target,
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
    from counterfact import fit_outcome_model, intervene, pass_rate_by_arm
    from counterfact.dag import build_dag

    if args.confound:
        runs = _synthetic_runs(n=args.synthetic_n, seed=args.seed, confound=True)
        source = (
            f"synthetic SCM (confounded, n={args.synthetic_n}, seed={args.seed})"
        )
    else:
        runs = _load_trace_dir(args.runs_dir)
        source = str(args.runs_dir)
        if not runs:
            runs = _synthetic_runs(n=args.synthetic_n, seed=args.seed)
            source = f"synthetic SCM (n={args.synthetic_n}, seed={args.seed})"

    decision_type = args.decision_type
    intervention_kind = _intervention_kind(decision_type)
    if args.target is not None:
        target = args.target
    elif args.confound and decision_type == "model_call":
        # Confounded showcase: sonnet is the headline arm (the one that looks
        # most inflated by the confounded marginal table).
        target = "sonnet"
    else:
        target = _first_arm(runs, decision_type)

    pass_count = sum(1 for run in runs if bool(run.outcome.value))
    print("counterfact demo: naive vs honest")
    print(f"data: {source}")
    print(f"outcomes: {pass_count} pass / {len(runs) - pass_count} fail")
    print()
    print("\n".join(_format_pass_rate_table(runs, decision_type)))
    print()

    model = None
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

    # Confounded showcase: when both arms have observed support, pair the
    # focal arm with its sibling and surface the naive-vs-causal contrast.
    if (
        args.confound
        and decision_type == "model_call"
        and estimate.outcome_delta is not None
        and model is not None
    ):
        table = pass_rate_by_arm(runs, decision_type)
        rates = {row.arm: row.pass_rate for row in table.rows}
        sibling = "haiku" if target == "sonnet" else "sonnet"
        if target in rates and sibling in rates:
            run_for_intervene, step = _first_step_for_decision_type(runs, decision_type)
            sibling_estimate = intervene(
                dag=build_dag(run_for_intervene),
                model=model,
                step=step,
                intervention={intervention_kind: sibling},
            )
            if sibling_estimate.outcome_delta is not None:
                naive_gap = rates[target] - rates[sibling]
                causal_gap = (
                    estimate.outcome_delta.point
                    - sibling_estimate.outcome_delta.point
                )
                if abs(naive_gap - causal_gap) >= _DEMO_CONTRAST_THRESHOLD:
                    print(
                        _DEMO_CONTRAST_TEMPLATE.format(
                            naive=naive_gap, causal=causal_gap
                        )
                    )
    return 0


def _bench_synthetic(args: argparse.Namespace) -> int:
    from bench.synthetic.generate import generate_corpus

    out = generate_corpus(n=args.n, seed=args.seed, output_dir=args.output_dir)
    print(f"Wrote {args.n} synthetic traces to {out}")
    return 0


def _format_report(report: Any, runs_dir: Path) -> str:
    """Plain-text rendering of a CorpusReadinessReport. Stable enough to grep."""
    from counterfact.corpus_analyzer import CorpusReadinessReport

    assert isinstance(report, CorpusReadinessReport)
    lines: list[str] = [
        f"counterfact analyze corpus: {runs_dir}",
        f"n_traces: {report.n_traces}",
        (
            f"outcome_balance: pass={report.outcome_balance.n_pass} "
            f"fail={report.outcome_balance.n_fail} "
            f"pass_rate={report.outcome_balance.pass_rate:.3f}"
        ),
    ]
    if report.arm_support:
        lines.append("arm_support:")
        lines.append("  decision_type    arm                  n  pass  rate")
        for row in report.arm_support:
            lines.append(
                f"  {row.decision_type:<14} {row.arm:<18} {row.n:>4} {row.pass_count:>5} "
                f"{row.pass_rate:>5.3f}"
            )
    else:
        lines.append("arm_support: (no observed arms on randomized decision types)")
    cov = report.identifiability_coverage
    reachable_str = ",".join(cov.reachable) if cov.reachable else "(none)"
    lines.append(
        f"identifiability_coverage: reachable={reachable_str} "
        f"unfittable_outcome_model={cov.unfittable_outcome_model}"
    )
    for c in report.criteria:
        prefix = "PASS" if c.passed else "FAIL"
        lines.append(f"{prefix} {c.reason}")
    lines.append(f"promote: {report.promote}")
    return "\n".join(lines)


def _analyze_corpus(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from counterfact.corpus_analyzer import RubricThresholds, analyze

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists() or not runs_dir.is_dir():
        print(
            f"counterfact analyze corpus: directory not found: {runs_dir}",
            file=sys.stderr,
        )
        return 2

    runs: list[Run] = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            runs.append(Run.model_validate_json(path.read_text()))
        except (ValidationError, ValueError) as exc:
            print(
                f"counterfact analyze corpus: failed to parse {path}: {exc}",
                file=sys.stderr,
            )
            return 2

    overrides: dict[str, Any] = {}
    if args.min_pass_rate is not None:
        overrides["min_pass_rate"] = args.min_pass_rate
    if args.max_pass_rate is not None:
        overrides["max_pass_rate"] = args.max_pass_rate
    if args.min_arms is not None:
        overrides["min_arms_per_decision_type"] = args.min_arms
    if args.min_n_per_arm is not None:
        overrides["min_n_per_arm"] = args.min_n_per_arm
    if args.min_identified is not None:
        overrides["min_identified_decision_types"] = args.min_identified
    thresholds = RubricThresholds(**overrides) if overrides else RubricThresholds()

    report = analyze(runs, thresholds=thresholds)
    print(_format_report(report, runs_dir))
    return 0 if report.promote else 1


def _explain(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from counterfact.explain import build_report, render_html

    run_path: Path = args.run_json
    if not run_path.exists() or not run_path.is_file():
        print(
            f"counterfact explain: run JSON not found: {run_path}",
            file=sys.stderr,
        )
        return 2

    try:
        focal = Run.model_validate_json(run_path.read_text())
    except (ValidationError, ValueError) as exc:
        print(
            f"counterfact explain: failed to parse {run_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    runs_dir: Path = args.runs_dir if args.runs_dir is not None else run_path.parent
    if not runs_dir.exists() or not runs_dir.is_dir():
        print(
            f"counterfact explain: corpus directory not found: {runs_dir}",
            file=sys.stderr,
        )
        return 2

    corpus: list[Run] = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            corpus.append(Run.model_validate_json(path.read_text()))
        except (ValidationError, ValueError) as exc:
            print(
                f"counterfact explain: failed to parse {path}: {exc}",
                file=sys.stderr,
            )
            return 2

    if focal.run_id not in {r.run_id for r in corpus}:
        print(
            f"counterfact explain: focal run_id={focal.run_id!r} not found in {runs_dir}",
            file=sys.stderr,
        )
        return 2

    output: Path = args.output if args.output is not None else (
        run_path.parent / f"counterfact-explain-{focal.run_id}.html"
    )

    report = build_report(
        focal,
        corpus,
        decision_type=args.decision_type,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    html = render_html(report)
    output.write_text(html)
    print(str(output.resolve()))
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
        default=Path("bench/real/smoke_mixed_outcome"),
        help="Directory of committed real traces (default: bench/real/smoke_mixed_outcome)",
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
    demo.add_argument("--bootstrap", type=_positive_int, default=200)
    demo.add_argument(
        "--confound",
        action="store_true",
        help=(
            "Run the confounded synthetic showcase: generate a fresh "
            "synthetic corpus where model_choice is biased by tool_choice, "
            "and surface the naive-vs-causal contrast."
        ),
    )
    demo.set_defaults(func=_demo)

    explain = sub.add_parser(
        "explain",
        help=(
            "Render a self-contained HTML report explaining one trace, "
            "grounded in CausalEstimate"
        ),
    )
    explain.add_argument(
        "run_json",
        type=Path,
        help="Path to a single Run JSON file (the focal trace)",
    )
    explain.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Corpus directory (defaults to the parent directory of run_json). "
            "Must contain the focal run."
        ),
    )
    explain.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
        help="Decision type to summarize (default: model_call)",
    )
    explain.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output HTML path (default: "
            "<run-json-parent>/counterfact-explain-<run_id>.html)"
        ),
    )
    explain.add_argument("--bootstrap", type=_positive_int, default=200)
    explain.add_argument("--seed", type=int, default=42)
    explain.set_defaults(func=_explain)

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
    real.add_argument("--output-dir", type=Path, default=Path("bench/real/pilot"))
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
        choices=[
            "v0",
            "easy",
            "hidden_v1",
            "hard_hidden_v1",
            "broad_calibration",
            "very_hard_hidden_v1",
            "stateful_calibration",
        ],
        help=(
            "Named fixture-set shortcut. Use 'broad_calibration' for broad "
            "date/rate-limit/version calibration, or 'stateful_calibration' "
            "for the streaming watermark fixture. Other choices are legacy "
            "harness fixtures kept for tests and historical calibration."
        ),
    )
    real.set_defaults(func=_bench_real)

    analyze = sub.add_parser(
        "analyze", help="Score a candidate corpus against the promotion rubric"
    )
    analyze_sub = analyze.add_subparsers(dest="analyze_kind", required=True)
    corpus = analyze_sub.add_parser(
        "corpus",
        help="Run the corpus-readiness analyzer on a directory of trace JSON files",
    )
    corpus.add_argument("runs_dir", type=Path, help="Directory of trace JSON files")
    corpus.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="Override RubricThresholds.min_pass_rate (default: 0.3)",
    )
    corpus.add_argument(
        "--max-pass-rate",
        type=float,
        default=None,
        help="Override RubricThresholds.max_pass_rate (default: 0.7)",
    )
    corpus.add_argument(
        "--min-arms",
        type=int,
        default=None,
        help="Override RubricThresholds.min_arms_per_decision_type (default: 2)",
    )
    corpus.add_argument(
        "--min-n-per-arm",
        type=int,
        default=None,
        help="Override RubricThresholds.min_n_per_arm (default: 5)",
    )
    corpus.add_argument(
        "--min-identified",
        type=int,
        default=None,
        help="Override RubricThresholds.min_identified_decision_types (default: 1)",
    )
    corpus.set_defaults(func=_analyze_corpus)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
