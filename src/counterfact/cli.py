"""`counterfact` CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from counterfact.compare import TraceComparison
from counterfact.corpus_analyzer import CorpusReadinessReport
from counterfact.diagnose import DiagnosisReport
from counterfact.errors import InvalidInterventionError
from counterfact.intervene.degenerate import degenerate_estimate, outcome_classes
from counterfact.intervene.estimate import CausalEstimate
from counterfact.schema import Decision, Run, Step

_DEFAULT_DEMO_RUNS_DIR = Path("bench/real/smoke_mixed_outcome")


def _load_trace_dir(path: Path, *, command: str) -> list[Run] | None:
    if not path.exists():
        return []
    if not path.is_dir():
        print(f"counterfact {command}: trace directory not found: {path}", file=sys.stderr)
        return None
    return _load_corpus_dir(path, command=command)


def _load_run_file(path: Path, *, command: str) -> Run | None:
    if not path.exists() or not path.is_file():
        print(f"counterfact {command}: run JSON not found: {path}", file=sys.stderr)
        return None
    try:
        return Run.model_validate_json(path.read_text())
    except (OSError, ValidationError, ValueError) as exc:
        print(f"counterfact {command}: failed to parse {path}: {exc}", file=sys.stderr)
        return None


def _load_corpus_dir(path: Path, *, command: str) -> list[Run] | None:
    if not path.exists() or not path.is_dir():
        print(f"counterfact {command}: corpus directory not found: {path}", file=sys.stderr)
        return None
    corpus: list[Run] = []
    for trace_path in sorted(path.glob("*.json")):
        if trace_path.name.endswith("receipt.json"):
            continue
        try:
            corpus.append(Run.model_validate_json(trace_path.read_text()))
        except (OSError, ValidationError, ValueError) as exc:
            print(
                f"counterfact {command}: failed to parse {trace_path}: {exc}",
                file=sys.stderr,
            )
            return None
    return corpus


def _require_focal_in_corpus(
    focal: Run, corpus: list[Run], runs_dir: Path, *, command: str
) -> bool:
    if focal.run_id in {run.run_id for run in corpus}:
        return True
    print(
        f"counterfact {command}: focal run_id={focal.run_id!r} not found in {runs_dir}",
        file=sys.stderr,
    )
    return False


def _positive_int(raw: str) -> int:
    """argparse type for ints that must be >= 1 (bootstrap counts)."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1; got {value}")
    return value


_BENCH_UNAVAILABLE_MESSAGE = (
    "counterfact bench: the bench harness is not included in the wheel. "
    "Install the development extras with `pip install counterfact[bench]` "
    'or use an editable dev install (`uv pip install -e ".[dev]"`).'
)


def _synthetic_runs(n: int, seed: int, confound: bool = False) -> list[Run]:
    try:
        from bench.synthetic import generate_traces
    except ImportError as exc:
        raise ImportError(_BENCH_UNAVAILABLE_MESSAGE) from exc

    return [
        Run.model_validate(trace) for trace in generate_traces(n=n, seed=seed, confound=confound)
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _demo_runs_dir(path: Path) -> tuple[Path, str]:
    if path.exists() or path != _DEFAULT_DEMO_RUNS_DIR:
        return path, str(path)
    repo_path = _repo_root() / _DEFAULT_DEMO_RUNS_DIR
    if repo_path.exists():
        return repo_path, _DEFAULT_DEMO_RUNS_DIR.as_posix()
    return path, str(path)


_DEMO_CONTRAST_THRESHOLD = 0.05
_DEMO_CONTRAST_TEMPLATE = (
    "naive_vs_causal_contrast: naive arm gap = {naive:+.3f}; "
    "causal arm gap (do-calculus, g-formula) = {causal:+.3f}; "
    "the marginal table overstates what the corpus supports — see "
    "DAG and assumptions."
)


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


def _decision_by_id(run: Run, decision_id: str) -> tuple[Step, Decision] | None:
    for step in run.steps:
        for decision in step.decisions:
            if decision.decision_id == decision_id:
                return step, decision
    return None


def _resolve_intervention_target(
    args: argparse.Namespace, focal: Run
) -> tuple[Step, Decision] | None:
    if args.decision_id is not None and args.step is not None:
        print(
            "counterfact intervene: only one targeting mode is allowed: --decision-id or --step",
            file=sys.stderr,
        )
        return None
    if args.decision_id is None and args.step is None:
        print(
            "counterfact intervene: specify --decision-id or --step",
            file=sys.stderr,
        )
        return None
    if args.decision_id is not None:
        resolved = _decision_by_id(focal, args.decision_id)
        if resolved is None:
            print(
                f"counterfact intervene: decision_id not found: {args.decision_id}",
                file=sys.stderr,
            )
            return None
        return resolved

    for step in focal.steps:
        if step.step_index != args.step:
            continue
        if not step.decisions:
            print(
                f"counterfact intervene: step {args.step} has no decisions",
                file=sys.stderr,
            )
            return None
        if len(step.decisions) > 1:
            ids = ", ".join(decision.decision_id for decision in step.decisions)
            print(
                f"counterfact intervene: step {args.step} has multiple decisions "
                f"({ids}); rerun with --decision-id",
                file=sys.stderr,
            )
            return None
        return step, step.decisions[0]

    print(f"counterfact intervene: step not found: {args.step}", file=sys.stderr)
    return None


def _parse_decision_edit(raw: str | None) -> tuple[str, str] | None:
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


def _add_cli_diagnostics(
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


def _format_intervention_estimate(
    *,
    estimate: CausalEstimate,
    run: Run,
    decision: Decision,
    step: Step,
    intervention_kind: str,
    target: str,
) -> str:
    lines = [
        f"counterfact intervene: {run.run_id}",
        (
            f"decision: {decision.decision_id} step={step.step_index} "
            f"type={decision.decision_type} chosen={decision.chosen_action}"
        ),
        f"edit: {intervention_kind}={target}",
        f"identifiability: {estimate.identifiability.value}",
    ]
    if estimate.outcome_delta is not None:
        delta = estimate.outcome_delta
        lines.append(f"outcome_delta: {delta.point:.3f} [{delta.ci_low:.3f}, {delta.ci_high:.3f}]")
    if estimate.reason:
        lines.append(f"reason: {estimate.reason}")
    if estimate.warnings:
        lines.append(f"warning: {estimate.warnings[0]}")
    missing_arms = estimate.next_step.payload.get("missing_arms")
    if missing_arms:
        lines.append(f"missing_arms: {', '.join(str(arm) for arm in missing_arms)}")
    localization_limit = estimate.next_step.payload.get("localization_limit")
    if localization_limit:
        lines.append(f"localization_limit: {localization_limit}")
    lines.append(f"next_step: {estimate.next_step.action} - {estimate.next_step.human_text}")
    return "\n".join(lines)


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
    from counterfact.outcome.binary import binary_outcome_value

    try:
        if args.confound:
            runs = _synthetic_runs(n=args.synthetic_n, seed=args.seed, confound=True)
            source = f"synthetic SCM (confounded, n={args.synthetic_n}, seed={args.seed})"
        else:
            runs_dir, source = _demo_runs_dir(args.runs_dir)
            _runs = _load_trace_dir(runs_dir, command="demo")
            if _runs is None:
                return 2
            runs = _runs
            if not runs:
                if not args.synthetic_fallback:
                    print(
                        "counterfact demo: no real traces found at "
                        f"{runs_dir}; pass --confound for the synthetic showcase or "
                        "--synthetic-fallback to opt into synthetic data.",
                        file=sys.stderr,
                    )
                    return 2
                runs = _synthetic_runs(n=args.synthetic_n, seed=args.seed)
                source = f"synthetic SCM (n={args.synthetic_n}, seed={args.seed})"
    except ImportError as exc:
        print(f"counterfact demo: {exc}", file=sys.stderr)
        return 2

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

    pass_count = sum(1 for run in runs if binary_outcome_value(run))
    print("counterfact demo: naive vs honest")
    print(f"data: {source}")
    print(f"outcomes: {pass_count} pass / {len(runs) - pass_count} fail")
    print()
    print("\n".join(_format_pass_rate_table(runs, decision_type)))
    print()

    model = None
    if len(outcome_classes(runs)) == 1:
        estimate = degenerate_estimate(
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
        print(f"outcome_delta: {delta.point:.3f} [{delta.ci_low:.3f}, {delta.ci_high:.3f}]")
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
                causal_gap = estimate.outcome_delta.point - sibling_estimate.outcome_delta.point
                if abs(naive_gap - causal_gap) >= _DEMO_CONTRAST_THRESHOLD:
                    print(_DEMO_CONTRAST_TEMPLATE.format(naive=naive_gap, causal=causal_gap))
    return 0


def _bench_synthetic(args: argparse.Namespace) -> int:
    try:
        from bench.synthetic.generate import generate_corpus
    except ImportError:
        print(_BENCH_UNAVAILABLE_MESSAGE, file=sys.stderr)
        return 2

    try:
        out = generate_corpus(n=args.n, seed=args.seed, output_dir=args.output_dir)
    except ValueError as exc:
        print(f"counterfact bench synthetic: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.n} synthetic traces to {out}")
    return 0


def _format_report(report: CorpusReadinessReport, runs_dir: Path) -> str:
    """Plain-text rendering of a CorpusReadinessReport. Stable enough to grep."""
    from counterfact.intervene.suggest import suggest_harness_command

    lines: list[str] = [
        f"counterfact analyze corpus support-readiness: {runs_dir}",
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
    lines.append(f"support_ready: {report.promote}")
    if report.promote:
        lines.append(
            "support_readiness: suitable for counterfactual-support workflows; "
            "use diagnose/intervene for trace-level causal questions."
        )
    else:
        lines.append("next_collection_guidance:")
        for c in report.criteria:
            if c.passed:
                continue
            if c.name == "outcome_balance" or "unfittable" in c.reason:
                lines.append(
                    "  - outcome model is unfittable for counterfactual support "
                    "without mixed pass/fail outcomes; collect mixed-outcome traces."
                )
            elif c.name == "model_arm_outcome_mix":
                lines.append(
                    "  - collect model_call support with both small and large arms "
                    "and mixed outcomes per arm."
                )
                suggestion = suggest_harness_command(
                    decision_type="model_call",
                    intervention_kind="model_choice",
                    action="broaden_arm_support",
                    arm_name="large",
                )
                if suggestion:
                    lines.append(f"    suggested_command: {suggestion}")
            elif c.name == "arm_support":
                lines.append(
                    "  - broaden randomized arm support for the decision type named "
                    "in the failed criterion."
                )
            elif c.name == "identifiability_coverage":
                lines.append(
                    "  - add support for at least one identifiable decision type "
                    "before relying on diagnose/intervene output."
                )
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
    try:
        thresholds = RubricThresholds(**overrides) if overrides else RubricThresholds()
    except ValidationError as exc:
        print(f"counterfact analyze corpus: invalid thresholds: {exc}", file=sys.stderr)
        return 2

    report = analyze(runs, thresholds=thresholds)
    print(_format_report(report, runs_dir))
    return 0 if report.promote else 1


def _explain(args: argparse.Namespace) -> int:
    from counterfact.explain import build_report, render_html

    run_path: Path = args.run_json
    focal = _load_run_file(run_path, command="explain")
    if focal is None:
        return 2

    runs_dir: Path = args.runs_dir if args.runs_dir is not None else run_path.parent
    corpus = _load_corpus_dir(runs_dir, command="explain")
    if corpus is None:
        return 2
    if not _require_focal_in_corpus(focal, corpus, runs_dir, command="explain"):
        return 2

    output: Path = (
        args.output
        if args.output is not None
        else (run_path.parent / f"counterfact-explain-{focal.run_id}.html")
    )

    report = build_report(
        focal,
        corpus,
        decision_type=args.decision_type,
        bootstrap=args.bootstrap,
        seed=args.seed,
        run_path=str(run_path),
        corpus_dir=str(runs_dir),
    )
    html = render_html(report)
    output.write_text(html)
    print(str(output.resolve()))
    return 0


def _intervene_cli(args: argparse.Namespace) -> int:
    from counterfact import fit_outcome_model, intervene
    from counterfact.dag import build_dag
    from counterfact.taxonomy import is_valid_intervention

    run_path: Path = args.run_json
    focal = _load_run_file(run_path, command="intervene")
    if focal is None:
        return 2

    runs_dir: Path = args.runs_dir if args.runs_dir is not None else run_path.parent
    corpus = _load_corpus_dir(runs_dir, command="intervene")
    if corpus is None:
        return 2
    if not _require_focal_in_corpus(focal, corpus, runs_dir, command="intervene"):
        return 2

    target = _resolve_intervention_target(args, focal)
    if target is None:
        return 2
    step, decision = target

    parsed_edit = _parse_decision_edit(args.set_value)
    if parsed_edit is None:
        return 2
    intervention_kind, target_value = parsed_edit
    if not is_valid_intervention(decision.decision_type, intervention_kind):
        print(
            "counterfact intervene: intervention "
            f"{intervention_kind!r} is not valid on decision type "
            f"{decision.decision_type!r}",
            file=sys.stderr,
        )
        return 2

    try:
        if len(outcome_classes(corpus)) == 1:
            estimate = degenerate_estimate(
                corpus,
                decision_type=decision.decision_type,
                intervention_kind=intervention_kind,
                target=target_value,
            )
        else:
            model = fit_outcome_model(corpus, n_bootstrap=args.bootstrap, seed=args.seed)
            estimate = intervene(
                dag=build_dag(focal),
                model=model,
                step=step.step_index,
                intervention={intervention_kind: target_value},
                decision_id=decision.decision_id if args.decision_id is not None else None,
            )
    except InvalidInterventionError as exc:
        print(f"counterfact intervene: {exc}", file=sys.stderr)
        return 2
    estimate = _add_cli_diagnostics(
        estimate,
        decision=decision,
        step=step,
        targeting_mode="decision_id" if args.decision_id is not None else "step",
    )

    estimate_json = estimate.model_dump_json(indent=2)
    if args.output is not None:
        args.output.write_text(estimate_json + "\n")
        print(str(args.output.resolve()), file=sys.stderr)

    if args.json:
        print(estimate_json)
    else:
        print(
            _format_intervention_estimate(
                estimate=estimate,
                run=focal,
                decision=decision,
                step=step,
                intervention_kind=intervention_kind,
                target=target_value,
            )
        )
    return 0


def _format_diagnosis(report: DiagnosisReport) -> str:
    lines = [
        report.summary,
        f"run_id: {report.run_id}",
        f"outcome: {report.outcome}",
        f"corpus_size: {report.corpus_size}",
    ]
    if not report.entries:
        lines.append("ranked_decisions: (none)")
        return "\n".join(lines)
    lines.append("ranked_decisions:")
    for idx, entry in enumerate(report.entries, start=1):
        step = entry.step if entry.step is not None else "?"
        lines.append(
            f"  {idx}. decision_id={entry.decision_id} step={step} "
            f"type={entry.decision_type} chosen={entry.chosen_action} "
            f"identifiability={entry.identifiability.value}"
        )
        if entry.outcome_delta is not None:
            delta = entry.outcome_delta
            lines.append(
                f"     outcome_delta: {delta.point:.3f} [{delta.ci_low:.3f}, {delta.ci_high:.3f}]"
            )
        if entry.reason:
            lines.append(f"     reason: {entry.reason}")
        lines.append(f"     next_step: {entry.next_step.action} - {entry.next_step.human_text}")
    return "\n".join(lines)


def _diagnose_cli(args: argparse.Namespace) -> int:
    from counterfact.diagnose import build_diagnosis_pair
    from counterfact.explain import render_html

    run_path: Path = args.run_json
    focal = _load_run_file(run_path, command="diagnose")
    if focal is None:
        return 2

    runs_dir: Path = args.runs_dir if args.runs_dir is not None else run_path.parent
    corpus = _load_corpus_dir(runs_dir, command="diagnose")
    if corpus is None:
        return 2
    if not _require_focal_in_corpus(focal, corpus, runs_dir, command="diagnose"):
        return 2

    try:
        report, html_report = build_diagnosis_pair(
            focal,
            corpus,
            decision_type=args.decision_type,
            top_k=args.top_k,
            bootstrap=args.bootstrap,
            seed=args.seed,
            run_path=str(run_path),
            corpus_dir=str(runs_dir),
        )
    except ValueError as exc:
        print(f"counterfact diagnose: {exc}", file=sys.stderr)
        return 2

    if args.html is not None:
        try:
            args.html.write_text(render_html(html_report))
        except OSError as exc:
            print(
                f"counterfact diagnose: failed to write HTML {args.html}: {exc}",
                file=sys.stderr,
            )
            return 2
        if args.json:
            print(str(args.html.resolve()), file=sys.stderr)

    if args.json:
        print(report.model_dump_json(indent=2, exclude_none=True))
    else:
        print(_format_diagnosis(report))
        if args.html is not None:
            print(f"html_report: {args.html.resolve()}")
    return 0


def _format_comparison(comparison: TraceComparison) -> str:
    lines = [
        "counterfact compare: descriptive trace diff",
        (
            f"left: {comparison.left_run_id} outcome={comparison.left_outcome} "
            f"steps={comparison.left_step_count}"
        ),
        (
            f"right: {comparison.right_run_id} outcome={comparison.right_outcome} "
            f"steps={comparison.right_step_count}"
        ),
        f"note: {comparison.note}",
    ]
    if comparison.decision_diffs:
        lines.append("decision_diffs:")
        for diff in comparison.decision_diffs:
            step = diff.step if diff.step is not None else "?"
            lines.append(
                f"  step={step} type={diff.decision_type} "
                f"left={diff.left_chosen_action} ({diff.left_decision_id}) "
                f"right={diff.right_chosen_action} ({diff.right_decision_id})"
            )
    else:
        lines.append("decision_diffs: (none)")
    if comparison.step_diffs:
        lines.append("step_diffs:")
        for step_diff in comparison.step_diffs:
            lines.append(
                f"  step={step_diff.step} decisions "
                f"{step_diff.left_decision_count}->{step_diff.right_decision_count}; "
                f"observations "
                f"{step_diff.left_observation_count}->{step_diff.right_observation_count}"
            )
    if comparison.diagnosis is not None:
        lines.append("diagnosis_overlay:")
        for line in _format_diagnosis(comparison.diagnosis).splitlines():
            lines.append(f"  {line}")
    return "\n".join(lines)


def _compare_cli(args: argparse.Namespace) -> int:
    from counterfact.compare import compare_traces
    from counterfact.diagnose import build_diagnosis

    left = _load_run_file(args.left_run_json, command="compare")
    if left is None:
        return 2
    right = _load_run_file(args.right_run_json, command="compare")
    if right is None:
        return 2

    diagnosis = None
    if args.runs_dir is not None:
        corpus = _load_corpus_dir(args.runs_dir, command="compare")
        if corpus is None:
            return 2
        focal = left if args.focal == "left" else right
        if not _require_focal_in_corpus(focal, corpus, args.runs_dir, command="compare"):
            return 2
        diagnosis = build_diagnosis(
            focal,
            corpus,
            decision_type=args.decision_type,
            top_k=args.top_k,
            bootstrap=args.bootstrap,
            seed=args.seed,
            run_path=str(args.left_run_json if args.focal == "left" else args.right_run_json),
            corpus_dir=str(args.runs_dir),
        )

    comparison = compare_traces(left, right, diagnosis=diagnosis)
    if args.json:
        print(comparison.model_dump_json(indent=2, exclude_none=True))
    else:
        print(_format_comparison(comparison))
    return 0


def _ingest_generic_jsonl_cli(args: argparse.Namespace) -> int:
    from counterfact.ingest import IngestError, ingest_generic_jsonl

    try:
        receipt = ingest_generic_jsonl(args.source_jsonl, args.mapping, args.output_dir)
    except IngestError as exc:
        print(f"counterfact ingest generic-jsonl: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact ingest generic-jsonl: wrote {receipt.generated_count} trace(s)")
        print(str((args.output_dir / "ingest-receipt.json").resolve()))
        for warning in receipt.warnings:
            print(f"warning: {warning}")
    return 0


def _ingest_claude_agent_sdk_cli(args: argparse.Namespace) -> int:
    from counterfact.adapters._common import IngestError
    from counterfact.adapters.claude_agent_sdk import ingest_claude_agent_sdk

    try:
        receipt = ingest_claude_agent_sdk(args.source_jsonl, args.output_dir)
    except IngestError as exc:
        print(f"counterfact ingest claude-agent-sdk: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact ingest claude-agent-sdk: wrote {receipt.generated_count} trace(s)")
        print(str((args.output_dir / "ingest-receipt.json").resolve()))
        for warning in receipt.warnings:
            print(f"warning: {warning}")
    return 0


def _ingest_openai_agents_cli(args: argparse.Namespace) -> int:
    from counterfact.adapters._common import IngestError
    from counterfact.adapters.openai_agents import ingest_openai_agents

    outcome: bool | None
    if args.outcome is None:
        outcome = None
    elif args.outcome == "pass":
        outcome = True
    elif args.outcome == "fail":
        outcome = False
    else:
        print(
            "counterfact ingest openai-agents: --outcome must be 'pass' or 'fail'",
            file=sys.stderr,
        )
        return 2

    try:
        receipt = ingest_openai_agents(args.source_json, args.output_dir, outcome=outcome)
    except IngestError as exc:
        print(f"counterfact ingest openai-agents: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact ingest openai-agents: wrote {receipt.generated_count} trace(s)")
        print(str((args.output_dir / "ingest-receipt.json").resolve()))
        for warning in receipt.warnings:
            print(f"warning: {warning}")
    return 0


def _export_runs_cli(args: argparse.Namespace) -> int:
    from counterfact.runrecord_export import export_runrecord_parquet

    if args.to != "runrecord-parquet":
        print(f"counterfact export-runs: unsupported target {args.to!r}", file=sys.stderr)
        return 2
    corpus = _load_corpus_dir(args.runs_dir, command="export-runs")
    if corpus is None:
        return 2
    output = args.output or (args.runs_dir / "runrecord.parquet")
    receipt = export_runrecord_parquet(corpus, source_corpus=args.runs_dir, output_path=output)
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact export-runs: wrote {Path(receipt.output_path).resolve()}")
        receipt_path = Path(receipt.output_path).with_suffix(
            Path(receipt.output_path).suffix + ".receipt.json"
        )
        print(f"receipt: {receipt_path.resolve()}")
        print(f"rows: {receipt.row_count}")
        if receipt.warnings:
            print(f"warnings: {len(receipt.warnings)}")
    return 0


def _bench_real(args: argparse.Namespace) -> int:
    try:
        from bench.real.coding_agent.agent import AgentRunConfig
        from bench.real.coding_agent.runner import run_real_corpus
    except ImportError:
        print(_BENCH_UNAVAILABLE_MESSAGE, file=sys.stderr)
        return 2

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
        tuple(s.strip() for s in args.fixtures.split(",") if s.strip()) if args.fixtures else None
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
        default=_DEFAULT_DEMO_RUNS_DIR,
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
        "--synthetic-fallback",
        action="store_true",
        help="Use a synthetic SCM corpus if --runs-dir has no trace JSON files",
    )
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
            "Render a self-contained HTML report explaining one trace, grounded in CausalEstimate"
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
        help=("Output HTML path (default: <run-json-parent>/counterfact-explain-<run_id>.html)"),
    )
    explain.add_argument("--bootstrap", type=_positive_int, default=200)
    explain.add_argument("--seed", type=int, default=42)
    explain.set_defaults(func=_explain)

    intervene_parser = sub.add_parser(
        "intervene",
        help="Estimate one decision edit on a trace and emit a CausalEstimate",
    )
    intervene_parser.add_argument(
        "run_json",
        type=Path,
        help="Path to a single Run JSON file (the focal trace)",
    )
    intervene_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Trace corpus directory (defaults to the parent directory of run_json). "
            "Must contain the focal run."
        ),
    )
    intervene_parser.add_argument(
        "--decision-id",
        default=None,
        help="Target a specific Decision.decision_id",
    )
    intervene_parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Target a single-decision step by step_index",
    )
    intervene_parser.add_argument(
        "--set",
        dest="set_value",
        required=True,
        help="Decision edit as key=value, e.g. model_choice=sonnet",
    )
    intervene_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only CausalEstimate JSON to stdout",
    )
    intervene_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the CausalEstimate JSON artifact to this path",
    )
    intervene_parser.add_argument("--bootstrap", type=_positive_int, default=200)
    intervene_parser.add_argument("--seed", type=int, default=42)
    intervene_parser.set_defaults(func=_intervene_cli)

    diagnose = sub.add_parser(
        "diagnose",
        help="Rank likely load-bearing decisions for one trace",
    )
    diagnose.add_argument("run_json", type=Path, help="Path to the focal Run JSON")
    diagnose.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Trace corpus directory (defaults to the parent directory of run_json)",
    )
    diagnose.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
        help="Only rank decisions of this type",
    )
    diagnose.add_argument("--top-k", type=_positive_int, default=3)
    diagnose.add_argument("--bootstrap", type=_positive_int, default=200)
    diagnose.add_argument("--seed", type=int, default=42)
    diagnose.add_argument("--json", action="store_true", help="Emit JSON only")
    diagnose.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Write a self-contained diagnosis-first HTML report to this path",
    )
    diagnose.set_defaults(func=_diagnose_cli)

    compare = sub.add_parser(
        "compare",
        help="Compare two traces descriptively, with optional diagnosis overlay",
    )
    compare.add_argument("left_run_json", type=Path)
    compare.add_argument("right_run_json", type=Path)
    compare.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Optional corpus directory for a diagnosis overlay",
    )
    compare.add_argument(
        "--focal",
        choices=["left", "right"],
        default="right",
        help="Which trace to diagnose when --runs-dir is supplied",
    )
    compare.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
    )
    compare.add_argument("--top-k", type=_positive_int, default=3)
    compare.add_argument("--bootstrap", type=_positive_int, default=200)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--json", action="store_true", help="Emit JSON only")
    compare.set_defaults(func=_compare_cli)

    ingest = sub.add_parser("ingest", help="Convert external trace data to native Run JSON")
    ingest.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported source formats and exit",
    )
    ingest_sub = ingest.add_subparsers(dest="ingest_kind", required=False)
    generic_jsonl = ingest_sub.add_parser(
        "generic-jsonl",
        help="Convert JSONL records through an explicit mapping file",
    )
    generic_jsonl.add_argument("source_jsonl", type=Path)
    generic_jsonl.add_argument("--mapping", type=Path, required=True)
    generic_jsonl.add_argument("--output-dir", type=Path, required=True)
    generic_jsonl.add_argument("--json", action="store_true", help="Emit receipt JSON")
    generic_jsonl.set_defaults(func=_ingest_generic_jsonl_cli)

    claude_sdk = ingest_sub.add_parser(
        "claude-agent-sdk",
        help=(
            "Convert a JSONL stream of Claude Agent SDK message dataclass dumps "
            "to native Run JSON (zero-config — no mapping required)"
        ),
    )
    claude_sdk.add_argument(
        "source_jsonl",
        type=Path,
        help=(
            'JSONL where each line is either {"messages": [...]} or a JSON list '
            "of message dicts captured from claude_agent_sdk.query()"
        ),
    )
    claude_sdk.add_argument("--output-dir", type=Path, required=True)
    claude_sdk.add_argument("--json", action="store_true", help="Emit receipt JSON")
    claude_sdk.set_defaults(func=_ingest_claude_agent_sdk_cli)

    openai_agents = ingest_sub.add_parser(
        "openai-agents",
        help=(
            "Convert an OpenAI Agents SDK trace export (one JSON file with a flat "
            "spans array) to native Run JSON"
        ),
    )
    openai_agents.add_argument(
        "source_json",
        type=Path,
        help="Path to a JSON file containing one trace or a list of traces",
    )
    openai_agents.add_argument("--output-dir", type=Path, required=True)
    openai_agents.add_argument(
        "--outcome",
        choices=["pass", "fail"],
        default=None,
        help=(
            "Explicit binary outcome for traces without a counterfact.outcome "
            "marker span and without a root error. The adapter never infers "
            "outcomes from the absence of error."
        ),
    )
    openai_agents.add_argument("--json", action="store_true", help="Emit receipt JSON")
    openai_agents.set_defaults(func=_ingest_openai_agents_cli)

    export_runs = sub.add_parser(
        "export-runs",
        help="Export native traces to another research artifact format",
    )
    export_runs.add_argument("runs_dir", type=Path, help="Directory of native trace JSON")
    export_runs.add_argument("--to", required=True, choices=["runrecord-parquet"])
    export_runs.add_argument("--output", type=Path, default=None)
    export_runs.add_argument("--json", action="store_true", help="Emit receipt JSON")
    export_runs.set_defaults(func=_export_runs_cli)

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

    real = bench_sub.add_parser("real", help="Generate real-agent traces (HUMAN GATE on first run)")
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
        "analyze",
        help="Check corpus support-readiness for counterfactual diagnosis",
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


_INGEST_FORMATS = [
    ("claude-agent-sdk", "Claude Agent SDK message JSONL (zero-config)"),
    ("openai-agents", "OpenAI Agents SDK trace JSON (requires --outcome unless derivable)"),
    ("generic-jsonl", "Any JSONL with an explicit user-supplied --mapping file"),
]


def _print_ingest_formats() -> int:
    width = max(len(name) for name, _ in _INGEST_FORMATS)
    for name, description in _INGEST_FORMATS:
        print(f"  {name:<{width}}  {description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if getattr(ns, "command", None) == "ingest" and getattr(ns, "list_formats", False):
        return _print_ingest_formats()
    func = getattr(ns, "func", None)
    if func is None:
        parser.parse_args([ns.command, "--help"])  # exits via argparse
        return 2
    return int(func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
