"""Plain-text formatters for CLI output."""

from __future__ import annotations

from pathlib import Path

from counterfact.corpus_analyzer import CorpusReadinessReport
from counterfact.diagnose import DiagnosisReport
from counterfact.intervene.estimate import CausalEstimate
from counterfact.schema import Decision, Run, Step


def format_intervention_estimate(
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


def format_pass_rate_table(runs: list[Run], decision_type: str) -> list[str]:
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


def format_report(report: CorpusReadinessReport, runs_dir: Path) -> str:
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


def format_diagnosis(report: DiagnosisReport) -> str:
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


def format_comparison(comparison: object) -> str:
    from counterfact.compare import TraceComparison

    if not isinstance(comparison, TraceComparison):
        raise TypeError(f"expected TraceComparison, got {type(comparison)!r}")
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
        for line in format_diagnosis(comparison.diagnosis).splitlines():
            lines.append(f"  {line}")
    return "\n".join(lines)
