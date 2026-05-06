"""Trace diagnosis orchestration.

`diagnose` is the high-level trace-forensics surface: it composes the existing
DAG, attribution, outcome-model, and intervention machinery into one
inspectable artifact. It does not introduce a second causal estimator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from counterfact.attribute import AttributionEntry, FailureAttribution
from counterfact.explain import ExplainReport, build_report
from counterfact.intervene.estimate import (
    CausalEstimate,
    DistributionSummary,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
)
from counterfact.schema import Decision, Run
from counterfact.taxonomy import identifiability_stance, valid_interventions


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class DiagnosisEntry(_Strict):
    decision_id: str
    step: int | None
    decision_type: str
    chosen_action: str | None
    influence: float
    identifiability: IdentifiabilityStatus
    next_step: NextStep
    estimate: CausalEstimate | None = None
    outcome_delta: DistributionSummary | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DiagnosisReport(_Strict):
    run_id: str
    outcome: str
    corpus_size: int
    summary: str
    entries: list[DiagnosisEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_path: str | None = None
    corpus_dir: str | None = None


def _outcome_label(run: Run) -> str:
    if run.outcome.kind == "binary":
        return "pass" if bool(run.outcome.value) else "fail"
    return f"{run.outcome.kind}={run.outcome.value!r}"


def _decision_index(run: Run) -> dict[str, tuple[int, Decision]]:
    out: dict[str, tuple[int, Decision]] = {}
    for step in run.steps:
        for decision in step.decisions:
            out[decision.decision_id] = (step.step_index, decision)
    return out


def _intervention_kind_for(decision_type: str) -> str | None:
    kinds = sorted(valid_interventions(decision_type))
    for kind in kinds:
        if identifiability_stance(decision_type, kind) == "requires-randomized-support":
            return kind
    return kinds[0] if kinds else None


def _unsupported_estimate(
    *,
    step: int | None,
    decision: Decision,
    reason: str,
) -> CausalEstimate:
    intervention_kind = _intervention_kind_for(decision.decision_type) or "unknown"
    return CausalEstimate(
        query=InterventionQuery(
            decision_type=decision.decision_type,
            intervention_kind=intervention_kind,
            target=decision.chosen_action,
            step=step if step is not None else -1,
        ),
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason=reason,
        next_step=NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": decision.decision_type,
                "missing_strata": [f"{decision.decision_type}::{decision.chosen_action}"],
                "observed_arms": [],
                "missing_arms": [],
            },
            human_text=(
                "Collect traces with alternative supported actions before "
                "estimating this decision-level counterfactual."
            ),
        ),
    )


def _entry_from_attribution(
    attribution_entry: AttributionEntry,
    decisions: dict[str, tuple[int, Decision]],
) -> DiagnosisEntry:
    resolved = decisions.get(attribution_entry.decision_id)
    step = resolved[0] if resolved else None
    decision = resolved[1] if resolved else None
    estimate = attribution_entry.estimate
    if estimate is None and decision is not None:
        estimate = _unsupported_estimate(
            step=step,
            decision=decision,
            reason="no observed sibling arm supports this decision contrast",
        )
    next_step = (
        estimate.next_step
        if estimate is not None
        else NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": attribution_entry.decision_type,
                "missing_strata": [attribution_entry.decision_type],
                "observed_arms": [],
                "missing_arms": [],
            },
            human_text="Collect traces with supported alternative actions.",
        )
    )
    return DiagnosisEntry(
        decision_id=attribution_entry.decision_id,
        step=step,
        decision_type=attribution_entry.decision_type,
        chosen_action=attribution_entry.chosen_action,
        influence=attribution_entry.influence,
        identifiability=attribution_entry.identifiability,
        next_step=next_step,
        estimate=estimate,
        outcome_delta=(
            estimate.outcome_delta
            if estimate is not None
            and attribution_entry.identifiability != IdentifiabilityStatus.UNIDENTIFIED
            else None
        ),
        reason=estimate.reason if estimate is not None else None,
        warnings=list(estimate.warnings) if estimate is not None else [],
    )


def _degenerate_entries(
    focal_run: Run,
    estimate: CausalEstimate,
    *,
    top_k: int,
) -> list[DiagnosisEntry]:
    entries: list[DiagnosisEntry] = []
    for step in focal_run.steps:
        for decision in step.decisions:
            if decision.chosen_action is None or not valid_interventions(decision.decision_type):
                continue
            query = estimate.query.model_copy(
                update={
                    "decision_type": decision.decision_type,
                    "intervention_kind": _intervention_kind_for(decision.decision_type)
                    or estimate.query.intervention_kind,
                    "target": decision.chosen_action,
                    "step": step.step_index,
                }
            )
            decision_estimate = estimate.model_copy(update={"query": query})
            entries.append(
                DiagnosisEntry(
                    decision_id=decision.decision_id,
                    step=step.step_index,
                    decision_type=decision.decision_type,
                    chosen_action=decision.chosen_action,
                    influence=0.0,
                    identifiability=IdentifiabilityStatus.UNIDENTIFIED,
                    next_step=decision_estimate.next_step,
                    estimate=decision_estimate,
                    reason=decision_estimate.reason,
                    warnings=list(decision_estimate.warnings),
                )
            )
            if len(entries) >= top_k:
                return entries
    return entries


def _summary_for(entries: list[DiagnosisEntry], run: Run) -> str:
    if not entries:
        return (
            f"Run {run.run_id} ({_outcome_label(run)}): no targetable decisions "
            "were found for counterfactual diagnosis."
        )
    supported = [
        entry
        for entry in entries
        if entry.identifiability != IdentifiabilityStatus.UNIDENTIFIED
    ]
    if supported:
        top = supported[0]
        return (
            f"Run {run.run_id} ({_outcome_label(run)}): most plausible supported "
            f"failure point is {top.decision_id} "
            f"({top.decision_type}={top.chosen_action}, "
            f"identifiability={top.identifiability.value})."
        )
    top = entries[0]
    return (
        f"Run {run.run_id} ({_outcome_label(run)}): no supported decision-level "
        f"counterfactual diagnosis is available; top candidate {top.decision_id} "
        f"is {top.identifiability.value} and needs {top.next_step.action}."
    )


def _diagnosis_from_explain_report(
    report: ExplainReport,
    *,
    top_k: int,
    run_path: str | None,
    corpus_dir: str | None,
) -> DiagnosisReport:
    decisions = _decision_index(report.run)
    if report.degenerate_estimate is not None:
        entries = _degenerate_entries(
            report.run, report.degenerate_estimate, top_k=top_k
        )
    else:
        entries = [
            _entry_from_attribution(entry, decisions)
            for entry in report.attribution.entries[:top_k]
        ]
    return DiagnosisReport(
        run_id=report.run.run_id,
        outcome=_outcome_label(report.run),
        corpus_size=report.corpus_size,
        summary=_summary_for(entries, report.run),
        entries=entries,
        warnings=list(report.notes),
        run_path=run_path,
        corpus_dir=corpus_dir,
    )


def explain_report_from_diagnosis(
    diagnosis: DiagnosisReport,
    base_report: ExplainReport,
) -> ExplainReport:
    """Adapt a `DiagnosisReport` into the existing HTML renderer input."""
    attribution = FailureAttribution(
        entries=[
            AttributionEntry(
                decision_id=entry.decision_id,
                decision_type=entry.decision_type,
                chosen_action=entry.chosen_action or "n/a",
                influence=entry.influence,
                identifiability=entry.identifiability,
                estimate=entry.estimate,
            )
            for entry in diagnosis.entries
        ]
    )
    return base_report.model_copy(
        update={
            "attribution": attribution,
            "diagnosis_summary": diagnosis.summary,
            "counterfactual_lookup": [
                entry.estimate
                for entry in diagnosis.entries
                if entry.estimate is not None
            ],
            "run_path": diagnosis.run_path or base_report.run_path,
            "corpus_dir": diagnosis.corpus_dir or base_report.corpus_dir,
        }
    )


def build_diagnosis_pair(
    focal_run: Run,
    corpus: list[Run],
    *,
    decision_type: Literal["model_call", "tool_call", "retry"] = "model_call",
    top_k: int = 3,
    bootstrap: int = 200,
    seed: int = 42,
    run_path: str | None = None,
    corpus_dir: str | None = None,
) -> tuple[DiagnosisReport, ExplainReport]:
    """Build one diagnosis and its diagnosis-first HTML report input."""
    explain_report = build_report(
        focal_run,
        corpus,
        decision_type=decision_type,
        bootstrap=bootstrap,
        seed=seed,
        run_path=run_path,
        corpus_dir=corpus_dir,
    )
    diagnosis = _diagnosis_from_explain_report(
        explain_report,
        top_k=top_k,
        run_path=run_path,
        corpus_dir=corpus_dir,
    )
    return diagnosis, explain_report_from_diagnosis(diagnosis, explain_report)


def build_diagnosis(
    focal_run: Run,
    corpus: list[Run],
    *,
    decision_type: Literal["model_call", "tool_call", "retry"] = "model_call",
    top_k: int = 3,
    bootstrap: int = 200,
    seed: int = 42,
    run_path: str | None = None,
    corpus_dir: str | None = None,
) -> DiagnosisReport:
    """Build a deterministic diagnosis artifact for one focal trace."""
    diagnosis, _ = build_diagnosis_pair(
        focal_run,
        corpus,
        decision_type=decision_type,
        top_k=top_k,
        bootstrap=bootstrap,
        seed=seed,
        run_path=run_path,
        corpus_dir=corpus_dir,
    )
    return diagnosis


__all__ = [
    "DiagnosisEntry",
    "DiagnosisReport",
    "build_diagnosis",
    "build_diagnosis_pair",
    "explain_report_from_diagnosis",
]
