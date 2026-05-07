"""Compose a typed `ExplainReport` from a focal trace plus its corpus.

The report carries everything the renderer needs to draw the page; all
causal verdicts are real `CausalEstimate` instances produced by the existing
engine, never invented here. The single-class corpus path delegates to
`counterfact.intervene.degenerate.degenerate_estimate` so the honest refusal
stays bit-identical to what `counterfact demo` already emits.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from counterfact.attribute import FailureAttribution, attribute_failure
from counterfact.baselines import PassRateTable, pass_rate_by_arm
from counterfact.dag import DAG, build_dag
from counterfact.errors import InsufficientOutcomeSupportError
from counterfact.intervene.degenerate import degenerate_estimate, outcome_classes
from counterfact.intervene.estimate import (
    CausalEstimate,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
)
from counterfact.outcome import fit_outcome_model
from counterfact.outcome.binary import binary_outcome_value
from counterfact.schema import DecisionTypeLiteral, Run

_INTERVENTION_KIND_BY_DECISION_TYPE: dict[str, str] = {
    "model_call": "model_choice",
    "tool_call": "tool_choice",
    "retry": "retry_policy",
}


class ExplainReport(BaseModel):
    """Inputs to the renderer. Everything the HTML needs lives here, typed."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run: Run
    corpus_size: int
    corpus_pass_rate: float | None
    pass_rate_table: PassRateTable
    dag: DAG
    attribution: FailureAttribution
    degenerate_estimate: CausalEstimate | None
    summary_decision_type: DecisionTypeLiteral
    decision_type_intervention_kind: str
    target_arm: str | None = None
    bootstrap: int = 200
    seed: int = 42
    run_path: str | None = None
    corpus_dir: str | None = None
    notes: list[str] = Field(default_factory=list)
    diagnosis_summary: str | None = None
    counterfactual_lookup: list[CausalEstimate] = Field(default_factory=list)


def _first_arm(runs: list[Run], decision_type: str) -> str | None:
    for run in runs:
        for step in run.steps:
            for decision in step.decisions:
                if decision.decision_type == decision_type and decision.chosen_action:
                    return decision.chosen_action
    return None


def _no_feature_estimate(
    *,
    decision_type: str,
    intervention_kind: str,
    target: str | None,
    reason: str,
) -> CausalEstimate:
    return CausalEstimate(
        query=InterventionQuery(
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
            step=-1,
        ),
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason=reason,
        warnings=["fit_outcome_model is skipped because the corpus has no features"],
        next_step=NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": decision_type,
                "missing_strata": [
                    "no intervenable decisions with chosen_action values were observed"
                ],
                "observed_arms": [],
                "missing_arms": [],
            },
            human_text=(
                "Collect traces with supported decisions and chosen actions before "
                "estimating decision-level effects."
            ),
        ),
    )


def build_report(
    focal_run: Run,
    corpus: list[Run],
    *,
    decision_type: Literal["model_call", "tool_call", "retry"] = "model_call",
    bootstrap: int = 200,
    seed: int = 42,
    run_path: str | None = None,
    corpus_dir: str | None = None,
) -> ExplainReport:
    """Compose an `ExplainReport` from one focal `Run` plus its corpus.

    Single-class corpora (no outcome variation) skip `fit_outcome_model`
    entirely and surface the shared degenerate refusal. Mixed-outcome
    corpora fit the outcome model once and run `attribute_failure` over the
    focal run's DAG; the resulting `FailureAttribution.entries` carry their
    own `CausalEstimate`s and `IdentifiabilityStatus`.
    """
    if not corpus:
        raise ValueError("build_report requires a non-empty corpus")
    if focal_run.run_id not in {r.run_id for r in corpus}:
        raise ValueError(
            f"focal run {focal_run.run_id!r} is not present in the supplied corpus"
        )

    intervention_kind = _INTERVENTION_KIND_BY_DECISION_TYPE[decision_type]
    table = pass_rate_by_arm(corpus, decision_type)
    dag = build_dag(focal_run)

    classes = outcome_classes(corpus)
    n_pass = sum(1 for r in corpus if binary_outcome_value(r))
    pass_rate: float | None = n_pass / len(corpus) if corpus else None

    if len(classes) < 2:
        target = _first_arm(corpus, decision_type)
        # Single-class corpora are not model-fit inputs (CLAUDE.md invariant);
        # surface the degenerate case as `unidentified` and skip the engine.
        estimate = degenerate_estimate(
            corpus,
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
        )
        return ExplainReport(
            run=focal_run,
            corpus_size=len(corpus),
            corpus_pass_rate=pass_rate,
            pass_rate_table=table,
            dag=dag,
            attribution=FailureAttribution(entries=[]),
            degenerate_estimate=estimate,
            summary_decision_type=decision_type,
            decision_type_intervention_kind=intervention_kind,
            target_arm=target,
            bootstrap=bootstrap,
            seed=seed,
            run_path=run_path,
            corpus_dir=corpus_dir,
        )

    try:
        model = fit_outcome_model(corpus, n_bootstrap=bootstrap, seed=seed)
    except InsufficientOutcomeSupportError as exc:
        target = _first_arm(corpus, decision_type)
        estimate = _no_feature_estimate(
            decision_type=decision_type,
            intervention_kind=intervention_kind,
            target=target,
            reason=str(exc),
        )
        return ExplainReport(
            run=focal_run,
            corpus_size=len(corpus),
            corpus_pass_rate=pass_rate,
            pass_rate_table=table,
            dag=dag,
            attribution=FailureAttribution(entries=[]),
            degenerate_estimate=estimate,
            summary_decision_type=decision_type,
            decision_type_intervention_kind=intervention_kind,
            target_arm=target,
            bootstrap=bootstrap,
            seed=seed,
            run_path=run_path,
            corpus_dir=corpus_dir,
            notes=[str(exc)],
        )
    attribution = attribute_failure(dag=dag, model=model)
    return ExplainReport(
        run=focal_run,
        corpus_size=len(corpus),
        corpus_pass_rate=pass_rate,
        pass_rate_table=table,
        dag=dag,
        attribution=attribution,
        degenerate_estimate=None,
        summary_decision_type=decision_type,
        decision_type_intervention_kind=intervention_kind,
        target_arm=_first_arm(corpus, decision_type),
        bootstrap=bootstrap,
        seed=seed,
        run_path=run_path,
        corpus_dir=corpus_dir,
    )
