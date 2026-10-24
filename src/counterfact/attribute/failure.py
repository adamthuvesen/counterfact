"""Failure attribution: which decision most influenced this outcome?

For each decision in the trace whose type accepts interventions, we estimate
the contrast between every observed sibling arm and the arm actually chosen.
The largest absolute supported contrast is the *influence score*; entries are
ranked by score.

Identifiability is inherited from the underlying `intervene` call:

* identified  → influence is |E[Y | do(sibling)] - E[Y | do(chosen)]| (AIPW)
* bounded     → influence is 0.0; the entry keeps its Manski bounds
* unidentified → influence is 0.0 and the entry is labeled unidentified

Per spec: each entry exposes `identifiability` so callers can filter or
re-rank by epistemic confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from counterfact.dag import DAG
from counterfact.intervene.api import intervene as _intervene
from counterfact.intervene.estimate import CausalEstimate, IdentifiabilityStatus
from counterfact.outcome.model import OutcomeModel
from counterfact.schema import Decision, Run, Step
from counterfact.taxonomy import attribution_intervention_kind, valid_interventions
from counterfact.trace_localization import decision_type_repeats_elsewhere


class AttributionEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    decision_id: str
    decision_type: str
    chosen_action: str
    influence: float
    identifiability: IdentifiabilityStatus
    estimate: CausalEstimate | None = None


@dataclass
class FailureAttribution:
    entries: list[AttributionEntry]

    def top_k(self, k: int) -> list[AttributionEntry]:
        return self.entries[:k]


def _sibling_actions(model: OutcomeModel, decision_type: str, chosen_action: str) -> list[str]:
    """Return every observed sibling arm for `decision_type`, in feature-index order.

    The caller picks among them. We deliberately do not preselect here —
    "most-different" is decided after the per-sibling intervene runs, since the
    feature-index alone says nothing about predicted outcome.
    """
    feat_index: dict[str, int] = getattr(model, "feature_index", {})
    return [
        k.split("::", 1)[1]
        for k in feat_index
        if k.startswith(f"{decision_type}::") and k != f"{decision_type}::{chosen_action}"
    ]


def _unidentified_entry(
    decision: Decision, *, estimate: CausalEstimate | None = None
) -> AttributionEntry:
    return AttributionEntry(
        decision_id=decision.decision_id,
        decision_type=decision.decision_type,
        chosen_action=decision.chosen_action or "",
        influence=0.0,
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        estimate=estimate,
    )


def _is_step_localizable(run: Run, step: Step, decision: Decision) -> bool:
    if len(step.decisions) > 1:
        return False
    return not decision_type_repeats_elsewhere(run, step.step_index, decision.decision_type)


_LABEL_RANK = {
    IdentifiabilityStatus.IDENTIFIED: 0,
    IdentifiabilityStatus.BOUNDED: 1,
    IdentifiabilityStatus.UNIDENTIFIED: 2,
}


def _influence(estimate: CausalEstimate) -> float:
    if estimate.identifiability != IdentifiabilityStatus.IDENTIFIED:
        return 0.0
    return abs(estimate.outcome_delta.point) if estimate.outcome_delta is not None else 0.0


def _entry_for_decision(
    *,
    dag: DAG,
    model: OutcomeModel,
    step: Step,
    decision: Decision,
    intervention_kind: str,
    siblings: list[str],
) -> AttributionEntry:
    if dag.run is None:
        return _unidentified_entry(decision)
    if not siblings:
        # No alternative arm with support: nothing to counterfact-factualize against.
        return _unidentified_entry(decision)
    if not _is_step_localizable(dag.run, step, decision):
        # The v0 engine cannot answer "intervene only here" for multi-decision
        # or repeated-type traces. Surface that uncertainty instead of scoring.
        return _unidentified_entry(decision)

    estimates = [
        _intervene(
            dag=dag,
            model=model,
            step=step.step_index,
            intervention={intervention_kind: sibling},
        )
        for sibling in siblings
    ]
    estimate = min(estimates, key=lambda e: (_LABEL_RANK[e.identifiability], -_influence(e)))
    influence = _influence(estimate)
    return AttributionEntry(
        decision_id=decision.decision_id,
        decision_type=decision.decision_type,
        chosen_action=decision.chosen_action or "",
        influence=influence,
        identifiability=estimate.identifiability,
        estimate=estimate,
    )


def attribute_failure(
    *,
    dag: DAG,
    model: OutcomeModel,
    outcome: str = "failed",
) -> FailureAttribution:
    """Rank decisions in `dag.run` by their estimated causal influence on the outcome."""
    if dag.run is None or not dag.nodes:
        return FailureAttribution(entries=[])

    entries: list[AttributionEntry] = []
    for step in dag.run.steps:
        for d in step.decisions:
            kinds = valid_interventions(d.decision_type)
            if not kinds or d.chosen_action is None:
                continue
            intervention_kind = attribution_intervention_kind(d.decision_type)
            if intervention_kind is None:
                continue
            siblings = _sibling_actions(model, d.decision_type, d.chosen_action)
            entries.append(
                _entry_for_decision(
                    dag=dag,
                    model=model,
                    step=step,
                    decision=d,
                    intervention_kind=intervention_kind,
                    siblings=siblings,
                )
            )

    entries.sort(key=lambda e: e.influence, reverse=True)
    return FailureAttribution(entries=entries)
