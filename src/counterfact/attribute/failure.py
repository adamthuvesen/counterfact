"""Failure attribution: which decision most influenced this outcome?

For each decision in the trace whose type accepts interventions, we compare
the predicted outcome under the actually-chosen action against the predicted
outcome under the most-different sibling arm. The absolute difference is the
*influence score*; entries are ranked by score.

Identifiability is inherited from the underlying `intervene` call:

* identified  → influence is the |g-formula counterfactual - actual|
* bounded     → influence is reported with the bound; ranking still shown
* unidentified → influence is 0.0 and the entry is labeled unidentified

Per spec: each entry exposes `identifiability` so callers can filter or
re-rank by epistemic confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from counterfact.dag import DAG
from counterfact.errors import UnknownDecisionTypeError
from counterfact.intervene.api import intervene as _intervene
from counterfact.intervene.estimate import CausalEstimate, IdentifiabilityStatus
from counterfact.taxonomy import valid_interventions


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


def _intervention_kind_for(decision_type: str) -> str | None:
    """Pick the canonical intervention kind for this decision type, if any.

    The taxonomy may declare multiple intervention kinds per type; for v0
    attribution we use the first one whose stance is randomized-support
    (those are the ones we can actually estimate). Falls back to the first
    declared kind.
    """
    from counterfact.taxonomy import identifiability_stance

    kinds = sorted(valid_interventions(decision_type))
    for k in kinds:
        try:
            if identifiability_stance(decision_type, k) == "requires-randomized-support":
                return k
        except UnknownDecisionTypeError:
            continue
    return kinds[0] if kinds else None


def _combined_identifiability(
    actual: CausalEstimate, counterfactual: CausalEstimate
) -> IdentifiabilityStatus:
    statuses = {actual.identifiability, counterfactual.identifiability}
    if IdentifiabilityStatus.UNIDENTIFIED in statuses:
        return IdentifiabilityStatus.UNIDENTIFIED
    if statuses == {IdentifiabilityStatus.IDENTIFIED}:
        return IdentifiabilityStatus.IDENTIFIED
    return IdentifiabilityStatus.BOUNDED


def _decision_type_repeats(run: object, step_index: int, decision_type: str) -> bool:
    """True iff `decision_type` appears in any *other* step in this run.

    The g-formula in `intervene` treats a (decision_type, action) one-hot as a
    single feature, so it cannot disambiguate between repeated occurrences in a
    single trace. Attribution mirrors `intervene`'s honest refusal here rather
    than catching `InvalidInterventionError` (which could mask unrelated bugs).
    """
    steps = getattr(run, "steps", []) or []
    for s in steps:
        if s.step_index == step_index:
            continue
        if any(d.decision_type == decision_type for d in s.decisions):
            return True
    return False


def _sibling_actions(model: object, decision_type: str, chosen_action: str) -> list[str]:
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


def attribute_failure(
    *,
    dag: DAG,
    model: object,
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
            intervention_kind = _intervention_kind_for(d.decision_type)
            if intervention_kind is None:
                continue
            siblings = _sibling_actions(model, d.decision_type, d.chosen_action)
            if not siblings:
                # No alternative arm with support: nothing to counterfact-factualize against.
                entries.append(
                    AttributionEntry(
                        decision_id=d.decision_id,
                        decision_type=d.decision_type,
                        chosen_action=d.chosen_action,
                        influence=0.0,
                        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
                    )
                )
                continue
            if len(step.decisions) > 1:
                entries.append(
                    AttributionEntry(
                        decision_id=d.decision_id,
                        decision_type=d.decision_type,
                        chosen_action=d.chosen_action,
                        influence=0.0,
                        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
                    )
                )
                continue
            if _decision_type_repeats(dag.run, step.step_index, d.decision_type):
                # The trace has another step of the same decision_type — the
                # corpus-wide g-formula cannot answer "intervene only here".
                # Surface as unidentified rather than producing the misleading
                # "set everywhere" estimate the v0 engine would otherwise emit.
                entries.append(
                    AttributionEntry(
                        decision_id=d.decision_id,
                        decision_type=d.decision_type,
                        chosen_action=d.chosen_action,
                        influence=0.0,
                        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
                    )
                )
                continue

            actual = _intervene(
                dag=dag,
                model=model,
                step=step.step_index,
                intervention={intervention_kind: d.chosen_action},
            )

            if actual.outcome_delta is None:
                entries.append(
                    AttributionEntry(
                        decision_id=d.decision_id,
                        decision_type=d.decision_type,
                        chosen_action=d.chosen_action,
                        influence=0.0,
                        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
                        estimate=actual,
                    )
                )
                continue

            # Evaluate every sibling arm and keep the one whose predicted
            # outcome is the most different from the actual arm's prediction.
            # This is the contrast the report ranks by — picking the first
            # sibling silently underestimated influence on three-or-more-arm
            # decision types.
            best_influence: float | None = None
            best_ident = IdentifiabilityStatus.UNIDENTIFIED
            for sibling in siblings:
                cf = _intervene(
                    dag=dag,
                    model=model,
                    step=step.step_index,
                    intervention={intervention_kind: sibling},
                )
                ident = _combined_identifiability(actual, cf)
                if cf.outcome_delta is not None and ident != IdentifiabilityStatus.UNIDENTIFIED:
                    influence = abs(actual.outcome_delta.point - cf.outcome_delta.point)
                else:
                    influence = 0.0
                if best_influence is None or influence > best_influence:
                    best_influence = influence
                    best_ident = ident

            entries.append(
                AttributionEntry(
                    decision_id=d.decision_id,
                    decision_type=d.decision_type,
                    chosen_action=d.chosen_action,
                    influence=best_influence or 0.0,
                    identifiability=best_ident,
                    estimate=actual,
                )
            )

    entries.sort(key=lambda e: e.influence, reverse=True)
    return FailureAttribution(entries=entries)
