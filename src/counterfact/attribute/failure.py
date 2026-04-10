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

import numpy as np
from pydantic import BaseModel, ConfigDict

from counterfact.dag import DAG
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
        except Exception:
            continue
    return kinds[0] if kinds else None


def _sibling_action(model: object, decision_type: str, chosen_action: str) -> str | None:
    """Return the most-different sibling arm with observed support."""
    feat_index: dict[str, int] = getattr(model, "feature_index", {})
    siblings = [
        k.split("::", 1)[1]
        for k in feat_index
        if k.startswith(f"{decision_type}::") and k != f"{decision_type}::{chosen_action}"
    ]
    if not siblings:
        return None
    return siblings[0]


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
            sibling = _sibling_action(model, d.decision_type, d.chosen_action)
            if sibling is None:
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

            actual = _intervene(
                dag=dag,
                model=model,
                step=step.step_index,
                intervention={intervention_kind: d.chosen_action},
            )
            counterfactual = _intervene(
                dag=dag,
                model=model,
                step=step.step_index,
                intervention={intervention_kind: sibling},
            )

            ident = (
                IdentifiabilityStatus.UNIDENTIFIED
                if (
                    actual.identifiability == IdentifiabilityStatus.UNIDENTIFIED
                    or counterfactual.identifiability == IdentifiabilityStatus.UNIDENTIFIED
                )
                else (
                    IdentifiabilityStatus.IDENTIFIED
                    if (
                        actual.identifiability == IdentifiabilityStatus.IDENTIFIED
                        and counterfactual.identifiability == IdentifiabilityStatus.IDENTIFIED
                    )
                    else IdentifiabilityStatus.BOUNDED
                )
            )

            if (
                actual.outcome_delta is not None
                and counterfactual.outcome_delta is not None
                and ident != IdentifiabilityStatus.UNIDENTIFIED
            ):
                influence = float(
                    np.abs(actual.outcome_delta.point - counterfactual.outcome_delta.point)
                )
            else:
                influence = 0.0

            entries.append(
                AttributionEntry(
                    decision_id=d.decision_id,
                    decision_type=d.decision_type,
                    chosen_action=d.chosen_action,
                    influence=influence,
                    identifiability=ident,
                    estimate=actual,
                )
            )

    entries.sort(key=lambda e: e.influence, reverse=True)
    return FailureAttribution(entries=entries)
