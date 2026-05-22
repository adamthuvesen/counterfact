"""HTML render honesty contract for causal estimates."""

from __future__ import annotations

from counterfact.attribute.failure import AttributionEntry
from counterfact.intervene.estimate import CausalEstimate, IdentifiabilityStatus


def shows_outcome_delta(estimate: CausalEstimate) -> bool:
    """True when numeric outcome_delta may be shown in HTML."""
    return (
        estimate.identifiability != IdentifiabilityStatus.UNIDENTIFIED
        and estimate.outcome_delta is not None
    )


def shows_numeric_attribution(entry: AttributionEntry) -> bool:
    """True when attribution influence may be shown as a number."""
    return entry.identifiability != IdentifiabilityStatus.UNIDENTIFIED
