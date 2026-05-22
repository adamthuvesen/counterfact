from __future__ import annotations

from counterfact.taxonomy import DECISION_TYPES, valid_interventions


def test_taxonomy_covers_adapter_decision_surface() -> None:
    """Adapters only emit decision types the taxonomy knows about."""
    adapter_types = {"model_call", "tool_call", "retry", "plan_step", "termination"}
    assert adapter_types <= set(DECISION_TYPES)


def test_randomized_intervention_kinds_have_stance() -> None:
    for dt in ("model_call", "tool_call", "retry"):
        kinds = valid_interventions(dt)
        assert kinds, f"{dt} should declare at least one intervention kind"
