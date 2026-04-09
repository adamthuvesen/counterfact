"""Tests for decision-taxonomy spec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from counter.schema import Decision, Run

# --- registry ---------------------------------------------------------------


def test_registry__all_six_types_are_present() -> None:
    """WHEN counter.taxonomy.DECISION_TYPES is enumerated
    THEN it contains exactly the six v0 names, no more and no fewer."""
    from counter.taxonomy import DECISION_TYPES

    assert set(DECISION_TYPES) == {
        "plan_step",
        "model_call",
        "tool_call",
        "memory_read",
        "retry",
        "termination",
    }


def test_registry__unknown_decision_types_are_rejected() -> None:
    """WHEN a trace with Decision(decision_type="reflect") is loaded
    THEN the system raises a pydantic.ValidationError referencing the unknown type."""
    with pytest.raises(ValidationError) as exc_info:
        Decision(decision_id="d-x", decision_type="reflect")  # type: ignore[arg-type]
    assert "reflect" in str(exc_info.value)


# --- valid interventions ----------------------------------------------------


def test_valid_interventions__tool_call_declares_tool_choice() -> None:
    """WHEN the runtime queries valid_interventions("tool_call")
    THEN the result includes "tool_choice"."""
    from counter.taxonomy import valid_interventions

    assert "tool_choice" in valid_interventions("tool_call")


def test_valid_interventions__tool_call_rejects_model_choice() -> None:
    """WHEN the runtime queries is_valid_intervention("tool_call", "model_choice")
    THEN the result is False."""
    from counter.taxonomy import is_valid_intervention

    assert is_valid_intervention("tool_call", "model_choice") is False


def test_valid_interventions__model_call_exact_set() -> None:
    """WHEN the runtime queries valid_interventions("model_call")
    THEN the result is exactly {model_choice, prompt_template, prompt_content, temperature}."""
    from counter.taxonomy import valid_interventions

    assert set(valid_interventions("model_call")) == {
        "model_choice",
        "prompt_template",
        "prompt_content",
        "temperature",
    }


# --- identifiability stance --------------------------------------------------


def test_identifiability_stance__prompt_content_is_always_replay() -> None:
    """WHEN identifiability_stance("model_call", "prompt_content") is queried
    THEN the result is "always-replay"."""
    from counter.taxonomy import identifiability_stance

    assert identifiability_stance("model_call", "prompt_content") == "always-replay"


def test_identifiability_stance__tool_choice_requires_randomized_support() -> None:
    """WHEN identifiability_stance("tool_call", "tool_choice") is queried
    THEN the result is "requires-randomized-support"."""
    from counter.taxonomy import identifiability_stance

    assert identifiability_stance("tool_call", "tool_choice") == "requires-randomized-support"


# --- parent declarations -----------------------------------------------------


def test_parent_types__tool_call_lists_plan_step_as_parent() -> None:
    """WHEN parent_types("tool_call") is queried
    THEN "plan_step" is in the result."""
    from counter.taxonomy import parent_types

    assert "plan_step" in parent_types("tool_call")


def test_parent_types__termination_has_no_required_parents() -> None:
    """WHEN parent_types("termination") is queried
    THEN the result is allowed to be empty (no required parents)."""
    from counter.taxonomy import parent_types

    parents = parent_types("termination")
    assert isinstance(parents, list | tuple | set | frozenset)
    # an empty result is acceptable; this scenario only requires that the call works
    assert all(isinstance(p, str) for p in parents)


# --- feature extraction ------------------------------------------------------


def test_feature_extraction__tool_call_extracts_tool_name_and_step_index() -> None:
    """WHEN extract_features(decision, run) is called for a tool_call decision
    THEN the returned dict contains keys "tool_name" and "step_index" with correct values."""
    from counter.schema import Step
    from counter.taxonomy import extract_features

    decision = Decision(
        decision_id="d-tool",
        decision_type="tool_call",
        chosen_action="run_tests",
    )
    step = Step(step_index=3, decisions=[decision])
    run = Run(
        schema_version="0.1.0",
        run_id="r-feat",
        steps=[step],
        outcome={"kind": "binary", "value": True, "verifier": "pytest"},
    )
    feats = extract_features(decision, run)
    assert feats["tool_name"] == "run_tests"
    assert feats["step_index"] == 3


def test_feature_extraction__unknown_decision_type_raises() -> None:
    """WHEN extract_features is called for a decision-type-string not in DECISION_TYPES
    THEN the system raises a KeyError or UnknownDecisionTypeError."""
    from counter.errors import UnknownDecisionTypeError
    from counter.taxonomy import extract_features

    # Bypass Pydantic validation by constructing a stand-in with an unknown type.
    class _Stub:
        decision_id = "d-x"
        decision_type = "reflect"  # not in DECISION_TYPES
        chosen_action = None

    run = Run(
        schema_version="0.1.0",
        run_id="r-x",
        steps=[],
        outcome={"kind": "binary", "value": True, "verifier": "pytest"},
    )
    with pytest.raises((KeyError, UnknownDecisionTypeError)):
        extract_features(_Stub(), run)  # type: ignore[arg-type]
