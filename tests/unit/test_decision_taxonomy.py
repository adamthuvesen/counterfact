"""Tests for decision-taxonomy spec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from counterfact.schema import Decision, Outcome, Run


def test_registry__all_six_types_are_present() -> None:
    from counterfact.taxonomy import DECISION_TYPES

    assert set(DECISION_TYPES) == {
        "plan_step",
        "model_call",
        "tool_call",
        "memory_read",
        "retry",
        "termination",
    }


def test_registry__unknown_decision_types_are_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Decision(decision_id="d-x", decision_type="reflect")  # pyright: ignore[reportArgumentType]  # unknown type on purpose
    assert "reflect" in str(exc_info.value)


def test_valid_interventions__tool_call_declares_tool_choice() -> None:
    from counterfact.taxonomy import valid_interventions

    assert "tool_choice" in valid_interventions("tool_call")


def test_valid_interventions__tool_call_rejects_model_choice() -> None:
    from counterfact.taxonomy import is_valid_intervention

    assert is_valid_intervention("tool_call", "model_choice") is False


def test_valid_interventions__model_call_exact_set() -> None:
    from counterfact.taxonomy import valid_interventions

    assert set(valid_interventions("model_call")) == {
        "model_choice",
        "prompt_template",
        "prompt_content",
        "temperature",
    }


def test_identifiability_stance__prompt_content_is_always_replay() -> None:
    from counterfact.taxonomy import identifiability_stance

    assert identifiability_stance("model_call", "prompt_content") == "always-replay"


def test_identifiability_stance__tool_choice_requires_randomized_support() -> None:
    from counterfact.taxonomy import identifiability_stance

    assert identifiability_stance("tool_call", "tool_choice") == "requires-randomized-support"


def test_parent_types__tool_call_lists_plan_step_as_parent() -> None:
    from counterfact.taxonomy import parent_types

    assert "plan_step" in parent_types("tool_call")


def test_parent_types__termination_has_no_required_parents() -> None:
    from counterfact.taxonomy import parent_types

    parents = parent_types("termination")
    assert isinstance(parents, list | tuple | set | frozenset)
    # an empty result is acceptable; this scenario only requires that the call works
    assert all(isinstance(p, str) for p in parents)


def test_feature_extraction__tool_call_extracts_tool_name_and_step_index() -> None:
    from counterfact.schema import Step
    from counterfact.taxonomy import extract_features

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
        outcome=Outcome(kind="binary", value=True, verifier="pytest"),
    )
    feats = extract_features(decision, run)
    assert feats["tool_name"] == "run_tests"
    assert feats["step_index"] == 3


def test_feature_extraction__unknown_decision_type_raises() -> None:
    from counterfact.errors import UnknownDecisionTypeError
    from counterfact.taxonomy import extract_features

    # Bypass Pydantic validation by constructing a stand-in with an unknown type.
    class _Stub:
        decision_id = "d-x"
        decision_type = "reflect"  # not in DECISION_TYPES
        chosen_action = None

    run = Run(
        schema_version="0.1.0",
        run_id="r-x",
        steps=[],
        outcome=Outcome(kind="binary", value=True, verifier="pytest"),
    )
    with pytest.raises((KeyError, UnknownDecisionTypeError)):
        extract_features(_Stub(), run)  # pyright: ignore[reportArgumentType]  # unknown type on purpose
