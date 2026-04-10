"""Acceptance tests for the structured NextStep model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from counterfact.intervene import NextStep


def test_next_step_increase_n_round_trips_through_json() -> None:
    ns = NextStep(
        action="increase_n",
        payload={"current_n": 30, "estimated_required_n": 1200, "target_ci_width": 0.10},
        human_text="Need ~1200 traces to reach CI width 0.10.",
    )
    blob = ns.model_dump_json()
    again = NextStep.model_validate_json(blob)
    assert again == ns


def test_next_step_increase_n_missing_required_keys_raises() -> None:
    with pytest.raises(Exception) as exc:
        NextStep(action="increase_n", payload={}, human_text="…")
    msg = str(exc.value)
    assert "current_n" in msg
    assert "estimated_required_n" in msg


def test_next_step_none_with_empty_payload_is_valid() -> None:
    ns = NextStep(action="none", payload={}, human_text="No further action.")
    assert ns.action == "none"
    assert ns.payload == {}


def test_next_step_replay_required_must_have_intervention_target() -> None:
    NextStep(
        action="replay_required",
        payload={"intervention_target": "prompt_content"},
        human_text="Replay needed.",
    )
    with pytest.raises(Exception) as exc:
        NextStep(action="replay_required", payload={}, human_text="…")
    assert "intervention_target" in str(exc.value)


def test_next_step_broaden_arm_support_requires_arm_and_strata() -> None:
    NextStep(
        action="broaden_arm_support",
        payload={"arm_name": "model_choice", "missing_strata": ["small"]},
        human_text="…",
    )
    with pytest.raises(Exception) as exc:
        NextStep(
            action="broaden_arm_support",
            payload={"arm_name": "model_choice"},
            human_text="…",
        )
    assert "missing_strata" in str(exc.value)


def test_next_step_add_arm_randomization_requires_policy() -> None:
    NextStep(
        action="add_arm_randomization",
        payload={"arm_name": "tool_choice", "current_policy": "always_inspect_file"},
        human_text="…",
    )
    with pytest.raises(Exception) as exc:
        NextStep(
            action="add_arm_randomization",
            payload={"arm_name": "tool_choice"},
            human_text="…",
        )
    assert "current_policy" in str(exc.value)


def test_next_step_unknown_action_raises() -> None:
    with pytest.raises(ValidationError):
        NextStep(action="teleport", payload={}, human_text="…")  # type: ignore[arg-type]
