"""Acceptance tests for the structured NextStep model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from counterfact.intervene import NextStep


def _arm_row(arm: str, n: int, k: int, lo: float, hi: float) -> dict:
    return {
        "arm": arm,
        "n": n,
        "pass_count": k,
        "pass_rate": k / n,
        "ci_low": lo,
        "ci_high": hi,
    }


def _full_increase_n_payload() -> dict:
    return {
        "current_n": 30,
        "estimated_required_n": 1200,
        "target_ci_width": 0.10,
        "power_method": "binomial_wald_two_arm",
        "arm_breakdown": [
            _arm_row("small", 10, 6, 0.3, 0.85),
            _arm_row("large", 20, 18, 0.7, 0.97),
        ],
    }


def _full_broaden_payload() -> dict:
    return {
        "arm_name": "model_call",
        "missing_strata": ["large"],
        "observed_arms": [_arm_row("small", 10, 5, 0.24, 0.76)],
        "missing_arms": ["large"],
    }


def _full_replay_payload() -> dict:
    return {
        "intervention_target": "prompt_content",
        "replay_inputs_required": ["prompt_template", "latent_state_at_step"],
        "note": (
            "v0 does not ship replay infrastructure; this next step is "
            "upstream of the bench harness."
        ),
    }


def test_next_step_increase_n_round_trips_through_json() -> None:
    ns = NextStep(
        action="increase_n",
        payload=_full_increase_n_payload(),
        human_text="Need ~1200 traces to reach CI width 0.10.",
    )
    blob = ns.model_dump_json()
    again = NextStep.model_validate_json(blob)
    assert again == ns


def test_next_step_increase_n_missing_required_keys_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        NextStep(action="increase_n", payload={}, human_text="…")
    msg = str(exc.value)
    assert "current_n" in msg
    assert "estimated_required_n" in msg
    assert "power_method" in msg
    assert "arm_breakdown" in msg


def test_next_step_increase_n_missing_only_new_keys_raises() -> None:
    """Old (pre-sharpening) payload is now incomplete and must raise."""
    with pytest.raises(ValidationError) as exc:
        NextStep(
            action="increase_n",
            payload={"current_n": 30, "estimated_required_n": 1200, "target_ci_width": 0.10},
            human_text="…",
        )
    msg = str(exc.value)
    assert "power_method" in msg
    assert "arm_breakdown" in msg


def test_next_step_none_with_empty_payload_is_valid() -> None:
    ns = NextStep(action="none", payload={}, human_text="No further action.")
    assert ns.action == "none"
    assert ns.payload == {}


def test_next_step_replay_required_full_payload_validates() -> None:
    NextStep(
        action="replay_required",
        payload=_full_replay_payload(),
        human_text="Replay needed.",
    )


def test_next_step_replay_required_missing_intervention_target_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        NextStep(action="replay_required", payload={}, human_text="…")
    assert "intervention_target" in str(exc.value)


def test_next_step_replay_required_missing_replay_inputs_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        NextStep(
            action="replay_required",
            payload={"intervention_target": "prompt_content"},
            human_text="…",
        )
    msg = str(exc.value)
    assert "replay_inputs_required" in msg
    assert "note" in msg


def test_next_step_broaden_arm_support_full_payload_validates() -> None:
    NextStep(
        action="broaden_arm_support",
        payload=_full_broaden_payload(),
        human_text="…",
    )


def test_next_step_broaden_arm_support_missing_strata_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": "model_choice",
                "observed_arms": [],
                "missing_arms": [],
            },
            human_text="…",
        )
    assert "missing_strata" in str(exc.value)


def test_next_step_broaden_arm_support_missing_observed_arms_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        NextStep(
            action="broaden_arm_support",
            payload={"arm_name": "model_choice", "missing_strata": ["small"]},
            human_text="…",
        )
    msg = str(exc.value)
    assert "observed_arms" in msg
    assert "missing_arms" in msg


def test_next_step_add_arm_randomization_requires_policy() -> None:
    NextStep(
        action="add_arm_randomization",
        payload={"arm_name": "tool_choice", "current_policy": "always_inspect_file"},
        human_text="…",
    )
    with pytest.raises(ValidationError) as exc:
        NextStep(
            action="add_arm_randomization",
            payload={"arm_name": "tool_choice"},
            human_text="…",
        )
    assert "current_policy" in str(exc.value)


def test_next_step_unknown_action_raises() -> None:
    with pytest.raises(ValidationError):
        NextStep(action="teleport", payload={}, human_text="…")  # type: ignore[arg-type]


def test_next_step_optional_suggested_command_is_accepted() -> None:
    payload = _full_broaden_payload()
    payload["suggested_command"] = "uv run counterfact bench real --n 30 --fixture-set hidden_v1"
    ns = NextStep(
        action="broaden_arm_support",
        payload=payload,
        human_text="…",
    )
    assert ns.payload["suggested_command"].startswith("uv run counterfact bench real")
