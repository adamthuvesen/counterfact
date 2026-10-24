"""Tests for `counterfact.intervene.suggest.suggest_harness_command`."""

from __future__ import annotations

import shlex

import pytest

from counterfact.cli import build_parser
from counterfact.intervene.suggest import suggest_harness_command


@pytest.mark.parametrize(
    ("decision_type", "intervention_kind", "action", "arm_name"),
    [
        ("model_call", "model_choice", "broaden_arm_support", "small"),
        ("model_call", "model_choice", "add_arm_randomization", "large"),
        ("model_call", "model_choice", "increase_n", "small"),
        ("retry", "retry_policy", "broaden_arm_support", "no_retry"),
        ("retry", "retry_policy", "increase_n", "retry_once"),
        ("tool_call", "tool_choice", "broaden_arm_support", "run_tests"),
        ("tool_call", "tool_choice", "increase_n", "inspect_file"),
    ],
)
def test_suggested_commands_parse_through_cli(
    decision_type: str,
    intervention_kind: str,
    action: str,
    arm_name: str,
) -> None:
    cmd = suggest_harness_command(
        decision_type=decision_type,
        intervention_kind=intervention_kind,
        action=action,
        arm_name=arm_name,
        estimated_required_n=60,
    )
    assert cmd is not None
    assert cmd.startswith("uv run counterfact bench real ")
    tail = cmd[len("uv run counterfact ") :]
    parser = build_parser()
    parser.parse_args(shlex.split(tail))


def test_replay_required_returns_none() -> None:
    cmd = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="prompt_content",
        action="replay_required",
    )
    assert cmd is None


def test_none_action_returns_none() -> None:
    cmd = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="model_choice",
        action="none",
    )
    assert cmd is None


def test_unknown_action_returns_none() -> None:
    cmd = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="model_choice",
        action="teleport",  # not a real NextStep action
    )
    assert cmd is None


def test_broaden_arm_support_on_unmapped_decision_type_returns_none() -> None:
    """plan_step has no harness arm; can't compose a meaningful broaden command."""
    cmd = suggest_harness_command(
        decision_type="plan_step",
        intervention_kind="plan_action",
        action="broaden_arm_support",
        arm_name="some_action",
    )
    assert cmd is None


def test_suggested_command_is_deterministic() -> None:
    a = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="model_choice",
        action="broaden_arm_support",
        arm_name="small",
        estimated_required_n=42,
    )
    b = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="model_choice",
        action="broaden_arm_support",
        arm_name="small",
        estimated_required_n=42,
    )
    assert a == b
    assert a is not None


def test_increase_n_command_uses_estimated_required_n() -> None:
    cmd = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="model_choice",
        action="increase_n",
        arm_name="large",
        estimated_required_n=180,
    )
    assert cmd is not None
    assert "--n 180" in cmd
    assert "--fixture-set broad_calibration" in cmd


def test_broaden_arm_support_command_uses_hard_hidden_fixture_set() -> None:
    cmd = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="model_choice",
        action="broaden_arm_support",
        arm_name="small",
    )
    assert cmd is not None
    assert "--fixture-set broad_calibration" in cmd


def test_increase_n_command_floors_at_minimum() -> None:
    cmd = suggest_harness_command(
        decision_type="model_call",
        intervention_kind="model_choice",
        action="increase_n",
        estimated_required_n=5,
    )
    assert cmd is not None
    assert "--n 30" in cmd


def test_suggestion_is_dropped_for_corpora_the_harness_did_not_produce() -> None:
    from bench.synthetic import generate_traces
    from counterfact import build_dag, fit_outcome_model, intervene
    from counterfact.schema import Run

    runs = [Run.model_validate(t) for t in generate_traces(n=120, seed=1)]
    est = intervene(
        dag=build_dag(runs[0]),
        model=fit_outcome_model(runs, seed=1),
        step=2,
        intervention={"model_choice": "sonnet"},
        baseline="haiku",
    )

    assert est.next_step.action == "increase_n"
    assert "suggested_command" not in est.next_step.payload


def test_suggestion_names_the_harness_corpus_fixtures() -> None:
    from counterfact import build_dag, fit_outcome_model, intervene
    from tests.fixtures.corpus_loaders import SMOKE_MIXED_OUTCOME_DIR, load_json_corpus

    runs = load_json_corpus(SMOKE_MIXED_OUTCOME_DIR)
    focal = runs[0]
    step = next(s.step_index for s in focal.steps if s.decisions[0].decision_type == "model_call")
    est = intervene(
        dag=build_dag(focal),
        model=fit_outcome_model(runs, seed=1),
        step=step,
        intervention={"model_choice": "small"},
        baseline="large",
    )

    command = est.next_step.payload["suggested_command"]
    assert "--fixtures streaming_watermark_dedupe" in command
    build_parser().parse_args(shlex.split(command)[3:])
