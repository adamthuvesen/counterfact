"""Unit coverage for `counterfact.runrecord_export` row derivations."""

from __future__ import annotations

from counterfact.runrecord_export import runs_to_runrecord_rows
from counterfact.schema import Decision, Metadata, Outcome, Run, Step


def _run(
    *,
    run_id: str,
    agent_name: str | None,
    decisions: list[Decision],
    outcome_value: bool = True,
) -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        metadata=Metadata(agent_name=agent_name),
        steps=[Step(step_index=0, decisions=decisions)] if decisions else [],
        outcome=Outcome(kind="binary", value=outcome_value, verifier="stub"),
    )


def test_model_id_uses_first_model_call_chosen_action() -> None:
    run = _run(
        run_id="r-with-model",
        agent_name="claude-agent-sdk",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="model_call",
                chosen_action="claude-sonnet-4-6",
            )
        ],
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["agent_id"] == "claude-agent-sdk"
    assert rows[0]["model_id"] == "claude-sonnet-4-6"


def test_model_id_is_none_when_no_model_call_exists() -> None:
    run = _run(
        run_id="r-no-model-call",
        agent_name="claude-agent-sdk",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="tool_call",
                chosen_action="bash",
            )
        ],
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["agent_id"] == "claude-agent-sdk"
    assert rows[0]["model_id"] is None


def test_model_id_picks_first_model_call_when_multiple_present() -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="r-multi",
        metadata=Metadata(agent_name="claude-agent-sdk"),
        steps=[
            Step(
                step_index=0,
                decisions=[
                    Decision(
                        decision_id="d0",
                        decision_type="tool_call",
                        chosen_action="bash",
                    )
                ],
            ),
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="d1",
                        decision_type="model_call",
                        chosen_action="gpt-4o",
                    )
                ],
            ),
            Step(
                step_index=2,
                decisions=[
                    Decision(
                        decision_id="d2",
                        decision_type="model_call",
                        chosen_action="gpt-4o-mini",
                    )
                ],
            ),
        ],
        outcome=Outcome(kind="binary", value=True, verifier="stub"),
    )
    rows, _ = runs_to_runrecord_rows([run])
    assert rows[0]["model_id"] == "gpt-4o"


def test_partial_credit_is_none_for_binary_outcomes() -> None:
    pass_run = _run(
        run_id="r-pass",
        agent_name="agent",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="model_call",
                chosen_action="m",
            )
        ],
        outcome_value=True,
    )
    fail_run = _run(
        run_id="r-fail",
        agent_name="agent",
        decisions=[
            Decision(
                decision_id="d0",
                decision_type="model_call",
                chosen_action="m",
            )
        ],
        outcome_value=False,
    )
    rows, _ = runs_to_runrecord_rows([pass_run, fail_run])
    # success still flips per outcome
    assert rows[0]["success"] is True
    assert rows[1]["success"] is False
    # partial_credit is None — we don't fake it as float(success)
    assert rows[0]["partial_credit"] is None
    assert rows[1]["partial_credit"] is None
