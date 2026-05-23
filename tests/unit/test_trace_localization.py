from __future__ import annotations

from counterfact.schema import Decision, Outcome, Run, Step
from counterfact.trace_localization import (
    decision_type_repeats_elsewhere,
    duplicate_decision_type_steps,
)


def _run_with_steps() -> Run:
    return Run(
        schema_version="0.1.0",
        run_id="r1",
        steps=[
            Step(
                step_index=1,
                decisions=[
                    Decision(
                        decision_id="d1",
                        decision_type="tool_call",
                        chosen_action="run_tests",
                    )
                ],
            ),
            Step(
                step_index=2,
                decisions=[
                    Decision(
                        decision_id="d2",
                        decision_type="model_call",
                        chosen_action="small",
                    )
                ],
            ),
            Step(
                step_index=3,
                decisions=[
                    Decision(
                        decision_id="d3",
                        decision_type="tool_call",
                        chosen_action="inspect_file",
                    )
                ],
            ),
        ],
        outcome=Outcome(kind="binary", value=True, verifier="stub"),
    )


def test_duplicate_decision_type_steps__finds_other_occurrences() -> None:
    run = _run_with_steps()
    assert duplicate_decision_type_steps(run, except_step=1, decision_type="tool_call") == [3]


def test_duplicate_decision_type_steps__none_when_unique() -> None:
    run = _run_with_steps()
    assert duplicate_decision_type_steps(run, except_step=2, decision_type="model_call") == []


def test_decision_type_repeats_elsewhere__matches_duplicate_list() -> None:
    run = _run_with_steps()
    assert decision_type_repeats_elsewhere(run, 1, "tool_call") is True
    assert decision_type_repeats_elsewhere(run, 2, "model_call") is False
