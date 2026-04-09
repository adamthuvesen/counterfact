"""Tests for trace-schema spec."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "canonical_run.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def test_native_json_format__round_trip_preserves_run_fixture() -> None:
    """WHEN a canonical fixture trace JSON is loaded into a `Run` model and re-serialized to JSON
    THEN the resulting JSON parses back into an equal `Run` instance with no field loss."""
    from counter.schema import Run

    raw = FIXTURE_PATH.read_text()
    run = Run.model_validate_json(raw)
    dumped = run.model_dump_json()
    reloaded = Run.model_validate_json(dumped)
    assert reloaded == run


def test_native_json_format__unknown_fields_are_rejected() -> None:
    """WHEN a JSON document containing a top-level key not present in the `Run` schema is loaded
    THEN the system raises a `pydantic.ValidationError` (no silent acceptance of extra fields)."""
    from counter.schema import Run

    payload = _fixture()
    payload["unexpected_field"] = "boom"
    with pytest.raises(ValidationError) as exc_info:
        Run.model_validate(payload)
    assert "unexpected_field" in str(exc_info.value)


def test_native_json_format__required_fields_are_enforced() -> None:
    """WHEN a JSON document is missing a required field on Run, Step, Decision, or Outcome
    THEN the system raises a pydantic.ValidationError naming the missing field."""
    from counter.schema import Run

    payload = _fixture()
    del payload["outcome"]
    with pytest.raises(ValidationError) as exc_info:
        Run.model_validate(payload)
    assert "outcome" in str(exc_info.value)


def test_outcome_tagged_union__binary_outcome_accepted_end_to_end() -> None:
    """WHEN a trace with Outcome(kind="binary", value=True, verifier="pytest") is loaded
    and passed to fit_outcome_model
    THEN the call succeeds and the outcome is fit as a binary target."""
    from counter import fit_outcome_model
    from counter.schema import Run

    # A binary outcome at the schema layer must reach fit_outcome_model
    # without tripping UnsupportedOutcomeError. We pass two traces (one of each
    # class) so logistic regression has two-class support.
    run_pos = Run.model_validate(_fixture())
    payload_neg = _fixture()
    payload_neg["run_id"] = "run-canonical-002"
    payload_neg["outcome"] = {"kind": "binary", "value": True, "verifier": "pytest", "metadata": {}}
    run_neg = Run.model_validate(payload_neg)

    model = fit_outcome_model([run_pos, run_neg], n_bootstrap=4)
    assert model is not None
    assert model.outcome_kind == "binary"


def test_outcome_tagged_union__categorical_outcome_rejected_at_fit_time() -> None:
    """WHEN a trace with Outcome(kind="categorical", value="tool_error", verifier="manual")
    is loaded and passed to fit_outcome_model
    THEN the system raises UnsupportedOutcomeError referencing kind="categorical"."""
    from counter import fit_outcome_model
    from counter.errors import UnsupportedOutcomeError
    from counter.schema import Run

    payload = _fixture()
    payload["outcome"] = {
        "kind": "categorical",
        "value": "tool_error",
        "verifier": "manual",
        "metadata": {},
    }
    run = Run.model_validate(payload)
    with pytest.raises(UnsupportedOutcomeError) as exc_info:
        fit_outcome_model([run])
    assert "categorical" in str(exc_info.value)


def test_outcome_tagged_union__continuous_outcome_rejected_at_intervene_time() -> None:
    """WHEN a model is somehow configured with kind="continuous" outcomes and intervene() is called
    THEN the system raises UnsupportedOutcomeError."""
    from counter import build_dag, intervene
    from counter.errors import UnsupportedOutcomeError
    from counter.schema import Run

    payload = _fixture()
    payload["outcome"] = {
        "kind": "continuous",
        "value": 0.42,
        "verifier": "score_fn",
        "metadata": {},
    }
    run = Run.model_validate(payload)
    dag = build_dag(run)

    class _ContinuousModel:
        outcome_kind = "continuous"

    with pytest.raises(UnsupportedOutcomeError):
        intervene(
            dag=dag,
            model=_ContinuousModel(),
            step=1,
            intervention={"tool_choice": "inspect_file"},
        )


def test_decisions_log_randomization__randomized_decision_round_trips_with_propensity() -> None:
    """WHEN a Decision with full randomization metadata is round-tripped through JSON
    THEN all six randomization fields are preserved."""
    from counter.schema import Decision

    src = Decision(
        decision_id="d-rand",
        decision_type="tool_call",
        chosen_action="run_tests",
        policy="epsilon_greedy",
        policy_params={"epsilon": 0.2},
        valid_actions=["run_tests", "inspect_file", "search_docs"],
        propensity=0.85,
        context_features={"step_index": 1},
    )
    raw = src.model_dump_json()
    dst = Decision.model_validate_json(raw)
    assert dst == src
    assert dst.policy == "epsilon_greedy"
    assert dst.policy_params == {"epsilon": 0.2}
    assert dst.valid_actions == ["run_tests", "inspect_file", "search_docs"]
    assert dst.chosen_action == "run_tests"
    assert dst.propensity == 0.85
    assert dst.context_features == {"step_index": 1}


def test_decisions_log_randomization__unrandomized_decision_has_no_propensity() -> None:
    """WHEN a Decision is created without randomization metadata
    THEN the decision serializes without a propensity field and is loadable again without error."""
    from counter.schema import Decision

    src = Decision(decision_id="d-plain", decision_type="plan_step", chosen_action="investigate")
    dumped = json.loads(src.model_dump_json())
    assert dumped.get("propensity") is None
    dst = Decision.model_validate(dumped)
    assert dst == src


def test_decisions_log_randomization__propensity_must_be_in_zero_one_inclusive() -> None:
    """WHEN a Decision is constructed with propensity=0.0 or propensity=1.5
    THEN the system raises a pydantic.ValidationError."""
    from counter.schema import Decision

    base = dict(decision_id="d-bad", decision_type="tool_call")
    with pytest.raises(ValidationError):
        Decision(**base, propensity=0.0)
    with pytest.raises(ValidationError):
        Decision(**base, propensity=1.5)
    # propensity=1.0 is allowed (inclusive upper bound)
    ok = Decision(**base, propensity=1.0)
    assert ok.propensity == 1.0


def test_schema_versioning__unrecognized_version_is_rejected() -> None:
    """WHEN a trace with schema_version="0.99.0" is loaded by a runtime that only recognizes "0.1.0"
    THEN the system raises a clear error naming both the seen and supported versions."""
    from counter.schema import SCHEMA_VERSION, Run

    payload = _fixture()
    payload["schema_version"] = "0.99.0"
    with pytest.raises(ValidationError) as exc_info:
        Run.model_validate(payload)
    msg = str(exc_info.value)
    assert "0.99.0" in msg
    assert SCHEMA_VERSION in msg


def test_schema_versioning__current_version_round_trips() -> None:
    """WHEN a trace produced by the current runtime is loaded by the same runtime
    THEN load succeeds and schema_version survives the round-trip."""
    from counter.schema import SCHEMA_VERSION, Run

    payload = _fixture()
    payload["schema_version"] = SCHEMA_VERSION
    run = Run.model_validate(payload)
    reloaded = Run.model_validate_json(run.model_dump_json())
    assert reloaded.schema_version == SCHEMA_VERSION
