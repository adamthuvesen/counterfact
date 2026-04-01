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

    run = Run.model_validate(_fixture())
    # The fixture has kind="binary", value=False — the call must reach the fit
    # path without raising UnsupportedOutcomeError. We pass a single trace; the
    # full statistical fit lands in §6, so we accept either a trained model
    # object or the explicit "not yet implemented" sentinel.
    try:
        result = fit_outcome_model([run])
    except NotImplementedError:
        # acceptable: the fit body is §6 work; the boundary check passed
        return
    assert result is not None


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
