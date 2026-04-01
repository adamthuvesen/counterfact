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
