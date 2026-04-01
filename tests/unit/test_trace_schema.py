"""Tests for trace-schema spec."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "canonical_run.json"


def test_native_json_format__round_trip_preserves_run_fixture() -> None:
    """WHEN a canonical fixture trace JSON is loaded into a `Run` model and re-serialized to JSON
    THEN the resulting JSON parses back into an equal `Run` instance with no field loss."""
    from counter.schema import Run

    raw = FIXTURE_PATH.read_text()
    run = Run.model_validate_json(raw)
    dumped = run.model_dump_json()
    reloaded = Run.model_validate_json(dumped)
    assert reloaded == run
