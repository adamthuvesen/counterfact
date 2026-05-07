"""End-to-end: ingest a fixture corpus through an SDK adapter, then diagnose it.

These tests assert that the headline 30-second story works:
    1. counterfact ingest <sdk-adapter> <fixture> --output-dir corpus/
    2. counterfact diagnose corpus/<run>.json --runs-dir corpus/

The diagnose output is asserted to be a valid structured artifact (no schema
violations, no spurious errors). Whether the small fixture corpus identifies
a load-bearing decision depends on the corpus shape; the round-trip itself is
the contract under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from counterfact.cli import main

CLAUDE_FIXTURE = (
    Path(__file__).parent.parent / "fixtures/adapters/claude_agent_sdk/minimal.jsonl"
)
OPENAI_FIXTURE_DIR = (
    Path(__file__).parent.parent / "fixtures/adapters/openai_agents"
)


def test_claude_round_trip_ingest_then_diagnose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "claude-agent-sdk",
            str(CLAUDE_FIXTURE),
            "--output-dir",
            str(corpus),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = main(
        [
            "diagnose",
            str(corpus / "sess-fail-002.json"),
            "--runs-dir",
            str(corpus),
            "--top-k",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out  # non-empty JSON payload
    import json as _json

    payload = _json.loads(captured.out)
    assert payload["run_id"] == "sess-fail-002"
    assert payload["outcome"] == "fail"
    assert isinstance(payload.get("entries", []), list)


def test_openai_round_trip_ingest_then_diagnose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "openai-agents",
            str(OPENAI_FIXTURE_DIR / "with_handoff.json"),
            "--output-dir",
            str(corpus),
        ]
    )
    assert rc == 0
    rc = main(
        [
            "ingest",
            "openai-agents",
            str(OPENAI_FIXTURE_DIR / "root_error.json"),
            "--output-dir",
            str(corpus),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = main(
        [
            "diagnose",
            str(corpus / "trace-ai-error.json"),
            "--runs-dir",
            str(corpus),
            "--top-k",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    import json as _json

    payload = _json.loads(captured.out)
    assert payload["run_id"] == "trace-ai-error"
    assert payload["outcome"] == "fail"
    assert isinstance(payload.get("entries", []), list)
