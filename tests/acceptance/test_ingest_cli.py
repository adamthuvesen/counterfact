"""End-to-end acceptance tests for `counterfact ingest` subcommands.

Runs `counterfact.cli.main([...])` directly (not subprocess) and asserts on
the resulting corpus directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact.cli import main
from counterfact.schema import Run

CLAUDE_FIXTURE = Path(__file__).parent.parent / "fixtures/adapters/claude_agent_sdk/minimal.jsonl"
OPENAI_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures/adapters/openai_agents"


def test_claude_agent_sdk_subcommand_writes_corpus_and_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "claude-agent-sdk",
            str(CLAUDE_FIXTURE),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote 2 trace(s)" in captured.out

    pass_run = Run.model_validate_json((out_dir / "sess-pass-001.json").read_text())
    assert pass_run.outcome.value is True
    fail_run = Run.model_validate_json((out_dir / "sess-fail-002.json").read_text())
    assert fail_run.outcome.value is False

    receipt = json.loads((out_dir / "ingest-receipt.json").read_text())
    assert receipt["source_format"] == "claude-agent-sdk"
    assert receipt["generated_count"] == 2
    assert any("randomization" in w for w in receipt["warnings"])


def test_claude_agent_sdk_subcommand_supports_json_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "claude-agent-sdk",
            str(CLAUDE_FIXTURE),
            "--output-dir",
            str(out_dir),
            "--json",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["source_format"] == "claude-agent-sdk"
    assert payload["generated_count"] == 2


def test_claude_agent_sdk_subcommand_aborts_on_unknown_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_source = tmp_path / "bad.jsonl"
    bad_source.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "__type__": "AssistantMessage",
                        "model": "claude-sonnet-4-6",
                        "session_id": "sess-bad",
                        "content": [{"__type__": "FutureWidgetBlock", "x": 1}],
                    },
                    {
                        "__type__": "ResultMessage",
                        "subtype": "success",
                        "session_id": "sess-bad",
                        "duration_ms": 1,
                        "duration_api_ms": 1,
                        "is_error": False,
                        "num_turns": 1,
                    },
                ]
            }
        )
        + "\n"
    )
    out_dir = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "claude-agent-sdk",
            str(bad_source),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert "FutureWidgetBlock" in captured.err
    # No partial corpus written.
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []


def test_openai_agents_subcommand_with_pass_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "openai-agents",
            str(OPENAI_FIXTURE_DIR / "minimal.json"),
            "--output-dir",
            str(out_dir),
            "--outcome",
            "pass",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote 1 trace(s)" in captured.out
    run = Run.model_validate_json((out_dir / "trace-ai-001.json").read_text())
    assert run.outcome.value is True


def test_openai_agents_subcommand_handoff_uses_marker_outcome(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "openai-agents",
            str(OPENAI_FIXTURE_DIR / "with_handoff.json"),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    run = Run.model_validate_json((out_dir / "trace-ai-handoff.json").read_text())
    assert run.outcome.value is True
    assert run.outcome.verifier == "counterfact_outcome_marker"
    assert any(d.chosen_action.startswith("handoff:") for s in run.steps for d in s.decisions)


def test_openai_agents_subcommand_root_error_outcome(tmp_path: Path) -> None:
    out_dir = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "openai-agents",
            str(OPENAI_FIXTURE_DIR / "root_error.json"),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    run = Run.model_validate_json((out_dir / "trace-ai-error.json").read_text())
    assert run.outcome.value is False
    assert run.outcome.verifier == "openai_agents_root_span_error"


def test_openai_agents_requires_outcome_when_indeterminate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "corpus"

    rc = main(
        [
            "ingest",
            "openai-agents",
            str(OPENAI_FIXTURE_DIR / "minimal.json"),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert "explicit outcome" in captured.err
    # No partial corpus.
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []
