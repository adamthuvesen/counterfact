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

import json
from pathlib import Path

import pytest

from counterfact.cli import main

CLAUDE_FIXTURE = Path(__file__).parent.parent / "fixtures/adapters/claude_agent_sdk/minimal.jsonl"
OPENAI_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures/adapters/openai_agents"


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

    source = tmp_path / "combined.json"
    source.write_text(
        json.dumps(
            [
                json.loads((OPENAI_FIXTURE_DIR / "with_handoff.json").read_text()),
                json.loads((OPENAI_FIXTURE_DIR / "root_error.json").read_text()),
            ]
        )
    )
    rc = main(["ingest", "openai-agents", str(source), "--output-dir", str(corpus)])
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


def test_ingest_analyze_diagnose_export_uses_one_corpus_boundary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import polars as pl

    corpus = tmp_path / "corpus"
    assert (
        main(["ingest", "claude-agent-sdk", str(CLAUDE_FIXTURE), "--output-dir", str(corpus)]) == 0
    )
    capsys.readouterr()
    assert main(["analyze", "corpus", str(corpus)]) in {0, 1}
    assert not capsys.readouterr().err
    assert (
        main(["diagnose", str(corpus / "sess-fail-002.json"), "--runs-dir", str(corpus), "--json"])
        == 0
    )
    capsys.readouterr()
    output = corpus / "runs.parquet"
    assert (
        main(
            [
                "export-runs",
                str(corpus),
                "--to",
                "runrecord-parquet",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    frame = pl.read_parquet(output)
    assert frame["reported_run_total_cost_usd"].to_list() == [0.004, 0.012]
    assert frame["tokens_in"].to_list() == [1800, 1500]
    assert main(["analyze", "corpus", str(corpus)]) in {0, 1}
    assert not capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["ingest", "openai-agents", "missing.json", "--output-dir", "out"],
        ["ingest", "claude-agent-sdk", "missing.jsonl", "--output-dir", "out"],
        [
            "ingest",
            "generic-jsonl",
            "missing.jsonl",
            "--mapping",
            "missing.yml",
            "--output-dir",
            "out",
        ],
        ["export-runs", "missing-corpus", "--to", "runrecord-parquet"],
    ],
)
def test_missing_io_paths_are_concise_cli_errors(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "counterfact" in captured.err
