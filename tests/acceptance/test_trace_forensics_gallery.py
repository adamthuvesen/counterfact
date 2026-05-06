"""Acceptance tests for the committed trace-forensics gallery."""

from __future__ import annotations

from pathlib import Path

import pytest

from counterfact.cli import main
from counterfact.schema import Run

GALLERY = Path("examples/trace-forensics")


def test_gallery__readme_lists_expected_cases() -> None:
    readme = (GALLERY / "README.md").read_text()

    for heading in [
        "Wrong Model Choice",
        "Bad Tool Choice",
        "Missed Retry",
        "Stopped Too Early",
        "Unsupported Counterfactual",
        "Single-Class Support Refusal",
        "Pass/Fail Trace Diff",
    ]:
        assert f"## {heading}" in readme
    assert "illustrative" in readme
    assert "not benchmark evidence" in readme
    assert "not a leaderboard" in readme


def test_gallery__fixtures_validate_as_native_runs() -> None:
    paths = sorted(GALLERY.glob("**/*.json"))
    assert paths

    for path in paths:
        run = Run.model_validate_json(path.read_text())
        assert run.run_id
        assert run.steps


def test_gallery__diagnose_and_compare_commands_run_without_provider_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html_path = tmp_path / "wrong-model.html"
    commands = [
        [
            "diagnose",
            "examples/trace-forensics/runs/syn-000000.json",
            "--runs-dir",
            "examples/trace-forensics/runs",
            "--decision-type",
            "model_call",
            "--bootstrap",
            "10",
            "--html",
            str(html_path),
        ],
        [
            "diagnose",
            "examples/trace-forensics/runs/syn-000000.json",
            "--runs-dir",
            "examples/trace-forensics/runs",
            "--decision-type",
            "tool_call",
            "--bootstrap",
            "10",
        ],
        [
            "diagnose",
            "examples/trace-forensics/runs/syn-000000.json",
            "--runs-dir",
            "examples/trace-forensics/runs",
            "--decision-type",
            "retry",
            "--bootstrap",
            "10",
        ],
        [
            "diagnose",
            "examples/trace-forensics/single-arm-model/single-arm-000000.json",
            "--runs-dir",
            "examples/trace-forensics/single-arm-model",
            "--decision-type",
            "model_call",
            "--bootstrap",
            "10",
        ],
        [
            "diagnose",
            "bench/real/single_class_refusal/real-csv_dedupe-000000.json",
            "--runs-dir",
            "bench/real/single_class_refusal",
            "--bootstrap",
            "10",
        ],
        [
            "compare",
            "examples/trace-forensics/stopped-early/pass.json",
            "examples/trace-forensics/stopped-early/fail.json",
        ],
        [
            "compare",
            "examples/trace-forensics/runs/syn-000003.json",
            "examples/trace-forensics/runs/syn-000000.json",
            "--runs-dir",
            "examples/trace-forensics/runs",
            "--focal",
            "right",
            "--bootstrap",
            "10",
        ],
    ]

    for command in commands:
        rc = main(command)
        assert rc == 0
        capsys.readouterr()

    assert html_path.exists()
