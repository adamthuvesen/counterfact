"""End-to-end tests for the `counterfact intervene` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact.cli import main
from counterfact.intervene.estimate import CausalEstimate, IdentifiabilityStatus
from tests.conftest import write_synthetic_corpus


def _decision_id_for(path: Path, decision_type: str) -> str:
    payload = json.loads(path.read_text())
    for step in payload["steps"]:
        for decision in step["decisions"]:
            if decision["decision_type"] == decision_type:
                return str(decision["decision_id"])
    raise AssertionError(f"no {decision_type} decision in {path}")


def _move_termination_into_model_step(path: Path) -> str:
    payload = json.loads(path.read_text())
    model_step = next(
        step
        for step in payload["steps"]
        if any(d["decision_type"] == "model_call" for d in step["decisions"])
    )
    term_step = next(
        step
        for step in payload["steps"]
        if any(d["decision_type"] == "termination" for d in step["decisions"])
    )
    term_decision = next(d for d in term_step["decisions"] if d["decision_type"] == "termination")
    model_step["decisions"].append(term_decision)
    payload["steps"] = [
        step for step in payload["steps"] if step["step_index"] != term_step["step_index"]
    ]
    path.write_text(json.dumps(payload) + "\n")
    return next(
        d["decision_id"] for d in model_step["decisions"] if d["decision_type"] == "model_call"
    )


def test_intervene__decision_id_json_round_trips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    focal = write_synthetic_corpus(runs_dir, n=80)[0]
    decision_id = _decision_id_for(focal, "model_call")

    rc = main(
        [
            "intervene",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--decision-id",
            decision_id,
            "--set",
            "model_choice=sonnet",
            "--bootstrap",
            "20",
            "--json",
        ]
    )

    assert rc == 0
    estimate = CausalEstimate.model_validate_json(capsys.readouterr().out)
    assert estimate.identifiability in {
        IdentifiabilityStatus.IDENTIFIED,
        IdentifiabilityStatus.UNIDENTIFIED,
    }
    assert estimate.query.intervention_kind == "model_choice"
    assert estimate.query.target == "sonnet"
    assert estimate.next_step.payload["decision_id"] == decision_id
    assert estimate.next_step.payload["targeting_mode"] == "decision_id"


def test_intervene__decision_id_targets_decision_inside_multi_decision_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    focal = write_synthetic_corpus(runs_dir, n=80)[0]
    decision_id = _move_termination_into_model_step(focal)

    rc = main(
        [
            "intervene",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--decision-id",
            decision_id,
            "--set",
            "model_choice=sonnet",
            "--bootstrap",
            "20",
            "--json",
        ]
    )

    assert rc == 0
    estimate = CausalEstimate.model_validate_json(capsys.readouterr().out)
    assert estimate.query.decision_type == "model_call"
    assert estimate.next_step.payload["decision_id"] == decision_id
    assert estimate.next_step.payload["targeting_mode"] == "decision_id"


def test_intervene__output_writes_json_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    focal = write_synthetic_corpus(runs_dir, n=80)[0]
    decision_id = _decision_id_for(focal, "tool_call")
    output = tmp_path / "estimate.json"

    rc = main(
        [
            "intervene",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--decision-id",
            decision_id,
            "--set",
            "tool_choice=run_tests",
            "--bootstrap",
            "20",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    CausalEstimate.model_validate_json(output.read_text())
    captured = capsys.readouterr()
    assert str(output.resolve()) in captured.err
    assert str(output.resolve()) not in captured.out


def test_intervene__json_plus_output_separates_streams(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    focal = write_synthetic_corpus(runs_dir, n=80)[0]
    decision_id = _decision_id_for(focal, "model_call")
    output = tmp_path / "estimate.json"

    rc = main(
        [
            "intervene",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--decision-id",
            decision_id,
            "--set",
            "model_choice=sonnet",
            "--bootstrap",
            "20",
            "--json",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    captured = capsys.readouterr()
    CausalEstimate.model_validate_json(captured.out)
    assert str(output.resolve()) not in captured.out
    assert str(output.resolve()) in captured.err


def test_intervene__human_output_names_decision_and_next_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    focal = write_synthetic_corpus(runs_dir, n=80)[0]
    decision_id = _decision_id_for(focal, "retry")

    rc = main(
        [
            "intervene",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--decision-id",
            decision_id,
            "--set",
            "retry_policy=retry_once",
            "--bootstrap",
            "20",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert decision_id in out
    assert "identifiability:" in out
    assert "next_step:" in out


def test_intervene__missing_arm_surfaces_support_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    focal = write_synthetic_corpus(runs_dir, n=80)[0]
    decision_id = _decision_id_for(focal, "model_call")

    rc = main(
        [
            "intervene",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--decision-id",
            decision_id,
            "--set",
            "model_choice=opus",
            "--bootstrap",
            "20",
            "--json",
        ]
    )

    estimate = CausalEstimate.model_validate_json(capsys.readouterr().out)
    assert rc == 0
    assert estimate.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert estimate.next_step.action == "broaden_arm_support"
    assert "opus" in estimate.next_step.payload["missing_arms"]


def test_intervene__invalid_inputs_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    focal = write_synthetic_corpus(runs_dir, n=80)[0]
    decision_id = _decision_id_for(focal, "model_call")

    assert (
        main(
            [
                "intervene",
                str(focal),
                "--runs-dir",
                str(runs_dir),
                "--decision-id",
                "missing",
                "--set",
                "model_choice=sonnet",
            ]
        )
        == 2
    )
    assert "missing" in capsys.readouterr().err

    assert (
        main(
            [
                "intervene",
                str(focal),
                "--runs-dir",
                str(runs_dir),
                "--decision-id",
                decision_id,
                "--set",
                "model_choice",
            ]
        )
        == 2
    )
    assert "key=value" in capsys.readouterr().err

    assert (
        main(
            [
                "intervene",
                str(focal),
                "--runs-dir",
                str(runs_dir),
                "--decision-id",
                decision_id,
                "--set",
                "tool_choice=run_tests",
            ]
        )
        == 2
    )
    assert "not valid" in capsys.readouterr().err

    assert (
        main(
            [
                "intervene",
                str(focal),
                "--runs-dir",
                str(runs_dir),
                "--decision-id",
                decision_id,
                "--step",
                "2",
                "--set",
                "model_choice=sonnet",
            ]
        )
        == 2
    )
    assert "only one targeting mode" in capsys.readouterr().err


def test_intervene__ambiguous_step_tells_user_to_use_decision_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    canonical = Path("tests/fixtures/canonical_run.json")
    focal = runs_dir / canonical.name
    focal.write_text(canonical.read_text())

    rc = main(
        [
            "intervene",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--step",
            "2",
            "--set",
            "model_choice=claude-sonnet-4-6",
        ]
    )

    err = capsys.readouterr().err
    assert rc == 2
    assert "multiple decisions" in err
    assert "--decision-id" in err
