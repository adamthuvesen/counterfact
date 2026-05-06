"""CLI acceptance coverage for trace-forensics workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.synthetic import generate_traces
from counterfact.cli import main
from counterfact.diagnose import DiagnosisReport
from counterfact.schema import Run


def _write_synthetic_corpus(target: Path, *, n: int = 48, seed: int = 42) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for trace in generate_traces(n=n, seed=seed):
        run = Run.model_validate(trace)
        path = target / f"{run.run_id}.json"
        path.write_text(run.model_dump_json())
        paths.append(path)
    return paths


def test_diagnose_cli__json_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _write_synthetic_corpus(tmp_path / "runs")

    rc = main(
        [
            "diagnose",
            str(paths[0]),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--bootstrap",
            "10",
            "--json",
        ]
    )

    assert rc == 0
    report = DiagnosisReport.model_validate_json(capsys.readouterr().out)
    assert report.entries
    assert report.entries[0].decision_id


def test_diagnose_cli__missing_run_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["diagnose", str(tmp_path / "missing.json")])

    assert rc == 2
    assert "run JSON not found" in capsys.readouterr().err


def test_diagnose_cli__focal_not_in_corpus_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_synthetic_corpus(tmp_path / "runs")
    other_dir = tmp_path / "other"
    other_paths = _write_synthetic_corpus(other_dir, seed=99)
    focal_payload = json.loads(paths[0].read_text())
    focal_payload["run_id"] = "intruder-run"
    paths[0].write_text(json.dumps(focal_payload))

    rc = main(["diagnose", str(paths[0]), "--runs-dir", str(other_dir)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err
    assert other_paths[0].exists()


def test_compare_cli__descriptive_without_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_synthetic_corpus(tmp_path / "runs", n=2)

    rc = main(["compare", str(paths[0]), str(paths[1])])

    assert rc == 0
    out = capsys.readouterr().out
    assert "descriptive trace diff" in out
    assert "outcome_delta:" not in out


def test_compare_cli__diagnosis_overlay_preserves_unidentified_hide_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    single = Path("bench/real/single_class_refusal")
    paths = sorted(single.glob("*.json"))

    rc = main(
        [
            "compare",
            str(paths[0]),
            str(paths[1]),
            "--runs-dir",
            str(single),
            "--focal",
            "right",
            "--bootstrap",
            "10",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "diagnosis_overlay:" in out
    assert "identifiability=unidentified" in out
    assert "outcome_delta:" not in out


def test_ingest_and_export_cli__write_receipts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "imported-1",
                "passed": True,
                "steps": [
                    {
                        "step_index": 0,
                        "decisions": [
                            {
                                "decision_id": "d1",
                                "decision_type": "model_call",
                                "chosen_action": "small",
                            }
                        ],
                        "observations": [],
                    }
                ],
            }
        )
        + "\n"
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "fields": {
                    "run_id": "id",
                    "steps": "steps",
                    "outcome.value": "passed",
                },
                "defaults": {
                    "schema_version": "0.1.0",
                    "outcome.kind": "binary",
                    "outcome.verifier": "imported",
                },
            }
        )
    )
    out_dir = tmp_path / "native"

    rc = main(
        [
            "ingest",
            "generic-jsonl",
            str(source),
            "--mapping",
            str(mapping),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "ingest-receipt.json").exists()
    assert "wrote 1 trace" in capsys.readouterr().out

    export_path = tmp_path / "eval-audit-runs.parquet"
    rc = main(
        [
            "export-runs",
            str(out_dir),
            "--to",
            "eval-audit-parquet",
            "--output",
            str(export_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert export_path.exists()
    assert export_path.with_suffix(".parquet.receipt.json").exists()
    for forbidden in ("switch", "hold", "hedge_on_cost", "drop_from_shortlist"):
        assert forbidden not in out
