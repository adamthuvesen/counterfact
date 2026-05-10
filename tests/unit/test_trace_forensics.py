"""Trace-forensics workflow tests for diagnose, compare, ingest, and export."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from counterfact.compare import compare_traces
from counterfact.diagnose import build_diagnosis
from counterfact.ingest import IngestError, ingest_generic_jsonl
from counterfact.intervene.estimate import IdentifiabilityStatus
from counterfact.runrecord_export import export_runrecord_parquet
from counterfact.schema import Decision, Run, Step
from tests.conftest import synthetic_corpus


def test_build_diagnosis__deterministic_for_fixed_inputs() -> None:
    corpus = synthetic_corpus(n=48, seed=42)
    a = build_diagnosis(corpus[0], corpus, bootstrap=10, seed=42)
    b = build_diagnosis(corpus[0], corpus, bootstrap=10, seed=42)

    assert a.model_dump(mode="python") == b.model_dump(mode="python")
    assert a.entries
    assert a.summary.startswith(f"Run {corpus[0].run_id}")


def test_build_diagnosis__decision_type_filters_ranked_entries() -> None:
    corpus = synthetic_corpus(n=60, seed=42)

    model_report = build_diagnosis(
        corpus[0],
        corpus,
        decision_type="model_call",
        top_k=10,
        bootstrap=10,
        seed=42,
    )
    tool_report = build_diagnosis(
        corpus[0],
        corpus,
        decision_type="tool_call",
        top_k=10,
        bootstrap=10,
        seed=42,
    )

    assert model_report.entries
    assert tool_report.entries
    assert {entry.decision_type for entry in model_report.entries} == {"model_call"}
    assert {entry.decision_type for entry in tool_report.entries} == {"tool_call"}


def test_build_diagnosis__single_class_refusal_has_no_outcome_delta() -> None:
    paths = sorted(Path("bench/real/single_class_refusal").glob("*.json"))
    corpus = [Run.model_validate_json(path.read_text()) for path in paths]

    report = build_diagnosis(corpus[0], corpus, bootstrap=10, seed=42)

    assert report.entries
    assert {entry.identifiability for entry in report.entries} == {
        IdentifiabilityStatus.UNIDENTIFIED
    }
    assert all(entry.outcome_delta is None for entry in report.entries)
    assert any(entry.next_step.action == "broaden_arm_support" for entry in report.entries)


def test_build_diagnosis__repeated_decision_type_is_unidentified() -> None:
    corpus = synthetic_corpus(n=60, seed=7)
    focal = corpus[0]
    repeated = Decision(
        decision_id="d-extra-model",
        decision_type="model_call",
        chosen_action="haiku",
    )
    focal = focal.model_copy(
        update={
            "steps": [
                *focal.steps,
                Step(step_index=99, decisions=[repeated], observations=[]),
            ]
        }
    )
    corpus = [focal, *corpus[1:]]

    report = build_diagnosis(focal, corpus, top_k=10, bootstrap=10, seed=42)
    model_entries = [entry for entry in report.entries if entry.decision_type == "model_call"]

    assert model_entries
    assert all(
        entry.identifiability == IdentifiabilityStatus.UNIDENTIFIED for entry in model_entries
    )
    assert "no supported" in report.summary or "most plausible" in report.summary


def test_compare_traces__descriptive_diff_without_diagnosis() -> None:
    left, right = synthetic_corpus(n=2, seed=11)
    comparison = compare_traces(left, right)

    assert comparison.left_run_id == left.run_id
    assert comparison.right_run_id == right.run_id
    assert "descriptive trace diff" in comparison.note
    assert comparison.diagnosis is None


def test_ingest_generic_jsonl__mapped_records_write_native_traces(tmp_path: Path) -> None:
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

    receipt = ingest_generic_jsonl(source, mapping, tmp_path / "out")

    assert receipt.generated_count == 1
    assert receipt.warnings
    Run.model_validate_json((tmp_path / "out" / "imported-1.json").read_text())
    assert (tmp_path / "out" / "ingest-receipt.json").exists()


def test_ingest_generic_jsonl__missing_required_mapping_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "bad"}) + "\n")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"fields": {"run_id": "id"}, "defaults": {}}))

    with pytest.raises(IngestError, match="missing required target mapping"):
        ingest_generic_jsonl(source, mapping, tmp_path / "out")


def test_ingest_generic_jsonl__partial_randomization_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "run_id": "partial-randomization",
                "steps": [
                    {
                        "step_index": 0,
                        "decisions": [
                            {
                                "decision_id": "d1",
                                "decision_type": "model_call",
                                "chosen_action": "small",
                                "policy": "epsilon_greedy",
                            }
                        ],
                        "observations": [],
                    }
                ],
                "outcome": {"kind": "binary", "value": False, "verifier": "x"},
            }
        )
        + "\n"
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"mode": "native"}))

    with pytest.raises(IngestError, match="randomized decisions"):
        ingest_generic_jsonl(source, mapping, tmp_path / "out")


def test_export_runrecord_parquet__writes_rows_and_receipt(tmp_path: Path) -> None:
    runs = synthetic_corpus(n=3, seed=42)
    output = tmp_path / "runs.parquet"

    receipt = export_runrecord_parquet(runs, source_corpus=tmp_path / "runs", output_path=output)

    assert output.exists()
    assert output.with_suffix(".parquet.receipt.json").exists()
    assert receipt.row_count == 3
    frame = pl.read_parquet(output)
    assert frame.height == 3
    assert set(frame["cost_provenance"].to_list()) == {"cost_not_available"}
