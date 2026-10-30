"""Publication failure, corpus identity, and sidecar boundary coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact import corpus_io
from counterfact.corpus_io import (
    CorpusIOError,
    IngestReceipt,
    load_corpus,
    recover_corpus,
    write_corpus,
)
from counterfact.schema import Outcome, Run


def _run(run_id: str = "r1") -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=run_id,
        outcome=Outcome(kind="binary", value=True, verifier="test"),
    )


def _receipt() -> IngestReceipt:
    return IngestReceipt(source_format="test", source_file="test", generated_count=999)


def test_offline_create_refuses_existing_destination(tmp_path: Path) -> None:
    with pytest.raises(CorpusIOError, match="cannot publish"):
        write_corpus([_run()], tmp_path, _receipt())
    assert list(tmp_path.glob("*.json")) == []


def test_append_counts_committed_files_and_refuses_collisions(tmp_path: Path) -> None:
    destination = tmp_path / "corpus"
    first = _receipt()
    committed = write_corpus([_run()], destination, first)
    assert first.generated_count == 999
    assert committed.generated_count == 1
    second = _receipt()
    committed = write_corpus([_run("r2")], destination, second, append=True)
    assert committed.generated_count == 2
    original = (destination / "r1.json").read_text()
    with pytest.raises(CorpusIOError, match="duplicate run_id"):
        write_corpus([_run()], destination, _receipt(), append=True)
    assert (destination / "r1.json").read_text() == original
    assert len(load_corpus(destination)) == 2


@pytest.mark.parametrize("failure_file", ["r3.json", "ingest-receipt.json"])
def test_interrupted_batch_rejects_reads_then_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_file: str
) -> None:
    destination = tmp_path / "corpus"
    write_corpus([_run()], destination, _receipt())
    original_receipt = (destination / "ingest-receipt.json").read_text()
    real_write = corpus_io.atomic_write

    def fail(path: Path, text: str) -> None:
        if path.name == failure_file:
            raise OSError("injected publication failure")
        real_write(path, text)

    with monkeypatch.context() as patch:
        patch.setattr(corpus_io, "atomic_write", fail)
        with pytest.raises(CorpusIOError, match="injected"):
            write_corpus([_run("r2"), _run("r3")], destination, _receipt(), append=True)
    assert (destination / "ingest-receipt.json").read_text() == original_receipt
    with pytest.raises(CorpusIOError, match="interrupted"):
        load_corpus(destination)
    recover_corpus(destination)
    assert [run.run_id for run in load_corpus(destination)] == ["r1"]
    write_corpus([_run("r2")], destination, _receipt(), append=True)
    assert len(load_corpus(destination)) == 2


def test_receipt_corruption_and_stale_count_are_errors(tmp_path: Path) -> None:
    (tmp_path / "ingest-receipt.json").write_text("{}")
    with pytest.raises(CorpusIOError, match="invalid ingest receipt"):
        load_corpus(tmp_path)
    (tmp_path / "ingest-receipt.json").write_text(_receipt().model_dump_json())
    with pytest.raises(CorpusIOError, match="count"):
        write_corpus([_run()], tmp_path, _receipt(), append=True)
    assert not (tmp_path / "r1.json").exists()


def test_legacy_receipts_read_but_arbitrary_receipt_json_does_not(tmp_path: Path) -> None:
    (tmp_path / "r1.json").write_text(_run().model_dump_json())
    receipt = _receipt().model_dump(exclude={"receipt_version"})
    receipt["generated_count"] = 1
    (tmp_path / "ingest-receipt.json").write_text(json.dumps(receipt))
    assert len(load_corpus(tmp_path)) == 1
    (tmp_path / "mystery-receipt.json").write_text("{}")
    with pytest.raises(CorpusIOError, match="failed to parse"):
        load_corpus(tmp_path)


def test_atomic_write_does_not_delete_another_writers_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "r.json.counterfact-tmp"
    temporary.write_text("other writer")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(corpus_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        corpus_io.atomic_write(tmp_path / "r.json", "our text")
    assert temporary.read_text() == "other writer"


def test_duplicate_files_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "one.json").write_text(_run().model_dump_json())
    (tmp_path / "two.json").write_text(_run().model_dump_json())
    with pytest.raises(CorpusIOError, match="duplicate run_id"):
        load_corpus(tmp_path)


def test_single_writer_lock_blocks_a_second_publisher(tmp_path: Path) -> None:
    import fcntl

    with (tmp_path / ".counterfact-write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(CorpusIOError, match="another writer"):
            write_corpus([_run()], tmp_path, _receipt(), append=True)
    assert not (tmp_path / "r1.json").exists()


def test_distinct_live_writers_share_committed_count(tmp_path: Path) -> None:
    from counterfact.tracing.openai_span_processor import CounterfactSpanProcessor

    first = CounterfactSpanProcessor(tmp_path, outcome_provider=lambda payload: True)
    second = CounterfactSpanProcessor(tmp_path, outcome_provider=lambda payload: True)
    for index, processor in enumerate([first, second, first, second]):
        trace_id = f"live-{index}"
        processor.on_span_end(
            {"id": "root", "trace_id": trace_id, "parent_id": None, "span_data": {"type": "agent"}}
        )
        processor.on_trace_end({"id": trace_id})
    assert len(load_corpus(tmp_path)) == 4
    assert json.loads((tmp_path / "ingest-receipt.json").read_text())["generated_count"] == 4


def test_custom_export_filename_sidecar_remains_loadable(tmp_path: Path) -> None:
    from counterfact.runrecord_export import export_runrecord_parquet

    destination = tmp_path / "corpus"
    write_corpus([_run()], destination, _receipt())
    export_runrecord_parquet(
        [_run()], source_corpus=destination, output_path=destination / "export.data"
    )
    assert [run.run_id for run in load_corpus(destination)] == ["r1"]


def test_cli_focal_requires_same_content_as_corpus(tmp_path: Path, capsys) -> None:
    from counterfact.cli.loaders import load_focal_and_corpus

    corpus = tmp_path / "corpus"
    write_corpus([_run()], corpus, _receipt())
    focal = _run()
    focal.metadata.notes = "different trace with the same identity"
    focal_path = tmp_path / "focal.json"
    focal_path.write_text(focal.model_dump_json())
    assert load_focal_and_corpus(focal_path, corpus, command="diagnose") is None
    captured = capsys.readouterr()
    assert not captured.out
    assert "differs" in captured.err


@pytest.mark.parametrize("run_id", ["a.json.receipt", "a.parquet.receipt"])
def test_receipt_like_native_run_ids_publish_and_load(tmp_path: Path, run_id: str) -> None:
    destination = tmp_path / "corpus"
    runs = [_run("a"), _run(run_id)]
    receipt = write_corpus(runs, destination, _receipt())
    assert receipt.generated_count == 2
    assert {run.run_id for run in load_corpus(destination)} == {run.run_id for run in runs}
