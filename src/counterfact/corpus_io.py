"""Native corpus files, ingest receipts, and single-writer publication."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from counterfact.schema import Run, validate_runs

_RECEIPT = "ingest-receipt.json"
_PENDING = ".counterfact-write-in-progress"


class CorpusIOError(RuntimeError):
    """A corpus cannot be read or published safely."""


class IngestReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_version: Literal["1", "2"] = "2"
    source_format: str
    source_file: str
    mapping_file: str = ""
    generated_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    dropped_fields: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class RunRecordExportReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    export_version: Literal["1", "2"] = "2"
    source_corpus: str
    output_path: str
    row_count: int
    field_derivations: dict[str, str]
    missing_cost_policy: str
    warnings: list[str] = Field(default_factory=list)
    migration_notes: list[str] = Field(default_factory=list)


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except (OSError, UnicodeError) as exc:
        raise CorpusIOError(f"cannot read {path}: {exc}") from exc


def read_json(path: Path) -> object:
    try:
        return json.loads(read_text(path))
    except ValueError as exc:
        raise CorpusIOError(f"invalid JSON in {path}: {exc}") from exc


def load_run(path: Path) -> Run:
    try:
        return Run.model_validate_json(read_text(path))
    except ValueError as exc:
        raise CorpusIOError(f"failed to parse {path}: {exc}") from exc


def _receipt(path: Path) -> IngestReceipt | None:
    receipt_path = path / _RECEIPT
    if not receipt_path.exists():
        return None
    try:
        return IngestReceipt.model_validate_json(read_text(receipt_path))
    except ValueError as exc:
        raise CorpusIOError(f"invalid ingest receipt {receipt_path}: {exc}") from exc


def _check_pending(path: Path) -> None:
    if (path / _PENDING).exists():
        raise CorpusIOError(
            f"interrupted corpus publication in {path}; "
            "call counterfact.corpus_io.recover_corpus(path) to roll back the interrupted write"
        )


def load_corpus(path: Path) -> list[Run]:
    if not path.is_dir():
        raise CorpusIOError(f"corpus directory not found: {path}")
    _check_pending(path)
    receipt = _receipt(path)
    runs: list[Run] = []
    try:
        files = sorted(path.iterdir())
        for file in files:
            if file.suffix != ".json" or file.name == _RECEIPT:
                continue
            text = read_text(file)
            try:
                run = Run.model_validate_json(text)
            except ValueError as run_error:
                artifact = file.with_name(file.name.removesuffix(".receipt.json"))
                if not file.name.endswith(".receipt.json") or not artifact.is_file():
                    raise CorpusIOError(f"failed to parse {file}: {run_error}") from run_error
                try:
                    export_receipt = RunRecordExportReceipt.model_validate_json(text)
                except ValueError as exc:
                    raise CorpusIOError(f"invalid export receipt {file}: {exc}") from exc
                if Path(export_receipt.output_path).name != artifact.name:
                    raise CorpusIOError(
                        f"export receipt names a different artifact: {file}"
                    ) from None
            else:
                runs.append(run)
        runs = validate_runs(runs)
    except (OSError, ValueError) as exc:
        raise CorpusIOError(f"cannot load corpus {path}: {exc}") from exc
    _check_pending(path)
    if receipt is not None and receipt.generated_count != len(runs):
        raise CorpusIOError(
            f"ingest receipt count {receipt.generated_count} does not match "
            f"{len(runs)} committed runs in {path}"
        )
    return runs


def _filename(run_id: str) -> str:
    if (
        not run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or "\x00" in run_id
        or run_id == "ingest-receipt"
    ):
        raise CorpusIOError(f"unsafe run_id for output filename: {run_id!r}")
    return f"{run_id}.json"


def atomic_write(path: Path, text: str) -> None:
    with NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".counterfact-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_corpus(
    runs: list[Run], output_dir: Path, receipt: IngestReceipt, *, append: bool = False
) -> IngestReceipt:
    receipt = receipt.model_copy(deep=True)
    try:
        runs = validate_runs(runs)
        filenames = [_filename(run.run_id) for run in runs]
        output_dir.mkdir(parents=True, exist_ok=append)
        with (output_dir / ".counterfact-write.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CorpusIOError(f"another writer owns corpus {output_dir}") from exc
            existing = load_corpus(output_dir)
            previous = _receipt(output_dir)
            if existing and previous is None:
                raise CorpusIOError("cannot append to an unreceipted legacy corpus")
            if previous is not None and previous.source_format != receipt.source_format:
                raise CorpusIOError("cannot append a different source format to this corpus")
            validate_runs([*existing, *runs])
            for filename in filenames:
                if (output_dir / filename).exists():
                    raise CorpusIOError(f"run output already exists: {output_dir / filename}")
            receipt.generated_count = len(existing) + len(runs)
            if previous is not None:
                receipt.warnings = list(dict.fromkeys([*previous.warnings, *receipt.warnings]))
                receipt.dropped_fields = sorted(
                    set(previous.dropped_fields + receipt.dropped_fields)
                )
            old_receipt = read_text(output_dir / _RECEIPT) if previous else None
            marker = {"filenames": filenames, "previous_receipt": old_receipt}
            atomic_write(output_dir / _PENDING, json.dumps(marker))
            # Leave the marker on failure. Readers reject the entire batch until explicit recovery.
            for run, filename in zip(runs, filenames, strict=True):
                atomic_write(output_dir / filename, run.model_dump_json(indent=2) + "\n")
            atomic_write(output_dir / _RECEIPT, receipt.model_dump_json(indent=2) + "\n")
            (output_dir / _PENDING).unlink()
            return receipt
    except (OSError, ValueError) as exc:
        raise CorpusIOError(f"cannot publish corpus {output_dir}: {exc}") from exc


class _InterruptedWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filenames: list[str]
    previous_receipt: str | None


def recover_corpus(path: Path) -> None:
    """Roll back only files named by an interrupted publication, preserving prior runs."""
    try:
        with (path / ".counterfact-write.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker = _InterruptedWrite.model_validate_json(read_text(path / _PENDING))
            for filename in marker.filenames:
                if not filename.endswith(".json") or _filename(filename[:-5]) != filename:
                    raise CorpusIOError("invalid interrupted publication filename")
            if marker.previous_receipt is not None:
                IngestReceipt.model_validate_json(marker.previous_receipt)
            for filename in marker.filenames:
                (path / filename).unlink(missing_ok=True)
            if marker.previous_receipt is None:
                (path / _RECEIPT).unlink(missing_ok=True)
            else:
                atomic_write(path / _RECEIPT, marker.previous_receipt)
            (path / _PENDING).unlink()
    except (OSError, ValidationError) as exc:
        raise CorpusIOError(f"cannot recover corpus {path}: {exc}") from exc
