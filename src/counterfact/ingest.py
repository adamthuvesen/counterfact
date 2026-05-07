"""Generic trace ingest boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from counterfact.adapters._common import (
    IngestError,
    IngestReceipt,
    per_decision_randomization_warnings,
    randomization_warning,
    write_corpus,
)
from counterfact.schema import Run


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _read_simple_yaml_mapping(text)


def _read_simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the tiny mapping subset documented for generic-jsonl.

    Supported shape:

        mode: mapped
        fields:
          run_id: id
        defaults:
          schema_version: 0.1.0

    This is intentionally narrow; users needing full YAML can pass JSON.
    """
    out: dict[str, Any] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, sep, value = line.partition(":")
            if not sep:
                raise IngestError(f"invalid mapping line: {raw_line!r}")
            key = key.strip()
            value = value.strip()
            if value:
                out[key] = _coerce_scalar(value)
                current = None
            else:
                out[key] = {}
                current = key
            continue
        if current is None:
            raise IngestError(f"indented mapping line without section: {raw_line!r}")
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise IngestError(f"invalid mapping line: {raw_line!r}")
        out[current][key.strip()] = _coerce_scalar(value.strip())
    return out


def _coerce_scalar(value: str) -> Any:
    if value in {"null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    return value.strip('"').strip("'")


def _get_path(payload: dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    cur = payload
    parts = path.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


_REQUIRED_TARGETS = {
    "schema_version",
    "run_id",
    "steps",
    "outcome.kind",
    "outcome.value",
    "outcome.verifier",
}


def _mapped_payload(
    source: dict[str, Any], mapping: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    fields = mapping.get("fields", {})
    defaults = mapping.get("defaults", {})
    if not isinstance(fields, dict) or not isinstance(defaults, dict):
        raise IngestError("mapping must contain object-valued 'fields' and 'defaults'")
    missing = sorted(
        target
        for target in _REQUIRED_TARGETS
        if target not in fields and target not in defaults
    )
    if missing:
        raise IngestError("missing required target mapping(s): " + ", ".join(missing))

    payload: dict[str, Any] = {}
    for target, value in defaults.items():
        _set_path(payload, str(target), value)
    for target, source_path in fields.items():
        try:
            value = _get_path(source, str(source_path))
        except KeyError as exc:
            raise IngestError(
                f"source path {source_path!r} for target {target!r} was not found"
            ) from exc
        _set_path(payload, str(target), value)
    payload.setdefault("metadata", {})

    used_top_level = {str(path).split(".", 1)[0] for path in fields.values()}
    dropped = sorted(k for k in source if k not in used_top_level)
    return payload, dropped


def ingest_generic_jsonl(
    source_path: Path,
    mapping_path: Path,
    output_dir: Path,
) -> IngestReceipt:
    mapping = _read_mapping(mapping_path)
    mode = mapping.get("mode", "mapped")
    runs: list[Run] = []
    warnings: list[str] = [randomization_warning("generic-jsonl")]
    dropped_fields: set[str] = set()
    validation_errors: list[str] = []

    for idx, line in enumerate(source_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError as exc:
            validation_errors.append(f"record {idx}: invalid JSON: {exc}")
            continue
        try:
            if mode == "native":
                payload = source
                dropped: list[str] = []
            elif mode == "mapped":
                payload, dropped = _mapped_payload(source, mapping)
            else:
                raise IngestError(f"unsupported ingest mode: {mode!r}")
            run = Run.model_validate(payload)
        except (IngestError, ValidationError, ValueError) as exc:
            validation_errors.append(f"record {idx}: {exc}")
            continue
        runs.append(run)
        dropped_fields.update(dropped)
        warnings.extend(per_decision_randomization_warnings(run, source_index=idx))

    if validation_errors:
        raise IngestError("; ".join(validation_errors))

    receipt = IngestReceipt(
        source_format="generic-jsonl",
        source_file=str(source_path),
        mapping_file=str(mapping_path),
        generated_count=len(runs),
        warnings=warnings,
        dropped_fields=sorted(dropped_fields),
        validation_errors=[],
    )
    write_corpus(runs, output_dir, receipt)
    return receipt


__all__ = ["IngestError", "IngestReceipt", "ingest_generic_jsonl"]
