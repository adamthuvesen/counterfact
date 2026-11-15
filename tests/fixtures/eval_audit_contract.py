"""Check an export against eval-audit's RunRecord JSON schema.

`eval_audit_run_record.schema.json` is a copy of eval-audit's
`docs/run-record.schema.json`. Copy it again when eval-audit changes RunRecord.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

CONTRACT: dict[str, Any] = json.loads(
    (Path(__file__).parent / "eval_audit_run_record.schema.json").read_text()
)


def _options(prop: dict[str, Any]) -> list[dict[str, Any]]:
    if "$ref" in prop:
        return [CONTRACT["$defs"][prop["$ref"].rsplit("/", 1)[1]]]
    return prop.get("anyOf", [prop])


def _conforms(value: object, option: dict[str, Any]) -> bool:
    kind = option["type"]
    minimum = option.get("minimum", -math.inf)
    if kind == "null":
        return value is None
    if "enum" in option:
        return value in option["enum"]
    if kind == "string" and option.get("format") == "date-time":
        return isinstance(value, datetime)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool) and value >= minimum
    if kind == "number":
        return isinstance(value, float) and math.isfinite(value) and value >= minimum
    if kind == "object":
        # eval-audit drops null struct fields: they mark absent keys.
        values = option["additionalProperties"]
        return isinstance(value, dict) and all(
            item is None or _conforms(item, values) for item in value.values()
        )
    raise AssertionError(f"contract uses a type this check does not know: {option}")


def assert_conforms_to_eval_audit(frame: pl.DataFrame) -> None:
    properties: dict[str, dict[str, Any]] = CONTRACT["properties"]
    assert sorted(frame.columns) == sorted(properties), "export columns differ from RunRecord"
    for index, row in enumerate(frame.iter_rows(named=True)):
        for field, prop in properties.items():
            value = row[field]
            assert any(_conforms(value, option) for option in _options(prop)), (
                f"row {index}, field {field!r}: {value!r} violates RunRecord {prop}"
            )
