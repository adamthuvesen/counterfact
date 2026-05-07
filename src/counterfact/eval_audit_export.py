"""Export native traces to eval-audit's RunRecord-shaped parquet contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from counterfact.schema import Run


class EvalAuditExportReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_corpus: str
    output_path: str
    row_count: int
    field_derivations: dict[str, str]
    missing_cost_policy: str
    warnings: list[str] = Field(default_factory=list)


def _agent_id(run: Run) -> str:
    if run.metadata.agent_name:
        return run.metadata.agent_name
    for step in run.steps:
        for decision in step.decisions:
            if decision.decision_type == "model_call" and decision.chosen_action:
                return decision.chosen_action
    return "counterfact-agent"


def _harness(run: Run) -> str:
    value = run.metadata.extra.get("harness")
    return str(value) if value else "counterfact-trace"


def _task_id(run: Run) -> str:
    value = run.metadata.extra.get("task_id")
    return str(value) if value else run.run_id


def _cost(run: Run) -> float | None:
    value = run.metadata.extra.get("cost_usd")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tokens(run: Run, key: str) -> int:
    value = run.metadata.extra.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def runs_to_eval_audit_rows(runs: list[Run]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run in runs:
        agent_id = _agent_id(run)
        cost = _cost(run)
        if cost is None:
            warnings.append(
                f"run {run.run_id}: no honest cost_usd metadata; exporting cost_not_available"
            )
        tokens_in = _tokens(run, "tokens_in")
        tokens_out = _tokens(run, "tokens_out")
        rows.append(
            {
                "agent_id": agent_id,
                "model_id": agent_id,
                "harness": _harness(run),
                "run_id": run.run_id,
                "task_id": _task_id(run),
                "task_category": run.metadata.extra.get("task_category"),
                "seed": run.metadata.extra.get("seed"),
                "success": bool(run.outcome.value) if run.outcome.kind == "binary" else None,
                "partial_credit": (
                    float(bool(run.outcome.value)) if run.outcome.kind == "binary" else None
                ),
                "outcome_status": "graded",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_in_by_model": {agent_id: tokens_in},
                "tokens_out_by_model": {agent_id: tokens_out},
                "latency_s": run.metadata.extra.get("latency_s"),
                "timestamp": run.metadata.extra.get("timestamp"),
                "reconstructed_per_task_cost_usd": cost,
                "reported_run_total_cost_usd": cost,
                "cost_provenance": "reconciled" if cost is not None else "cost_not_available",
                "rerun_metadata": {
                    "source": "counterfact export-runs",
                    "counterfact_run_id": run.run_id,
                },
            }
        )
    return rows, warnings


_FIELD_DERIVATIONS = {
    "agent_id": (
        "Run.metadata.agent_name, else first model_call chosen_action, "
        "else counterfact-agent"
    ),
    "harness": "Run.metadata.extra['harness'], else counterfact-trace",
    "task_id": "Run.metadata.extra['task_id'], else Run.run_id",
    "success": "Run.outcome.value for binary outcomes",
    "cost_provenance": (
        "reconciled when Run.metadata.extra['cost_usd'] is present, "
        "else cost_not_available"
    ),
}


def export_eval_audit_parquet(
    runs: list[Run], *, source_corpus: Path, output_path: Path
) -> EvalAuditExportReceipt:
    rows, warnings = runs_to_eval_audit_rows(runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, strict=False).write_parquet(output_path)
    receipt = EvalAuditExportReceipt(
        source_corpus=str(source_corpus),
        output_path=str(output_path),
        row_count=len(rows),
        field_derivations=_FIELD_DERIVATIONS,
        missing_cost_policy=(
            "Rows without Run.metadata.extra['cost_usd'] are exported with "
            "cost_provenance='cost_not_available' and null cost fields."
        ),
        warnings=warnings,
    )
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt.model_dump(), indent=2, sort_keys=True) + "\n")
    return receipt


__all__ = [
    "EvalAuditExportReceipt",
    "export_eval_audit_parquet",
    "runs_to_eval_audit_rows",
]
