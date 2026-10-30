"""Versioned RunRecord Parquet export with source-derived usage."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import polars as pl

from counterfact.corpus_io import CorpusIOError, RunRecordExportReceipt, atomic_write
from counterfact.schema import Run, validate_runs


def _number(value: object, *, field: str, integer: bool = False) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorpusIOError(f"{field} must be a non-negative number or null")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise CorpusIOError(
            f"{field} must be a finite non-negative {'integer' if integer else 'number'}"
        )
    return int(value) if integer else float(value)


def _token(usage: object, *keys: str) -> int | None:
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise CorpusIOError("usage must be an object or null")
    for key in keys:
        if key in usage:
            value = _number(usage[key], field=key, integer=True)
            return int(value) if value is not None else None
    return None


def _sum_known(values: Sequence[int | float | None]) -> int | float | None:
    return (
        sum(value for value in values if value is not None)
        if values and None not in values
        else None
    )


def _model_identity(action: str | None, metadata: dict[str, Any]) -> str | None:
    value = metadata.get("model_id") or metadata.get("provider_model") or action
    return str(value) if value else None


def _reconstructed_cost(run: Run) -> float | int | None:
    observations = [
        (step_index, observation_index, observation.content)
        for step_index, step in enumerate(run.steps)
        for observation_index, observation in enumerate(step.observations)
        if "cost_usd" in observation.content
    ]
    costs: list[float | int | None] = []
    call_ids: set[str] = set()
    observation_ids: set[str] = set()
    for step_index, observation_index, content in observations:
        observation_id = run.steps[step_index].observations[observation_index].observation_id
        if observation_id in observation_ids:
            return None
        observation_ids.add(observation_id)
        call_id = content.get("call_id")
        if call_id is not None:
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                return None
            call_ids.add(call_id)
        costs.append(_number(content["cost_usd"], field="observation.content.cost_usd"))
    used: set[tuple[int, int]] = set()
    for step_index, step in enumerate(run.steps):
        for decision in step.decisions:
            if decision.decision_type != "model_call":
                continue
            call_id = decision.metadata.get("call_id")
            if call_id is not None:
                if not isinstance(call_id, str) or not call_id:
                    return None
                candidates = [entry for entry in observations if entry[2].get("call_id") == call_id]
            elif len(step.decisions) == 1:
                candidates = [entry for entry in observations if entry[0] == step_index]
            else:
                return None
            if len(candidates) != 1:
                return None
            source_step, observation_index, _ = candidates[0]
            source = (source_step, observation_index)
            if source in used:
                return None
            used.add(source)
    return _sum_known(costs)


def runs_to_runrecord_rows(runs: list[Run]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run in validate_runs(runs):
        extra = run.metadata.extra
        calls = [d for s in run.steps for d in s.decisions if d.decision_type == "model_call"]
        first_action = next((d.chosen_action for d in calls if d.chosen_action), None)
        models = list(
            dict.fromkeys(
                model for d in calls if (model := _model_identity(d.chosen_action, d.metadata))
            )
        )
        by_model: dict[str, tuple[int | None, int | None]] = {}
        model_usage = extra.get("model_usage")
        if model_usage is not None:
            if not isinstance(model_usage, dict):
                raise CorpusIOError(f"run {run.run_id}: model_usage must be an object")
            for model, usage in model_usage.items():
                by_model[model] = (
                    _token(usage, "input_tokens", "inputTokens", "prompt_tokens"),
                    _token(usage, "output_tokens", "outputTokens", "completion_tokens"),
                )
        else:
            for model in models:
                usages = [
                    d.metadata.get("usage")
                    for d in calls
                    if _model_identity(d.chosen_action, d.metadata) == model
                ]
                total_in = _sum_known([_token(u, "input_tokens", "prompt_tokens") for u in usages])
                total_out = _sum_known(
                    [_token(u, "output_tokens", "completion_tokens") for u in usages]
                )
                by_model[model] = (
                    int(total_in) if total_in is not None else None,
                    int(total_out) if total_out is not None else None,
                )
        missing_models = sorted(set(models) - set(by_model)) if model_usage is not None else []
        if missing_models:
            for model in missing_models:
                by_model[model] = (None, None)
            warnings.append(
                f"run {run.run_id}: model_usage omits known models {missing_models}; "
                "aggregate usage is unavailable without explicit run totals"
            )
        tokens: list[int | None] = []
        token_sources: list[str] = []
        for index, (canonical, aliases) in enumerate(
            (
                ("tokens_in", ("input_tokens", "prompt_tokens")),
                ("tokens_out", ("output_tokens", "completion_tokens")),
            )
        ):
            value = _token(extra, canonical)
            source = f"metadata.extra.{canonical}"
            if value is None:
                value = _token(extra.get("usage"), *aliases)
                source = "metadata.extra.usage"
            if value is None:
                values = (
                    [pair[index] for pair in by_model.values()]
                    if model_usage is not None
                    else [_token(d.metadata.get("usage"), *aliases) for d in calls]
                )
                total = _sum_known(values) if not missing_models else None
                value = int(total) if total is not None else None
                source = (
                    "metadata.extra.model_usage"
                    if model_usage is not None
                    else "decisions.metadata.usage"
                )
            tokens.append(value)
            token_sources.append(source if value is not None else "not_available")
        if len(models) == 1 and model_usage is None and all(v is None for v in by_model[models[0]]):
            by_model[models[0]] = (tokens[0], tokens[1])
        reported_key = "cost_usd" if extra.get("cost_usd") is not None else "total_cost_usd"
        reported = _number(extra.get(reported_key), field=reported_key)
        if reported is None and isinstance(model_usage, dict):
            model_costs = []
            for usage in model_usage.values():
                if not isinstance(usage, dict):
                    raise CorpusIOError("model_usage entries must be objects")
                model_costs.append(
                    _number(
                        usage.get("cost_usd", usage.get("costUSD")), field="model_usage.costUSD"
                    )
                )
            reported = _sum_known(model_costs) if not missing_models else None
            reported_key = "model_usage"
        reconstructed = _reconstructed_cost(run)
        if calls and reconstructed is None:
            warnings.append(
                f"run {run.run_id}: incomplete per-call cost coverage or ambiguous association; "
                "exporting null"
            )
        cost_sources = [f"metadata.extra.{reported_key}"] if reported is not None else []
        if reconstructed is not None:
            cost_sources.append("observations.content.cost_usd")
        if reported is not None and reconstructed is not None:
            provenance = (
                "reconciled"
                if math.isclose(reported, reconstructed, abs_tol=1e-9)
                else "discrepant"
            )
            if provenance == "discrepant":
                warnings.append(f"run {run.run_id}: reported and reconstructed costs differ")
        elif reported is not None:
            provenance = "reported"
        elif reconstructed is not None:
            provenance = "reconstructed"
        else:
            provenance = "cost_not_available"
            warnings.append(f"run {run.run_id}: no recorded cost; exporting null cost fields")
        rows.append(
            {
                "agent_id": run.metadata.agent_name or first_action or "counterfact-agent",
                "model_id": first_action,
                "model_ids": models,
                "model_identity_provenance": (
                    "model_id: first chosen_action; model_ids: logged model_id/provider_model, "
                    "else chosen_action"
                ),
                "harness": str(extra.get("harness") or "counterfact-trace"),
                "run_id": run.run_id,
                "task_id": str(extra.get("task_id") or run.run_id),
                "task_category": extra.get("task_category"),
                "seed": extra.get("seed"),
                "success": run.outcome.value if run.outcome.kind == "binary" else None,
                "partial_credit": None,
                "outcome_status": "graded" if run.outcome.kind == "binary" else "ungraded",
                "tokens_in": tokens[0],
                "tokens_out": tokens[1],
                "tokens_in_by_model": [
                    {"model_id": m, "tokens": pair[0]} for m, pair in by_model.items()
                ],
                "tokens_out_by_model": [
                    {"model_id": m, "tokens": pair[1]} for m, pair in by_model.items()
                ],
                "tokens_in_provenance": token_sources[0],
                "tokens_out_provenance": token_sources[1],
                "latency_s": extra.get("latency_s"),
                "timestamp": extra.get("timestamp"),
                "reconstructed_per_task_cost_usd": reconstructed,
                "reported_run_total_cost_usd": reported,
                "cost_provenance": provenance,
                "cost_sources": cost_sources,
                "rerun_metadata": {
                    "source": "counterfact export-runs",
                    "counterfact_run_id": run.run_id,
                },
            }
        )
    return rows, warnings


_MODEL_TOKENS = pl.List(pl.Struct({"model_id": pl.String, "tokens": pl.Int64}))
_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "agent_id": pl.String,
    "model_id": pl.String,
    "model_ids": pl.List(pl.String),
    "model_identity_provenance": pl.String,
    "harness": pl.String,
    "run_id": pl.String,
    "task_id": pl.String,
    "task_category": pl.String,
    "seed": pl.Int64,
    "success": pl.Boolean,
    "partial_credit": pl.Float64,
    "outcome_status": pl.String,
    "tokens_in": pl.Int64,
    "tokens_out": pl.Int64,
    "tokens_in_by_model": _MODEL_TOKENS,
    "tokens_out_by_model": _MODEL_TOKENS,
    "tokens_in_provenance": pl.String,
    "tokens_out_provenance": pl.String,
    "latency_s": pl.Float64,
    "timestamp": pl.String,
    "reconstructed_per_task_cost_usd": pl.Float64,
    "reported_run_total_cost_usd": pl.Float64,
    "cost_provenance": pl.String,
    "cost_sources": pl.List(pl.String),
    "rerun_metadata": pl.Struct({"source": pl.String, "counterfact_run_id": pl.String}),
}
_FIELD_DERIVATIONS = {
    "agent_id": (
        "Run.metadata.agent_name, else first model_call chosen_action, else counterfact-agent"
    ),
    "model_id": "first model_call chosen_action, preserving the legacy identity",
    "model_ids": "distinct model_call model_id/provider_model, else chosen_action",
    "success": "binary outcome value; null for other outcome kinds",
    "partial_credit": "null because no native partial-credit grader is recorded",
    "tokens": "run totals, else model totals, else decision usage; never add totals to calls",
    "cost": (
        "reported provider total remains separate from unique observed task costs; "
        "reconstruction requires complete unambiguous model-call cost coverage"
    ),
}


def export_runrecord_parquet(
    runs: list[Run], *, source_corpus: Path, output_path: Path
) -> RunRecordExportReceipt:
    rows, warnings = runs_to_runrecord_rows(runs)
    receipt = RunRecordExportReceipt(
        source_corpus=str(source_corpus),
        output_path=str(output_path),
        row_count=len(rows),
        field_derivations=_FIELD_DERIVATIONS,
        missing_cost_policy="Missing costs are null with cost_provenance=cost_not_available.",
        warnings=warnings,
        migration_notes=[
            "Version 2 replaces fabricated zero token counts with null.",
            "Per-model usage is a list of model_id/tokens records instead of agent-keyed structs.",
            "Reported totals stay separate from observed task costs with model-call coverage.",
            "Legacy receipts without export_version remain readable.",
        ],
    )
    temporary: Path | None = None
    published = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            output_path.exists()
            or output_path.with_suffix(output_path.suffix + ".receipt.json").exists()
        ):
            raise CorpusIOError(f"export output already exists: {output_path}")
        with NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=".counterfact-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            pl.DataFrame(rows, schema=_SCHEMA, strict=True).write_parquet(stream)
        os.link(temporary, output_path)
        published = True
        receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
        atomic_write(receipt_path, receipt.model_dump_json(indent=2) + "\n")
    except (OSError, pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        if published:
            output_path.unlink(missing_ok=True)
        raise CorpusIOError(f"cannot export {output_path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt


__all__ = ["RunRecordExportReceipt", "export_runrecord_parquet", "runs_to_runrecord_rows"]
