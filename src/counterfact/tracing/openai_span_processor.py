"""`CounterfactSpanProcessor` — OpenAI Agents SDK trace processor.

Compatible with `agents.add_trace_processor()`. Buffers spans per `trace_id`
in memory; on `on_trace_end` reconstructs the trace dict and reuses
`run_from_trace` to write a native `Run` JSON.

The OpenAI Agents SDK is an OPTIONAL dependency. Importing this module works
without it; instantiating the processor surfaces SDK absence naturally on
adapter import.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from counterfact.adapters._common import (
    IngestError,
    IngestReceipt,
    randomization_warning,
    read_existing_receipt_count,
    write_corpus,
)
from counterfact.adapters.openai_agents import SOURCE_FORMAT, run_from_trace
from counterfact.schema import Outcome, Run

logger = logging.getLogger(__name__)

OutcomeProvider = Callable[[dict[str, Any]], "bool | str | None"]


class CounterfactSpanProcessor:
    """OpenAI Agents SDK trace processor that emits native `Run` JSON.

    Construct with an `output_dir`. By default the processor refuses to invent
    a binary outcome — if the trace has neither a root error nor an explicit
    `counterfact.outcome` marker span, the run is written with a categorical
    outcome `"unknown"` and the receipt records a warning. Pass an
    `outcome_provider` callback to derive outcomes from your own evaluator.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        outcome_verifier: str = "counterfact_span_processor",
        outcome_provider: OutcomeProvider | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.outcome_verifier = outcome_verifier
        self.outcome_provider = outcome_provider
        self._spans_by_trace: dict[str, list[dict[str, Any]]] = {}
        self._trace_metadata: dict[str, dict[str, Any]] = {}
        # Track traces written by this processor instance; seeded from any
        # pre-existing receipt on first write so a restarted session keeps
        # extending the count rather than overwriting it with 1.
        self._cumulative_count = 0
        self._seeded_from_disk = False

    # --- TracingProcessor interface ------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        meta = _maybe_export(trace)
        if meta and "id" in meta:
            self._trace_metadata[meta["id"]] = meta

    def on_trace_end(self, trace: Any) -> None:
        trace_meta = _maybe_export(trace) or {}
        trace_id = trace_meta.get("id") or _extract_attr(trace, "trace_id")
        if trace_id is None:
            return
        spans = self._spans_by_trace.pop(trace_id, [])
        self._trace_metadata.pop(trace_id, None)
        if not spans:
            return
        run = self._build_run(trace_id=str(trace_id), spans=spans)
        if not self._seeded_from_disk:
            self._cumulative_count = read_existing_receipt_count(self.output_dir)
            self._seeded_from_disk = True
        self._cumulative_count += 1
        receipt = IngestReceipt(
            source_format=SOURCE_FORMAT,
            source_file="<live-processor>",
            generated_count=self._cumulative_count,
            warnings=[randomization_warning(SOURCE_FORMAT)],
        )
        if run.outcome.kind != "binary":
            receipt.warnings.append(
                f"trace {trace_id!r}: no binary outcome could be derived; "
                f"wrote categorical outcome={run.outcome.value!r} so the trace "
                "is preserved without faking success/fail. Provide an "
                "outcome_provider to label it."
            )
        write_corpus([run], self.output_dir, receipt)

    def on_span_start(self, span: Any) -> None:
        # Most useful state arrives at span_end (timings, outputs).
        return None

    def on_span_end(self, span: Any) -> None:
        exported = _maybe_export(span)
        if exported is None:
            return
        trace_id = exported.get("trace_id")
        if trace_id is None:
            return
        self._spans_by_trace.setdefault(str(trace_id), []).append(exported)

    def shutdown(self) -> None:
        self._spans_by_trace.clear()
        self._trace_metadata.clear()

    def force_flush(self) -> None:
        # In-process buffering only — nothing async to drain.
        return None

    # --- helpers -------------------------------------------------------------

    def _build_run(self, *, trace_id: str, spans: list[dict[str, Any]]) -> Run:
        trace_payload = {"trace_id": trace_id, "spans": spans}

        override: bool | str | None = None
        if self.outcome_provider is not None:
            override = self.outcome_provider(trace_payload)

        try:
            return run_from_trace(trace_payload, outcome=override)
        except IngestError:
            # Fall back to a categorical "unknown" outcome rather than dropping
            # the trace. Preserves the run for inspection while refusing to
            # invent a binary success/fail. Other exceptions (real bugs)
            # propagate so they surface instead of being swallowed.
            return _run_with_unknown_outcome(
                trace_payload,
                verifier=self.outcome_verifier,
            )


def _maybe_export(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    export = getattr(obj, "export", None)
    if callable(export):
        try:
            value = export()
        except Exception as exc:
            # SDK Span.export() is third-party code; we can't enumerate its
            # failure modes. Log the failure so a degraded trace doesn't
            # silently lose data, then drop the span.
            logger.warning(
                "Span/trace export() raised %s: %s; dropping object",
                type(exc).__name__,
                exc,
            )
            return None
        if isinstance(value, dict):
            return value
    return None


def _extract_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _run_with_unknown_outcome(trace_payload: dict[str, Any], *, verifier: str) -> Run:
    """Build a Run that mirrors `run_from_trace` but stamps a categorical 'unknown' outcome."""

    # Borrow run_from_trace by injecting a sentinel outcome marker, then rewrite.
    sentinel_spans = list(trace_payload["spans"])
    # Use a True override so the call succeeds; we'll overwrite outcome below.
    run = run_from_trace(
        {"trace_id": trace_payload["trace_id"], "spans": sentinel_spans},
        outcome=True,
    )
    return run.model_copy(
        update={
            "outcome": Outcome(
                kind="categorical",
                value="unknown",
                verifier=verifier,
                metadata={"note": "no binary outcome derivable; preserved for inspection"},
            )
        }
    )


__all__ = ["CounterfactSpanProcessor"]
