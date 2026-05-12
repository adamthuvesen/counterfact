"""Shared formatting helpers used across renderers and CLI surfaces.

These helpers exist so that user-visible labels stay consistent across the
HTML report, the diagnose CLI, and the trace-comparison output. Add a helper
here only when at least two callers need it; otherwise keep formatting local.
"""

from __future__ import annotations

from counterfact.schema import Run


def outcome_label(run: Run) -> str:
    """Render a `Run.outcome` as a short human-facing label.

    Binary outcomes collapse to `pass`/`fail`; everything else falls back to
    `<kind>=<repr(value)>` so consumers can see both the verifier kind and the
    raw value at a glance.
    """
    if run.outcome.kind == "binary":
        return "pass" if bool(run.outcome.value) else "fail"
    return f"{run.outcome.kind}={run.outcome.value!r}"
