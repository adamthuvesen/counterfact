"""Outcome model fitting.

The full logistic-regression + bootstrap implementation lands in tasks §6
(causal-engine). This module currently only enforces the v0 binary-outcome
boundary (design.md D2): non-binary outcomes are rejected at the entrypoint
so that downstream code can assume `Outcome.kind == "binary"`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from counter.errors import UnsupportedOutcomeError
from counter.schema import Run


def _assert_binary(traces: Iterable[Run]) -> list[Run]:
    materialized = list(traces)
    for run in materialized:
        kind = run.outcome.kind
        if kind != "binary":
            raise UnsupportedOutcomeError(
                f"v0 only supports binary outcomes; got kind={kind!r} on run {run.run_id!r}"
            )
    return materialized


def fit_outcome_model(
    traces: Iterable[Run],
    *,
    schema: Any | None = None,
    outcome: str = "success",
    n_bootstrap: int = 200,
    seed: int | None = None,
) -> Any:
    """Fit a logistic outcome model on the given traces.

    v0 boundary check is in place; the statistical fit body is §6 work.
    """
    _assert_binary(traces)
    raise NotImplementedError(
        "fit_outcome_model body lands in causal-engine §6 (logistic + bootstrap)"
    )
