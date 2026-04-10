"""Naive marginal estimators — the comparison baseline, not the recommendation.

Lab researchers will compute these numbers anyway. Exposing them cleanly lets
the demo present the naive read alongside the honest causal verdict from
`counterfact.intervene` rather than letting consumers roll their own.

Use `intervene` for causal queries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from counterfact.schema import Run


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PassRateRow(_Strict):
    """One row of the per-arm pass-rate table."""

    arm: str
    n: int
    pass_count: int
    pass_rate: float
    ci_low: float
    ci_high: float


class PassRateTable(_Strict):
    """Per-arm pass-rate breakdown for a single decision type.

    This is the **naive marginal estimator**. It ignores confounding, propensity,
    and identifiability. Use `counterfact.intervene` for causal queries.
    """

    decision_type: str
    rows: list[PassRateRow] = Field(default_factory=list)


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pass_rate_by_arm(
    corpus: Iterable[Run], decision_type: str
) -> PassRateTable:
    """Naive marginal pass-rate per `chosen_action` for `decision_type`.

    This is the **naive marginal estimator** — comparison baseline, not the
    recommended estimator. Use `counterfact.intervene` for causal queries; that
    estimator carries an `identifiability` label and respects the DAG.

    Returns a `PassRateTable` with one row per observed arm. Each row carries:

    - `arm`: the chosen action
    - `n`: how many decisions of `decision_type` chose this arm
    - `pass_count`: how many of those came from traces with `Outcome.value=True`
    - `pass_rate`: `pass_count / n`
    - `ci_low`, `ci_high`: 95% Wilson-score interval on `pass_rate`

    A run can contribute multiple decisions of the same type (e.g. two
    `tool_call` decisions in one trace). Each decision is counted once.
    """
    bucket_n: dict[str, int] = {}
    bucket_pass: dict[str, int] = {}
    for run in corpus:
        outcome_pass = bool(run.outcome.value)
        for step in run.steps:
            for d in step.decisions:
                if d.decision_type != decision_type:
                    continue
                if d.chosen_action is None:
                    continue
                bucket_n[d.chosen_action] = bucket_n.get(d.chosen_action, 0) + 1
                if outcome_pass:
                    bucket_pass[d.chosen_action] = (
                        bucket_pass.get(d.chosen_action, 0) + 1
                    )
    rows: list[PassRateRow] = []
    for arm in sorted(bucket_n):
        n = bucket_n[arm]
        k = bucket_pass.get(arm, 0)
        ci_low, ci_high = _wilson_ci(k, n)
        rows.append(
            PassRateRow(
                arm=arm,
                n=n,
                pass_count=k,
                pass_rate=k / n if n else 0.0,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return PassRateTable(decision_type=decision_type, rows=rows)
