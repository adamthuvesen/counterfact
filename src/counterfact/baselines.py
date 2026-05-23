"""Naive marginal estimators — the comparison baseline, not the recommendation.

Lab researchers will compute these numbers anyway. Exposing them cleanly lets
the demo present the naive read alongside the honest causal verdict from
`counterfact.intervene` rather than letting consumers roll their own.

Use `intervene` for causal queries.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from counterfact.outcome.binary import binary_outcome_value
from counterfact.schema import Run
from counterfact.stats import wilson_ci


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


def pass_rate_by_arm(corpus: Iterable[Run], decision_type: str) -> PassRateTable:
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
        outcome_pass = binary_outcome_value(run)
        for step in run.steps:
            for d in step.decisions:
                if d.decision_type != decision_type:
                    continue
                if d.chosen_action is None:
                    continue
                bucket_n[d.chosen_action] = bucket_n.get(d.chosen_action, 0) + 1
                if outcome_pass:
                    bucket_pass[d.chosen_action] = bucket_pass.get(d.chosen_action, 0) + 1
    rows: list[PassRateRow] = []
    for arm in sorted(bucket_n):
        n = bucket_n[arm]
        k = bucket_pass.get(arm, 0)
        ci_low, ci_high = wilson_ci(k, n)
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
