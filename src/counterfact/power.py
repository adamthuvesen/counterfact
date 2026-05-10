"""Binomial-CI power analysis: how many traces to tighten the CI on a 2-arm marginal effect.

This is the v0 power helper. Scope is deliberately narrow per design.md D2:
the only question it answers is "given the current per-arm pass rates and arm
fractions, how large must n be for a 95% bootstrap CI on the marginal-effect
difference to be at most `target_ci_width` wide?"

Real power analysis (effect-size hypotheses, multi-arm corrections, sequential
designs) is its own research project and is out of v0 scope.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from counterfact.baselines import _Z_95, pass_rate_by_arm
from counterfact.schema import Run


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PowerReport(_Strict):
    """Result of `power_analysis`. See `power_analysis.__doc__` for the
    definition of `current_ci_width` and the binomial assumption.
    """

    decision_type: str
    arms: tuple[str, str]
    target_ci_width: float
    current_n: int
    current_ci_width: float | None = None
    estimated_required_n: int | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _two_arm_ci_width(
    *, p_a: float, p_b: float, n_a: int, n_b: int
) -> float:
    """95% CI full width on (p_a - p_b) under the binomial-Wald approximation."""
    if n_a <= 0 or n_b <= 0:
        return float("inf")
    var = p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b
    return 2.0 * _Z_95 * math.sqrt(var)


def power_analysis(
    corpus: Iterable[Run],
    *,
    decision_type: str,
    arms: tuple[str, str],
    target_ci_width: float = 0.10,
) -> PowerReport:
    """Estimate `n` to bring 95% CI width on `(arms[0] - arms[1])` ≤ `target_ci_width`.

    Method (binomial-Wald):

        Var(p̂_a - p̂_b) ≈ p_a (1 - p_a) / n_a + p_b (1 - p_b) / n_b

    where `n_a = f_a · n_total` and `f_a` is the observed arm fraction. Solve for
    the smallest `n_total` such that `2 · z_{0.975} · √Var ≤ target_ci_width`,
    holding `(p_a, p_b, f_a, f_b)` fixed at observed values.

    Returns a `PowerReport`. If either arm has zero observations, returns
    `estimated_required_n=None` and a warning naming the missing arm — the
    binomial formula is ill-posed there, and the right next step is broader
    arm support, not bigger n.
    """
    table = pass_rate_by_arm(corpus, decision_type)
    by_arm = {row.arm: row for row in table.rows}

    arm_a, arm_b = arms
    missing = [arm for arm in arms if arm not in by_arm or by_arm[arm].n == 0]

    n_total = sum(row.n for row in table.rows)
    base_assumption = (
        "binomial-Wald approximation; assumes per-arm pass rates and arm "
        "fractions remain at their currently observed values"
    )

    if missing:
        return PowerReport(
            decision_type=decision_type,
            arms=arms,
            target_ci_width=target_ci_width,
            current_n=n_total,
            estimated_required_n=None,
            assumptions=[base_assumption],
            warnings=[
                f"missing arm support for {missing!r} on decision_type={decision_type!r}; "
                f"power analysis is ill-posed — broaden arm randomization first"
            ],
        )

    row_a = by_arm[arm_a]
    row_b = by_arm[arm_b]
    p_a = row_a.pass_rate
    p_b = row_b.pass_rate
    n_a = row_a.n
    n_b = row_b.n
    f_a = n_a / (n_a + n_b)
    f_b = n_b / (n_a + n_b)

    current_width = _two_arm_ci_width(p_a=p_a, p_b=p_b, n_a=n_a, n_b=n_b)

    # If there's no variance to bound (e.g. p_a = p_b = 0 or 1), every n
    # satisfies the constraint — current_n suffices.
    var_per_n = p_a * (1 - p_a) / f_a + p_b * (1 - p_b) / f_b
    if var_per_n <= 0:
        return PowerReport(
            decision_type=decision_type,
            arms=arms,
            target_ci_width=target_ci_width,
            current_n=n_total,
            current_ci_width=current_width,
            estimated_required_n=n_total,
            assumptions=[
                base_assumption,
                "observed per-arm pass rates are degenerate (0 or 1); width "
                "is exactly 0 in expectation under the binomial approximation",
            ],
        )

    # Solve 2·z·√(var_per_n / n) ≤ W  →  n ≥ var_per_n · (2z / W)²
    n_required = math.ceil(var_per_n * (2.0 * _Z_95 / target_ci_width) ** 2)

    return PowerReport(
        decision_type=decision_type,
        arms=arms,
        target_ci_width=target_ci_width,
        current_n=n_total,
        current_ci_width=current_width,
        estimated_required_n=max(n_required, n_total),
        assumptions=[base_assumption],
    )
