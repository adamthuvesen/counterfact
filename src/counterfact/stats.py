"""Shared binomial statistics helpers."""

from __future__ import annotations

import math

# 95% two-sided normal critical value for Wilson / Wald intervals.
Z_95: float = 1.959963984540054


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95%)."""
    if n == 0:
        return (0.0, 0.0)
    z = Z_95
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_difference_ci(
    k_target: int, n_target: int, k_baseline: int, n_baseline: int
) -> tuple[float, float]:
    """Newcombe's independent-proportions interval combining Wilson scores.

    Source: Newcombe (1998), https://pubmed.ncbi.nlm.nih.gov/9595617/.
    """
    if not (0 <= k_target <= n_target and 0 <= k_baseline <= n_baseline):
        raise ValueError("success counts must lie between zero and their sample sizes")
    if n_target <= 0 or n_baseline <= 0:
        raise ValueError("both arms need observations for a difference interval")
    target, baseline = k_target / n_target, k_baseline / n_baseline
    target_low, target_high = wilson_ci(k_target, n_target)
    baseline_low, baseline_high = wilson_ci(k_baseline, n_baseline)
    delta = target - baseline
    return (
        delta - math.hypot(target - target_low, baseline_high - baseline),
        delta + math.hypot(target_high - target, baseline - baseline_low),
    )
