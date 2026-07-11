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
