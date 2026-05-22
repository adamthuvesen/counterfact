"""Re-export bench flag constants (canonical home: `counterfact.bench_flags`)."""

from counterfact.bench_flags import (
    FIXTURE_SET_CHOICES,
    MIN_BENCH_N,
    MODEL_GREEDY_CHOICES,
    RANDOMIZATION_EPSILON,
    RETRY_GREEDY_CHOICES,
    SUGGESTED_FIXTURE_SET,
)

__all__ = [
    "FIXTURE_SET_CHOICES",
    "MIN_BENCH_N",
    "MODEL_GREEDY_CHOICES",
    "RANDOMIZATION_EPSILON",
    "RETRY_GREEDY_CHOICES",
    "SUGGESTED_FIXTURE_SET",
]
