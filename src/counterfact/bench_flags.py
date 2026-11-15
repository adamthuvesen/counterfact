"""Bench harness constants shared by CLI argparse and next-step suggestions."""

from __future__ import annotations

SUGGESTED_RANDOMIZATION_EPSILON = 0.5
MIN_BENCH_N = 30
SUGGESTED_FIXTURE_SET = "broad_calibration"

MODEL_GREEDY_CHOICES = ("small", "large")
RETRY_GREEDY_CHOICES = ("no_retry", "retry_once")
