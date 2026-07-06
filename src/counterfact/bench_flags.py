"""Bench harness constants shared by CLI argparse and next-step suggestions."""

from __future__ import annotations

SUGGESTED_RANDOMIZATION_EPSILON = 0.5
MIN_BENCH_N = 30
SUGGESTED_FIXTURE_SET = "broad_calibration"

MODEL_GREEDY_CHOICES = ("small", "large")
RETRY_GREEDY_CHOICES = ("no_retry", "retry_once")

FIXTURE_SET_CHOICES = (
    "v0",
    "easy",
    "hidden_v1",
    "hard_hidden_v1",
    "broad_calibration",
    "very_hard_hidden_v1",
    "stateful_calibration",
)
