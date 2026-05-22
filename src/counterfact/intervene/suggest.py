"""Compose copy-pasteable harness commands for `NextStep.payload`.

The engine returns refusals with structured `next_step` actions; for actions
where running the bench harness with different flags would produce the missing
data, this module composes the exact `uv run counterfact bench …` invocation.

Kept out of `intervene/api.py` so the engine doesn't import the CLI module.
Flag defaults live in `counterfact.cli.bench_flags`; `tests/unit/test_suggest.py`
parses every returned command through `build_parser()` to keep CLI and suggest
in lockstep.
"""

from __future__ import annotations

from counterfact.bench_flags import (
    MIN_BENCH_N,
    MODEL_GREEDY_CHOICES,
    RANDOMIZATION_EPSILON,
    RETRY_GREEDY_CHOICES,
    SUGGESTED_FIXTURE_SET,
)

# Canonical arm sets per (decision_type, intervention_kind), pulled from the
# CLI's flag `choices=...` lists. When a kind is absent, arms are free-form
# (e.g. tool_choice) and `missing_arms` cannot be enumerated from the schema
# alone.
_KNOWN_ARMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("model_call", "model_choice"): MODEL_GREEDY_CHOICES,
    ("retry", "retry_policy"): RETRY_GREEDY_CHOICES,
}


def known_arms(decision_type: str, intervention_kind: str) -> tuple[str, ...]:
    """Return the canonical arm set for a (decision_type, intervention_kind), or ()."""
    return _KNOWN_ARMS.get((decision_type, intervention_kind), ())


def _model_flags(arm_name: str | None) -> list[str]:
    arm = arm_name if arm_name in MODEL_GREEDY_CHOICES else "large"
    return [
        "--model-greedy",
        arm,
        "--model-epsilon",
        f"{RANDOMIZATION_EPSILON}",
    ]


def _retry_flags(arm_name: str | None) -> list[str]:
    arm = arm_name if arm_name in RETRY_GREEDY_CHOICES else "retry_once"
    return [
        "--retry-greedy",
        arm,
        "--retry-epsilon",
        f"{RANDOMIZATION_EPSILON}",
    ]


def _tool_flags(arm_name: str | None) -> list[str]:
    arm = arm_name or "inspect_file"
    return [
        "--tool-greedy",
        arm,
        "--tool-epsilon",
        f"{RANDOMIZATION_EPSILON}",
    ]


def _arm_flags(decision_type: str, arm_name: str | None) -> list[str]:
    if decision_type == "model_call":
        return _model_flags(arm_name)
    if decision_type == "retry":
        return _retry_flags(arm_name)
    if decision_type == "tool_call":
        return _tool_flags(arm_name)
    return []


def suggest_harness_command(
    *,
    decision_type: str,
    intervention_kind: str,
    action: str,
    arm_name: str | None = None,
    estimated_required_n: int | None = None,
) -> str | None:
    """Compose a `uv run counterfact bench …` invocation for a NextStep, or None.

    `replay_required` and `none` always return None — replay is upstream of the
    bench, and `none` means no further action is needed. For `add_arm_randomization`
    and `broaden_arm_support`, the command introduces ε at the named arm. For
    `increase_n`, the command preserves whatever ε is already logged and just
    grows `--n` to the estimated required count.

    Output is deterministic for a given input tuple — the suggest tests pin
    that contract.
    """
    if action in {"replay_required", "none"}:
        return None
    if action not in {"broaden_arm_support", "add_arm_randomization", "increase_n"}:
        return None

    arm_flags = _arm_flags(decision_type, arm_name)
    if action == "increase_n":
        n = max(int(estimated_required_n or MIN_BENCH_N), MIN_BENCH_N)
        # Preserve existing ε implicitly by not passing per-decision ε flags.
        parts = [
            "uv run counterfact bench real",
            f"--n {n}",
            f"--fixture-set {SUGGESTED_FIXTURE_SET}",
        ]
        return " ".join(parts)

    # broaden_arm_support / add_arm_randomization: introduce ε at the named arm.
    if not arm_flags:
        # No canonical flag mapping for this decision_type; the bench can't
        # generate the missing arm directly.
        return None
    n = max(int(estimated_required_n or MIN_BENCH_N), MIN_BENCH_N)
    parts = [
        "uv run counterfact bench real",
        f"--n {n}",
        f"--fixture-set {SUGGESTED_FIXTURE_SET}",
        *arm_flags,
    ]
    return " ".join(parts)
