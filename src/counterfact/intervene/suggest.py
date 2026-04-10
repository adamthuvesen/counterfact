"""Compose copy-pasteable harness commands for `NextStep.payload`.

The engine returns refusals with structured `next_step` actions; for actions
where running the bench harness with different flags would produce the missing
data, this module composes the exact `uv run counterfact bench …` invocation.

Kept out of `intervene/api.py` so the engine doesn't import the CLI module.
The flag names mirror those in `counterfact.cli.build_parser()`; the
`tests/unit/test_suggest.py` test parses every returned command back through
that parser to keep the two in lockstep.
"""

from __future__ import annotations

# Canonical arm sets per (decision_type, intervention_kind), pulled from the
# CLI's flag `choices=...` lists. When a kind is absent, arms are free-form
# (e.g. tool_choice) and `missing_arms` cannot be enumerated from the schema
# alone.
_KNOWN_ARMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("model_call", "model_choice"): ("small", "large"),
    ("retry", "retry_policy"): ("no_retry", "retry_once"),
}


def known_arms(decision_type: str, intervention_kind: str) -> tuple[str, ...]:
    """Return the canonical arm set for a (decision_type, intervention_kind), or ()."""
    return _KNOWN_ARMS.get((decision_type, intervention_kind), ())


# Default ε to suggest when we want to *introduce* randomization at an arm.
# Matches the CLI's default ε for un-tuned decision types.
_RANDOMIZATION_EPSILON = 0.5

# Floor for `--n` on broaden-arm or add-arm commands. Below this the binomial
# math is too noisy to be useful.
_MIN_BENCH_N = 30


def _model_flags(arm_name: str | None) -> list[str]:
    arm = arm_name if arm_name in {"small", "large"} else "large"
    return [
        "--model-greedy",
        arm,
        "--model-epsilon",
        f"{_RANDOMIZATION_EPSILON}",
    ]


def _retry_flags(arm_name: str | None) -> list[str]:
    arm = arm_name if arm_name in {"no_retry", "retry_once"} else "retry_once"
    return [
        "--retry-greedy",
        arm,
        "--retry-epsilon",
        f"{_RANDOMIZATION_EPSILON}",
    ]


def _tool_flags(arm_name: str | None) -> list[str]:
    arm = arm_name or "inspect_file"
    return [
        "--tool-greedy",
        arm,
        "--tool-epsilon",
        f"{_RANDOMIZATION_EPSILON}",
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
        n = max(int(estimated_required_n or _MIN_BENCH_N), _MIN_BENCH_N)
        # Preserve existing ε implicitly by not passing per-decision ε flags.
        parts = [
            "uv run counterfact bench real",
            f"--n {n}",
            "--fixture-set hidden_v1",
        ]
        return " ".join(parts)

    # broaden_arm_support / add_arm_randomization: introduce ε at the named arm.
    if not arm_flags:
        # No canonical flag mapping for this decision_type; the bench can't
        # generate the missing arm directly.
        return None
    n = max(int(estimated_required_n or _MIN_BENCH_N), _MIN_BENCH_N)
    parts = [
        "uv run counterfact bench real",
        f"--n {n}",
        "--fixture-set hidden_v1",
        *arm_flags,
    ]
    return " ".join(parts)
