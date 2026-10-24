from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from counterfact.cli import loaders
from counterfact.errors import CounterError
from counterfact.policy import Policy, PolicyEstimate, evaluate_policy


def _load_policy(path: Path) -> Policy | None:
    try:
        return Policy.model_validate_json(path.read_text())
    except (OSError, ValidationError, ValueError) as exc:
        print(f"counterfact evaluate-policy: invalid policy {path}: {exc}", file=sys.stderr)
        return None


def _format(estimate: PolicyEstimate) -> str:
    lines = [
        f"counterfact evaluate-policy: {estimate.estimand}",
        f"identifiability: {estimate.identifiability.value}",
    ]
    if estimate.outcome_delta is not None:
        d = estimate.outcome_delta
        lines.append(
            f"outcome_delta: {d.point:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]  (sequential DR)"
        )
    if estimate.diagnostics is not None:
        g = estimate.diagnostics
        lines.append(f"cross-check: ICE {g.ice_delta:+.3f}, SNIS {g.snis_delta:+.3f}")
        if g.matched_runs_delta is not None:
            lines.append(f"matched runs (descriptive, not causal): {g.matched_runs_delta:+.3f}")
        lines.append(
            f"support: {g.n_matching_runs}/{g.n_runs} runs follow the policy; "
            f"effective n {g.effective_n:.0f}, max weight {g.max_weight:.1f}"
        )
    if estimate.bounds is not None:
        b = estimate.bounds
        lines.append(f"bounds ({b.technique}): [{b.lower:+.3f}, {b.upper:+.3f}]")
    if estimate.reason:
        lines.append(f"reason: {estimate.reason}")
    lines.extend(f"warning: {w}" for w in estimate.warnings)
    lines.append(f"next_step: {estimate.next_step.action} - {estimate.next_step.human_text}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    runs = loaders.load_corpus_dir(args.runs_dir, command="evaluate-policy")
    if runs is None:
        return 2
    if not runs:
        print(f"counterfact evaluate-policy: no traces in {args.runs_dir}", file=sys.stderr)
        return 2
    policy = _load_policy(args.policy)
    baseline = _load_policy(args.baseline) if args.baseline is not None else None
    if policy is None or (args.baseline is not None and baseline is None):
        return 2

    try:
        estimate = evaluate_policy(
            runs,
            policy,
            baseline=baseline,
            observation_keys=set(args.observation_key or ()),
            seed=args.seed,
        )
    except CounterError as exc:
        print(f"counterfact evaluate-policy: {exc}", file=sys.stderr)
        return 2
    print(estimate.model_dump_json(indent=2) if args.json else _format(estimate))
    return 0
