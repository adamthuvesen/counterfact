"""Compute the 2x2 (public_pass, hidden_pass) contingency from a hidden-fixture run.

Used by the hidden-test-fixtures pilot decision gate (tasks 6.2-6.4): if the
cell `(public_pass=True, hidden_pass=False)` contains >=3 traces, the
generalization gap exists and we replicate to date_window + rate_limit.
Otherwise we stop and rethink.

Run:
    uv run python -m bench.real.analyze_pilot bench/real/pilot_<YYYY-MM-DD>
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from counterfact.corpus_io import CorpusIOError, load_corpus
from counterfact.schema import Run

_FAILURE_MODES = (
    "pass",
    "format_failure",
    "public_failure",
    "hidden_semantic_failure",
    "timeout",
    "output_limit",
    "exception",
)


class PilotReport(TypedDict):
    n: int
    fixtures: dict[str, int]
    cost_usd: float
    extraction_failures: dict[str, int]
    failure_modes: dict[str, int]
    failure_modes_by_fixture: dict[str, int]
    failure_modes_by_model: dict[str, int]
    table: dict[str, int]
    gate_passed: bool
    showcase_gate_passed: bool


class _HiddenOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fixture_id: str = Field(min_length=1)
    public_pass: bool
    hidden_pass: bool


def _model_attempts(run: Run) -> list[tuple[str, object]]:
    attempts: list[tuple[str, object]] = []
    for step in run.steps:
        model_arm = next(
            (
                decision.chosen_action
                for decision in step.decisions
                if decision.decision_type == "model_call"
            ),
            None,
        )
        if model_arm is not None:
            attempts.extend(
                (model_arm, obs.content.get("extracted_code")) for obs in step.observations
            )
    return attempts


def _stdout_tails(run: Run) -> list[str]:
    return [
        tail
        for step in run.steps
        for obs in step.observations
        if isinstance(tail := obs.content.get("stdout_tail"), str)
    ]


def _execution_statuses(run: Run) -> list[str]:
    executions = [run.outcome.metadata.get("execution")]
    executions.extend(
        obs.content.get("execution") for step in run.steps for obs in step.observations
    )
    return [
        str(value["status"])
        for value in executions
        if isinstance(value, dict) and "status" in value
    ]


def _failure_mode(run: Run, outcome: _HiddenOutcome) -> str:
    statuses = _execution_statuses(run)
    if run.outcome.verifier == "infrastructure_error" or "infrastructure_error" in statuses:
        raise ValueError(f"{run.run_id}: infrastructure attempt cannot define a task outcome")
    executions = [run.outcome.metadata.get("execution")]
    executions.extend(
        obs.content.get("execution") for step in run.steps for obs in step.observations
    )
    if any(
        isinstance(value, dict) and value.get("failure_reason") == "output_limit"
        for value in executions
    ):
        return "output_limit"
    if outcome.hidden_pass:
        return "pass"
    tails = _stdout_tails(run)
    if "workload_timeout" in statuses or any("<pytest timed out" in tail for tail in tails):
        return "timeout"
    if any("traceback" in tail.lower() for tail in tails):
        return "exception"
    if not outcome.public_pass:
        return "public_failure"
    attempts = _model_attempts(run)
    if not attempts or attempts[-1][1] is None:
        return "format_failure"
    return "hidden_semantic_failure"


def _final_model_arm(run: Run) -> str:
    attempts = _model_attempts(run)
    return attempts[-1][0] if attempts else "unknown"


def _showcase_gate(failure_modes: Counter[str], by_model: Counter[str]) -> bool:
    failures = sum(count for mode, count in failure_modes.items() if mode != "pass")
    if failures == 0:
        return False
    if failure_modes["format_failure"] > failures / 2:
        return False
    return (
        by_model["model=small,mode=hidden_semantic_failure"] > 0
        and by_model["model=large,mode=hidden_semantic_failure"] > 0
    )


def analyze(run_dir: Path) -> PilotReport:
    runs = load_corpus(run_dir)
    if not runs:
        raise ValueError(f"no traces under {run_dir}")
    cells: Counter[tuple[bool, bool]] = Counter()
    cost_total = 0.0
    fixtures: Counter[str] = Counter()
    extraction_failures: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    failure_modes_by_fixture: Counter[str] = Counter()
    failure_modes_by_model: Counter[str] = Counter()
    for run in runs:
        metadata = _HiddenOutcome.model_validate(
            {key: run.outcome.metadata.get(key) for key in _HiddenOutcome.model_fields}
        )
        cells[(metadata.public_pass, metadata.hidden_pass)] += 1
        fixtures[metadata.fixture_id] += 1
        mode = _failure_mode(run, metadata)
        model_arm = _final_model_arm(run)
        failure_modes[mode] += 1
        failure_modes_by_fixture[f"fixture={metadata.fixture_id},mode={mode}"] += 1
        failure_modes_by_model[f"model={model_arm},mode={mode}"] += 1
        for step in run.steps:
            chosen_model = next(
                (
                    decision.chosen_action
                    for decision in step.decisions
                    if decision.decision_type == "model_call"
                ),
                None,
            )
            for obs in step.observations:
                content = obs.content
                cost = content.get("cost_usd", 0.0)
                if (
                    isinstance(cost, bool)
                    or not isinstance(cost, int | float)
                    or not math.isfinite(cost)
                    or cost < 0
                ):
                    raise ValueError(f"{run.run_id}: invalid cost_usd {cost!r}")
                cost_total += cost
                if chosen_model is not None and content.get("extracted_code") is None:
                    extraction_failures["total"] += 1
                    extraction_failures[f"fixture={metadata.fixture_id}"] += 1
                    extraction_failures[f"model={chosen_model}"] += 1
    n = sum(cells.values())
    return {
        "n": n,
        "fixtures": dict(fixtures),
        "cost_usd": cost_total,
        "extraction_failures": dict(extraction_failures),
        "failure_modes": {mode: failure_modes[mode] for mode in _FAILURE_MODES},
        "failure_modes_by_fixture": dict(failure_modes_by_fixture),
        "failure_modes_by_model": dict(failure_modes_by_model),
        "table": {
            "public=T,hidden=T": cells[(True, True)],
            "public=T,hidden=F (gap)": cells[(True, False)],
            "public=F,hidden=T": cells[(False, True)],
            "public=F,hidden=F": cells[(False, False)],
        },
        "gate_passed": cells[(True, False)] >= 3,
        "showcase_gate_passed": _showcase_gate(failure_modes, failure_modes_by_model),
    }


def _count_table(title: str, rows: dict[str, int]) -> list[str]:
    return [
        f"## {title}",
        "| slice | count |",
        "|-------|------:|",
        *(f"| {key} | {value} |" for key, value in sorted(rows.items())),
    ]


def render(report: PilotReport) -> str:
    table = report["table"]
    n = report["n"]
    cost = report["cost_usd"]
    fixtures = report["fixtures"]
    extraction_failures = report["extraction_failures"]
    gate = "PASS" if report["gate_passed"] else "FAIL"
    showcase_gate = "PASS" if report["showcase_gate_passed"] else "FAIL"
    label = "/".join(sorted(fixtures)) if fixtures else "unknown"
    extraction_lines = ["## Extraction failures", "No fenced-code extraction failures."]
    if extraction_failures:
        extraction_lines = _count_table(
            "Extraction failures",
            {key: value for key, value in extraction_failures.items() if key != "total"},
        )
        extraction_lines.insert(
            1,
            f"Total model calls with no extracted code: {extraction_failures['total']}",
        )
    failure_mode_lines = _count_table(
        "Failure modes",
        report["failure_modes"],
    )
    failure_fixture_lines = _count_table(
        "Failure modes by fixture",
        report["failure_modes_by_fixture"],
    )
    failure_model_lines = _count_table(
        "Failure modes by model",
        report["failure_modes_by_model"],
    )
    return "\n".join(
        [
            f"# {label} pilot — n={n}",
            f"Cost: ${cost:.4f}  ({cost / n:.4f} / trace)",
            f"Fixtures: {fixtures}",
            "",
            "## 2x2 contingency (public_pass x hidden_pass)",
            "|             | hidden=T | hidden=F |",
            "|-------------|---------:|---------:|",
            (
                f"| **public=T**| {table['public=T,hidden=T']:8d} | "
                f"{table['public=T,hidden=F (gap)']:8d} |"
            ),
            (
                f"| **public=F**| {table['public=F,hidden=T']:8d} | "
                f"{table['public=F,hidden=F']:8d} |"
            ),
            "",
            "## Decision gate",
            (
                "Generalization-gap cell `(public=T, hidden=F)`: "
                f"{table['public=T,hidden=F (gap)']} (need >=3)"
            ),
            f"Gate: **{gate}**",
            "",
            *failure_mode_lines,
            "",
            *failure_fixture_lines,
            "",
            *failure_model_lines,
            "",
            "## Showcase composition gate",
            (
                "Requires hidden-semantic failures in both model arms and "
                "format failures not to be the majority of failures."
            ),
            f"Gate: **{showcase_gate}**",
            "",
            *extraction_lines,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.run_dir)
        text = render(report)
        notes = args.run_dir / "PILOT_NOTES.md"
        notes.write_text(text + "\n")
    except (CorpusIOError, ValueError, OSError) as exc:
        print(f"counterfact analyze-pilot: {exc}", file=sys.stderr)
        return 2
    print(text)
    print(f"\n(wrote {notes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
