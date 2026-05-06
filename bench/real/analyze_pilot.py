"""Compute the 2x2 (public_pass, hidden_pass) contingency from a hidden-fixture run.

Used by the hidden-test-fixtures pilot decision gate (tasks 6.2-6.4): if the
cell `(public_pass=True, hidden_pass=False)` contains >=3 traces, the
generalization gap exists and we replicate to date_window + rate_limit.
Otherwise we stop and rethink.

Run:
    uv run python -m bench.real.analyze_pilot bench/real/pilot_<YYYY-MM-DD>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_FAILURE_MODES = (
    "pass",
    "format_failure",
    "public_failure",
    "hidden_semantic_failure",
    "timeout",
    "exception",
)


def _model_attempts(run: dict[str, object]) -> list[tuple[str, object]]:
    attempts: list[tuple[str, object]] = []
    for step in run["steps"]:  # type: ignore[index]
        model_arm = None
        for decision in step.get("decisions", []) or []:
            if decision.get("decision_type") == "model_call":
                model_arm = decision.get("chosen_action")
        if model_arm is None:
            continue
        for obs in step.get("observations", []) or []:
            content = obs.get("content", {})
            attempts.append((str(model_arm), content.get("extracted_code")))
    return attempts


def _stdout_tails(run: dict[str, object]) -> list[str]:
    tails: list[str] = []
    for step in run["steps"]:  # type: ignore[index]
        for obs in step.get("observations", []) or []:
            tail = obs.get("content", {}).get("stdout_tail")
            if isinstance(tail, str):
                tails.append(tail)
    return tails


def _failure_mode(run: dict[str, object]) -> str:
    md = run["outcome"]["metadata"]  # type: ignore[index]
    if md["hidden_pass"]:
        return "pass"

    tails = _stdout_tails(run)
    if any("<pytest timed out" in tail for tail in tails):
        return "timeout"
    if any("traceback" in tail.lower() for tail in tails):
        return "exception"
    if not md["public_pass"]:
        return "public_failure"

    attempts = _model_attempts(run)
    if not attempts or attempts[-1][1] is None:
        return "format_failure"
    return "hidden_semantic_failure"


def _final_model_arm(run: dict[str, object]) -> str:
    attempts = _model_attempts(run)
    if not attempts:
        return "unknown"
    return attempts[-1][0]


def _showcase_gate(failure_modes: Counter[str], by_model: Counter[str]) -> bool:
    failures = sum(
        count for mode, count in failure_modes.items() if mode != "pass"
    )
    if failures == 0:
        return False
    if failure_modes["format_failure"] > failures / 2:
        return False
    return (
        by_model["model=small,mode=hidden_semantic_failure"] > 0
        and by_model["model=large,mode=hidden_semantic_failure"] > 0
    )


def analyze(run_dir: Path) -> dict[str, object]:
    traces = sorted(run_dir.glob("real-*.json"))
    if not traces:
        raise SystemExit(f"no traces under {run_dir}")
    cells: Counter[tuple[bool, bool]] = Counter()
    cost_total = 0.0
    fixtures: Counter[str] = Counter()
    extraction_failures: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    failure_modes_by_fixture: Counter[str] = Counter()
    failure_modes_by_model: Counter[str] = Counter()
    for path in traces:
        run = json.loads(path.read_text())
        md = run["outcome"]["metadata"]
        if "public_pass" not in md or "hidden_pass" not in md:
            raise SystemExit(
                f"{path.name}: missing public_pass/hidden_pass in metadata "
                "(is this a v0 trace?)"
            )
        cells[(md["public_pass"], md["hidden_pass"])] += 1
        fixtures[md["fixture_id"]] += 1
        mode = _failure_mode(run)
        model_arm = _final_model_arm(run)
        failure_modes[mode] += 1
        failure_modes_by_fixture[f"fixture={md['fixture_id']},mode={mode}"] += 1
        failure_modes_by_model[f"model={model_arm},mode={mode}"] += 1
        for step in run["steps"]:
            model_arm = None
            for decision in step.get("decisions", []) or []:
                if decision.get("decision_type") == "model_call":
                    model_arm = decision.get("chosen_action")
            for obs in step.get("observations", []) or []:
                content = obs.get("content", {})
                cost_total += content.get("cost_usd", 0.0)
                if model_arm is not None and content.get("extracted_code") is None:
                    extraction_failures["total"] += 1
                    extraction_failures[f"fixture={md['fixture_id']}"] += 1
                    extraction_failures[f"model={model_arm}"] += 1
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


def render(report: dict[str, object]) -> str:
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
            {
                key: value
                for key, value in extraction_failures.items()
                if key != "total"
            },
        )
        extraction_lines.insert(
            1,
            f"Total model calls with no extracted code: {extraction_failures['total']}",
        )
    failure_mode_lines = _count_table(
        "Failure modes",
        report["failure_modes"],  # type: ignore[arg-type]
    )
    failure_fixture_lines = _count_table(
        "Failure modes by fixture",
        report["failure_modes_by_fixture"],  # type: ignore[arg-type]
    )
    failure_model_lines = _count_table(
        "Failure modes by model",
        report["failure_modes_by_model"],  # type: ignore[arg-type]
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
            "## Decision gate (task 6.3)",
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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_dir = Path(argv[1])
    report = analyze(run_dir)
    print(render(report))
    notes = run_dir / "PILOT_NOTES.md"
    notes.write_text(render(report) + "\n")
    print(f"\n(wrote {notes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
