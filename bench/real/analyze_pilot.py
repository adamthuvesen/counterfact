"""Compute the 2x2 (public_pass, hidden_pass) contingency from a hidden-fixture run.

Used by the hidden-test-fixtures pilot decision gate (tasks 6.2-6.4): if the
cell `(public_pass=True, hidden_pass=False)` contains >=3 traces, the
generalization gap exists and we replicate to date_window + rate_limit.
Otherwise we stop and rethink.

Run:
    uv run python -m bench.real.analyze_pilot bench/real/runs_v3_pilot
"""

from __future__ import annotations

import json
import sys
from collections import Counterfact
from pathlib import Path


def analyze(run_dir: Path) -> dict[str, object]:
    traces = sorted(run_dir.glob("real-*.json"))
    if not traces:
        raise SystemExit(f"no traces under {run_dir}")
    cells: Counterfact[tuple[bool, bool]] = Counterfact()
    cost_total = 0.0
    fixtures: Counterfact[str] = Counterfact()
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
        for step in run["steps"]:
            for obs in step.get("observations", []) or []:
                cost_total += obs.get("content", {}).get("cost_usd", 0.0)
    n = sum(cells.values())
    return {
        "n": n,
        "fixtures": dict(fixtures),
        "cost_usd": cost_total,
        "table": {
            "public=T,hidden=T": cells[(True, True)],
            "public=T,hidden=F (gap)": cells[(True, False)],
            "public=F,hidden=T": cells[(False, True)],
            "public=F,hidden=F": cells[(False, False)],
        },
        "gate_passed": cells[(True, False)] >= 3,
    }


def render(report: dict[str, object]) -> str:
    table = report["table"]
    n = report["n"]
    cost = report["cost_usd"]
    fixtures = report["fixtures"]
    gate = "PASS" if report["gate_passed"] else "FAIL"
    return "\n".join(
        [
            f"# csv_dedupe pilot — n={n}",
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
