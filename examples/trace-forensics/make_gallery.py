"""Materialize the trace-forensics gallery fixtures.

The gallery is illustrative: it gives stable local traces for the docs and
tests, not benchmark evidence. Running this script should be a byte-stable
regeneration of the committed JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.synthetic import generate_traces
from counterfact.schema import Run

ROOT = Path(__file__).resolve().parent


def _write_run(path: Path, payload: dict[str, Any]) -> None:
    Run.model_validate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_synthetic_corpus() -> list[dict[str, Any]]:
    traces = list(generate_traces(n=60, seed=42))
    out = ROOT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for trace in traces:
        _write_run(out / f"{trace['run_id']}.json", trace)
    return traces


def _write_single_arm_mixed_corpus(traces: list[dict[str, Any]]) -> None:
    out = ROOT / "single-arm-model"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for idx, trace in enumerate(traces[:12]):
        payload = json.loads(json.dumps(trace))
        payload["run_id"] = f"single-arm-{idx:06d}"
        for step in payload["steps"]:
            for decision in step["decisions"]:
                decision["decision_id"] = decision["decision_id"].replace(
                    trace["run_id"].replace("syn-", "d-"),
                    f"d-single-arm-{idx:06d}",
                )
                if decision["decision_type"] == "model_call":
                    decision["chosen_action"] = "haiku"
        # Keep outcomes mixed while the model arm has no sibling support.
        payload["outcome"]["value"] = idx % 2 == 0
        _write_run(out / f"{payload['run_id']}.json", payload)


def _write_stopped_early_pair(traces: list[dict[str, Any]]) -> None:
    fail = json.loads(json.dumps(traces[0]))
    passed = json.loads(json.dumps(traces[3]))
    fail["run_id"] = "stopped-early-fail"
    passed["run_id"] = "stopped-early-pass"
    for payload, termination in ((fail, "stop"), (passed, "success")):
        payload["outcome"]["value"] = termination == "success"
        for step in payload["steps"]:
            for decision in step["decisions"]:
                decision["decision_id"] = decision["decision_id"].replace(
                    "d-000000",
                    f"d-{payload['run_id']}",
                ).replace(
                    "d-000003",
                    f"d-{payload['run_id']}",
                )
                if decision["decision_type"] == "termination":
                    decision["chosen_action"] = termination
    _write_run(ROOT / "stopped-early" / "fail.json", fail)
    _write_run(ROOT / "stopped-early" / "pass.json", passed)


def main() -> None:
    traces = _write_synthetic_corpus()
    _write_single_arm_mixed_corpus(traces)
    _write_stopped_early_pair(traces)


if __name__ == "__main__":
    main()
