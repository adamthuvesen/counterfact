from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from counterfact.cli import formatters
from counterfact.schema import Run

format_report = formatters.format_report


def run(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from counterfact.corpus_analyzer import RubricThresholds, analyze

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists() or not runs_dir.is_dir():
        print(
            f"counterfact analyze corpus: directory not found: {runs_dir}",
            file=sys.stderr,
        )
        return 2

    runs: list[Run] = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            runs.append(Run.model_validate_json(path.read_text()))
        except (ValidationError, ValueError) as exc:
            print(
                f"counterfact analyze corpus: failed to parse {path}: {exc}",
                file=sys.stderr,
            )
            return 2

    overrides: dict[str, Any] = {}
    if args.min_pass_rate is not None:
        overrides["min_pass_rate"] = args.min_pass_rate
    if args.max_pass_rate is not None:
        overrides["max_pass_rate"] = args.max_pass_rate
    if args.min_arms is not None:
        overrides["min_arms_per_decision_type"] = args.min_arms
    if args.min_n_per_arm is not None:
        overrides["min_n_per_arm"] = args.min_n_per_arm
    if args.min_identified is not None:
        overrides["min_identified_decision_types"] = args.min_identified
    try:
        thresholds = RubricThresholds(**overrides) if overrides else RubricThresholds()
    except ValidationError as exc:
        print(f"counterfact analyze corpus: invalid thresholds: {exc}", file=sys.stderr)
        return 2

    report = analyze(runs, thresholds=thresholds)
    print(format_report(report, runs_dir))
    return 0 if report.promote else 1
