from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from counterfact.cli import formatters
from counterfact.corpus_analyzer import RubricThresholds
from counterfact.schema import Run

format_report = formatters.format_report


def _load_runs(runs_dir: Path) -> list[Run] | None:
    if not runs_dir.exists() or not runs_dir.is_dir():
        print(
            f"counterfact analyze corpus: directory not found: {runs_dir}",
            file=sys.stderr,
        )
        return None

    runs: list[Run] = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            runs.append(Run.model_validate_json(path.read_text()))
        except (ValidationError, ValueError) as exc:
            print(
                f"counterfact analyze corpus: failed to parse {path}: {exc}",
                file=sys.stderr,
            )
            return None
    return runs


def _threshold_overrides(args: argparse.Namespace) -> dict[str, Any]:
    fields = {
        "min_pass_rate": args.min_pass_rate,
        "max_pass_rate": args.max_pass_rate,
        "min_arms_per_decision_type": args.min_arms,
        "min_n_per_arm": args.min_n_per_arm,
        "min_identified_decision_types": args.min_identified,
    }
    return {key: value for key, value in fields.items() if value is not None}


def _thresholds_from_args(args: argparse.Namespace) -> RubricThresholds | None:
    try:
        overrides = _threshold_overrides(args)
        thresholds = RubricThresholds(**overrides) if overrides else RubricThresholds()
    except ValidationError as exc:
        print(f"counterfact analyze corpus: invalid thresholds: {exc}", file=sys.stderr)
        return None
    return thresholds


def run(args: argparse.Namespace) -> int:
    from counterfact.corpus_analyzer import analyze

    runs_dir: Path = args.runs_dir
    runs = _load_runs(runs_dir)
    if runs is None:
        return 2

    thresholds = _thresholds_from_args(args)
    if thresholds is None:
        return 2

    report = analyze(runs, thresholds=thresholds)
    print(format_report(report, runs_dir))
    return 0 if report.promote else 1
