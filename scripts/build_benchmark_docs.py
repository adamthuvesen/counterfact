"""Render the benchmark table from its committed JSON results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.synthetic.suite import to_markdown
from scripts._benchmark_results import BenchmarkResults
from scripts._generated import write_or_check

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        results = BenchmarkResults.model_validate_json(
            (ROOT / "docs" / "benchmark-results.json").read_text()
        )
        text = to_markdown(
            results.scenarios, results.policies, seeds=results.seeds, command=results.command
        )
        return (
            0
            if write_or_check(ROOT / "docs" / "benchmark-results.md", text, check=args.check)
            else 1
        )
    except (OSError, ValueError) as exc:
        print(f"benchmark docs: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
