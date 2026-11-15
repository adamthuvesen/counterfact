"""Update the generated routed-trial result block from committed traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.real.analyze_routed_trial import analyze, to_markdown
from counterfact.corpus_io import CorpusIOError, load_corpus
from scripts._generated import write_or_check

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN GENERATED ROUTED RESULTS -->"
END = "<!-- END GENERATED ROUTED RESULTS -->"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = ROOT / "bench" / "real" / "routed_trial" / "RESULTS.md"
    try:
        original = path.read_text()
        if original.count(BEGIN) != 1 or original.count(END) != 1:
            raise ValueError(f"{path} must have exactly one generated-results marker pair")
        before, remaining = original.split(BEGIN)
        _, after = remaining.split(END)
        text = to_markdown(analyze(load_corpus(path.parent)))
        updated = before + BEGIN + "\n\n" + text.rstrip() + "\n\n" + END + after
        return 0 if write_or_check(path, updated, check=args.check) else 1
    except (CorpusIOError, OSError, ValueError) as exc:
        print(f"routed results: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
