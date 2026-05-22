from __future__ import annotations

import argparse
import sys

from counterfact.cli.constants import BENCH_UNAVAILABLE_MESSAGE


def run(args: argparse.Namespace) -> int:
    try:
        from bench.synthetic.generate import generate_corpus
    except ImportError:
        print(BENCH_UNAVAILABLE_MESSAGE, file=sys.stderr)
        return 2

    try:
        out = generate_corpus(n=args.n, seed=args.seed, output_dir=args.output_dir)
    except ValueError as exc:
        print(f"counterfact bench synthetic: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.n} synthetic traces to {out}")
    return 0
