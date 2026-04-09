"""`counter` CLI. Subcommands: `bench synthetic`, `bench real` (real lands in §12)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bench_synthetic(args: argparse.Namespace) -> int:
    from bench.synthetic.generate import generate_corpus

    out = generate_corpus(n=args.n, seed=args.seed, output_dir=args.output_dir)
    print(f"Wrote {args.n} synthetic traces to {out}")
    return 0


def _bench_real(args: argparse.Namespace) -> int:
    # Stub: real-agent harness lands in §11/§12 with HUMAN GATE.
    print(
        "real-agent corpus generator not yet implemented (tasks.md §11-§12).",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="counter", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Generate CounterBench corpora")
    bench_sub = bench.add_subparsers(dest="bench_kind", required=True)

    syn = bench_sub.add_parser(
        "synthetic", help="Generate synthetic SCM traces (no LLM, deterministic)"
    )
    syn.add_argument("--n", type=int, required=True, help="Number of traces to generate")
    syn.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    syn.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench/synthetic/_out"),
        help="Where to write traces (default: bench/synthetic/_out)",
    )
    syn.set_defaults(func=_bench_synthetic)

    real = bench_sub.add_parser(
        "real", help="Generate real-agent traces (HUMAN GATE on first run)"
    )
    real.add_argument("--n", type=int, required=True)
    real.add_argument("--budget-cap", type=float, default=50.0)
    real.add_argument("--output-dir", type=Path, default=Path("bench/real/runs"))
    real.set_defaults(func=_bench_real)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
