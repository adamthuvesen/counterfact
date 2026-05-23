from __future__ import annotations

import argparse
import sys

from counterfact.cli.constants import BENCH_UNAVAILABLE_MESSAGE


def run(args: argparse.Namespace) -> int:
    try:
        from bench.real.coding_agent.agent import AgentRunConfig
        from bench.real.coding_agent.runner import run_real_corpus
    except ImportError:
        print(BENCH_UNAVAILABLE_MESSAGE, file=sys.stderr)
        return 2

    config = AgentRunConfig(
        seed=args.seed,
        epsilon=args.epsilon,
        tool_greedy=args.tool_greedy,
        tool_epsilon=args.tool_epsilon,
        model_greedy=args.model_greedy,
        model_epsilon=args.model_epsilon,
        retry_greedy=args.retry_greedy,
        retry_epsilon=args.retry_epsilon,
    )
    fixture_ids = (
        tuple(s.strip() for s in args.fixtures.split(",") if s.strip()) if args.fixtures else None
    )
    return run_real_corpus(
        n=args.n,
        budget_cap_usd=args.budget_cap,
        output_dir=args.output_dir,
        config=config,
        fixture_ids=fixture_ids,
        fixture_set=args.fixture_set,
    )
