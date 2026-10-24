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

    try:
        route = _parse_route(args.model_route_by_fixture)
    except ValueError as exc:
        print(f"counterfact bench real: {exc}", file=sys.stderr)
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
        model_route_by_fixture=route,
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


def _parse_route(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    route: dict[str, float] = {}
    for pair in (p.strip() for p in raw.split(",") if p.strip()):
        fixture_id, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--model-route-by-fixture: expected fixture_id=P, got {pair!r}")
        if fixture_id.strip() in route:
            raise ValueError(f"--model-route-by-fixture: {fixture_id.strip()!r} appears twice")
        try:
            route[fixture_id.strip()] = float(value)
        except ValueError:
            raise ValueError(f"--model-route-by-fixture: {value!r} is not a probability") from None
    return route
