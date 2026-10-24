from __future__ import annotations

import argparse
import shlex
import sys

from counterfact.cli.constants import BENCH_UNAVAILABLE_MESSAGE


def run(args: argparse.Namespace) -> int:
    try:
        from bench.synthetic.suite import run_suite, to_markdown, write_outputs
    except ImportError:
        print(BENCH_UNAVAILABLE_MESSAGE, file=sys.stderr)
        return 2

    try:
        summaries = run_suite(args.seeds, scenario_names=args.scenario, workers=args.workers)
    except KeyError as exc:
        print(f"counterfact bench suite: {exc.args[0]}", file=sys.stderr)
        return 2
    command = "uv run counterfact " + shlex.join(sys.argv[1:])
    if args.markdown is None:
        print(to_markdown(summaries, seeds=args.seeds, command=command), end="")
    write_outputs(
        summaries, seeds=args.seeds, command=command, markdown=args.markdown, json_path=args.json
    )
    for path in (args.markdown, args.json):
        if path is not None:
            print(str(path.resolve()), file=sys.stderr)
    return 0
