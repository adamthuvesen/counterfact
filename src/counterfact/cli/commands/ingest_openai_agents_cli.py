from __future__ import annotations

import argparse
import sys


def run(args: argparse.Namespace) -> int:
    from counterfact.adapters._common import IngestError
    from counterfact.adapters.openai_agents import ingest_openai_agents

    outcome: bool | None
    if args.outcome is None:
        outcome = None
    elif args.outcome == "pass":
        outcome = True
    elif args.outcome == "fail":
        outcome = False
    else:
        print(
            "counterfact ingest openai-agents: --outcome must be 'pass' or 'fail'",
            file=sys.stderr,
        )
        return 2

    try:
        receipt = ingest_openai_agents(args.source_json, args.output_dir, outcome=outcome)
    except IngestError as exc:
        print(f"counterfact ingest openai-agents: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact ingest openai-agents: wrote {receipt.generated_count} trace(s)")
        print(str((args.output_dir / "ingest-receipt.json").resolve()))
        for warning in receipt.warnings:
            print(f"warning: {warning}")
    return 0
