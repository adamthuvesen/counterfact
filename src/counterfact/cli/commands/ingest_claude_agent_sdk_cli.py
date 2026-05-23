from __future__ import annotations

import argparse
import sys


def run(args: argparse.Namespace) -> int:
    from counterfact.adapters._common import IngestError
    from counterfact.adapters.claude_agent_sdk import ingest_claude_agent_sdk

    try:
        receipt = ingest_claude_agent_sdk(args.source_jsonl, args.output_dir)
    except IngestError as exc:
        print(f"counterfact ingest claude-agent-sdk: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact ingest claude-agent-sdk: wrote {receipt.generated_count} trace(s)")
        print(str((args.output_dir / "ingest-receipt.json").resolve()))
        for warning in receipt.warnings:
            print(f"warning: {warning}")
    return 0
