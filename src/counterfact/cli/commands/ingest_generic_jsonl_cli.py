from __future__ import annotations

import argparse
import sys


def run(args: argparse.Namespace) -> int:
    from counterfact.ingest import IngestError, ingest_generic_jsonl

    try:
        receipt = ingest_generic_jsonl(args.source_jsonl, args.mapping, args.output_dir)
    except IngestError as exc:
        print(f"counterfact ingest generic-jsonl: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact ingest generic-jsonl: wrote {receipt.generated_count} trace(s)")
        print(str((args.output_dir / "ingest-receipt.json").resolve()))
        for warning in receipt.warnings:
            print(f"warning: {warning}")
    return 0
