"""List supported ingest formats."""

from __future__ import annotations

from counterfact.cli.constants import INGEST_FORMATS


def run() -> int:
    width = max(len(name) for name, _ in INGEST_FORMATS)
    for name, description in INGEST_FORMATS:
        print(f"  {name:<{width}}  {description}")
    return 0
