"""`counterfact` CLI package."""

from __future__ import annotations

import argparse

from counterfact.cli.constants import DEMO_CONTRAST_THRESHOLD

# Tests monkeypatch this name on the cli module.
_DEMO_CONTRAST_THRESHOLD = DEMO_CONTRAST_THRESHOLD


def build_parser() -> argparse.ArgumentParser:
    from counterfact.cli.parser import build_parser as _build_parser

    return _build_parser()


def main(argv: list[str] | None = None) -> int:
    from counterfact.cli.commands import ingest_formats

    parser = build_parser()
    ns = parser.parse_args(argv)
    if getattr(ns, "command", None) == "ingest" and getattr(ns, "list_formats", False):
        return ingest_formats.run()
    func = getattr(ns, "func", None)
    if func is None:
        parser.parse_args([ns.command, "--help"])
        return 2
    return int(func(ns))


__all__ = ["DEMO_CONTRAST_THRESHOLD", "build_parser", "main"]
