"""Synthetic-corpus generator. Pure-Python; no LLM calls; deterministic per seed."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from bench.synthetic.scm import SyntheticSCM


def generate_traces(n: int, seed: int = 42) -> Iterator[dict]:
    """Yield `n` traces deterministically given the seed."""
    scm = SyntheticSCM(seed=seed)
    for i in range(n):
        yield scm.sample_run(i)


def generate_corpus(n: int, seed: int, output_dir: str | Path) -> Path:
    """Write `n` traces to `output_dir` as `syn-<i>.json` files. Returns the dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for trace in generate_traces(n, seed=seed):
        trace_path = out / f"{trace['run_id']}.json"
        trace_path.write_text(json.dumps(trace, sort_keys=True, indent=2))
    return out
