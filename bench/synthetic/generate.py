"""Synthetic-corpus generator. Pure-Python; no LLM calls; deterministic per seed."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from bench.synthetic.scm import SyntheticSCM


def generate_traces(n: int, seed: int = 42, confound: bool = False) -> Iterator[dict]:
    """Yield `n` traces deterministically given the seed.

    `confound=True` opts into the showcase mode where `model_choice` is biased
    by the run's earlier `tool_choice` (see `bench.synthetic.scm`). Default is
    the existing uniform-randomization mode.
    """
    scm = SyntheticSCM(seed=seed, confound=confound)
    for i in range(n):
        yield scm.sample_run(i)


def generate_corpus(
    n: int, seed: int, output_dir: str | Path, confound: bool = False
) -> Path:
    """Write `n` traces to `output_dir` as `syn-<i>.json` files. Returns the dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.json"))
    if existing:
        raise ValueError(
            f"output directory already contains JSON trace files: {out}. "
            "Use an empty directory for synthetic generation."
        )
    for trace in generate_traces(n, seed=seed, confound=confound):
        trace_path = out / f"{trace['run_id']}.json"
        trace_path.write_text(json.dumps(trace, sort_keys=True, indent=2))
    return out
