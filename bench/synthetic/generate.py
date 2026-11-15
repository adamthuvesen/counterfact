"""Synthetic-corpus generator. Pure-Python; no LLM calls; deterministic per seed."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bench.synthetic.scm import SyntheticSCM
from counterfact.corpus_io import IngestReceipt, write_corpus
from counterfact.schema import Run


def generate_traces(n: int, seed: int = 42, confound: bool = False) -> Iterator[dict[str, Any]]:
    """Yield `n` traces deterministically given the seed.

    `confound=True` opts into the showcase mode where `model_choice` is biased
    by the run's earlier `tool_choice` (see `bench.synthetic.scm`). Default is
    the existing uniform-randomization mode.
    """
    scm = SyntheticSCM(seed=seed, confound=confound)
    for i in range(n):
        yield scm.sample_run(i)


def generate_corpus(n: int, seed: int, output_dir: str | Path, confound: bool = False) -> Path:
    """Publish `n` traces to `output_dir` as `syn-<i>.json` files with an ingest receipt."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.json"))
    if existing:
        raise ValueError(
            f"output directory already contains JSON trace files: {out}. "
            "Use an empty directory for synthetic generation."
        )
    runs = [Run.model_validate(trace) for trace in generate_traces(n, seed=seed, confound=confound)]
    receipt = IngestReceipt(
        source_format="bench-synthetic",
        source_file=f"seed={seed} confound={confound}",
        generated_count=n,
    )
    write_corpus(runs, out, receipt, append=True)
    return out
