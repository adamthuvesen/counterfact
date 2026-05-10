"""Shared test helpers for synthetic and committed corpora.

These factory functions are imported directly by test modules instead of being
exposed as `pytest.fixture`s because call sites pass varied `n` and `seed`
combinations and several need both the in-memory `Run` list and the on-disk
JSON layout. Keeping them as plain functions preserves the original call-site
ergonomics while removing the duplicated bodies that previously lived in
six test files.
"""

from __future__ import annotations

from pathlib import Path

from counterfact.schema import Run


def synthetic_corpus(n: int = 16, seed: int = 42) -> list[Run]:
    """Generate a synthetic SCM corpus as in-memory `Run` objects."""
    from bench.synthetic import generate_traces

    return [Run.model_validate(trace) for trace in generate_traces(n=n, seed=seed)]


def write_synthetic_corpus(target: Path, *, n: int = 16, seed: int = 42) -> list[Path]:
    """Materialize a synthetic SCM corpus to disk; return the written paths.

    Writes are routed through `Run.model_dump_json()` so the on-disk shape
    matches the strict schema producers rely on.
    """
    from bench.synthetic import generate_traces

    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for trace in generate_traces(n=n, seed=seed):
        run = Run.model_validate(trace)
        path = target / f"{run.run_id}.json"
        path.write_text(run.model_dump_json())
        paths.append(path)
    return paths


def single_class_refusal_corpus() -> list[Run]:
    """Load the committed single-class refusal corpus."""
    paths = sorted(Path("bench/real/single_class_refusal").glob("*.json"))
    assert paths, "single_class_refusal corpus must be committed for this test"
    return [Run.model_validate_json(p.read_text()) for p in paths]
