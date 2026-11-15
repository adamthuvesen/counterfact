"""Trace and corpus loading for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

from counterfact.cli.constants import BENCH_UNAVAILABLE_MESSAGE, DEFAULT_DEMO_RUNS_DIR
from counterfact.corpus_io import CorpusIOError, load_corpus, load_run
from counterfact.schema import Run, require_focal


def load_trace_dir(path: Path, *, command: str) -> list[Run] | None:
    return load_corpus_dir(path, command=command)


def load_run_file(path: Path, *, command: str) -> Run | None:
    try:
        return load_run(path)
    except CorpusIOError as exc:
        print(f"counterfact {command}: {exc}", file=sys.stderr)
        return None


def load_corpus_dir(path: Path, *, command: str) -> list[Run] | None:
    try:
        return load_corpus(path)
    except CorpusIOError as exc:
        print(f"counterfact {command}: {exc}", file=sys.stderr)
        return None


def require_focal_in_corpus(focal: Run, corpus: list[Run], runs_dir: Path, *, command: str) -> bool:
    try:
        require_focal(focal, corpus)
    except ValueError as exc:
        print(f"counterfact {command}: {exc} in {runs_dir}", file=sys.stderr)
        return False
    return True


def load_focal_and_corpus(
    run_path: Path,
    runs_dir: Path | None,
    *,
    command: str,
) -> tuple[Run, list[Run], Path] | None:
    focal = load_run_file(run_path, command=command)
    if focal is None:
        return None

    resolved_runs_dir = runs_dir if runs_dir is not None else run_path.parent
    corpus = load_corpus_dir(resolved_runs_dir, command=command)
    if corpus is None:
        return None
    if not require_focal_in_corpus(focal, corpus, resolved_runs_dir, command=command):
        return None
    return focal, corpus, resolved_runs_dir


def synthetic_runs(n: int, seed: int, confound: bool = False) -> list[Run]:
    try:
        from bench.synthetic import generate_traces
    except ImportError as exc:
        raise ImportError(BENCH_UNAVAILABLE_MESSAGE) from exc

    return [
        Run.model_validate(trace) for trace in generate_traces(n=n, seed=seed, confound=confound)
    ]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def demo_runs_dir(path: Path) -> tuple[Path, str]:
    if path.exists() or path != DEFAULT_DEMO_RUNS_DIR:
        return path, str(path)
    repo_path = repo_root() / DEFAULT_DEMO_RUNS_DIR
    if repo_path.exists():
        return repo_path, DEFAULT_DEMO_RUNS_DIR.as_posix()
    return path, str(path)
