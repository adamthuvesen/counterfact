"""Shared loaders for committed real corpora used in acceptance tests."""

from __future__ import annotations

from pathlib import Path

from counterfact.corpus_io import load_corpus
from counterfact.schema import Run

REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_CLASS_REFUSAL_DIR = REPO_ROOT / "bench" / "real" / "single_class_refusal"
SMOKE_MIXED_OUTCOME_DIR = REPO_ROOT / "bench" / "real" / "smoke_mixed_outcome"


def repo_root() -> Path:
    return REPO_ROOT


def load_json_corpus(runs_dir: Path) -> list[Run]:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"required committed corpus is missing or empty: {runs_dir}")
    runs = load_corpus(runs_dir)
    if not runs:
        raise FileNotFoundError(f"required committed corpus is missing or empty: {runs_dir}")
    return runs


def load_single_class_refusal() -> list[Run]:
    return load_json_corpus(SINGLE_CLASS_REFUSAL_DIR)


def load_smoke_mixed_outcome() -> list[Run]:
    return load_json_corpus(SMOKE_MIXED_OUTCOME_DIR)
