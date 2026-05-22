"""Shared loaders for committed real corpora used in acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from counterfact.schema import Run

REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_CLASS_REFUSAL_DIR = REPO_ROOT / "bench" / "real" / "single_class_refusal"
SMOKE_MIXED_OUTCOME_DIR = REPO_ROOT / "bench" / "real" / "smoke_mixed_outcome"


def repo_root() -> Path:
    return REPO_ROOT


def load_json_corpus(runs_dir: Path) -> list[Run]:
    return [Run.model_validate_json(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]


def load_single_class_refusal() -> list[Run]:
    if not SINGLE_CLASS_REFUSAL_DIR.exists():
        pytest.skip(f"single_class_refusal corpus absent at {SINGLE_CLASS_REFUSAL_DIR}")
    return load_json_corpus(SINGLE_CLASS_REFUSAL_DIR)


def load_smoke_mixed_outcome() -> list[Run]:
    if not SMOKE_MIXED_OUTCOME_DIR.exists():
        pytest.skip(f"smoke_mixed_outcome corpus absent at {SMOKE_MIXED_OUTCOME_DIR}")
    return load_json_corpus(SMOKE_MIXED_OUTCOME_DIR)
