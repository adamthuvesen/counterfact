"""§14 acceptance: hand-labeled root cause is the top-1 attribution.

The labels file at `bench/real/coding_agent/labels.json` is filled in by the
human (§14.1 HUMAN GATE). Once at least one label is present and the smoke_mixed_outcome
corpus exists, this test:

* loads each label entry,
* loads the labeled run from `bench/real/smoke_mixed_outcome/<run_id>.json`,
* fits the outcome model on the full smoke_mixed_outcome corpus,
* calls `attribute_failure(dag, model)` on the labeled run,
* asserts `top_k(1)[0].decision_id == label.root_cause_decision_id`.

The test SKIPS (not fails) when labels is empty or smoke_mixed_outcome is absent — that
state is the §14.1 gate sitting open, not a regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact import attribute_failure, build_dag, fit_outcome_model
from counterfact.schema import Run

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "bench" / "real" / "coding_agent" / "labels.json"
CORPUS_DIR = REPO_ROOT / "bench" / "real" / "smoke_mixed_outcome"


def _load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    payload = json.loads(LABELS_PATH.read_text())
    return list(payload.get("labels", []))


def _load_corpus() -> list[Run]:
    if not CORPUS_DIR.exists():
        return []
    return [Run.model_validate_json(p.read_text()) for p in sorted(CORPUS_DIR.glob("*.json"))]


@pytest.fixture(scope="module")
def labels() -> list[dict]:
    items = _load_labels()
    if not items:
        pytest.skip(
            "§14.1 HUMAN GATE open: no entries in labels.json. "
            "Add at least one label to enable this acceptance test."
        )
    return items


@pytest.fixture(scope="module")
def corpus() -> list[Run]:
    runs = _load_corpus()
    if not runs:
        pytest.skip(
            f"smoke_mixed_outcome corpus absent at {CORPUS_DIR}. "
            f"Promote a real corpus per bench/real/README.md before running this test."
        )
    return runs


@pytest.fixture(scope="module")
def fitted_model(corpus: list[Run]) -> object:
    return fit_outcome_model(corpus, n_bootstrap=100, seed=42)


def _run_by_id(corpus: list[Run], run_id: str) -> Run | None:
    for r in corpus:
        if r.run_id == run_id:
            return r
    return None


def test_labels_file_is_present_and_well_formed() -> None:
    """The labels file must exist and parse, even when empty."""
    assert LABELS_PATH.exists(), f"missing labels file at {LABELS_PATH}"
    payload = json.loads(LABELS_PATH.read_text())
    assert "labels" in payload and isinstance(payload["labels"], list)


def test_top1_attribution_matches_hand_labeled_root_cause(
    labels: list[dict], corpus: list[Run], fitted_model: object
) -> None:
    """For each label, the attribution top-1 must match the hand-labeled decision."""
    failures: list[str] = []
    for label in labels:
        run_id = label["run_id"]
        expected = label["root_cause_decision_id"]
        run = _run_by_id(corpus, run_id)
        if run is None:
            failures.append(
                f"label references run_id={run_id!r} which is not present in {CORPUS_DIR}"
            )
            continue
        if run.outcome.value is True:
            failures.append(
                f"label references run_id={run_id!r} but its outcome is success "
                f"(labeling is only meaningful for failed runs)"
            )
            continue
        attribution = attribute_failure(dag=build_dag(run), model=fitted_model)
        top = attribution.top_k(1)
        if not top:
            failures.append(f"attribute_failure returned empty top_k for run_id={run_id!r}")
            continue
        actual = top[0].decision_id
        if actual != expected:
            failures.append(f"run_id={run_id!r}: top_1={actual!r}, labeled root cause={expected!r}")
    assert not failures, (
        "Top-1 attribution disagrees with hand-labeled root cause for "
        f"{len(failures)}/{len(labels)} label(s):\n  - " + "\n  - ".join(failures)
    )
