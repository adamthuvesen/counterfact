"""Composition tests for `counterfact.explain.build_report`.

These tests assert behavior of the typed `ExplainReport` model — ranking,
identifiability propagation, the single-class refusal contract, and
determinism — without going through the HTML renderer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from counterfact.explain import build_report
from counterfact.intervene.estimate import IdentifiabilityStatus
from counterfact.schema import Run


def _synthetic_corpus(n: int = 24, seed: int = 7) -> list[Run]:
    from bench.synthetic import generate_traces

    return [Run.model_validate(t) for t in generate_traces(n=n, seed=seed)]


def _runs_v1_corpus() -> list[Run]:
    paths = sorted(Path("bench/real/runs_v1").glob("*.json"))
    assert paths, "runs_v1 corpus must be committed for this test"
    return [Run.model_validate_json(p.read_text()) for p in paths]


def test_build_report__mixed_outcome_corpus_has_ranked_attribution() -> None:
    """A small synthetic corpus with both pass and fail outcomes must
    produce a non-empty attribution ranked by influence descending."""
    corpus = _synthetic_corpus()
    classes = {bool(r.outcome.value) for r in corpus}
    assert len(classes) == 2, "synthetic corpus should not be single-class"

    focal = corpus[0]
    report = build_report(
        focal, corpus, decision_type="model_call", bootstrap=20, seed=42
    )

    assert report.degenerate_estimate is None
    assert report.attribution.entries, "mixed-outcome corpus must yield entries"
    influences = [e.influence for e in report.attribution.entries]
    assert influences == sorted(influences, reverse=True)


def test_build_report__single_class_corpus_returns_degenerate_refusal() -> None:
    """The runs_v1 single-class corpus must surface the unidentified
    refusal and yield zero attribution entries."""
    corpus = _runs_v1_corpus()
    focal = corpus[0]

    report = build_report(
        focal, corpus, decision_type="model_call", bootstrap=20, seed=42
    )

    assert report.attribution.entries == []
    assert report.degenerate_estimate is not None
    estimate = report.degenerate_estimate
    assert estimate.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert estimate.next_step.action == "broaden_arm_support"


def test_build_report__never_calls_fit_outcome_model_on_single_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The honesty contract: no model fit on single-class corpora."""
    import counterfact.explain.report as report_module

    calls = {"n": 0}

    def fail_if_called(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError(
            "fit_outcome_model must not be called on a single-class corpus"
        )

    monkeypatch.setattr(report_module, "fit_outcome_model", fail_if_called)

    corpus = _runs_v1_corpus()
    focal = corpus[0]
    report = build_report(focal, corpus, bootstrap=20, seed=42)

    assert calls["n"] == 0
    assert report.degenerate_estimate is not None


def test_build_report__deterministic_for_fixed_inputs() -> None:
    corpus = _synthetic_corpus()
    focal = corpus[0]

    a = build_report(focal, corpus, bootstrap=20, seed=42)
    b = build_report(focal, corpus, bootstrap=20, seed=42)

    # Pydantic equality covers attribution.entries via the inner models.
    assert a.model_dump(mode="python") == b.model_dump(mode="python")


def test_build_report__rejects_focal_run_not_in_corpus() -> None:
    corpus = _synthetic_corpus()
    intruder = _runs_v1_corpus()[0]
    with pytest.raises(ValueError, match="not present in the supplied corpus"):
        build_report(intruder, corpus, bootstrap=20, seed=42)
