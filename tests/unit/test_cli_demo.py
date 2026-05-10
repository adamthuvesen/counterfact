from __future__ import annotations

import builtins
from pathlib import Path

from counterfact.cli import main


def test_demo__defaults_to_smoke_mixed_outcome_and_reports_identified_verdict(
    capsys,
) -> None:
    """Default --runs-dir is smoke_mixed_outcome.

    The engine must fit the outcome model and produce an `identified` verdict
    with a finite outcome_delta and a structured next_step.
    """
    rc = main(["demo", "--bootstrap", "20"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "counterfact demo: naive vs honest" in out
    assert "data: bench/real/smoke_mixed_outcome" in out
    assert "pass_rate_by_arm(model_call)" in out
    assert "intervene(model_call ->" in out
    assert "identifiability: identified" in out
    assert "outcome_delta:" in out
    assert "next_step:" in out


def test_demo__on_single_class_refusal_reports_honest_refusal(capsys) -> None:
    """The single_class_refusal corpus is single-class by construction.

    With --runs-dir explicitly pointed at it, the engine must surface the
    degenerate-corpus refusal rather than fitting a one-class model.
    """
    rc = main(["demo", "--runs-dir", "bench/real/single_class_refusal", "--bootstrap", "20"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "data: bench/real/single_class_refusal" in out
    assert "identifiability: unidentified" in out
    assert "next_step: broaden_arm_support" in out
    assert "suggested_command: uv run counterfact bench real " in out


def test_demo__falls_back_to_synthetic_without_real_harness_import(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("bench.real.coding_agent.runner"):
            raise AssertionError("counterfact demo must not import the real-agent runner")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    rc = main(
        [
            "demo",
            "--runs-dir",
            str(tmp_path / "missing"),
            "--synthetic-n",
            "120",
            "--bootstrap",
            "20",
            "--target",
            "sonnet",
            "--synthetic-fallback",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "data: synthetic SCM" in out
    assert "pass_rate_by_arm(model_call)" in out
    assert "intervene(model_call -> sonnet)" in out
    assert "identifiability: identified" in out
    # When the synthetic-tight branch fires, next_step.action is "none" and
    # no suggested_command line is printed; when it's wide, an increase_n
    # suggestion is printed. Either is valid.
    if "next_step: none" in out:
        assert "suggested_command:" not in out


def test_demo__missing_real_corpus_errors_without_explicit_synthetic_fallback(
    tmp_path: Path, capsys
) -> None:
    rc = main(["demo", "--runs-dir", str(tmp_path / "missing")])
    captured = capsys.readouterr()

    assert rc == 2
    assert "no real traces found" in captured.err
    assert "--synthetic-fallback" in captured.err


def test_demo__confound_runs_confounded_synthetic_showcase(capsys) -> None:
    """`counterfact demo --confound` generates a confounded synthetic corpus,
    runs the naive-vs-causal flow, and surfaces the contrast line. No real
    corpus is touched. The contrast line uses the project's epistemic voice."""
    rc = main(
        [
            "demo",
            "--confound",
            "--synthetic-n",
            "1000",
            "--seed",
            "42",
            "--bootstrap",
            "20",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "data: synthetic SCM (confounded, n=1000, seed=42)" in out
    assert "pass_rate_by_arm(model_call)" in out
    assert "intervene(model_call ->" in out
    assert "identifiability: identified" in out
    assert "naive_vs_causal_contrast:" in out
    # No real corpus paths in --confound mode.
    assert "bench/real" not in out
    # Voice contract: don't undersell or accuse the descriptive baseline.
    for forbidden in ("lie", "lies", "lying", "false", "fake"):
        assert forbidden not in out, f"forbidden token {forbidden!r} found in output"


def test_demo__default_invocation_unchanged(capsys) -> None:
    """Regression: without --confound, default `counterfact demo` still
    points at the committed smoke_mixed_outcome corpus."""
    rc = main(["demo", "--bootstrap", "20"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "data: bench/real/smoke_mixed_outcome" in out
    # Confounded contrast line is suppressed in default mode.
    assert "naive_vs_causal_contrast:" not in out


def test_demo__confound_suppresses_contrast_line_when_gap_below_threshold(
    monkeypatch, capsys
) -> None:
    """When the measured naive-vs-causal gap is below the threshold, the
    contrast line is suppressed — the demo never asserts a contrast the
    sample does not show."""
    import counterfact.cli as cli

    # Push the threshold above any measurable gap so the contrast can never fire.
    monkeypatch.setattr(cli, "_DEMO_CONTRAST_THRESHOLD", 10.0)

    rc = cli.main(
        [
            "demo",
            "--confound",
            "--synthetic-n",
            "1000",
            "--seed",
            "42",
            "--bootstrap",
            "20",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "naive_vs_causal_contrast:" not in out
