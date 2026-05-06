"""Tests for the `counterfact analyze corpus` CLI subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.synthetic import generate_traces
from counterfact.cli import main
from counterfact.schema import Run

REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_CLASS_REFUSAL_DIR = REPO_ROOT / "bench" / "real" / "single_class_refusal"


def test_analyze_single_class_refusal_exits_1_with_outcome_balance_failure(capsys) -> None:
    if not SINGLE_CLASS_REFUSAL_DIR.exists():
        pytest.skip(f"single_class_refusal corpus absent at {SINGLE_CLASS_REFUSAL_DIR}")
    rc = main(["analyze", "corpus", str(SINGLE_CLASS_REFUSAL_DIR)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "support-readiness" in out
    assert "FAIL outcome_balance:" in out
    assert "promote: False" in out
    assert "next_collection_guidance:" in out
    assert "mixed pass/fail outcomes" in out


def test_analyze_synthetic_corpus_exits_0(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "syn"
    out_dir.mkdir()
    for i, trace in enumerate(generate_traces(n=200, seed=42)):
        run = Run.model_validate(trace)
        (out_dir / f"r-{i:04d}.json").write_text(run.model_dump_json())

    rc = main(["analyze", "corpus", str(out_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "support-readiness" in out
    assert "promote: True" in out
    assert "suitable for counterfactual-support workflows" in out
    # Every criterion should pass on the default rubric for synthetic SCM
    pass_lines = [line for line in out.splitlines() if line.startswith("PASS ")]
    fail_lines = [line for line in out.splitlines() if line.startswith("FAIL ")]
    assert pass_lines, "no PASS lines in synthetic analyzer output"
    assert not fail_lines, f"unexpected FAIL lines: {fail_lines}"


def test_analyze_missing_directory_exits_2(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "no-such-dir"
    rc = main(["analyze", "corpus", str(missing)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "directory not found" in captured.err
    assert "promote:" not in captured.out


def test_analyze_unparseable_json_exits_2(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "bad"
    out_dir.mkdir()
    (out_dir / "bad.json").write_text('{"not": "a run"}')
    rc = main(["analyze", "corpus", str(out_dir)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "failed to parse" in captured.err
    assert "bad.json" in captured.err


def test_analyze_thresholds_can_be_overridden_via_flags(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "syn-strict"
    out_dir.mkdir()
    for i, trace in enumerate(generate_traces(n=120, seed=3)):
        run = Run.model_validate(trace)
        (out_dir / f"r-{i:04d}.json").write_text(run.model_dump_json())

    # With default thresholds: promote=True
    rc_default = main(["analyze", "corpus", str(out_dir)])
    out_default = capsys.readouterr().out
    assert rc_default == 0
    assert "promote: True" in out_default

    # Tightening max_pass_rate well below the corpus's actual pass rate
    # should flip the verdict.
    rc_strict = main(
        [
            "analyze",
            "corpus",
            str(out_dir),
            "--min-pass-rate",
            "0.0",
            "--max-pass-rate",
            "0.05",
        ]
    )
    out_strict = capsys.readouterr().out
    assert rc_strict == 1
    assert "promote: False" in out_strict
    assert "FAIL outcome_balance:" in out_strict
    assert "next_collection_guidance:" in out_strict
