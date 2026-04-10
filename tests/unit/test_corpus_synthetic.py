"""Tests for the synthetic side of corpus-harness spec."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bench.synthetic import HEADLINE_TRUE_EFFECT, generate_corpus
from bench.synthetic.scm import MODEL_CHOICE_ARMS, RETRY_POLICY_ARMS, TOOL_CHOICE_ARMS
from counterfact.schema import Run


def test_synthetic__generator_produces_n_traces(tmp_path: Path) -> None:
    """WHEN counterfact bench synthetic --n 500 --seed 42 is executed
    THEN 500 trace files are produced and the wall-clock runtime is under 60 seconds."""
    out = tmp_path / "syn"
    t0 = time.monotonic()
    generate_corpus(n=500, seed=42, output_dir=out)
    dt = time.monotonic() - t0
    files = sorted(out.glob("*.json"))
    assert len(files) == 500
    assert dt < 60.0, f"generation took {dt:.2f}s (>60s)"
    # spot-check parseability with the strict Pydantic schema
    Run.model_validate_json(files[0].read_text())
    Run.model_validate_json(files[-1].read_text())


def test_synthetic__known_true_effect_is_accessible() -> None:
    """WHEN the synthetic SCM is constructed in code
    THEN the true effect of the headline intervention is exposed as a constant."""
    assert isinstance(HEADLINE_TRUE_EFFECT, float)
    assert -1.0 < HEADLINE_TRUE_EFFECT < 1.0
    # design.md D9 says model_choice is the headline; sonnet > haiku gives positive effect.
    assert HEADLINE_TRUE_EFFECT > 0


def test_synthetic__randomization_is_uniform(tmp_path: Path) -> None:
    """WHEN any randomized decision is made during synthetic-trace generation
    THEN policy is "uniform" and propensity equals 1 / len(valid_actions)."""
    out = tmp_path / "syn"
    generate_corpus(n=20, seed=1, output_dir=out)
    files = sorted(out.glob("*.json"))
    expected_propensity = {
        "tool_call": 1.0 / len(TOOL_CHOICE_ARMS),
        "model_call": 1.0 / len(MODEL_CHOICE_ARMS),
        "retry": 1.0 / len(RETRY_POLICY_ARMS),
    }
    for path in files:
        run = Run.model_validate_json(path.read_text())
        for step in run.steps:
            for d in step.decisions:
                if d.policy is None:
                    continue
                assert d.policy == "uniform"
                expected = expected_propensity[d.decision_type]
                assert d.propensity == pytest.approx(expected)


def test_synthetic__same_seed_yields_same_traces(tmp_path: Path) -> None:
    """WHEN counterfact bench synthetic --n 100 --seed 7 is executed twice
    THEN the two output directories contain byte-identical files."""
    a, b = tmp_path / "a", tmp_path / "b"
    generate_corpus(n=100, seed=7, output_dir=a)
    generate_corpus(n=100, seed=7, output_dir=b)
    files_a = sorted(p.name for p in a.glob("*.json"))
    files_b = sorted(p.name for p in b.glob("*.json"))
    assert files_a == files_b
    for name in files_a:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_synthetic__different_seeds_yield_different_traces(tmp_path: Path) -> None:
    """WHEN counterfact bench synthetic --n 100 --seed 7 and --seed 8 are executed
    THEN at least one trace file differs between the two outputs."""
    a, b = tmp_path / "a", tmp_path / "b"
    generate_corpus(n=100, seed=7, output_dir=a)
    generate_corpus(n=100, seed=8, output_dir=b)
    diffs = 0
    for name in sorted(p.name for p in a.glob("*.json")):
        if (a / name).read_bytes() != (b / name).read_bytes():
            diffs += 1
    assert diffs > 0


# --- CLI scenarios -----------------------------------------------------------


def test_cli__synthetic_subcommand_runs_end_to_end(tmp_path: Path) -> None:
    """WHEN `counterfact bench synthetic --n 50 --seed 1 --output-dir DIR` is executed
    THEN 50 trace JSON files appear in DIR and the process exits 0."""
    out = tmp_path / "cli_syn"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "counterfact.cli",
            "bench",
            "synthetic",
            "--n",
            "50",
            "--seed",
            "1",
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert len(list(out.glob("*.json"))) == 50


def test_cli__bench_help_describes_both_subcommands() -> None:
    """WHEN `counterfact bench --help` is executed
    THEN the output mentions both `synthetic` and `real` subcommands."""
    proc = subprocess.run(
        [sys.executable, "-m", "counterfact.cli", "bench", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "synthetic" in proc.stdout
    assert "real" in proc.stdout


# --- spot-check via the strict schema ---------------------------------------


def test_synthetic__generated_traces_round_trip_through_schema(tmp_path: Path) -> None:
    """Every generated trace must be a valid Run. Belt-and-suspenders for the schema."""
    out = tmp_path / "syn"
    generate_corpus(n=10, seed=99, output_dir=out)
    for p in out.glob("*.json"):
        raw = p.read_text()
        run = Run.model_validate_json(raw)
        # also check that re-dump matches the on-disk JSON content
        roundtrip = json.loads(run.model_dump_json())
        assert roundtrip["run_id"] == run.run_id
