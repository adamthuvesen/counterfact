"""Failure checks for generated artifacts, import boundaries, and coverage."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import coverage
import pytest

from scripts import (
    build_benchmark_docs,
    build_demo_notebook,
    build_figures,
    check_container,
    check_wheel,
)
from scripts._generated import write_or_check
from tests.fixtures.corpus_loaders import load_json_corpus

ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE_", "_COV_"))
    }


def test_generated_text_check_rejects_missing_and_stale_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "generated.txt"
    assert not write_or_check(path, "expected", check=True)
    assert not path.exists()
    assert write_or_check(path, "expected", check=False)
    assert write_or_check(path, "expected", check=True)
    path.write_text("stale")
    assert not write_or_check(path, "expected", check=True)
    assert path.read_text() == "stale"


def test_notebook_generator_check_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "demo.ipynb"
    monkeypatch.setattr(build_demo_notebook, "NB_PATH", path)
    assert build_demo_notebook.main([]) == 0
    assert build_demo_notebook.main(["--check"]) == 0
    path.write_text("{}")
    assert build_demo_notebook.main(["--check"]) == 1
    assert path.read_text() == "{}"


def test_benchmark_generators_derive_outputs_and_reject_stale_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "benchmark-results.json"
    shutil.copy(ROOT / "docs" / "benchmark-results.json", source)
    monkeypatch.setattr(build_benchmark_docs, "ROOT", tmp_path)
    monkeypatch.setattr(build_figures, "RESULTS", source)
    monkeypatch.setattr(build_figures, "OUT", docs / "figures")
    for module in (build_benchmark_docs, build_figures):
        assert module.main([]) == 0
        assert module.main(["--check"]) == 0
    (docs / "benchmark-results.md").write_text("stale")
    (docs / "figures" / "policy-bias.svg").write_text("stale")
    assert build_benchmark_docs.main(["--check"]) == 1
    assert build_figures.main(["--check"]) == 1
    source.write_text("{}")
    assert build_benchmark_docs.main([]) == 1
    assert build_figures.main([]) == 1


@pytest.mark.parametrize("exists", [False, True])
def test_required_corpus_missing_or_empty_fails(tmp_path: Path, exists: bool) -> None:
    path = tmp_path / "required-corpus"
    if exists:
        path.mkdir()
    with pytest.raises(FileNotFoundError, match="required committed corpus"):
        load_json_corpus(path)


def test_import_contract_rejects_scientific_dependency_on_cli(tmp_path: Path) -> None:
    source = tmp_path / "src"
    shutil.copytree(
        ROOT / "src" / "counterfact",
        source / "counterfact",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (source / "bench").mkdir()
    (source / "bench" / "__init__.py").write_text("")
    (source / "counterfact" / "analysis" / "boundary_probe.py").write_text(
        "import counterfact.cli\n"
    )
    env = _subprocess_env()
    env["PYTHONPATH"] = str(source)
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "lint-imports"),
            "--no-cache",
            "--config",
            str(ROOT / "pyproject.toml"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Scientific code has no presentation or integration dependencies BROKEN" in result.stdout
    assert "boundary_probe" in result.stdout


@pytest.mark.slow
def test_subprocess_branch_coverage_excludes_fixture_inputs(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.py"
    fixture.write_text("value = 1\n")
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys\nimport bench, scripts\n"
        "subprocess.run([sys.executable, '-c', "
        "'from counterfact.stats import wilson_ci; wilson_ci(1, 2)'], check=True)\n"
        f"subprocess.run([sys.executable, {str(fixture)!r}], check=True)\n"
    )
    env = _subprocess_env()
    env["COVERAGE_FILE"] = str(tmp_path / ".coverage")
    command = [sys.executable, "-m", "coverage"]
    for args in (
        ["run", f"--rcfile={ROOT / 'pyproject.toml'}", str(parent)],
        ["combine", f"--rcfile={ROOT / 'pyproject.toml'}"],
    ):
        result = subprocess.run(
            [*command, *args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stdout + result.stderr
    data = coverage.CoverageData(basename=str(tmp_path / ".coverage"))
    data.read()
    assert data.has_arcs()
    stats = next(path for path in data.measured_files() if path.endswith("counterfact/stats.py"))
    assert data.lines(stats)
    assert any(path.endswith("bench/real/container/verify.py") for path in data.measured_files())
    assert not any("fixture.py" in path or "parent.py" in path for path in data.measured_files())


def test_tool_prerequisite_failures_are_explicit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("required runtime unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(sys, "argv", ["check_container.py", "--build"])
    assert check_container.main() == 1
    assert "required runtime unavailable" in capsys.readouterr().err
    assert check_wheel.main() == 1
    assert "wheel verification failed" in capsys.readouterr().err


def test_pilot_cli_help_is_not_treated_as_a_corpus(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bench.real.analyze_pilot import main

    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["missing", "empty", "malformed", "invalid-metadata"])
def test_pilot_cli_rejects_invalid_input_without_notes_or_partial_output(
    state: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from bench.real.analyze_pilot import main
    from counterfact.schema import Outcome, Run

    corpus = tmp_path / "corpus"
    if state != "missing":
        corpus.mkdir()
    if state == "malformed":
        (corpus / "run.json").write_text("not JSON")
    if state == "invalid-metadata":
        run = Run(
            schema_version="0.1.0",
            run_id="missing-pilot-fields",
            steps=[],
            outcome=Outcome(kind="binary", value=False, verifier="test"),
        )
        (corpus / "run.json").write_text(run.model_dump_json())
    assert main([str(corpus)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "counterfact analyze-pilot:" in captured.err
    assert "Traceback" not in captured.err
    assert not (corpus / "PILOT_NOTES.md").exists()
