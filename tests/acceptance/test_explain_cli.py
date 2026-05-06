"""End-to-end acceptance tests for the `counterfact explain` CLI.

Runs `counterfact.cli.main([...])` directly (not subprocess) and asserts on
the resulting HTML file. Covers the synthetic-corpus path, the single_class_refusal
single-class path, and the missing-input error path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfact.cli import main


def _write_synthetic_corpus(target: Path, *, n: int = 16, seed: int = 42) -> Path:
    """Materialize a small synthetic SCM corpus to disk for CLI input."""
    from bench.synthetic import generate_traces

    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for trace in generate_traces(n=n, seed=seed):
        path = target / f"{trace['run_id']}.json"
        path.write_text(json.dumps(trace))
        paths.append(path)
    return paths[0]


def test_explain__synthetic_run_writes_identified_report(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    focal = _write_synthetic_corpus(runs_dir)
    output = tmp_path / "report.html"

    rc = main(
        [
            "explain",
            str(focal),
            "--runs-dir",
            str(runs_dir),
            "--bootstrap",
            "20",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    html = output.read_text()
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert 'class="badge ident-identified"' in html
    # Self-containment: no external network references.
    assert "https://" not in html.replace("https://www.w3.org/2000/svg", "")


def test_explain__single_class_refusal_run_writes_unidentified_report(tmp_path: Path) -> None:
    single_class_refusal = Path("bench/real/single_class_refusal")
    focal = sorted(single_class_refusal.glob("*.json"))[0]
    output = tmp_path / "report.html"

    rc = main(
        [
            "explain",
            str(focal),
            "--runs-dir",
            str(single_class_refusal),
            "--bootstrap",
            "20",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    html = output.read_text()
    assert 'class="badge ident-unidentified"' in html
    assert "broaden_arm_support" in html


def test_explain__missing_run_json_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist.json"
    rc = main(["explain", str(missing)])
    assert rc == 2
    err = capsys.readouterr().err
    assert str(missing) in err
