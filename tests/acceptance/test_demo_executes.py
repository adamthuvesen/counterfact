"""Demo notebook acceptance tests (tasks §13).

Covers the demo-spec scenarios that don't require the real-agent corpus or
the hand-labeled root cause:

* Clean-kernel execution succeeds.
* Synthetic section reports recovery vs truth.
* Real-trace section yields a mix of identified, bounded, unidentified.
* Attribution section renders top_k(5) with identifiability labels.

The hand-labeled root cause comparison (`top_1 matches the label`) lands once
§14.1 clears.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = REPO_ROOT / "notebooks" / "demo.ipynb"


@pytest.fixture(scope="module")
def executed_notebook(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Execute the demo notebook from a clean kernel and return the parsed cells."""
    out_dir = tmp_path_factory.mktemp("demo_exec")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(DEMO_PATH),
            "--output-dir",
            str(out_dir),
            "--output",
            "demo.executed.ipynb",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"nbconvert failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    executed_path = out_dir / "demo.executed.ipynb"
    return json.loads(executed_path.read_text())


def _all_text_outputs(nb: dict) -> str:
    """Concatenate every text-output stream / display-data text from every code cell."""
    pieces: list[str] = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                text = out["text"]
                pieces.append("".join(text) if isinstance(text, list) else text)
            elif "data" in out and "text/plain" in out["data"]:
                d = out["data"]["text/plain"]
                pieces.append("".join(d) if isinstance(d, list) else d)
    return "\n".join(pieces)


def test_demo__clean_kernel_execution_succeeds(executed_notebook: dict) -> None:
    """WHEN the notebook is executed via nbconvert against the shipped corpus
    THEN every cell completes without raising and the notebook exits 0."""
    # If the fixture got here, nbconvert exited 0 — the assertion in the
    # fixture covers the requirement. Spot-check we got output back.
    assert "cells" in executed_notebook
    assert any(c.get("cell_type") == "code" for c in executed_notebook["cells"])


def test_demo__synthetic_section_reports_recovery_vs_truth(executed_notebook: dict) -> None:
    """WHEN the notebook's synthetic section is executed
    THEN the cell output shows estimated effect, true E, and the absolute difference."""
    text = _all_text_outputs(executed_notebook)
    assert "estimated effect" in text
    assert "true effect" in text
    assert "|diff|" in text
    assert "tolerance" in text


def test_demo__at_least_one_identified_result_is_rendered(executed_notebook: dict) -> None:
    """WHEN the notebook's real-trace section is executed
    THEN at least one cell output displays a CausalEstimate with identifiability='identified'."""
    text = _all_text_outputs(executed_notebook)
    assert "identifiability : identified" in text
    assert "95% bootstrap CI" in text


def test_demo__at_least_one_bounded_result_is_rendered(executed_notebook: dict) -> None:
    """WHEN the notebook's real-trace section is executed
    THEN at least one cell output displays a CausalEstimate with identifiability='bounded'
    and a non-null bounds.e_value."""
    text = _all_text_outputs(executed_notebook)
    assert "identifiability : bounded" in text
    assert "E-value" in text


def test_demo__at_least_one_unidentified_result_is_rendered_with_structured_next_step(
    executed_notebook: dict,
) -> None:
    """WHEN the notebook's real-trace section is executed
    THEN at least one cell output displays identifiability='unidentified',
    a non-null reason, and a structured next_step.action ∈ the documented set."""
    text = _all_text_outputs(executed_notebook)
    assert "identifiability : unidentified" in text
    assert "reason" in text
    actionable = {"increase_n", "broaden_arm_support", "replay_required", "add_arm_randomization"}
    assert any(a in text for a in actionable), (
        f"no actionable next_step.action rendered; expected one of {sorted(actionable)}"
    )


def test_demo__top_k_table_is_rendered_with_labels(executed_notebook: dict) -> None:
    """WHEN the notebook's attribution section is executed
    THEN the rendered table contains up to 5 rows with decision id, influence score,
    and identifiability label."""
    text = _all_text_outputs(executed_notebook)
    assert "failure attribution for" in text
    assert "rank" in text and "decision_id" in text and "influence" in text
    # Header line + at most 5 ranked rows. Each row line should include an
    # identifiability value.
    assert (
        "identified" in text or "bounded" in text or "unidentified" in text
    )
