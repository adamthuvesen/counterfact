from __future__ import annotations

import builtins
from pathlib import Path

from counterfact.cli import main


def test_demo__defaults_to_runs_v2_and_reports_identified_verdict(
    capsys,
) -> None:
    """Default --runs-dir is runs_v2 (mixed-outcome). The engine must fit the
    outcome model and produce an `identified` verdict with a finite
    outcome_delta and a structured next_step."""
    rc = main(["demo", "--bootstrap", "20"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "counterfact demo: naive vs honest" in out
    assert "data: bench/real/runs_v2" in out
    assert "pass_rate_by_arm(model_call)" in out
    assert "intervene(model_call ->" in out
    assert "identifiability: identified" in out
    assert "outcome_delta:" in out
    assert "next_step:" in out


def test_demo__on_runs_v1_reports_honest_refusal(capsys) -> None:
    """The legacy runs_v1 corpus is single-class by construction. With
    --runs-dir explicitly pointed at it, the engine must surface the
    degenerate-corpus refusal rather than fitting a one-class model."""
    rc = main(
        ["demo", "--runs-dir", "bench/real/runs_v1", "--bootstrap", "20"]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "data: bench/real/runs_v1" in out
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
