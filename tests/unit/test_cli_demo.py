from __future__ import annotations

import builtins
from pathlib import Path

from counter.cli import main


def test_demo__uses_local_real_corpus_and_reports_degenerate_verdict(
    capsys,
) -> None:
    rc = main(["demo", "--bootstrap", "20"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "counter demo: naive vs honest" in out
    assert "data: bench/real/runs_v1" in out
    assert "pass_rate_by_arm(model_call)" in out
    assert "intervene(model_call ->" in out
    assert "identifiability: unidentified" in out
    assert "next_step: broaden_arm_support" in out


def test_demo__falls_back_to_synthetic_without_real_harness_import(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("bench.real.coding_agent.runner"):
            raise AssertionError("counter demo must not import the real-agent runner")
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
