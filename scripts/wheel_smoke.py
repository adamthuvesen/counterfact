"""Exercise the installed wheel from a clean directory and isolated interpreter."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import polars as pl

import counterfact
from counterfact import AnalysisCorpus, contrast
from counterfact.corpus_io import load_corpus
from counterfact.schema import Decision, Observation, Outcome, Run, Step


def _cli(*args: str, expected: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "counterfact.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in expected:
        raise RuntimeError(f"CLI {args} failed: {result.stderr or result.stdout}")
    return result.stdout + result.stderr


def main() -> None:
    root = Path.cwd()
    assert Path(counterfact.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert importlib.util.find_spec("bench") is None
    corpus_dir = root / "native"
    corpus_dir.mkdir()
    runs = []
    for index in range(80):
        run = Run(
            schema_version="0.1.0",
            run_id=f"wheel-{index}",
            steps=[
                Step(
                    step_index=0,
                    decisions=[
                        Decision(
                            decision_id="model",
                            decision_type="model_call",
                            chosen_action="large" if index % 2 else "small",
                            policy="random",
                            policy_params={},
                            valid_actions=["small", "large"],
                            propensity=0.5,
                            context_features={},
                            metadata={"model_id": "smoke-model"},
                        )
                    ],
                    observations=[Observation(observation_id="cost", content={"cost_usd": 0.0})],
                )
            ],
            outcome=Outcome(kind="binary", value=index % 5 != 0, verifier="smoke"),
        )
        runs.append(run)
        (corpus_dir / f"{run.run_id}.json").write_text(run.model_dump_json())
    corpus = AnalysisCorpus(runs)
    estimate = contrast(
        corpus=corpus,
        decision_type="model_call",
        intervention={"model_choice": "large"},
        baseline="small",
    )
    assert estimate.identifiability.value == "identified"
    assert estimate.outcome_delta is not None
    analyzed = _cli("analyze", "corpus", str(corpus_dir), expected=(0, 1))
    assert "support-readiness" in analyzed and "support_ready:" in analyzed
    report = root / "explain.html"
    _cli("explain", str(corpus_dir / "wheel-0.json"), "--output", str(report))
    assert "<html" in report.read_text().lower()
    exported = root / "runs.parquet"
    _cli("export-runs", str(corpus_dir), "--to", "runrecord-parquet", "--output", str(exported))
    assert pl.read_parquet(exported).height == len(runs)
    _cli(
        "ingest",
        "claude-agent-sdk",
        str(root / "adapter.jsonl"),
        "--output-dir",
        str(root / "ingested"),
    )
    assert len(load_corpus(root / "ingested")) == 2
    unavailable = _cli("bench", "synthetic", "--n", "1", expected=(2,))
    assert "source checkout" in unavailable and "not included in the wheel" in unavailable
    print(f"Installed wheel passed API, ingest, analyze, explain, export: {counterfact.__file__}")


if __name__ == "__main__":
    main()
