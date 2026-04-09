"""Corpus runner: budget-aware, checkpointable, mockable LLM client."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
from bench.real.coding_agent.budget import BudgetExceeded, BudgetTracker
from bench.real.coding_agent.fixtures import FIXTURES, FixtureSpec
from bench.real.coding_agent.llm import LiteLLMClient, LLMClient

APPROVAL_MARKER = Path(".counter") / "approved"


def first_run_gate_check(*, marker_path: Path | None = None) -> bool:
    """Return True iff the harness is approved to make external API calls.

    Per design.md autonomy contract / §12.2: the first real-agent run requires
    explicit human approval. The autonomous loop MUST NOT create the marker.
    """
    return (marker_path or APPROVAL_MARKER).exists()


def print_approval_prompt(stream=sys.stderr) -> None:
    """Render the first-run prompt the user sees before any API call."""
    print(
        "------------------------------------------------------------\n"
        "counter bench real — first-run HUMAN GATE (§12.3)\n"
        "------------------------------------------------------------\n"
        "This will make external LLM API calls and incur USD spend.\n"
        "Before proceeding, the user must:\n"
        "  1. Run a tiny smoke corpus:\n"
        "       counter bench real --n 5 --budget-cap 5\n"
        "  2. Eyeball the resulting traces under bench/real/runs/.\n"
        "  3. If sane, create the approval marker:\n"
        "       mkdir -p .counter && touch .counter/approved\n"
        "Re-run after the marker exists.\n",
        file=stream,
    )


def _checkpoint_dir(output_dir: Path) -> Path:
    d = output_dir / ".checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _completed_indices(output_dir: Path) -> set[int]:
    """Indices of already-written traces, for resume."""
    out: set[int] = set()
    for p in output_dir.glob("real-*.json"):
        # filename: real-<fixture>-<index>.json
        try:
            idx = int(p.stem.rsplit("-", 1)[-1])
            out.add(idx)
        except ValueError:
            continue
    return out


def _fixture_for_index(index: int) -> FixtureSpec:
    return FIXTURES[index % len(FIXTURES)]


def run_real_corpus(
    *,
    n: int,
    budget_cap_usd: float,
    output_dir: Path,
    llm_client_factory: Callable[[], LLMClient] | None = None,
    config: AgentRunConfig | None = None,
    sandbox_root: Path | None = None,
    marker_path: Path | None = None,
    write_to_stream=sys.stdout,
) -> int:
    """Generate `n` real-agent traces. Returns process-style exit code.

    HUMAN GATE: if the approval marker is missing, prints the prompt and
    returns 2 without making any external call.
    """
    if not first_run_gate_check(marker_path=marker_path):
        print_approval_prompt(stream=write_to_stream)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or AgentRunConfig()
    sandbox = sandbox_root or Path(tempfile.mkdtemp(prefix="counter-real-"))
    budget = BudgetTracker(cap_usd=budget_cap_usd)
    llm = (llm_client_factory or LiteLLMClient)()

    done = _completed_indices(output_dir)
    progress_path = _checkpoint_dir(output_dir) / "progress.jsonl"

    written = 0
    try:
        for i in range(n):
            if i in done:
                continue
            fixture = _fixture_for_index(i)
            run = run_one_trace(
                fixture,
                run_index=i,
                llm=llm,
                budget=budget,
                sandbox_root=sandbox,
                config=config,
            )
            out_path = output_dir / f"real-{fixture.fixture_id}-{i:06d}.json"
            out_path.write_text(run.model_dump_json(indent=2))
            written += 1
            with progress_path.open("a") as f:
                f.write(json.dumps({"index": i, "fixture": fixture.fixture_id}) + "\n")
    except BudgetExceeded as exc:
        print(
            f"Budget cap {int(exc.halt_fraction * 100)}% reached: "
            f"${exc.spent:.4f} of ${exc.cap:.2f}\n"
            f"Wrote {written} traces; resume to continue.",
            file=sys.stderr,
        )
        return 3

    print(f"Wrote {written} new traces to {output_dir}", file=write_to_stream)
    return 0
