"""Corpus runner: budget-aware, checkpointable, mockable LLM client."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
from bench.real.coding_agent.budget import BudgetExceeded, BudgetTracker
from bench.real.coding_agent.fixtures import (
    EASY_FIXTURES,
    FIXTURES,
    HIDDEN_FIXTURES,
    FixtureSpec,
)
from bench.real.coding_agent.llm import ROLE_TO_MODEL, LiteLLMClient, LLMClient

APPROVAL_MARKER = Path(".counter") / "approved"

# Provider credential lookup table. Keys are env-var names that satisfy each
# provider; the first non-empty hit wins. Adding a new provider means adding a
# row here and an entry in `bench.real.coding_agent.llm.ROLE_TO_MODEL`.
_PROVIDER_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
}


def first_run_gate_check(*, marker_path: Path | None = None) -> bool:
    """Return True iff the harness is approved to make external API calls.

    Per design.md autonomy contract / §12.2: the first real-agent run requires
    explicit human approval. The autonomous loop MUST NOT create the marker.
    """
    return (marker_path or APPROVAL_MARKER).exists()


def _provider_for_model(model_name: str) -> str:
    """Infer the provider key from a litellm model identifier."""
    name = model_name.lower()
    if name.startswith(("claude", "anthropic/")):
        return "anthropic"
    if name.startswith(("gpt", "openai/", "o1", "o3", "o4")):
        return "openai"
    # Default: best-effort; unknown providers will surface their own errors.
    return name.split("/", 1)[0] if "/" in name else "unknown"


def check_credentials(
    role_to_model: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> str | None:
    """Return None if every required provider credential is present, else a
    user-facing error message naming the provider and how to fix it."""
    role_to_model = role_to_model or ROLE_TO_MODEL
    env = env if env is not None else dict(os.environ)
    missing: list[str] = []
    for role, model in role_to_model.items():
        provider = _provider_for_model(model)
        candidates = _PROVIDER_CREDENTIALS.get(provider)
        if candidates is None:
            continue  # unknown provider — let it surface its own error at call time
        if not any(env.get(name) for name in candidates):
            missing.append(
                f"  - role={role!r} (model={model!r}) needs one of: "
                + " | ".join(candidates)
            )
    if not missing:
        return None
    lines = [
        "Provider credentials missing — refusing to start the corpus run.",
        "",
        *missing,
        "",
        "Set the variable(s) before running, e.g.:",
        "  export ANTHROPIC_API_KEY='your-key-here'",
        "If you use 1Password:",
        '  export ANTHROPIC_API_KEY="$(op read \'op://<vault>/<item>/credential\')"',
    ]
    return "\n".join(lines)


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


def _fixture_for_index(index: int, fixtures: tuple[FixtureSpec, ...]) -> FixtureSpec:
    return fixtures[index % len(fixtures)]


_FIXTURE_SETS: dict[str, tuple[FixtureSpec, ...]] = {
    "v0": FIXTURES,
    "easy": EASY_FIXTURES,
    "hidden_v1": HIDDEN_FIXTURES,
}


def resolve_fixtures(
    fixture_ids: tuple[str, ...] | None = None,
    fixture_set: str | None = None,
) -> tuple[FixtureSpec, ...]:
    """Resolve which fixtures the runner should iterate over.

    Precedence: explicit `fixture_ids` > named `fixture_set` > the default v0
    `FIXTURES` (preserving existing behavior).
    """
    if fixture_ids:
        registry = {
            fx.fixture_id: fx
            for fx in (*FIXTURES, *EASY_FIXTURES, *HIDDEN_FIXTURES)
        }
        unknown = [fid for fid in fixture_ids if fid not in registry]
        if unknown:
            raise ValueError(f"unknown fixture id(s): {unknown}")
        return tuple(registry[fid] for fid in fixture_ids)
    if fixture_set:
        if fixture_set not in _FIXTURE_SETS:
            raise ValueError(
                f"unknown fixture-set {fixture_set!r}; "
                f"choices: {sorted(_FIXTURE_SETS)}"
            )
        return _FIXTURE_SETS[fixture_set]
    return FIXTURES


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
    fixture_ids: tuple[str, ...] | None = None,
    fixture_set: str | None = None,
) -> int:
    """Generate `n` real-agent traces. Returns process-style exit code.

    HUMAN GATE: if the approval marker is missing, prints the prompt and
    returns 2 without making any external call.
    """
    if not first_run_gate_check(marker_path=marker_path):
        print_approval_prompt(stream=write_to_stream)
        return 2

    # Credential pre-flight. Catches the missing-key case before any agent
    # work starts — otherwise the first model_call dies mid-trace and you
    # wake up to a half-written corpus. Skipped when a custom client factory
    # is passed (test path; caller takes responsibility).
    if llm_client_factory is None:
        cred_error = check_credentials()
        if cred_error is not None:
            print(cred_error, file=sys.stderr)
            return 4

    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or AgentRunConfig()
    sandbox = sandbox_root or Path(tempfile.mkdtemp(prefix="counter-real-"))
    budget = BudgetTracker(cap_usd=budget_cap_usd)
    llm = (llm_client_factory or LiteLLMClient)()

    fixtures = resolve_fixtures(fixture_ids, fixture_set)
    done = _completed_indices(output_dir)
    progress_path = _checkpoint_dir(output_dir) / "progress.jsonl"

    written = 0
    try:
        for i in range(n):
            if i in done:
                continue
            fixture = _fixture_for_index(i, fixtures)
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
