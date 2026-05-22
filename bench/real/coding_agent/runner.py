"""Corpus runner: budget-aware, checkpointable, mockable LLM client."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
from bench.real.coding_agent.budget import BudgetExceeded, BudgetTracker
from bench.real.coding_agent.fixtures import (
    BROAD_CALIBRATION_FIXTURES,
    EASY_FIXTURES,
    FIXTURES,
    FixtureSpec,
    fixtures_by_id,
)
from bench.real.coding_agent.llm import (
    ROLE_TO_MODEL,
    CostUnknownError,
    LiteLLMClient,
    LLMClient,
)

APPROVAL_MARKER = Path(".counterfact") / "approved"
APPROVAL_SCHEMA_VERSION = 1

# Provider credential lookup table. Keys are env-var names that satisfy each
# provider; the first non-empty hit wins. Adding a new provider means adding a
# row here and an entry in `bench.real.coding_agent.llm.ROLE_TO_MODEL`.
_PROVIDER_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
}


class BudgetLedgerError(RuntimeError):
    """Raised when resume spend cannot be read safely."""


def first_run_gate_check(*, marker_path: Path | None = None) -> bool:
    """Return True iff an approval receipt exists.

    The first real-agent run requires explicit human approval — the operator
    must create `.counterfact/approved` after eyeballing a smoke corpus.
    Autonomous loops MUST NOT create the marker.
    """
    return (marker_path or APPROVAL_MARKER).exists()


def approval_receipt_template(
    *,
    n: int,
    budget_cap_usd: float,
    output_dir: Path,
    fixtures: tuple[FixtureSpec, ...],
    config: AgentRunConfig,
    role_to_model: dict[str, str],
) -> dict[str, object]:
    """Return the approval receipt shape expected for this real-agent command."""
    identity = _run_identity(fixtures=fixtures, config=config, role_to_model=role_to_model)
    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approved_at": "<ISO-8601 timestamp>",
        "max_traces": n,
        "budget_cap_usd": budget_cap_usd,
        "output_dir": str(output_dir),
        **identity,
    }


def _approval_error(marker_path: Path, expected: dict[str, object]) -> str | None:
    if not marker_path.exists():
        return f"approval receipt not found at {marker_path}"
    try:
        receipt = json.loads(marker_path.read_text())
    except json.JSONDecodeError as exc:
        return f"approval receipt is not valid JSON: {exc}"
    if not isinstance(receipt, dict):
        return "approval receipt must be a JSON object"
    if receipt.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        return (
            "approval receipt schema_version mismatch; expected "
            f"{APPROVAL_SCHEMA_VERSION}"
        )
    approved_at = receipt.get("approved_at")
    if not isinstance(approved_at, str) or not approved_at.strip():
        return "approval receipt must include a non-empty approved_at timestamp"
    if approved_at == "<ISO-8601 timestamp>":
        return "approval receipt approved_at still contains the template placeholder"

    max_traces = receipt.get("max_traces")
    if not isinstance(max_traces, int) or max_traces < int(expected["max_traces"]):
        return (
            "approval receipt max_traces is lower than this invocation "
            f"({max_traces!r} < {expected['max_traces']!r})"
        )

    budget = receipt.get("budget_cap_usd")
    if not isinstance(budget, int | float) or not math.isclose(
        float(budget), float(expected["budget_cap_usd"])
    ):
        return (
            "approval receipt budget_cap_usd does not match this invocation "
            f"({budget!r} != {expected['budget_cap_usd']!r})"
        )

    for key in ("output_dir", "fixtures", "config", "role_to_model"):
        if receipt.get(key) != expected[key]:
            return (
                f"approval receipt {key} does not match this invocation; "
                "create a new receipt for the exact command you intend to run"
            )
    return None


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
                f"  - role={role!r} (model={model!r}) needs one of: " + " | ".join(candidates)
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
        "  export ANTHROPIC_API_KEY=\"$(op read 'op://<vault>/<item>/credential')\"",
    ]
    return "\n".join(lines)


def print_approval_prompt(
    *,
    marker_path: Path | None = None,
    expected_receipt: dict[str, object] | None = None,
    reason: str | None = None,
    stream=sys.stderr,
) -> None:
    """Render the first-run prompt the user sees before any API call."""
    marker = marker_path or APPROVAL_MARKER
    receipt = (
        json.dumps(expected_receipt, indent=2, sort_keys=True)
        if expected_receipt is not None
        else "{}"
    )
    reason_line = f"Refusing to start: {reason}\n\n" if reason else ""
    print(
        "------------------------------------------------------------\n"
        "counterfact bench real — first-run HUMAN GATE\n"
        "------------------------------------------------------------\n"
        f"{reason_line}"
        "This will make external LLM API calls and incur USD spend.\n"
        "Before proceeding, write an approval receipt for the exact command/config "
        "you intend to run. Approve a tiny smoke command first, inspect those "
        "traces, then write a separate receipt before scaling up.\n"
        f"Create {marker} with this JSON, replacing approved_at with the current "
        "UTC timestamp:\n"
        f"{receipt}\n",
        file=stream,
    )


def _checkpoint_dir(output_dir: Path) -> Path:
    d = output_dir / ".checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_identity(
    *,
    fixtures: tuple[FixtureSpec, ...],
    config: AgentRunConfig,
    role_to_model: dict[str, str],
) -> dict[str, object]:
    return {
        "fixtures": [fx.fixture_id for fx in fixtures],
        "config": {
            "seed": config.seed,
            "max_steps": config.max_steps,
            "tool_greedy": config.tool_greedy,
            "tool_epsilon": config.resolved_tool_epsilon(),
            "model_greedy": config.model_greedy,
            "model_epsilon": config.resolved_model_epsilon(),
            "retry_greedy": config.retry_greedy,
            "retry_epsilon": config.resolved_retry_epsilon(),
        },
        "role_to_model": dict(sorted(role_to_model.items())),
    }


def _identity_error(
    checkpoint_dir: Path,
    identity: dict[str, object],
    *,
    has_completed_traces: bool,
) -> str | None:
    identity_path = checkpoint_dir / "identity.json"
    if not identity_path.exists():
        if has_completed_traces:
            return (
                "Existing real-agent traces do not have resume identity metadata. "
                "Use a new output directory, or regenerate the corpus with this "
                "version before resuming."
            )
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        return None
    existing = json.loads(identity_path.read_text())
    if existing == identity:
        return None
    return (
        "Existing real-agent corpus was created with a different fixture or "
        "randomization configuration. Use a new output directory, or rerun with "
        f"the original identity recorded in {identity_path}."
    )


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


def _completed_spend(output_dir: Path) -> float:
    """Sum cost observations from already-written traces."""
    spent = 0.0
    for path in output_dir.glob("real-*.json"):
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BudgetLedgerError(
                f"cannot read completed trace spend from {path}: {exc}"
            ) from exc
        if not isinstance(run, dict):
            raise BudgetLedgerError(
                f"cannot read completed trace spend from {path}: expected JSON object"
            )
        for step in run.get("steps", []):
            if not isinstance(step, dict):
                raise BudgetLedgerError(
                    f"cannot read completed trace spend from {path}: step is not an object"
                )
            for obs in step.get("observations", []) or []:
                if not isinstance(obs, dict):
                    raise BudgetLedgerError(
                        f"cannot read completed trace spend from {path}: "
                        "observation is not an object"
                    )
                content = obs.get("content", {})
                if not isinstance(content, dict):
                    raise BudgetLedgerError(
                        f"cannot read completed trace spend from {path}: "
                        "observation content is not an object"
                    )
                cost = content.get("cost_usd")
                if cost is not None:
                    spent += _spend_value(cost, source=f"{path}: observation cost_usd")
    return spent


def _ledger_spend(ledger_path: Path) -> float:
    """Sum unflushed spend from the budget ledger.

    The agent appends each priced LLM response to this file *before* calling
    `budget.add`. After a trace JSON is successfully written, the runner
    truncates the ledger — so any entries that survive represent costs that
    never made it into a committed trace (typically because `BudgetExceeded`
    fired). Resume initializes spend from `_completed_spend() + _ledger_spend()`
    so the budget gate cannot be re-crossed for free.
    """
    spent = 0.0
    try:
        with ledger_path.open() as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    spent += float(json.loads(line)["cost_usd"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise BudgetLedgerError(
                        f"{ledger_path}:{line_number}: cannot read cost_usd; "
                        "refusing to resume because prior spend cannot be "
                        "accounted safely"
                    ) from exc
    except FileNotFoundError:
        return 0.0
    return spent


def _spend_value(value: object, *, source: str) -> float:
    try:
        cost = float(value)
    except (TypeError, ValueError) as exc:
        raise BudgetLedgerError(f"{source} is not a number: {value!r}") from exc
    if not math.isfinite(cost):
        raise BudgetLedgerError(f"{source} is not finite: {value!r}")
    if cost < 0:
        raise BudgetLedgerError(f"{source} is negative: {value!r}")
    return cost


def _fixture_for_index(index: int, fixtures: tuple[FixtureSpec, ...]) -> FixtureSpec:
    return fixtures[index % len(fixtures)]


_FIXTURE_SETS: dict[str, tuple[FixtureSpec, ...]] = {
    "v0": FIXTURES,
    "easy": EASY_FIXTURES,
    "hidden_v1": fixtures_by_id("csv_dedupe"),
    "hard_hidden_v1": fixtures_by_id("date_window"),
    "broad_calibration": BROAD_CALIBRATION_FIXTURES,
    "very_hard_hidden_v1": fixtures_by_id("unicode_normalize"),
    "stateful_calibration": fixtures_by_id("streaming_watermark_dedupe"),
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
        try:
            return fixtures_by_id(*fixture_ids)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    if fixture_set:
        if fixture_set not in _FIXTURE_SETS:
            raise ValueError(
                f"unknown fixture-set {fixture_set!r}; choices: {sorted(_FIXTURE_SETS)}"
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

    HUMAN GATE: if the approval receipt is missing, invalid, or for a different
    command/config, prints the prompt and returns 2 without making any external
    call.
    """
    config = config or AgentRunConfig()
    fixtures = resolve_fixtures(fixture_ids, fixture_set)
    approval_path = marker_path or APPROVAL_MARKER
    expected_receipt = approval_receipt_template(
        n=n,
        budget_cap_usd=budget_cap_usd,
        output_dir=output_dir,
        fixtures=fixtures,
        config=config,
        role_to_model=ROLE_TO_MODEL,
    )
    approval_error = _approval_error(approval_path, expected_receipt)
    if approval_error is not None:
        print_approval_prompt(
            marker_path=approval_path,
            expected_receipt=expected_receipt,
            reason=approval_error,
            stream=write_to_stream,
        )
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
    checkpoint_dir = _checkpoint_dir(output_dir)
    done = _completed_indices(output_dir)
    identity = _run_identity(fixtures=fixtures, config=config, role_to_model=ROLE_TO_MODEL)
    identity_error = _identity_error(
        checkpoint_dir,
        identity,
        has_completed_traces=bool(done),
    )
    if identity_error is not None:
        print(identity_error, file=sys.stderr)
        return 6

    ledger_path = checkpoint_dir / "budget_ledger.jsonl"
    try:
        spent_usd = _completed_spend(output_dir) + _ledger_spend(ledger_path)
    except BudgetLedgerError as exc:
        print(f"Budget accounting failed: {exc}", file=sys.stderr)
        return 7
    budget = BudgetTracker(cap_usd=budget_cap_usd, spent_usd=spent_usd)
    if budget.spent_usd >= budget.halt_threshold:
        print(
            f"Budget cap {int(budget.halt_fraction * 100)}% reached: "
            f"${budget.spent_usd:.4f} of ${budget.cap_usd:.2f}\n"
            "Wrote 0 traces; resume to continue.",
            file=sys.stderr,
        )
        return 3
    llm = (llm_client_factory or LiteLLMClient)()

    progress_path = checkpoint_dir / "progress.jsonl"

    written = 0
    with contextlib.ExitStack() as stack:
        if sandbox_root is None:
            sandbox = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="counterfact-real-"))
            )
        else:
            sandbox = sandbox_root
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
                    ledger_path=ledger_path,
                )
                out_path = output_dir / f"real-{fixture.fixture_id}-{i:06d}.json"
                out_path.write_text(run.model_dump_json(indent=2))
                # Trace owns its costs now; drop the unflushed-spend ledger so
                # the next iteration starts with a clean slate.
                ledger_path.unlink(missing_ok=True)
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
        except CostUnknownError as exc:
            print(f"Cost accounting failed: {exc}", file=sys.stderr)
            return 5

    print(f"Wrote {written} new traces to {output_dir}", file=write_to_stream)
    return 0
