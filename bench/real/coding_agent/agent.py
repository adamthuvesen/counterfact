"""Coding-agent loop for the real-agent corpus.

The loop is small and inspectable. Per design.md D16/D18, the agent decides
its retry budget *upfront* (after inspect_file, before the first model call)
so that `retry_policy` is logged on every trace — not just the ones whose
first attempt fails. When the retry branch does fire, the second model call's
prompt includes the failing test output, so the retry is informed rather than
a blind coin flip.

Randomized decision types (per the v0 commitment in design.md D5/D9):
* `model_call.model_choice` — which model role (small/large) drafts the patch
* `tool_call.tool_choice`   — which tool comes first (run_tests vs inspect_file)
* `retry.retry_policy`      — attempt budget: no_retry (1 attempt) or retry_once (2)

The loop emits a counterfact-native `Run` per fixture. LLM calls go through the
`LLMClient` protocol so tests can mock them.
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from bench.real.coding_agent.budget import BudgetTracker
from bench.real.coding_agent.fixtures import (
    FixtureSpec,
    build_hidden_eval_workspace,
    is_hidden_fixture,
    run_pytest,
    run_pytest_hidden,
    run_pytest_public,
    snapshot_fixture,
)
from bench.real.coding_agent.llm import LLMClient
from bench.real.coding_agent.randomize import EpsilonGreedy
from counterfact.schema import Decision, Observation, Outcome, Run, Step

DEFAULT_MAX_STEPS = 8
# `search_docs` is intentionally excluded: the agent loop has no distinct
# observation for it (would dispatch identically to `inspect_file`), so logging
# it as a separate arm would be a fictitious decision.
TOOL_ARMS: list[str] = ["inspect_file", "run_tests"]
MODEL_ARMS: list[str] = ["small", "large"]
RETRY_ARMS: list[str] = ["no_retry", "retry_once"]

_FIX_PROMPT = """\
You are a small coding agent. The repository under {root} contains a Python
source file with a bug; the test below pins the expected behavior. Read both,
identify the bug, and reply with the FULL corrected source file inside a
fenced ```python``` block. Do not include any other commentary.

--- SOURCE ({source_relpath}) ---
{source}

--- TEST ({test_relpath}) ---
{test}
"""

_HIDDEN_FIX_PROMPT = """\
You are a small coding agent. The repository under {root} contains a Python
source file with a bug. The requirements live in spec.md (the source of truth)
and a small set of public tests under tests_public/ exercises a subset of
those requirements. Implement the full spec, not just enough to pass the
public tests. Reply with the FULL corrected source file inside a fenced
```python``` block. Do not include any other commentary.

--- SPEC (spec.md) ---
{spec}

--- SOURCE ({source_relpath}) ---
{source}

--- PUBLIC TESTS ({public_tests_relpath}) ---
{public_tests}
"""

_RETRY_PROMPT_SUFFIX = """

Your previous patch was applied but the tests still failed. Here is the test
output (last lines). Identify what your patch missed and reply with the FULL
corrected source file inside a fenced ```python``` block.

--- TEST OUTPUT (tail) ---
{test_output}
"""

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class _ExpectedSymbol:
    name: str
    kind: str


@dataclass(frozen=True)
class _ExtractionResult:
    code: str | None
    status: str
    reason: str | None


def _defines_symbol(source: str, expected: _ExpectedSymbol) -> bool:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    node_type = ast.ClassDef if expected.kind == "class" else ast.FunctionDef
    return any(isinstance(node, node_type) and node.name == expected.name for node in module.body)


def _first_public_symbol(source: str) -> _ExpectedSymbol | None:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            return _ExpectedSymbol(name=node.name, kind=kind)
    return None


def _extract_python_source(
    text: str,
    *,
    expected_symbol: _ExpectedSymbol | None = None,
) -> _ExtractionResult:
    m = _CODE_FENCE.search(text)
    if m:
        code = m.group(1).rstrip() + "\n"
        return _ExtractionResult(code=code, status="extracted", reason=None)
    if expected_symbol is None:
        return _ExtractionResult(
            code=None,
            status="failed",
            reason="no_fenced_python_block_or_expected_symbol",
        )
    source = text.strip()
    if not source:
        return _ExtractionResult(code=None, status="failed", reason="empty_response")
    try:
        ast.parse(source)
    except SyntaxError:
        return _ExtractionResult(code=None, status="failed", reason="raw_response_not_python")
    if not _defines_symbol(source, expected_symbol):
        return _ExtractionResult(
            code=None,
            status="failed",
            reason=f"missing_expected_{expected_symbol.kind}:{expected_symbol.name}",
        )
    return _ExtractionResult(code=source.rstrip() + "\n", status="extracted", reason=None)


def _extract_python_block(text: str, *, expected_function: str | None = None) -> str | None:
    expected_symbol = (
        _ExpectedSymbol(name=expected_function, kind="function")
        if expected_function is not None
        else None
    )
    return _extract_python_source(text, expected_symbol=expected_symbol).code


def _model_observation_content(
    resp_text: str,
    cost_usd: float,
    extraction: _ExtractionResult,
    finish_reason: str | None,
) -> dict[str, object]:
    return {
        "response_chars": len(resp_text),
        "cost_usd": cost_usd,
        "finish_reason": finish_reason,
        "extraction_status": extraction.status,
        "extraction_failure_reason": extraction.reason,
        "extracted_code": extraction.code,
    }


def _charge(budget: BudgetTracker, cost_usd: float, ledger_path: Path | None) -> None:
    """Record `cost_usd` to the resume ledger, then add it to the live budget.

    The ledger entry is written *before* `budget.add` so a `BudgetExceeded`
    raise still leaves an on-disk record of the paid call. Resume includes this
    ledger alongside completed trace spend.
    """
    if ledger_path is not None:
        with ledger_path.open("a") as f:
            f.write(json.dumps({"cost_usd": cost_usd}) + "\n")
    budget.add(cost_usd)


def build_fix_prompt(fixture: FixtureSpec, sandbox: Path) -> str:
    """Build the initial fix prompt for `fixture` against its `sandbox` snapshot.

    Hidden-test fixtures get a prompt that cites spec.md as the requirements
    source and tests_public/ as feedback — the prompt never references
    tests_hidden/ or its filenames. v0 fixtures get the original
    test-in-prompt template.
    """
    src_path = sandbox / "src" / fixture.source_relpath
    src_text = src_path.read_text()
    if is_hidden_fixture(fixture):
        spec_text = (sandbox / "spec.md").read_text()
        assert fixture.public_tests_relpath is not None
        public_path = sandbox / "tests_public" / fixture.public_tests_relpath
        public_text = public_path.read_text()
        return _HIDDEN_FIX_PROMPT.format(
            root=sandbox,
            spec=spec_text,
            source_relpath=fixture.source_relpath,
            source=src_text,
            public_tests_relpath=fixture.public_tests_relpath,
            public_tests=public_text,
        )
    assert fixture.test_relpath is not None  # non-hidden fixtures always set it
    test_path = sandbox / "tests" / fixture.test_relpath
    test_text = test_path.read_text()
    return _FIX_PROMPT.format(
        root=sandbox,
        source_relpath=fixture.source_relpath,
        source=src_text,
        test_relpath=fixture.test_relpath,
        test=test_text,
    )


@dataclass
class AgentRunConfig:
    """Per-decision policy knobs for the agent loop.

    Each randomized decision type has its own (greedy_action, epsilon) pair so
    experimental conditions can be set explicitly per pilot. The actual values
    used for a decision are logged in that decision's `policy_params` so traces
    are self-describing.

    `epsilon` is the legacy fallback used when *_epsilon fields are at their
    default; setting `epsilon` propagates to all three. Per-decision fields
    take precedence when explicitly set to a non-default value.
    """

    epsilon: float = 0.2
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = 0
    # Per-decision policy knobs (D20)
    tool_greedy: str = "inspect_file"
    tool_epsilon: float | None = None
    model_greedy: str = "large"
    model_epsilon: float | None = None
    retry_greedy: str = "retry_once"
    retry_epsilon: float | None = None

    def resolved_tool_epsilon(self) -> float:
        return self.epsilon if self.tool_epsilon is None else self.tool_epsilon

    def resolved_model_epsilon(self) -> float:
        return self.epsilon if self.model_epsilon is None else self.model_epsilon

    def resolved_retry_epsilon(self) -> float:
        return self.epsilon if self.retry_epsilon is None else self.retry_epsilon


def run_one_trace(
    fixture: FixtureSpec,
    *,
    run_index: int,
    llm: LLMClient,
    budget: BudgetTracker,
    sandbox_root: Path,
    config: AgentRunConfig,
    ledger_path: Path | None = None,
) -> Run:
    """Run the agent against one fixture sandbox; return a counterfact-native Run."""
    tool_eps = config.resolved_tool_epsilon()
    model_eps = config.resolved_model_epsilon()
    retry_eps = config.resolved_retry_epsilon()

    rng = random.Random(config.seed ^ run_index)
    eg_tool = EpsilonGreedy(epsilon=tool_eps, seed=rng.randint(0, 2**31 - 1))
    eg_model = EpsilonGreedy(epsilon=model_eps, seed=rng.randint(0, 2**31 - 1))
    eg_retry = EpsilonGreedy(epsilon=retry_eps, seed=rng.randint(0, 2**31 - 1))

    sandbox = snapshot_fixture(fixture, sandbox_root)
    src_path = sandbox / "src" / fixture.source_relpath
    hidden = is_hidden_fixture(fixture)
    expected_symbol = _first_public_symbol(src_path.read_text())

    def _run_tests_in_loop() -> tuple[bool, str]:
        return run_pytest_public(sandbox) if hidden else run_pytest(sandbox)

    steps: list[Step] = []
    step_index = 0

    # ----- Step 0: plan_step (deterministic) -----
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-plan",
                    decision_type="plan_step",
                    chosen_action="begin",
                )
            ],
        )
    )
    step_index += 1

    # ----- Step 1: tool_call (ε-greedy on tool_choice) -----
    tool_action, tool_prop = eg_tool.choose(TOOL_ARMS, greedy=config.tool_greedy)
    src_text = src_path.read_text()
    if hidden:
        assert fixture.public_tests_relpath is not None
        test_chars = len((sandbox / "tests_public" / fixture.public_tests_relpath).read_text())
    else:
        assert fixture.test_relpath is not None
        test_chars = len((sandbox / "tests" / fixture.test_relpath).read_text())
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-inspect",
                    decision_type="tool_call",
                    chosen_action=tool_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": tool_eps, "greedy": config.tool_greedy},
                    valid_actions=list(TOOL_ARMS),
                    propensity=tool_prop,
                    context_features={"first_action": True},
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-inspect",
                    content={
                        "source_chars": len(src_text),
                        "test_chars": test_chars,
                    },
                )
            ],
        )
    )
    step_index += 1

    # ----- Step 2: retry_policy decided UPFRONT (D18) -----
    retry_action, retry_prop = eg_retry.choose(RETRY_ARMS, greedy=config.retry_greedy)
    attempts_remaining = 1 + (1 if retry_action == "retry_once" else 0)
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-retry",
                    decision_type="retry",
                    chosen_action=retry_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": retry_eps, "greedy": config.retry_greedy},
                    valid_actions=list(RETRY_ARMS),
                    propensity=retry_prop,
                    context_features={"step_index": step_index, "upfront": True},
                )
            ],
        )
    )
    step_index += 1

    # ----- Step 3: model_call (ε-greedy on model_choice) -----
    model_action, model_prop = eg_model.choose(MODEL_ARMS, greedy=config.model_greedy)
    prompt = build_fix_prompt(fixture, sandbox)
    resp = llm.call(role=model_action, prompt=prompt)
    _charge(budget, resp.cost_usd, ledger_path)
    extraction = _extract_python_source(resp.text, expected_symbol=expected_symbol)
    patched = extraction.code
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-model-1",
                    decision_type="model_call",
                    chosen_action=model_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": model_eps, "greedy": config.model_greedy},
                    valid_actions=list(MODEL_ARMS),
                    propensity=model_prop,
                    context_features={"prompt_chars": len(prompt), "attempt": 1},
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-model-1",
                    content=_model_observation_content(
                        resp.text,
                        resp.cost_usd,
                        extraction,
                        resp.finish_reason,
                    ),
                )
            ],
        )
    )
    step_index += 1

    if patched is not None:
        src_path.write_text(patched)

    # ----- Step 4: tool_call run_tests (deterministic) -----
    passed, tail = _run_tests_in_loop()
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-runtests-1",
                    decision_type="tool_call",
                    chosen_action="run_tests",
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-runtests-1",
                    content={"passed": passed, "stdout_tail": tail[-500:]},
                )
            ],
        )
    )
    step_index += 1
    attempts_remaining -= 1

    if passed:
        steps.append(
            Step(
                step_index=step_index,
                decisions=[
                    Decision(
                        decision_id=f"d-{run_index:06d}-term",
                        decision_type="termination",
                        chosen_action="success",
                    )
                ],
            )
        )
        return _finalize_run(run_index, fixture, steps, sandbox, sandbox_root, public_pass=True)

    if attempts_remaining <= 0:
        # no_retry was chosen; we ran the one allotted attempt and it failed
        steps.append(
            Step(
                step_index=step_index,
                decisions=[
                    Decision(
                        decision_id=f"d-{run_index:06d}-term",
                        decision_type="termination",
                        chosen_action="give_up",
                    )
                ],
            )
        )
        return _finalize_run(run_index, fixture, steps, sandbox, sandbox_root, public_pass=False)

    # ----- Step 5: model_call retry, with failure context (D18) -----
    retry_prompt = build_fix_prompt(fixture, sandbox) + _RETRY_PROMPT_SUFFIX.format(
        test_output=tail[-1000:]
    )
    resp2 = llm.call(role=model_action, prompt=retry_prompt)
    _charge(budget, resp2.cost_usd, ledger_path)
    extraction2 = _extract_python_source(resp2.text, expected_symbol=expected_symbol)
    patched2 = extraction2.code
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-model-2",
                    decision_type="model_call",
                    chosen_action=model_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": model_eps, "greedy": config.model_greedy},
                    valid_actions=list(MODEL_ARMS),
                    propensity=model_prop,
                    context_features={
                        "prompt_chars": len(retry_prompt),
                        "attempt": 2,
                        "informed_retry": True,
                    },
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-model-2",
                    content=_model_observation_content(
                        resp2.text,
                        resp2.cost_usd,
                        extraction2,
                        resp2.finish_reason,
                    ),
                )
            ],
        )
    )
    step_index += 1

    if patched2 is not None:
        src_path.write_text(patched2)

    # ----- Step 6: tool_call run_tests (second attempt) -----
    passed2, tail2 = _run_tests_in_loop()
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-runtests-2",
                    decision_type="tool_call",
                    chosen_action="run_tests",
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-runtests-2",
                    content={"passed": passed2, "stdout_tail": tail2[-500:]},
                )
            ],
        )
    )
    step_index += 1

    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-term",
                    decision_type="termination",
                    chosen_action="success" if passed2 else "give_up",
                )
            ],
        )
    )
    return _finalize_run(run_index, fixture, steps, sandbox, sandbox_root, public_pass=passed2)


def _finalize_run(
    run_index: int,
    fixture: FixtureSpec,
    steps: list[Step],
    sandbox: Path,
    sandbox_root: Path,
    *,
    public_pass: bool,
) -> Run:
    """Build the final Run, branching on whether `fixture` is hidden.

    For v0 fixtures, `Outcome.value` is the in-loop pytest result and
    `verifier="pytest"`. For hidden fixtures, the in-loop pytest result is
    `public_pass`; the harness then runs `tests_hidden/` once in a separate
    workspace, and `Outcome.value` is the hidden-test result.
    """
    if not is_hidden_fixture(fixture):
        return Run(
            schema_version="0.1.0",
            run_id=f"real-{fixture.fixture_id}-{run_index:06d}",
            steps=steps,
            outcome=Outcome(
                kind="binary",
                value=public_pass,
                verifier="pytest",
                metadata={"fixture_id": fixture.fixture_id},
            ),
        )

    eval_workspace = build_hidden_eval_workspace(fixture, sandbox, sandbox_root / "_hidden_eval")
    hidden_pass, _ = run_pytest_hidden(eval_workspace)
    return Run(
        schema_version="0.1.0",
        run_id=f"real-{fixture.fixture_id}-{run_index:06d}",
        steps=steps,
        outcome=Outcome(
            kind="binary",
            value=hidden_pass,
            verifier="pytest_hidden",
            metadata={
                "fixture_id": fixture.fixture_id,
                "public_pass": public_pass,
                "hidden_pass": hidden_pass,
                "generalization_gap": public_pass and not hidden_pass,
            },
        ),
    )


__all__ = [
    "DEFAULT_MAX_STEPS",
    "MODEL_ARMS",
    "RETRY_ARMS",
    "TOOL_ARMS",
    "AgentRunConfig",
    "run_one_trace",
]
