"""Acceptance tests for hidden real-agent fixture behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

# --- Phase A: csv_dedupe fixture content + registry ------------------------


def test_csv_dedupe_is_registered() -> None:
    """Req: csv_dedupe is the first hidden-test fixture
    WHEN the hidden-fixture registry is queried
    THEN csv_dedupe is among the registered fixtures."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    ids = [fx.fixture_id for fx in HIDDEN_FIXTURES]
    assert "csv_dedupe" in ids


def test_date_window_is_registered() -> None:
    """Req: date_window is registered as a hard hidden-test fixture
    WHEN the hidden-fixture registry is queried
    THEN date_window is among the registered fixtures."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    ids = [fx.fixture_id for fx in HIDDEN_FIXTURES]
    assert "date_window" in ids


def test_hidden_fixture_root_has_four_required_entries() -> None:
    """Req: Hidden-test fixtures use a public/hidden split layout
    WHEN a hidden-test fixture is registered
    THEN its root directory contains src/, tests_public/, tests_hidden/, spec.md
    AND the harness exposes public/hidden tests path as distinct attributes."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    for fx in HIDDEN_FIXTURES:
        assert (fx.root / "src").is_dir(), f"{fx.fixture_id}: src/ missing"
        assert (fx.root / "tests_public").is_dir(), (
            f"{fx.fixture_id}: tests_public/ missing"
        )
        assert (fx.root / "tests_hidden").is_dir(), (
            f"{fx.fixture_id}: tests_hidden/ missing"
        )
        assert (fx.root / "spec.md").is_file(), f"{fx.fixture_id}: spec.md missing"
        # Distinct path attributes
        assert fx.public_tests_relpath is not None
        assert fx.hidden_tests_relpath is not None
        assert fx.public_tests_relpath != fx.hidden_tests_relpath


def test_csv_dedupe_spec_md_exists_and_is_nonempty() -> None:
    """Req: Hidden tests are derivable from spec.md
    WHEN a hidden-test fixture is registered
    THEN spec.md exists at the fixture root and is non-empty."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    spec = csv_dedupe.root / "spec.md"
    assert spec.is_file()
    text = spec.read_text()
    assert len(text.strip()) > 0
    # Sanity: spec.md must mention the four normalization rules so reviewers
    # can predict hidden-test categories.
    lower = text.lower()
    for keyword in ("whitespace", "case", "nfc", "bom"):
        assert keyword in lower, f"spec.md must mention {keyword!r}"


def test_hidden_registry_is_distinct_from_v0() -> None:
    """Req: Hidden-test fixtures use a public/hidden split layout
    WHEN the harness lists registered fixture sets
    THEN v0 and hidden-test sets are addressable as distinct registries."""
    from bench.real.coding_agent.fixtures import (
        EASY_FIXTURES,
        FIXTURES,
        HIDDEN_FIXTURES,
    )

    v0_ids = {fx.fixture_id for fx in EASY_FIXTURES} | {fx.fixture_id for fx in FIXTURES}
    hidden_ids = {fx.fixture_id for fx in HIDDEN_FIXTURES}
    # Disjoint by id — no overlap between v0 and hidden registries.
    assert v0_ids.isdisjoint(hidden_ids), (
        f"hidden registry must not reuse v0 fixture ids: {v0_ids & hidden_ids}"
    )
    # And hidden fixtures must declare both relpath fields, while v0 may not.
    for fx in HIDDEN_FIXTURES:
        assert fx.public_tests_relpath is not None
        assert fx.hidden_tests_relpath is not None


def test_csv_dedupe_public_covers_only_exact_string_dedup() -> None:
    """Req: csv_dedupe is the first hidden-test fixture
    WHEN pytest tests_public/ runs on csv_dedupe's reference (correct) source
    THEN all public tests pass and the public test set exercises only exact-string dedup."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    public_dir = csv_dedupe.root / "tests_public"
    test_files = list(public_dir.glob("test_*.py"))
    assert len(test_files) >= 1
    text = "\n".join(p.read_text() for p in test_files)
    # Public tests must NOT reference normalization concerns by name.
    forbidden = [
        "whitespace",
        "casefold",
        "lower(",
        ".upper(",
        "nfc",
        "unicodedata",
        "bom",
        "\\ufeff",
    ]
    for kw in forbidden:
        assert kw.lower() not in text.lower(), (
            f"public tests must not test normalization rule: found {kw!r}"
        )


def test_csv_dedupe_hidden_covers_four_normalization_rules() -> None:
    """Req: csv_dedupe is the first hidden-test fixture
    WHEN pytest tests_hidden/ runs on csv_dedupe's reference source
    THEN all hidden tests pass and the hidden set covers each of four normalization rules."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    hidden_dir = csv_dedupe.root / "tests_hidden"
    test_files = list(hidden_dir.glob("test_*.py"))
    assert len(test_files) >= 1
    text = "\n".join(p.read_text() for p in test_files).lower()
    # At least one test per rule — we look for the rule keyword in test names.
    for keyword in ("whitespace", "case", "nfc", "bom"):
        assert keyword in text, (
            f"hidden tests must include at least one test for {keyword!r} rule"
        )


def test_csv_dedupe_reference_implementation_passes_both_sets(tmp_path: Path) -> None:
    """End-to-end sanity for fixture content: when the buggy src/ is replaced with
    the reference implementation, BOTH tests_public/ and tests_hidden/ pass."""
    import shutil
    import subprocess
    import sys

    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    # Copy whole fixture into tmp
    workspace = tmp_path / "csv_dedupe"
    shutil.copytree(csv_dedupe.root, workspace)

    # Reference implementation lives next to the buggy one for review purposes.
    reference = csv_dedupe.root / "src" / "_dedupe_reference.py"
    assert reference.is_file(), (
        "csv_dedupe must ship a reference implementation at src/_dedupe_reference.py "
        "so the fixture's hidden tests are demonstrably satisfiable"
    )
    # Replace buggy module with reference.
    target = workspace / "src" / csv_dedupe.source_relpath
    target.write_text(reference.read_text())

    # Public tests pass.
    proc_pub = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_public/", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_pub.returncode == 0, (
        f"reference fails public tests:\n{proc_pub.stdout}\n{proc_pub.stderr}"
    )
    # Hidden tests pass.
    proc_hid = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_hidden/", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_hid.returncode == 0, (
        f"reference fails hidden tests:\n{proc_hid.stdout}\n{proc_hid.stderr}"
    )


def test_csv_dedupe_buggy_src_fails_hidden_passes_public(tmp_path: Path) -> None:
    """The shipped (buggy) src/ must pass tests_public/ but fail tests_hidden/.
    This is the corpus's invariant — without it, no generalization gap is possible."""
    import shutil
    import subprocess
    import sys

    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    workspace = tmp_path / "csv_dedupe"
    shutil.copytree(csv_dedupe.root, workspace)

    proc_pub = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_public/", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_pub.returncode == 0, (
        f"buggy src must satisfy public tests:\n{proc_pub.stdout}"
    )
    proc_hid = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_hidden/", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_hid.returncode != 0, (
        "buggy src must fail at least one hidden test (otherwise no generalization gap is possible)"
    )


def _date_window_known_good_source() -> str:
    return '''"""Known-good date_window implementation for fixture self-tests."""

from __future__ import annotations

from datetime import date
import re

_DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$")


def _parse(value: str) -> date:
    if not _DATE_RE.match(value):
        raise ValueError(f"invalid date: {value!r}")
    return date.fromisoformat(value)


def in_any_window(target_date: str, windows: list[tuple[str, str]]) -> bool:
    target = _parse(target_date)
    for start_raw, end_raw in windows:
        start = _parse(start_raw)
        end = _parse(end_raw)
        if start > end:
            raise ValueError("window start must be <= end")
        if start <= target <= end:
            return True
    return False
'''


def test_date_window_spec_md_names_hidden_categories() -> None:
    """Req: date_window is registered as a hard hidden-test fixture
    WHEN the fixture's spec.md is inspected
    THEN it names the hidden-test categories."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    date_window = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "date_window")
    text = (date_window.root / "spec.md").read_text().lower()
    for keyword in (
        "yyyy-mm-dd",
        "inclusive",
        "any order",
        "start > end",
        "leap",
        "malformed",
    ):
        assert keyword in text, f"date_window spec.md must mention {keyword!r}"


def test_date_window_buggy_src_fails_hidden_passes_public(tmp_path: Path) -> None:
    """Req: date_window is registered as a hard hidden-test fixture
    WHEN the pristine date_window source is tested
    THEN public tests pass and hidden tests fail."""
    import shutil
    import subprocess
    import sys

    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    date_window = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "date_window")
    workspace = tmp_path / "date_window"
    shutil.copytree(date_window.root, workspace)

    proc_pub = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_public/", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_pub.returncode == 0, (
        f"buggy src must satisfy public tests:\n{proc_pub.stdout}\n{proc_pub.stderr}"
    )
    proc_hid = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_hidden/", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_hid.returncode != 0, "buggy src must fail hidden tests"


def test_date_window_known_good_source_passes_hidden(tmp_path: Path) -> None:
    """Req: date_window is registered as a hard hidden-test fixture
    WHEN source is replaced with a known-good implementation
    THEN hidden tests pass."""
    import shutil
    import subprocess
    import sys

    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    date_window = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "date_window")
    workspace = tmp_path / "date_window"
    shutil.copytree(date_window.root, workspace)
    target = workspace / "src" / date_window.source_relpath
    target.write_text(_date_window_known_good_source())

    proc_hid = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_hidden/", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_hid.returncode == 0, (
        f"known-good source fails hidden tests:\n{proc_hid.stdout}\n{proc_hid.stderr}"
    )


# --- Phase B: sandbox + public-pytest isolation ----------------------------


def test_sandbox_snapshot_omits_tests_hidden(tmp_path: Path) -> None:
    """Req: Hidden tests are not present in the agent sandbox
    WHEN snapshot_fixture is called on a hidden-test fixture
    THEN the sandbox contains src/, tests_public/, spec.md and does NOT
         contain tests_hidden/ at any depth."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES, snapshot_fixture

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    sandbox = snapshot_fixture(csv_dedupe, tmp_path)
    assert (sandbox / "src").is_dir()
    assert (sandbox / "tests_public").is_dir()
    assert (sandbox / "spec.md").is_file()
    # tests_hidden/ must be absent at any depth
    hidden_dirs = [p for p in sandbox.rglob("tests_hidden") if p.is_dir()]
    hidden_files = [p for p in sandbox.rglob("*") if p.name.startswith("test_dedupe_hidden")]
    assert hidden_dirs == [], f"tests_hidden/ leaked into sandbox: {hidden_dirs}"
    assert hidden_files == [], f"hidden test file leaked into sandbox: {hidden_files}"


def test_sandbox_stays_clean_through_run_pytest_public(tmp_path: Path) -> None:
    """Req: Hidden tests are not present in the agent sandbox
    WHEN the agent runs run_pytest_public against the sandbox
    THEN tests_hidden/ remains absent from the sandbox after execution."""
    from bench.real.coding_agent.fixtures import (
        HIDDEN_FIXTURES,
        run_pytest_public,
        snapshot_fixture,
    )

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    sandbox = snapshot_fixture(csv_dedupe, tmp_path)
    # Run public tests — must not introduce tests_hidden/ as a side effect.
    passed, _ = run_pytest_public(sandbox)
    assert passed is True  # buggy-but-public-passing is the fixture's invariant
    hidden_dirs = [p for p in sandbox.rglob("tests_hidden") if p.is_dir()]
    assert hidden_dirs == []


def test_run_pytest_public_targets_only_tests_public(tmp_path: Path) -> None:
    """Req: Public-test verifier runs only public tests during the agent loop
    WHEN the agent's run_tests step executes on a hidden-fixture sandbox
    THEN pytest collects zero items from tests_hidden/ (defensive: hidden dir is also
         absent from the sandbox per the sandbox snapshot requirement)."""
    import subprocess
    import sys

    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES, snapshot_fixture

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    sandbox = snapshot_fixture(csv_dedupe, tmp_path)
    # Belt-and-braces: even if a future regression copies tests_hidden/ into
    # the sandbox, pytest should still only collect from tests_public/ when
    # invoked with that path.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_public/", "--collect-only", "-q"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"collect-only failed: {proc.stdout}\n{proc.stderr}"
    # No collected node should reference tests_hidden.
    assert "tests_hidden" not in proc.stdout, proc.stdout
    assert "test_dedupe_hidden" not in proc.stdout, proc.stdout


def test_date_window_sandbox_omits_hidden_tests(tmp_path: Path) -> None:
    """Req: date_window is registered as a hard hidden-test fixture
    WHEN snapshot_fixture is called on date_window
    THEN tests_hidden/ is absent from the sandbox."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES, snapshot_fixture

    date_window = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "date_window")
    sandbox = snapshot_fixture(date_window, tmp_path)
    assert (sandbox / "src").is_dir()
    assert (sandbox / "tests_public").is_dir()
    assert (sandbox / "spec.md").is_file()
    assert not (sandbox / "tests_hidden").exists()
    assert [p for p in sandbox.rglob("*date_window_hidden*")] == []


# --- Phase C: prompt content discipline ------------------------------------


def test_hidden_fixture_prompt_contains_spec_md_and_public_test(tmp_path: Path) -> None:
    """Req: Agent prompt references spec.md and public tests only
    WHEN the agent constructs the fix prompt for a hidden-test fixture
    THEN the prompt contains the full contents of spec.md and the full
         contents of every file under tests_public/."""
    from bench.real.coding_agent.agent import build_fix_prompt
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES, snapshot_fixture

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    sandbox = snapshot_fixture(csv_dedupe, tmp_path)
    prompt = build_fix_prompt(csv_dedupe, sandbox)
    spec_text = (csv_dedupe.root / "spec.md").read_text()
    public_text = (
        csv_dedupe.root / "tests_public" / csv_dedupe.public_tests_relpath  # type: ignore[arg-type]
    ).read_text()
    src_text = csv_dedupe.source_path.read_text()
    # Pick a few stable substrings that must survive whatever framing the
    # builder adds; full-text equality would be brittle to wrapping.
    for needle in ("BOM strip", "case fold", "Unicode NFC"):
        assert needle in prompt, f"spec.md content missing from prompt: {needle!r}"
    assert "test_exact_duplicates_collapse_to_one" in prompt
    assert "test_first_occurrence_is_preserved" in prompt
    # Sanity: at least the bulk of each document made it in.
    assert len(spec_text) >= 500 and spec_text[:200] in prompt
    assert public_text[:200] in prompt
    assert src_text[:200] in prompt


def test_hidden_fixture_prompt_does_not_mention_tests_hidden(tmp_path: Path) -> None:
    """Req: Agent prompt references spec.md and public tests only
    WHEN the agent constructs the fix prompt for a hidden-test fixture
    THEN the prompt does not contain the substring 'tests_hidden', does
         not contain any filename from tests_hidden/, and does not contain
         the body of any test under tests_hidden/."""
    from bench.real.coding_agent.agent import build_fix_prompt
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES, snapshot_fixture

    csv_dedupe = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    sandbox = snapshot_fixture(csv_dedupe, tmp_path)
    prompt = build_fix_prompt(csv_dedupe, sandbox)
    assert "tests_hidden" not in prompt
    hidden_path = csv_dedupe.hidden_test_path
    assert hidden_path is not None
    assert hidden_path.name not in prompt
    # Hidden test bodies must not appear. Pick stable substrings from the
    # actual hidden test file.
    hidden_text = hidden_path.read_text()
    for needle in (
        "test_whitespace_rule_strips_outer_whitespace",
        "test_nfc_rule_treats_nfc_and_nfd_as_equal",
        "test_combined_rules_treat_full_normalization_as_equal",
    ):
        assert needle in hidden_text, "test fixture self-check (hidden file unchanged)"
        assert needle not in prompt, (
            f"hidden test name leaked into prompt: {needle}"
        )


def test_date_window_prompt_contains_spec_and_public_only(tmp_path: Path) -> None:
    """Req: Agent prompt references spec.md and public tests only
    WHEN the agent constructs the fix prompt for date_window
    THEN it includes spec/public/source text and omits hidden-test text."""
    from bench.real.coding_agent.agent import build_fix_prompt
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES, snapshot_fixture

    date_window = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "date_window")
    sandbox = snapshot_fixture(date_window, tmp_path)
    prompt = build_fix_prompt(date_window, sandbox)
    spec_text = (date_window.root / "spec.md").read_text()
    public_path = date_window.public_test_path
    hidden_path = date_window.hidden_test_path
    assert public_path is not None
    assert hidden_path is not None
    public_text = public_path.read_text()
    src_text = date_window.source_path.read_text()

    for needle in ("YYYY-MM-DD", "inclusive", "Leap days"):
        assert needle in prompt
    assert public_text[:200] in prompt
    assert src_text[:200] in prompt
    assert spec_text[:200] in prompt
    assert "tests_hidden" not in prompt
    assert hidden_path.name not in prompt
    for needle in (
        "test_end_boundary_is_inclusive",
        "test_unsorted_windows_are_all_checked",
        "test_non_leap_day_is_rejected",
    ):
        assert needle in hidden_path.read_text()
        assert needle not in prompt


# --- Phase D: hidden eval + Outcome metadata wiring ------------------------


def _stub_llm_returning(text: str) -> object:
    """Build a one-shot stub LLM that always returns `text`."""
    from bench.real.coding_agent.llm import LLMResponse

    class _Stub:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text=text, cost_usd=0.0)

    return _Stub()


def _csv_dedupe_reference_fence() -> str:
    """The reference implementation source, fenced for the prompt protocol."""
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    ref = (csv.root / "src" / "_dedupe_reference.py").read_text()
    return f"```python\n{ref}```"


def _csv_dedupe_buggy_fence() -> str:
    """A patch that satisfies public tests but fails hidden tests."""
    return (
        "```python\n"
        '"""Public-passing, hidden-failing patch."""\n'
        "from __future__ import annotations\n"
        "def dedupe(rows: list[str]) -> list[str]:\n"
        "    seen: set[str] = set()\n"
        "    out: list[str] = []\n"
        "    for row in rows:\n"
        "        if row in seen:\n"
        "            continue\n"
        "        seen.add(row)\n"
        "        out.append(row)\n"
        "    return out\n"
        "```"
    )


def test_hidden_outcome_when_reference_passes_both(tmp_path: Path) -> None:
    """Req: Hidden-test verifier runs once after the loop and defines Outcome
    Req: Outcome metadata records public, hidden, and generalization gap
    Req: Hidden-fixture traces use a distinct verifier label
    WHEN a stub LLM returns the reference implementation
    THEN Outcome.value=True, verifier='pytest_hidden', metadata records
         public_pass=True, hidden_pass=True, generalization_gap=False."""
    from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
    from bench.real.coding_agent.budget import BudgetTracker
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    budget = BudgetTracker(cap_usd=1.0)
    run = run_one_trace(
        csv,
        run_index=0,
        llm=_stub_llm_returning(_csv_dedupe_reference_fence()),  # type: ignore[arg-type]
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=42),
    )
    assert run.outcome.kind == "binary"
    assert run.outcome.value is True
    assert run.outcome.verifier == "pytest_hidden"
    md = run.outcome.metadata
    assert md["fixture_id"] == "csv_dedupe"
    assert md["public_pass"] is True
    assert md["hidden_pass"] is True
    assert md["generalization_gap"] is False


def test_hidden_outcome_records_generalization_gap(tmp_path: Path) -> None:
    """Req: Outcome metadata records public, hidden, and generalization gap
    Req: Hidden-test verifier runs once after the loop and defines Outcome
    WHEN a stub LLM returns a fix that passes public but fails hidden
    THEN Outcome.value=False, public_pass=True, hidden_pass=False,
         generalization_gap=True."""
    from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
    from bench.real.coding_agent.budget import BudgetTracker
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    budget = BudgetTracker(cap_usd=1.0)
    run = run_one_trace(
        csv,
        run_index=0,
        llm=_stub_llm_returning(_csv_dedupe_buggy_fence()),  # type: ignore[arg-type]
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=42),
    )
    assert run.outcome.value is False
    md = run.outcome.metadata
    assert md["public_pass"] is True
    assert md["hidden_pass"] is False
    assert md["generalization_gap"] is True


def test_hidden_outcome_when_no_patch_extracted(tmp_path: Path) -> None:
    """When the model returns no fenced block, src/ stays buggy. Public tests
    still pass on the buggy src (by fixture invariant), so we must still see
    public_pass=True / hidden_pass=False / gap=True."""
    from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
    from bench.real.coding_agent.budget import BudgetTracker
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES

    csv = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    budget = BudgetTracker(cap_usd=1.0)
    run = run_one_trace(
        csv,
        run_index=0,
        llm=_stub_llm_returning("(no fence)"),  # type: ignore[arg-type]
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=42),
    )
    md = run.outcome.metadata
    assert md["public_pass"] is True
    assert md["hidden_pass"] is False
    assert md["generalization_gap"] is True
    assert run.outcome.value is False


def test_hidden_eval_runs_exactly_once_per_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Req: Hidden-test verifier runs once after the loop and defines Outcome
    WHEN a single trace completes on a hidden-test fixture
    THEN run_pytest_hidden is invoked exactly once after the agent has terminated."""
    from bench.real.coding_agent import fixtures as fixtures_mod
    from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
    from bench.real.coding_agent.budget import BudgetTracker

    csv = next(fx for fx in fixtures_mod.HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    calls: list[Path] = []
    real_hidden = fixtures_mod.run_pytest_hidden

    def _spy(workspace: Path, *, timeout_s: int = 30) -> tuple[bool, str]:
        calls.append(workspace)
        return real_hidden(workspace, timeout_s=timeout_s)

    # Patch the symbol that agent.py imports so the spy is observed there.
    import bench.real.coding_agent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "run_pytest_hidden", _spy, raising=True)
    budget = BudgetTracker(cap_usd=1.0)
    run_one_trace(
        csv,
        run_index=0,
        llm=_stub_llm_returning("(no fence)"),  # type: ignore[arg-type]
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=42),
    )
    assert len(calls) == 1, f"expected hidden eval to run once, got {len(calls)}"


def test_retry_prompt_tail_comes_from_public_not_hidden(tmp_path: Path) -> None:
    """Req: Public-test verifier runs only public tests during the agent loop
    WHEN the agent's first run_tests step on a hidden-fixture reports a failure
    THEN the failure tail piped into the retry prompt comes from tests_public/
         and contains no text originating from tests_hidden/."""
    from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
    from bench.real.coding_agent.budget import BudgetTracker
    from bench.real.coding_agent.fixtures import HIDDEN_FIXTURES
    from bench.real.coding_agent.llm import LLMResponse

    csv = next(fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe")
    captured: list[str] = []
    # Patch that BREAKS public tests (raises on import), so a public failure
    # drives the retry prompt.
    public_breaking_fence = (
        "```python\n"
        "raise RuntimeError('public-breaking patch')\n"
        "```"
    )

    class _Stub:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            captured.append(prompt)
            return LLMResponse(text=public_breaking_fence, cost_usd=0.0)

    budget = BudgetTracker(cap_usd=1.0)
    # Force retry_once via greedy + epsilon=0
    cfg = AgentRunConfig(
        epsilon=0.0, seed=0, retry_greedy="retry_once", retry_epsilon=0.0
    )
    run_one_trace(
        csv,
        run_index=0,
        llm=_Stub(),  # type: ignore[arg-type]
        budget=budget,
        sandbox_root=tmp_path,
        config=cfg,
    )
    assert len(captured) == 2, (
        f"expected initial + retry prompts, got {len(captured)}"
    )
    retry_prompt = captured[1]
    # The retry prompt's failure tail must mention public test output, not hidden.
    assert "tests_public" in retry_prompt or "test_dedupe.py" in retry_prompt
    assert "tests_hidden" not in retry_prompt
    assert "test_dedupe_hidden" not in retry_prompt
    assert "whitespace_rule" not in retry_prompt
    assert "nfc_rule" not in retry_prompt


# --- Phase E: v0 backward compat -------------------------------------------


def test_v0_fixture_outcome_unchanged(tmp_path: Path) -> None:
    """Req: Hidden-fixture traces use a distinct verifier label
    Req: MODIFIED Real-agent family generates traces from a coding-agent loop
    WHEN a v0 (single-tests-dir) fixture run completes
    THEN Outcome.verifier='pytest' and metadata has only fixture_id
         (no public_pass / hidden_pass / generalization_gap keys)."""
    from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
    from bench.real.coding_agent.budget import BudgetTracker
    from bench.real.coding_agent.fixtures import EASY_FIXTURES
    from bench.real.coding_agent.llm import LLMResponse

    fixture = EASY_FIXTURES[0]  # string-utils — known-good patch is small

    class _PerfectLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            fixed = (
                "```python\n"
                '"""Patched."""\n'
                "from __future__ import annotations\n"
                "import re\n"
                "def normalize_name(name: str) -> str:\n"
                '    return re.sub(r"\\s+", " ", name.strip().lower())\n'
                "```"
            )
            return LLMResponse(text=fixed, cost_usd=0.0)

    budget = BudgetTracker(cap_usd=1.0)
    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_PerfectLLM(),  # type: ignore[arg-type]
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=42),
    )
    assert run.outcome.value is True
    assert run.outcome.verifier == "pytest"
    assert run.outcome.metadata == {"fixture_id": fixture.fixture_id}
    # Specifically, none of the hidden-fixture metadata keys leaked into v0.
    for forbidden in ("public_pass", "hidden_pass", "generalization_gap"):
        assert forbidden not in run.outcome.metadata


# --- Phase F: CLI --fixtures flag ------------------------------------------


def test_run_real_corpus_with_fixtures_csv_dedupe(tmp_path: Path) -> None:
    """Req: Pilot gate before scaling to additional hidden fixtures
    WHEN the corpus generator is invoked with --fixtures csv_dedupe
    THEN the harness runs only csv_dedupe and produces traces tagged with that
         fixture_id (and verifier='pytest_hidden')."""
    import json

    from bench.real.coding_agent.llm import LLMResponse
    from bench.real.coding_agent.runner import run_real_corpus

    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    class _NullLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    rc = run_real_corpus(
        n=2,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _NullLLM(),
        marker_path=marker,
        fixture_ids=("csv_dedupe",),
    )
    assert rc == 0
    written = sorted(output.glob("real-*.json"))
    assert len(written) == 2
    for path in written:
        assert "csv_dedupe" in path.name
        data = json.loads(path.read_text())
        assert data["outcome"]["verifier"] == "pytest_hidden"
        assert data["outcome"]["metadata"]["fixture_id"] == "csv_dedupe"


def test_cli_real_subcommand_accepts_fixtures_flag(tmp_path: Path) -> None:
    """The CLI parses --fixtures as a comma-separated list and forwards it
    through to run_real_corpus."""
    import subprocess
    import sys

    out = tmp_path / "out"
    # Without an approval marker, the run will exit 2 — we only want to assert
    # the CLI accepts the flag without an argparse error.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "counterfact.cli",
            "bench",
            "real",
            "--n",
            "1",
            "--budget-cap",
            "1",
            "--output-dir",
            str(out),
            "--fixtures",
            "csv_dedupe",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    # Exit 2 is the HUMAN-GATE refusal, which proves the CLI parsed the args.
    assert proc.returncode == 2, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    # Argparse errors on unknown flags exit 2 *with* an "unrecognized arguments"
    # message; the gate refusal does not. Distinguish.
    assert "unrecognized arguments" not in (proc.stdout + proc.stderr)
    assert "HUMAN GATE" in (proc.stdout + proc.stderr)


def test_hard_hidden_v1_fixture_set_resolves_to_date_window() -> None:
    """Req: hard hidden fixture set is selectable
    WHEN fixture_set='hard_hidden_v1' is resolved
    THEN date_window is included."""
    from bench.real.coding_agent.runner import resolve_fixtures

    fixtures = resolve_fixtures(fixture_set="hard_hidden_v1")
    ids = [fx.fixture_id for fx in fixtures]
    assert ids == ["date_window"]


def test_hidden_v1_fixture_set_stays_csv_dedupe_only() -> None:
    """Req: hard hidden fixture set is selectable
    WHEN fixture_set='hidden_v1' is resolved
    THEN it remains the historical csv_dedupe calibration set."""
    from bench.real.coding_agent.runner import resolve_fixtures

    fixtures = resolve_fixtures(fixture_set="hidden_v1")
    ids = [fx.fixture_id for fx in fixtures]
    assert ids == ["csv_dedupe"]


def test_cli_real_subcommand_accepts_hard_hidden_v1() -> None:
    """Req: hard hidden fixture set is selectable
    WHEN the CLI parser sees --fixture-set hard_hidden_v1
    THEN argparse accepts it."""
    from counterfact.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "bench",
            "real",
            "--n",
            "1",
            "--fixture-set",
            "hard_hidden_v1",
        ]
    )
    assert ns.fixture_set == "hard_hidden_v1"
