"""Fixture registry + deterministic pytest verifier."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class FixtureSpec:
    """A single coding-agent fixture.

    v0 fixtures use a single `tests/` directory; hidden-test fixtures (per the
    `hidden-test-fixtures` change) split into `tests_public/` and
    `tests_hidden/` and add a `spec.md`. The two layouts are distinguished by
    whether `public_tests_relpath` and `hidden_tests_relpath` are set.
    """

    fixture_id: str
    source_relpath: str  # path of the source file under <fixture>/src/
    test_relpath: str  # path of the test file under <fixture>/tests/ (v0)
    public_tests_relpath: str | None = None  # under <fixture>/tests_public/
    hidden_tests_relpath: str | None = None  # under <fixture>/tests_hidden/

    @property
    def root(self) -> Path:
        return FIXTURES_ROOT / self.fixture_id

    @property
    def source_path(self) -> Path:
        return self.root / "src" / self.source_relpath

    @property
    def test_path(self) -> Path:
        return self.root / "tests" / self.test_relpath

    @property
    def public_test_path(self) -> Path | None:
        if self.public_tests_relpath is None:
            return None
        return self.root / "tests_public" / self.public_tests_relpath

    @property
    def hidden_test_path(self) -> Path | None:
        if self.hidden_tests_relpath is None:
            return None
        return self.root / "tests_hidden" / self.hidden_tests_relpath


def is_hidden_fixture(fixture: FixtureSpec) -> bool:
    """A fixture is hidden iff it declares both public and hidden test paths."""
    return (
        fixture.public_tests_relpath is not None
        and fixture.hidden_tests_relpath is not None
    )


# The original three fixtures are easy bugs that capable LLMs one-shot every
# time (see design.md D19 + the post-mortem on the v0 200-trace corpus). They
# are kept for harness-integration testing but excluded from the demo's causal
# corpus by virtue of living in `EASY_FIXTURES`. The harder set in `FIXTURES`
# is what `counterfact bench real` exercises.
EASY_FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec("string-utils", "normalize.py", "test_normalize.py"),
    FixtureSpec("date-utils", "parse.py", "test_parse.py"),
    FixtureSpec("csv-stats", "agg.py", "test_agg.py"),
)

FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec("regex-anchors", "match.py", "test_match.py"),
    FixtureSpec("iso-week-dates", "iso_week.py", "test_iso_week.py"),
    FixtureSpec("agg-with-groups", "agg.py", "test_agg.py"),
)

# Hidden-test fixtures (per the `hidden-test-fixtures` change). The agent sees
# `src/`, `tests_public/`, and `spec.md`; `Outcome` is determined by
# `tests_hidden/`, which is never copied into the agent's sandbox.
HIDDEN_FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        "csv_dedupe",
        source_relpath="dedupe.py",
        test_relpath="",  # unused for hidden fixtures
        public_tests_relpath="test_dedupe.py",
        hidden_tests_relpath="test_dedupe_hidden.py",
    ),
    FixtureSpec(
        "date_window",
        source_relpath="date_window.py",
        test_relpath="",  # unused for hidden fixtures
        public_tests_relpath="test_date_window.py",
        hidden_tests_relpath="test_date_window_hidden.py",
    ),
)

HIDDEN_V1_FIXTURES: tuple[FixtureSpec, ...] = tuple(
    fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe"
)

HARD_HIDDEN_V1_FIXTURES: tuple[FixtureSpec, ...] = tuple(
    fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "date_window"
)


def _run_pytest_at(
    fixture_root: Path, target: str, *, timeout_s: int = 30
) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", target, "-q", "--tb=line"],
            cwd=fixture_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        # An untrusted patch or test suite hung past `timeout_s`. Surface this
        # as a normal verifier failure with diagnostic tail rather than letting
        # it crash the corpus runner.
        captured = exc.stdout or exc.stderr or b""
        if isinstance(captured, bytes):
            captured = captured.decode(errors="replace")
        tail = f"<pytest timed out after {timeout_s}s>\n{captured[-2000:]}"
        return False, tail
    tail = proc.stdout[-2000:] if proc.stdout else proc.stderr[-2000:]
    return proc.returncode == 0, tail


def run_pytest(fixture_root: Path, *, timeout_s: int = 30) -> tuple[bool, str]:
    """Run pytest in the fixture's `tests/` directory (v0 layout).

    Returns `(passed, stdout_tail)`.
    """
    return _run_pytest_at(fixture_root, "tests/", timeout_s=timeout_s)


def run_pytest_public(sandbox_root: Path, *, timeout_s: int = 30) -> tuple[bool, str]:
    """Run pytest in the sandbox's `tests_public/` directory (hidden-fixture layout).

    The agent invokes this during its loop; it never sees `tests_hidden/`.
    """
    return _run_pytest_at(sandbox_root, "tests_public/", timeout_s=timeout_s)


def run_pytest_hidden(
    eval_workspace: Path, *, timeout_s: int = 30
) -> tuple[bool, str]:
    """Run pytest in a hidden-eval workspace's `tests_hidden/` directory.

    The harness invokes this exactly once per trace, after the agent loop
    has terminated, in a workspace separate from the agent's sandbox.
    """
    return _run_pytest_at(eval_workspace, "tests_hidden/", timeout_s=timeout_s)


def snapshot_fixture(fixture: FixtureSpec, dest_root: Path) -> Path:
    """Copy a fixture into a sandbox so the agent can edit without dirtying source.

    For hidden-test fixtures, `tests_hidden/` is excluded — the agent must not
    see hidden tests, by structural guarantee (per design.md D2).
    """
    dest = dest_root / fixture.fixture_id
    if dest.exists():
        shutil.rmtree(dest)
    if is_hidden_fixture(fixture):
        shutil.copytree(
            fixture.root,
            dest,
            ignore=shutil.ignore_patterns("tests_hidden", "_*reference*"),
        )
    else:
        shutil.copytree(fixture.root, dest)
    return dest


def build_hidden_eval_workspace(
    fixture: FixtureSpec, sandbox_root: Path, dest_root: Path
) -> Path:
    """Stage a hidden-eval workspace by copying the agent's patched src/ next
    to the canonical tests_hidden/ from the fixture's source tree.

    The agent's sandbox is read; the fixture's source is read; nothing is
    written back to either. The returned workspace lives under `dest_root`
    and is the cwd for `run_pytest_hidden`.
    """
    if not is_hidden_fixture(fixture):
        raise ValueError(
            f"build_hidden_eval_workspace called on non-hidden fixture {fixture.fixture_id!r}"
        )
    workspace = dest_root / f"{fixture.fixture_id}-hidden-eval"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    # Patched src/ from the agent's sandbox.
    shutil.copytree(sandbox_root / "src", workspace / "src")
    # Canonical tests_hidden/ from the fixture's source tree (NOT from the
    # sandbox — the sandbox doesn't have it).
    shutil.copytree(fixture.root / "tests_hidden", workspace / "tests_hidden")
    return workspace
