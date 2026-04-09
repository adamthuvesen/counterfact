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
# is what `counter bench real` exercises.
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
)


def run_pytest(fixture_root: Path, *, timeout_s: int = 30) -> tuple[bool, str]:
    """Run pytest in the fixture's `tests/` directory.

    Returns `(passed, stdout_tail)`.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
        cwd=fixture_root,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    tail = proc.stdout[-2000:] if proc.stdout else proc.stderr[-2000:]
    return proc.returncode == 0, tail


def snapshot_fixture(fixture: FixtureSpec, dest_root: Path) -> Path:
    """Copy a fixture into a sandbox so the agent can edit without dirtying source."""
    dest = dest_root / fixture.fixture_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(fixture.root, dest)
    return dest
