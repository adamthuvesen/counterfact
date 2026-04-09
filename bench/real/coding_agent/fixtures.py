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
    """A single coding-agent fixture."""

    fixture_id: str
    source_relpath: str  # path of the source file under <fixture>/src/
    test_relpath: str  # path of the test file under <fixture>/tests/

    @property
    def root(self) -> Path:
        return FIXTURES_ROOT / self.fixture_id

    @property
    def source_path(self) -> Path:
        return self.root / "src" / self.source_relpath

    @property
    def test_path(self) -> Path:
        return self.root / "tests" / self.test_relpath


FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec("string-utils", "normalize.py", "test_normalize.py"),
    FixtureSpec("date-utils", "parse.py", "test_parse.py"),
    FixtureSpec("csv-stats", "agg.py", "test_agg.py"),
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
