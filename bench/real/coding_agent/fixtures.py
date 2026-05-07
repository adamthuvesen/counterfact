"""Fixture registry + deterministic pytest verifier."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
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
    FixtureSpec(
        "rate_limit",
        source_relpath="rate_limit.py",
        test_relpath="",  # unused for hidden fixtures
        public_tests_relpath="test_rate_limit.py",
        hidden_tests_relpath="test_rate_limit_hidden.py",
    ),
    FixtureSpec(
        "version_range",
        source_relpath="version_range.py",
        test_relpath="",  # unused for hidden fixtures
        public_tests_relpath="test_version_range.py",
        hidden_tests_relpath="test_version_range_hidden.py",
    ),
    FixtureSpec(
        "unicode_normalize",
        source_relpath="unicode_normalize.py",
        test_relpath="",  # unused for hidden fixtures
        public_tests_relpath="test_unicode_normalize.py",
        hidden_tests_relpath="test_unicode_normalize_hidden.py",
    ),
    FixtureSpec(
        "streaming_watermark_dedupe",
        source_relpath="watermark_dedupe.py",
        test_relpath="",  # unused for hidden fixtures
        public_tests_relpath="test_watermark_dedupe.py",
        hidden_tests_relpath="test_watermark_dedupe_hidden.py",
    ),
)

HIDDEN_V1_FIXTURES: tuple[FixtureSpec, ...] = tuple(
    fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "csv_dedupe"
)

HARD_HIDDEN_V1_FIXTURES: tuple[FixtureSpec, ...] = tuple(
    fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "date_window"
)

BROAD_CALIBRATION_FIXTURES: tuple[FixtureSpec, ...] = tuple(
    fx
    for fx in HIDDEN_FIXTURES
    if fx.fixture_id in {"date_window", "rate_limit", "version_range"}
)

VERY_HARD_HIDDEN_V1_FIXTURES: tuple[FixtureSpec, ...] = tuple(
    fx for fx in HIDDEN_FIXTURES if fx.fixture_id == "unicode_normalize"
)

STATEFUL_CALIBRATION_FIXTURES: tuple[FixtureSpec, ...] = tuple(
    fx
    for fx in HIDDEN_FIXTURES
    if fx.fixture_id == "streaming_watermark_dedupe"
)


_PROVIDER_ENV_NEEDLES = ("API_KEY", "AUTH_TOKEN", "ACCESS_TOKEN", "SECRET")
_OUTPUT_TAIL_CHARS = 2000


def _scrubbed_pytest_env(home: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return {
        key: value
        for key, value in env.items()
        if not any(needle in key for needle in _PROVIDER_ENV_NEEDLES)
    }


def _tail_text(path: Path, limit: int = _OUTPUT_TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        try:
            f.seek(-limit, os.SEEK_END)
        except OSError:
            f.seek(0)
        return f.read().decode(errors="replace")


def _run_pytest_at(
    fixture_root: Path, target: str, *, timeout_s: int = 30
) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="counterfact-pytest-") as tmp:
        tmp_path = Path(tmp)
        stdout_path = tmp_path / "stdout.txt"
        stderr_path = tmp_path / "stderr.txt"
        home = tmp_path / "home"
        home.mkdir()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", target, "-q", "--tb=line"],
                    cwd=fixture_root,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout_s,
                    env=_scrubbed_pytest_env(home),
                )
            except subprocess.TimeoutExpired:
                tail = _tail_text(stdout_path) or _tail_text(stderr_path)
                return False, f"<pytest timed out after {timeout_s}s>\n{tail}"
        tail = _tail_text(stdout_path) or _tail_text(stderr_path)
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
