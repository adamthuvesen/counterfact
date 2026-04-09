"""Acceptance tests for the hidden-test-fixtures change.

Scenarios are sourced verbatim from
`openspec/changes/hidden-test-fixtures/specs/corpus-harness/spec.md`.
Each test name encodes the requirement and scenario it pins.
"""

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
    forbidden = ["whitespace", "casefold", "lower(", ".upper(", "nfc", "unicodedata", "bom", "\\ufeff"]
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
