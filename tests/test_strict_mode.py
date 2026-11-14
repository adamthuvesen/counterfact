"""Type-check suppressions stay narrow: no file-level downgrade, no inline ignore without a rule and reason."""

import io
import os
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Any file-level `# pyright:` comment except `strict` turns a check off for the whole file.
DOWNGRADE = re.compile(r"^\s*#\s*pyright:(?!\s*(?:strict\s*$|ignore\b))")
INLINE_IGNORE = re.compile(r"#\s*(?:pyright|type):\s*ignore")
NAMED_WITH_REASON = re.compile(r"#\s*(?:pyright|type):\s*ignore\[[^\]]+\]\s*#\s*\S")


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def checkout_python_files() -> list[Path]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (ROOT / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file() and is_python_file(path)]


def test_no_file_downgrades_type_checking_mode() -> None:
    downgrades = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in checkout_python_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if DOWNGRADE.match(line)
    ]
    assert not downgrades, "file-level `# pyright:` comment turns strict mode off: " + ", ".join(
        downgrades
    )


def test_downgrade_pattern_finds_a_known_mode_comment() -> None:
    assert DOWNGRADE.match("# pyright: basic")
    assert DOWNGRADE.match("#pyright: reportMissingImports=false, strict=false")
    assert DOWNGRADE.match("# pyright: reportAssignmentType=false, reportArgumentType=false")
    assert not DOWNGRADE.match("# pyright: strict")
    assert not DOWNGRADE.match("# pyright: ignore[reportUnknownMemberType]")


def unexplained_ignores(source: str) -> list[int]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        token.start[0]
        for token in tokens
        if token.type == tokenize.COMMENT
        and INLINE_IGNORE.search(token.string)
        and not NAMED_WITH_REASON.search(token.string)
    ]


def test_inline_ignores_name_a_rule_and_give_a_reason() -> None:
    hits = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in checkout_python_files()
        for line in unexplained_ignores(path.read_text(errors="replace"))
    ]
    assert not hits, "write `# pyright: ignore[rule]  # reason`: " + ", ".join(hits)


def test_ignore_pattern_finds_unexplained_ignores() -> None:
    source = (
        "a = 1  # pyright: ignore\n"
        "b = 1  # pyright: ignore[reportArgumentType]\n"
        "c = 1  # type: ignore[attr-defined]\n"
        "d = 1  # pyright: ignore[reportArgumentType]  # invalid on purpose\n"
        'e = "# pyright: ignore"\n'
    )
    assert unexplained_ignores(source) == [1, 2, 3]
