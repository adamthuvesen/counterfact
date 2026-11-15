"""Suppressions stay narrow: no file-level downgrade, no inline ignore without a rule and reason, no skip without a reason."""

import ast
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
FILE_WIDE_NOQA = re.compile(r"#\s*ruff:\s*noqa", re.IGNORECASE)
NOQA_ON_BAN = re.compile(r"#\s*noqa\b.*\bTID251", re.IGNORECASE)
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
    assert not downgrades, (
        "file-level `# pyright:` comment turns strict mode off; remove it and fix the type error, or use an inline `# pyright: ignore[rule]  # reason`: "
        + ", ".join(downgrades)
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
    assert not hits, (
        "pyright ignore needs a rule and a reason; write `# pyright: ignore[ruleName]  # why`: "
        + ", ".join(hits)
    )


def test_ignore_pattern_finds_unexplained_ignores() -> None:
    source = (
        "a = 1  # pyright: ignore\n"
        "b = 1  # pyright: ignore[reportArgumentType]\n"
        "c = 1  # type: ignore[attr-defined]\n"
        "d = 1  # pyright: ignore[reportArgumentType]  # invalid on purpose\n"
        'e = "# pyright: ignore"\n'
    )
    assert unexplained_ignores(source) == [1, 2, 3]


def lifted_bans(source: str) -> list[int]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        token.start[0]
        for token in tokens
        if token.type == tokenize.COMMENT
        and (FILE_WIDE_NOQA.search(token.string) or NOQA_ON_BAN.search(token.string))
    ]


def test_no_noqa_lifts_a_ban_or_a_whole_file() -> None:
    hits = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in checkout_python_files()
        for line in lifted_bans(path.read_text(errors="replace"))
    ]
    assert not hits, (
        "remove the file-wide or TID251 noqa; fix each finding, or give the owner file a TID251 per-file-ignore in pyproject.toml: "
        + ", ".join(hits)
    )


def test_lifted_ban_pattern_finds_known_cases() -> None:
    source = (
        "# ruff: noqa\n"
        "import subprocess  # noqa: TID251\n"
        "import os  # noqa: F401\n"
        'x = "# noqa: TID251"\n'
    )
    assert lifted_bans(source) == [1, 2]


SKIP_NAMES = {"pytest.skip", "unittest.skip", "pytest.mark.skip", "pytest.mark.xfail"}
# Only these two take the reason as a first positional argument. For the marks, the first
# positional argument is the xfail condition, so they take the reason as a keyword alone.
POSITIONAL_REASON = {"pytest.skip", "unittest.skip"}


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _reason(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == "reason":
            return keyword.value
    if _dotted_name(call.func) in POSITIONAL_REASON and call.args:
        return call.args[0]
    return None


def _has_reason(value: ast.expr | None) -> bool:
    if value is None:
        return False
    return not (
        isinstance(value, ast.Constant) and isinstance(value.value, str) and not value.value.strip()
    )


def reasonless_skips(source: str) -> list[int]:
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _dotted_name(node.func) in SKIP_NAMES:
            if not _has_reason(_reason(node)):
                found.append(node.lineno)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            for decorator in node.decorator_list:
                if _dotted_name(decorator) in SKIP_NAMES:
                    found.append(decorator.lineno)
    return sorted(found)


def test_every_skip_gives_a_reason() -> None:
    bare = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in checkout_python_files()
        for line in reasonless_skips(path.read_text(errors="replace"))
    ]
    assert not bare, (
        'give each skip or xfail a non-empty reason string, e.g. reason="needs Docker": '
        + ", ".join(bare)
    )


def test_skip_scan_finds_known_reasonless_skips() -> None:
    source = (
        "import pytest\n"
        "@pytest.mark.skip\n"
        "def test_bare_decorator(): ...\n"
        "@pytest.mark.skip()\n"
        "def test_empty_mark(): ...\n"
        "pytest.skip()\n"
        "pytest.skip('')\n"
        "pytest.mark.xfail(False)\n"
        "unittest.skip('the reason is positional')\n"
        "pytest.mark.skip(reason='the reason is a keyword')\n"
        "pytest.mark.xfail(False, reason='the reason is a keyword')\n"
        "pytest.skip(reason='the reason is a keyword')\n"
    )
    assert reasonless_skips(source) == [2, 4, 6, 7, 8]
