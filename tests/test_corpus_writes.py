"""Run files are written through corpus_io.write_corpus, not by hand."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# runner.py publishes each graded trace at once so a resume can count spend.
ALLOWED = {"src/counterfact/corpus_io.py", "bench/real/coding_agent/runner.py"}


# The AST has no types, so any dumped model counts as a Run unless its name says otherwise.
NOT_RUN_NAME = re.compile(r"(^|_)receipt$")
WRITERS = {"write_text", "write", "atomic_write"}


def _callee_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _dumps_run_json(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        receiver = n.func.value
        name = receiver.id if isinstance(receiver, ast.Name) else getattr(receiver, "attr", "")
        if n.func.attr == "model_dump_json" and not NOT_RUN_NAME.search(name):
            return True
    return False


def test_run_json_is_not_written_by_hand() -> None:
    hits = []
    for path in sorted(
        [*ROOT.glob("src/**/*.py"), *ROOT.glob("bench/**/*.py"), *ROOT.glob("scripts/*.py")]
    ):
        relative = path.relative_to(ROOT).as_posix()
        if relative in ALLOWED or "/fixtures/" in relative:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and _callee_name(node.func) in WRITERS
                and any(_dumps_run_json(arg) for arg in node.args)
            ):
                hits.append(f"{relative}:{node.lineno}")
    assert not hits, "Publish runs with counterfact.corpus_io.write_corpus: " + ", ".join(hits)


def test_detector_finds_a_known_hand_written_run() -> None:
    written = ast.parse("path.write_text(run.model_dump_json())").body[0]
    assert isinstance(written, ast.Expr) and isinstance(written.value, ast.Call)
    assert _dumps_run_json(written.value.args[0])
    assert _dumps_run_json(ast.parse("trace.model_dump_json()").body[0])
    assert not _dumps_run_json(ast.parse("receipt.model_dump_json()").body[0])
