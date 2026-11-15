"""Run files are written through corpus_io.write_corpus, not by hand."""

import ast
import re
from pathlib import Path

from tests.test_strict_mode import checkout_python_files

ROOT = Path(__file__).resolve().parents[1]
# runner.py publishes each graded trace at once so a resume can count spend.
ALLOWED = {"src/counterfact/corpus_io.py", "bench/real/coding_agent/runner.py"}


# The AST has no types, so any dumped model counts as a Run unless its name says otherwise.
NOT_RUN_NAME = re.compile(r"(^|_)(receipt|estimate|report|comparison)$")
DUMPS = {"model_dump_json", "model_dump"}
WRITERS = {"write_text", "write_bytes", "write", "writelines", "atomic_write", "dump"}


def _callee_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _dumps_run(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        receiver = n.func.value
        name = receiver.id if isinstance(receiver, ast.Name) else getattr(receiver, "attr", "")
        if n.func.attr in DUMPS and not NOT_RUN_NAME.search(name):
            return True
    return False


def _writes_to_a_file(call: ast.Call) -> bool:
    if _callee_name(call.func) == "print":
        return any(keyword.arg == "file" for keyword in call.keywords)
    return _callee_name(call.func) in WRITERS


def hand_written_runs(source: str) -> list[int]:
    """Lines that pass a dumped model, or a name assigned from one, to a file writer."""
    tree = ast.parse(source)
    dumped_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value and _dumps_run(node.value)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _writes_to_a_file(node)
        and any(
            _dumps_run(arg)
            or any(isinstance(n, ast.Name) and n.id in dumped_names for n in ast.walk(arg))
            for arg in [*node.args, *(keyword.value for keyword in node.keywords)]
        )
    )


def test_run_json_is_not_written_by_hand() -> None:
    hits = []
    for path in checkout_python_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in ALLOWED or "/fixtures/" in relative or relative.startswith("tests/"):
            continue
        hits += [f"{relative}:{line}" for line in hand_written_runs(path.read_text())]
    assert not hits, (
        "Do not write run files by hand; publish runs with counterfact.corpus_io.write_corpus (src/counterfact/corpus_io.py): "
        + ", ".join(hits)
    )


def test_detector_finds_each_hand_written_form() -> None:
    source = (
        "path.write_text(run.model_dump_json())\n"
        "path.write_text(trace.model_dump_json())\n"
        "text = trace.model_dump_json()\n"
        "path.write_text(text)\n"
        "path.write_bytes(trace.model_dump_json().encode())\n"
        "json.dump(trace.model_dump(mode='json'), handle)\n"
        "atomic_write(path, trace.model_dump_json())\n"
        "atomic_write(path=p, text=run.model_dump_json())\n"
        "handle.writelines([run.model_dump_json()])\n"
        "print(run.model_dump_json(), file=handle)\n"
        "print(run.model_dump_json())\n"
    )
    assert hand_written_runs(source) == [1, 2, 4, 5, 6, 7, 8, 9, 10]
    assert hand_written_runs("atomic_write(path, receipt.model_dump_json())\nprint(x)") == []
