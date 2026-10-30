"""Every row in the AGENTS.md Enforced Rules table cites a file that defines the named check."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _keys(data: object, prefix: str = "") -> set[str]:
    if not isinstance(data, dict):
        return set()
    found = set()
    for key, value in data.items():
        found |= {f"{prefix}{key}"} | _keys(value, f"{prefix}{key}.")
    return found


def _defines(path: Path, name: str) -> bool:
    text = path.read_text()
    if path.suffix == ".toml":
        return name in _keys(tomllib.loads(text))
    if path.suffix == ".json":
        return name in _keys(json.loads(text))
    if path.name == "Makefile":
        return re.search(rf"^{re.escape(name)}:", text, re.M) is not None
    return re.search(rf"^\s*(async\s+)?(def|class)\s+{re.escape(name)}\b", text, re.M) is not None


def test_enforced_rules_cite_existing_files_and_names() -> None:
    sections = re.split(r"^## ", (ROOT / "AGENTS.md").read_text(), flags=re.M)
    table = next((s for s in sections if s.startswith("Enforced Rules\n")), None)
    assert table is not None, "AGENTS.md needs a `## Enforced Rules` section"
    rows = [line for line in table.splitlines() if line.startswith("| ")][2:]
    problems = []
    for row in rows:
        citations = re.findall(r"`([^`]+)`", row.split("|")[2])
        if not citations:
            problems.append(f"row cites nothing: {row}")
        for citation in citations:
            path, _, name = citation.partition("::")
            if not (ROOT / path).is_file():
                problems.append(f"{citation}: no file {path}")
            elif name and not _defines(ROOT / path, name):
                problems.append(f"{citation}: {path} does not define {name!r}")
    assert rows and not problems, "Fix the AGENTS.md Enforced Rules table:\n" + "\n".join(problems)
