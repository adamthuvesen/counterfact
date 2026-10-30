"""Every `Enforced by` citation in AGENTS.md names a real file and symbol."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_enforced_rules_cite_existing_files_and_names() -> None:
    text = (ROOT / "AGENTS.md").read_text()
    table = text.split("## Enforced Rules", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    citations = [c for row in rows[1:] for c in re.findall(r"`([^`]+)`", row.split("|")[2])]
    assert citations, "AGENTS.md ## Enforced Rules has no `path` or `path::name` citations"
    missing = []
    for citation in citations:
        path, _, name = citation.partition("::")
        if not (ROOT / path).is_file():
            missing.append(f"{citation}: no file {path}")
        elif name and name not in (ROOT / path).read_text():
            missing.append(f"{citation}: {name!r} not found in {path}")
    assert not missing, "Fix the AGENTS.md Enforced Rules citations:\n" + "\n".join(missing)
