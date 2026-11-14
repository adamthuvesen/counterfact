"""Claude Code and Codex PreToolUse hook: agents never write the spend approval receipt."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# A command that names both parts, e.g. `cd .counterfact && touch approved`, counts.
NAMES_RECEIPT = (re.compile(r"\.counterfact\b"), re.compile(r"\bapproved\b"))
WRITES = re.compile(r">(?!\s*(/dev/null|&\d))")
# A command that names the receipt may only read it: every segment starts with a read-only tool.
SEGMENT_SPLIT = re.compile(r"\|\|?|&&|;|\n|&(?![0-9>])")
READ_ONLY = {
    "cat",
    "ls",
    "test",
    "[",
    "stat",
    "head",
    "tail",
    "grep",
    "rg",
    "wc",
    "file",
    "cd",
    "echo",
}
MESSAGE = (
    "Stop. Agents never create or edit .counterfact/approved. "
    "Ask Adam to approve the spend and write the approval receipt himself."
)


def denial(payload: dict[str, object]) -> str | None:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if isinstance(path, str):
        target = Path(str(payload.get("cwd") or ".")) / path
        return MESSAGE if target.parts[-2:] == (".counterfact", "approved") else None
    command = tool_input.get("command") or tool_input.get("input") or ""
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    text = str(command)
    named = all(pattern.search(text) for pattern in NAMES_RECEIPT)
    if not named:
        return None
    return MESSAGE if WRITES.search(text) or not _only_reads(text) else None


def _only_reads(command: str) -> bool:
    for segment in SEGMENT_SPLIT.split(command):
        words = segment.split()
        if not words:
            continue
        if words[0] not in READ_ONLY:
            return False
    return True


def main() -> int:
    reason = denial(json.load(sys.stdin))
    if reason is not None:
        decision = {"permissionDecision": "deny", "permissionDecisionReason": reason}
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", **decision}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
