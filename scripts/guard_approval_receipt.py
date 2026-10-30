"""Claude Code PreToolUse hook: agents never write the real-agent spend approval receipt."""

from __future__ import annotations

import json
import re
import sys

RECEIPT = re.compile(r"\.counterfact/approved\b")
SHELL_WRITE = re.compile(
    r"(?<![0-9&])>(?!\s*/dev/null)|\bsed\s+-i"
    r"|\b(tee|cp|mv|touch|ln|rm|install|dd|truncate|python3?|uv|node|perl|ruby)\b"
)
MESSAGE = (
    "Stop. Agents never create or edit .counterfact/approved. "
    "Ask Adam to approve the spend and write the approval receipt himself."
)


def denial(payload: dict[str, object]) -> str | None:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    if payload.get("tool_name") == "Bash":
        command = str(tool_input.get("command", ""))
        return MESSAGE if RECEIPT.search(command) and SHELL_WRITE.search(command) else None
    path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    return MESSAGE if RECEIPT.search(path) else None


def main() -> int:
    reason = denial(json.load(sys.stdin))
    if reason is None:
        return 0
    print(reason, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
