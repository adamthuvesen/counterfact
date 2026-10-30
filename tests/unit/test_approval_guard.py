"""The repo PreToolUse hook refuses agent writes to the spend approval receipt."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.guard_approval_receipt import MESSAGE, denial

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "/repo/.counterfact/approved"


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "Write", "tool_input": {"file_path": RECEIPT, "content": "{}"}},
        {"tool_name": "Edit", "tool_input": {"file_path": RECEIPT}},
        {"tool_name": "Bash", "tool_input": {"command": "echo '{}' > .counterfact/approved"}},
        {"tool_name": "Bash", "tool_input": {"command": "cp receipt.json .counterfact/approved"}},
        {
            "tool_name": "Bash",
            "tool_input": {"command": 'python3 -c \'open(".counterfact/approved", "w")\''},
        },
    ],
)
def test_receipt_writes_are_denied(payload: dict[str, object]) -> None:
    assert denial(payload) == MESSAGE


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "Write", "tool_input": {"file_path": "/repo/docs/usage.md"}},
        {"tool_name": "Bash", "tool_input": {"command": "cat .counterfact/approved 2>/dev/null"}},
        {"tool_name": "Bash", "tool_input": {"command": "uv run counterfact bench real --n 1"}},
    ],
)
def test_reads_and_unrelated_tools_are_allowed(payload: dict[str, object]) -> None:
    assert denial(payload) is None


def test_settings_hook_exits_2_with_the_stop_message() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "scripts/guard_approval_receipt.py" in command
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "guard_approval_receipt.py")],
        input=json.dumps({"tool_name": "Write", "tool_input": {"file_path": RECEIPT}}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Ask Adam to approve the spend" in result.stderr
