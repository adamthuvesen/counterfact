"""The repo PreToolUse hook refuses agent writes to the spend approval receipt."""

import json
import os
import subprocess
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
        {"tool_name": "Bash", "tool_input": {"command": "echo '{}' 1> .counterfact/approved"}},
        {"tool_name": "Bash", "tool_input": {"command": "echo '{}' &> .counterfact/approved"}},
        {"tool_name": "Bash", "tool_input": {"command": "cp receipt.json .counterfact/approved"}},
        {"tool_name": "Bash", "tool_input": {"command": "cd .counterfact && touch approved"}},
        {
            "tool_name": "Bash",
            "tool_input": {"command": "gawk -i inplace '{print}' .counterfact/approved"},
        },
        {"tool_name": "Bash", "tool_input": {"command": "curl -o .counterfact/approved http://x"}},
        {"tool_name": "Bash", "tool_input": {"command": "rsync r.json .counterfact/approved"}},
        {
            "tool_name": "Bash",
            "tool_input": {"command": "git checkout stash -- .counterfact/approved"},
        },
        {
            "tool_name": "Bash",
            "tool_input": {"command": "sed -E -i 's/a/b/' .counterfact/approved"},
        },
        {
            "tool_name": "Bash",
            "tool_input": {"command": "sed --in-place s/a/b/ .counterfact/approved"},
        },
        {
            "tool_name": "Write",
            "cwd": "/repo/.counterfact",
            "tool_input": {"file_path": "approved"},
        },
        {
            "tool_name": "apply_patch",
            "tool_input": {"command": "*** Begin Patch\n*** Add File: .counterfact/approved\n+{}"},
        },
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
        {"tool_name": "Bash", "tool_input": {"command": "cat .counterfact/approved 2>&1"}},
        {"tool_name": "Bash", "tool_input": {"command": "sed -n 1p .counterfact/approved"}},
        {
            "tool_name": "Bash",
            "tool_input": {"command": "ls .counterfact && cat .counterfact/approved"},
        },
        {"tool_name": "Bash", "tool_input": {"command": "uv run counterfact bench real --n 1"}},
    ],
)
def test_reads_and_unrelated_tools_are_allowed(payload: dict[str, object]) -> None:
    assert denial(payload) is None


@pytest.mark.parametrize(
    ("settings", "client_dir"),
    [(".claude/settings.json", "$CLAUDE_PROJECT_DIR"), (".codex/hooks.json", "git rev-parse")],
)
def test_client_hook_denies_with_the_stop_message(settings: str, client_dir: str) -> None:
    command = json.loads((ROOT / settings).read_text())["hooks"]["PreToolUse"][0]["hooks"][0][
        "command"
    ]
    assert client_dir in command and "scripts/guard_approval_receipt.py" in command
    result = subprocess.run(
        command,
        shell=True,
        cwd=ROOT,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(ROOT)},
        input=json.dumps({"tool_name": "Write", "tool_input": {"file_path": RECEIPT}}),
        capture_output=True,
        text=True,
        check=False,
    )
    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "Ask Adam to approve the spend" in output["permissionDecisionReason"]
