from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = REPO_ROOT / "notebooks" / "demo.ipynb"
EXECUTED_NOTEBOOK_NAME = "demo.executed.ipynb"


def execute_demo_notebook(out_dir: Path) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(DEMO_PATH),
            "--output-dir",
            str(out_dir),
            "--output",
            EXECUTED_NOTEBOOK_NAME,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"nbconvert failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads((out_dir / EXECUTED_NOTEBOOK_NAME).read_text())


def all_text_outputs(nb: dict) -> str:
    pieces: list[str] = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                text = out["text"]
                pieces.append("".join(text) if isinstance(text, list) else text)
            elif "data" in out and "text/plain" in out["data"]:
                data = out["data"]["text/plain"]
                pieces.append("".join(data) if isinstance(data, list) else data)
    return "\n".join(pieces)
