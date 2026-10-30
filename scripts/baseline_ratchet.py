"""Fail when the basedpyright baseline holds more errors than on origin/main."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASELINE = ".basedpyright/baseline.json"


def count_errors(text: str) -> int:
    return sum(len(entries) for entries in json.loads(text)["files"].values())


def main() -> int:
    path = Path(BASELINE)
    current = count_errors(path.read_text()) if path.exists() else 0
    show = subprocess.run(
        ["git", "show", f"origin/main:{BASELINE}"], capture_output=True, text=True, check=False
    )
    if show.returncode != 0:
        if subprocess.run(["git", "rev-parse", "origin/main"], capture_output=True).returncode:
            print("baseline ratchet: origin/main is not fetched", file=sys.stderr)
            return 2
        return 0  # main has no baseline yet: the first baseline is allowed
    on_main = count_errors(show.stdout)
    if current > on_main:
        print(
            f"baseline ratchet: {current} baselined errors, {on_main} on origin/main. "
            "Fix the new errors instead of baselining them.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
