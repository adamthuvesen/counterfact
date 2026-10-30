"""Write generated text, or check it without changing the tree."""

from __future__ import annotations

import sys
from pathlib import Path


def write_or_check(path: Path, text: str, *, check: bool) -> bool:
    if check:
        if not path.is_file() or path.read_text() != text:
            print(f"stale or missing generated artifact: {path}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")
    return True
