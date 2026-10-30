"""Build or verify the generated-code Docker boundary without model calls."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.real.coding_agent.executor import (  # noqa: E402
    DockerExecutor,
    ExecutorUnavailable,
    build_verifier_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", action="store_true", help="Build the pinned verifier image first"
    )
    args = parser.parse_args()
    try:
        if args.build:
            build_verifier_image()
        DockerExecutor().preflight()
    except (ExecutorUnavailable, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "container",
            "--no-cov",
            "tests/unit/test_corpus_real.py",
            "tests/acceptance/test_corpus_real_hidden.py",
            "tests/acceptance/test_container_executor.py",
        ],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
