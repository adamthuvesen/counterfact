"""Build and exercise a wheel outside the source checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
        and not key.startswith(("COVERAGE_", "_COV_"))
    }
    try:
        with tempfile.TemporaryDirectory(prefix="counterfact-wheel-") as temporary:
            work = Path(temporary)
            distribution = work / "dist"
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(distribution)],
                cwd=ROOT,
                env=env,
                check=True,
            )
            wheels = list(distribution.glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError("wheel build must produce exactly one wheel")
            with zipfile.ZipFile(wheels[0]) as archive:
                if any(
                    name.startswith(("bench/", "scripts/", "tests/")) for name in archive.namelist()
                ):
                    raise ValueError("wheel contains checkout-only files")
            requirements = work / "runtime.txt"
            subprocess.run(
                [
                    "uv",
                    "export",
                    "--locked",
                    "--no-dev",
                    "--no-emit-project",
                    "--output-file",
                    str(requirements),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            environment = work / "venv"
            subprocess.run(
                ["uv", "venv", "--python", sys.executable, str(environment)],
                cwd=work,
                env=env,
                check=True,
            )
            python = environment / "bin" / "python"
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "-r",
                    str(requirements),
                    str(wheels[0]),
                ],
                cwd=work,
                env=env,
                check=True,
            )
            shutil.copy(ROOT / "scripts" / "wheel_smoke.py", work / "smoke.py")
            shutil.copy(
                ROOT / "tests/fixtures/adapters/claude_agent_sdk/minimal.jsonl",
                work / "adapter.jsonl",
            )
            subprocess.run(
                [str(python), "-I", str(work / "smoke.py")], cwd=work, env=env, check=True
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"wheel verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
