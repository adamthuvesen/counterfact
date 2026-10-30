"""Trusted container entrypoint. Workload time excludes container startup."""

import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if sys.argv[1] == "--preflight":
        import pytest

        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "pytest": pytest.__version__,
                    "uid": os.getuid(),
                    "locale": os.environ["LC_ALL"],
                    "tz": os.environ["TZ"],
                }
            )
        )
        return
    target, timeout = sys.argv[1], float(sys.argv[2])
    encoded = Path("/input/manifest.json").read_bytes()
    manifest = json.loads(encoded)
    if hashlib.sha256(encoded).hexdigest() != sys.argv[3]:
        print(
            json.dumps(
                {"status": "infrastructure_error", "output_tail": "Source manifest digest mismatch"}
            )
        )
        return
    for relative, expected in manifest.items():
        content = (Path("/input/src") / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            print(
                json.dumps(
                    {
                        "status": "infrastructure_error",
                        "output_tail": f"Source digest mismatch: {relative}",
                    }
                )
            )
            return
        target_path = Path("/workspace/src") / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
    with Path("/workspace/output").open("w+b") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "pytest", target, "-q", "--tb=line", "-p", "no:cacheprovider"],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            status = (
                "passed"
                if code == 0
                else "infrastructure_error"
                if code in (3, 4, 5)
                else "test_failure"
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            status = "workload_timeout"
        output.seek(0, os.SEEK_END)
        output.seek(max(0, output.tell() - 2000))
        tail = output.read().decode(errors="replace")
    print(json.dumps({"status": status, "output_tail": tail}))


if __name__ == "__main__":
    main()
