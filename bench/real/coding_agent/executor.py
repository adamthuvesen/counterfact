"""Docker execution boundary for model-generated Python."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

CONTAINER_ROOT = Path(__file__).resolve().parents[1] / "container"
SUPPORTED_PLATFORMS = ("linux/arm64", "linux/amd64")
MAX_CONTAINER_OUTPUT_BYTES = 65_536


def verifier_image() -> str:
    digest = hashlib.sha256()
    for name in ("Dockerfile", "verify.py"):
        digest.update(name.encode())
        digest.update((CONTAINER_ROOT / name).read_bytes())
    return f"counterfact-verifier:{digest.hexdigest()[:20]}"


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["passed", "test_failure", "workload_timeout", "infrastructure_error"]
    output_tail: str
    failure_reason: Literal["output_limit"] | None = None
    source_digest: str | None = None
    image_id: str | None = None
    platform: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class ExecutorUnavailable(RuntimeError):
    """The isolated verifier cannot start safely."""


class _OutputLimitExceeded(RuntimeError):
    """The container exceeded the host capture limit."""


def _capture(command: list[str], *, timeout_s: int) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    output = bytearray()
    deadline = time.monotonic() + timeout_s
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(timeout=remaining):
                    raise subprocess.TimeoutExpired(command, timeout_s)
                chunk = os.read(
                    process.stdout.fileno(), min(8192, MAX_CONTAINER_OUTPUT_BYTES - len(output) + 1)
                )
                if not chunk:
                    selector.unregister(process.stdout)
                    break
                if len(output) + len(chunk) > MAX_CONTAINER_OUTPUT_BYTES:
                    raise _OutputLimitExceeded(
                        f"Verifier output exceeded {MAX_CONTAINER_OUTPUT_BYTES} bytes"
                    )
                output.extend(chunk)
        code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        return subprocess.CompletedProcess(command, code, stdout=bytes(output))
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=15)
        finally:
            process.stdout.close()


def _stage_source(source: Path, destination: Path) -> str:
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    destination.chmod(0o755)
    staged = destination / "src"
    staged.mkdir(mode=0o755)
    staged.chmod(0o755)
    manifest = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"source symlink cannot be staged: {path}")
        if not path.is_file():
            continue
        content = path.read_bytes()
        target = staged / relative
        parent = staged
        for part in relative.parts[:-1]:
            parent /= part
            parent.mkdir(exist_ok=True)
            parent.chmod(0o755)
        target.write_bytes(content)
        target.chmod(0o644)
        manifest[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    encoded = json.dumps(manifest, sort_keys=True).encode()
    manifest_path = destination / "manifest.json"
    manifest_path.write_bytes(encoded)
    manifest_path.chmod(0o644)
    return hashlib.sha256(encoded).hexdigest()


class DockerExecutor:
    def __init__(self) -> None:
        self.image_id: str | None = None
        self.platform: str | None = None
        self._ready = False

    def preflight(self) -> None:
        if self._ready:
            return
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            platform = result.stdout.strip().replace("aarch64", "arm64").replace("x86_64", "amd64")
            if result.returncode or platform not in SUPPORTED_PLATFORMS:
                raise ExecutorUnavailable("Docker must run Linux arm64 or amd64 containers")
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    verifier_image(),
                    "--format",
                    "{{.Id}} {{.Os}}/{{.Architecture}}",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode or not result.stdout.strip().endswith(platform):
                raise ExecutorUnavailable(
                    f"Docker verifier unavailable. Build with: docker build "
                    f"-t {verifier_image()} {CONTAINER_ROOT}"
                )
            self.image_id = result.stdout.split()[0]
            self.platform = platform
            name = f"counterfact-{uuid.uuid4().hex}"
            probe_error: BaseException | None = None
            try:
                probe = _capture([*self._command(name), self.image_id, "--preflight"], timeout_s=60)
                expected = {
                    "python": "3.13.11",
                    "pytest": "8.3.5",
                    "uid": 65534,
                    "locale": "C.UTF-8",
                    "tz": "UTC",
                }
                if probe.returncode or json.loads(probe.stdout) != expected:
                    tail = probe.stdout[-2000:].decode(errors="replace")
                    raise ExecutorUnavailable(f"Docker verifier runtime probe failed: {tail}")
            except BaseException as exc:
                probe_error = exc
                raise
            finally:
                try:
                    self._remove(name)
                except BaseException as exc:
                    if probe_error is not None:
                        probe_error.add_note(str(exc))
                    else:
                        raise
            self._ready = True
        except (OSError, ValueError, subprocess.TimeoutExpired, _OutputLimitExceeded) as exc:
            raise ExecutorUnavailable(f"Docker verifier unavailable: {exc}") from exc

    def _command(self, name: str) -> list[str]:
        assert self.platform is not None
        return [
            "docker",
            "run",
            "--name",
            name,
            "--platform",
            self.platform,
            "--network=none",
            "--log-driver=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--memory=256m",
            "--memory-swap=256m",
            "--cpus=1",
            "--ulimit",
            "nofile=128:128",
            "--tmpfs",
            "/workspace:rw,nosuid,nodev,size=32m,mode=1777",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m,mode=1777",
        ]

    def _remove(self, name: str) -> None:
        try:
            result = subprocess.run(
                ["docker", "rm", "--force", name], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutorUnavailable(
                f"Container cleanup failed: {exc}. Run: docker rm --force {name}"
            ) from exc
        if result.returncode:
            raise ExecutorUnavailable(
                f"Container cleanup failed: {result.stderr}. Run: docker rm --force {name}"
            )

    def run(
        self, source: Path, tests: Path, *, target: str, timeout_s: int = 30
    ) -> ExecutionResult:
        self.preflight()
        assert self.image_id is not None
        source_digest = None
        snapshot = tempfile.TemporaryDirectory(prefix="counterfact-source-", delete=False)
        staged = Path(snapshot.name)
        retain_snapshot = False
        cancellation: BaseException | None = None
        try:
            try:
                source_digest = _stage_source(source, staged)
            except (OSError, ValueError) as exc:
                return ExecutionResult(
                    status="infrastructure_error",
                    output_tail=str(exc),
                    image_id=self.image_id,
                    platform=self.platform,
                )
            name = f"counterfact-{uuid.uuid4().hex}"
            command = [
                *self._command(name),
                "--mount",
                f"type=bind,src={staged.resolve()},dst=/input,readonly",
                "--mount",
                f"type=bind,src={tests.resolve()},dst=/workspace/{target},readonly",
                self.image_id,
                target,
                str(timeout_s),
                source_digest,
            ]
            try:
                process = _capture(command, timeout_s=timeout_s + 60)
                text = process.stdout.decode(errors="replace")
                if process.returncode:
                    result = ExecutionResult(
                        status="infrastructure_error", output_tail=text[-2000:]
                    )
                else:
                    try:
                        result = ExecutionResult.model_validate_json(text)
                    except ValueError:
                        result = ExecutionResult(
                            status="infrastructure_error",
                            output_tail=f"Invalid verifier response: {text[-2000:]}",
                        )
            except _OutputLimitExceeded as exc:
                result = ExecutionResult(
                    status="test_failure", failure_reason="output_limit", output_tail=str(exc)
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                result = ExecutionResult(status="infrastructure_error", output_tail=str(exc))
            except BaseException as exc:
                cancellation = exc
                raise
            finally:
                try:
                    self._remove(name)
                except BaseException as exc:
                    retain_snapshot = True
                    message = f"{exc}. Staged source retained at {staged}"
                    if cancellation is not None:
                        cancellation.add_note(message)
                    elif isinstance(exc, ExecutorUnavailable):
                        result = ExecutionResult(status="infrastructure_error", output_tail=message)
                    else:
                        exc.add_note(message)
                        raise
        except BaseException as exc:
            cancellation = exc
            raise
        finally:
            if not retain_snapshot:
                try:
                    snapshot.cleanup()
                except OSError as exc:
                    if cancellation is not None:
                        cancellation.add_note(f"Cannot remove staged source {staged}: {exc}")
                    else:
                        raise
        return result.model_copy(
            update={
                "image_id": self.image_id,
                "platform": self.platform,
                "source_digest": source_digest,
            }
        )


def build_verifier_image() -> None:
    subprocess.run(
        ["docker", "build", "-t", verifier_image(), str(CONTAINER_ROOT)],
        check=True,
    )
