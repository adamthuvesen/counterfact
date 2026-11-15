"""Real Docker checks for generated-code execution. No provider calls."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bench.real.coding_agent.executor import DockerExecutor
from bench.real.coding_agent.fixtures import (
    EASY_FIXTURES,
    fixtures_by_id,
    run_pytest,
    run_pytest_hidden,
    run_pytest_public,
)
from tests.fixtures.coding_agent import REFERENCES_ROOT

pytestmark = pytest.mark.container


@pytest.fixture(scope="module")
def executor() -> DockerExecutor:
    result = DockerExecutor()
    result.preflight()
    return result


def _probe(tmp_path: Path, source: str) -> tuple[Path, Path]:
    src, tests = tmp_path / "src", tmp_path / "tests"
    src.mkdir()
    tests.mkdir()
    (src / "probe.py").write_text(source)
    (tests / "test_probe.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
        "def test_probe():\n    import probe\n"
    )
    return src, tests


def test_generated_code_cannot_reach_host_or_mutate_verifier(
    executor: DockerExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = tmp_path / "host-canary"
    canary.write_text("unchanged")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    source = f"""
import os
import socket
from pathlib import Path
assert os.getuid() == 65534
assert os.getenv('ANTHROPIC_API_KEY') is None
assert not Path({str(canary)!r}).exists()
assert not Path('/var/run/docker.sock').exists()
assert os.environ['HOME'] == '/workspace'
assert not Path('/workspace/.aws/credentials').exists()
assert not Path('/workspace/tests_hidden').exists()
for path in ('/workspace/tests/test_probe.py', '/opt/counterfact/verify.py'):
    try:
        Path(path).write_text('tampered')
    except OSError:
        pass
    else:
        raise AssertionError('verifier is writable')
try:
    socket.create_connection(('1.1.1.1', 53), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError('network is reachable')
assert not Path('/workspace/sentinel').exists()
Path('/workspace/sentinel').write_text('private')
"""
    src, tests = _probe(tmp_path, source)
    for _ in range(2):
        result = executor.run(src, tests, target="tests")
        assert result.status == "passed", result
        assert result.image_id and result.platform in ("linux/arm64", "linux/amd64")
    assert canary.read_text() == "unchanged"
    assert "def test_probe" in (tests / "test_probe.py").read_text()
    assert not (tmp_path / "sentinel").exists()


def test_workload_timeout_is_distinct_from_runtime_failure(
    executor: DockerExecutor, tmp_path: Path
) -> None:
    src, tests = _probe(tmp_path, "while True:\n    pass\n")
    assert executor.run(src, tests, target="tests", timeout_s=1).status == "workload_timeout"
    assert executor.run(src, tmp_path / "missing", target="tests").status == "infrastructure_error"


def test_generated_output_is_bounded(executor: DockerExecutor, tmp_path: Path) -> None:
    src, tests = _probe(tmp_path, "print('x' * 4_000_000)\nraise ValueError('failure')\n")
    result = executor.run(src, tests, target="tests")
    assert result.status == "test_failure"
    assert len(result.output_tail) <= 2000


def test_no_collected_tests_is_infrastructure_failure(
    executor: DockerExecutor, tmp_path: Path
) -> None:
    src, tests = _probe(tmp_path, "pass\n")
    (tests / "test_probe.py").write_text("# no verifier tests\n")
    result = executor.run(src, tests, target="tests")
    assert result.status == "infrastructure_error"


@pytest.mark.parametrize("fixture_id", ["csv_dedupe", "streaming_watermark_dedupe"])
def test_trusted_buggy_and_reference_host_container_parity(
    executor: DockerExecutor, tmp_path: Path, fixture_id: str
) -> None:
    fixture = fixtures_by_id(fixture_id)[0]
    workspace = tmp_path / fixture_id
    shutil.copytree(fixture.root, workspace)
    references = {
        "csv_dedupe": "_dedupe_reference.py",
        "streaming_watermark_dedupe": "_watermark_dedupe_reference.py",
    }
    for reference in (False, True, False, True):
        source_path = (
            (REFERENCES_ROOT / references[fixture_id]) if reference else fixture.source_path
        )
        (workspace / "src" / fixture.source_relpath).write_bytes(source_path.read_bytes())
        host_public, _ = run_pytest_public(workspace)
        host_hidden, _ = run_pytest_hidden(workspace)
        public = executor.run(
            workspace / "src", fixture.root / "tests_public", target="tests_public"
        )
        hidden = executor.run(
            workspace / "src", fixture.root / "tests_hidden", target="tests_hidden"
        )
        assert public.passed == host_public is True, public
        assert public.source_digest and hidden.source_digest == public.source_digest
        assert hidden.passed == host_hidden == reference
        assert hidden.status in ("passed", "test_failure")


def test_trusted_single_suite_host_container_parity(
    executor: DockerExecutor, tmp_path: Path
) -> None:
    fixture = EASY_FIXTURES[0]
    workspace = tmp_path / "single"
    shutil.copytree(fixture.root, workspace)
    for source in (
        None,
        "import re\ndef normalize_name(name):\n"
        "    return re.sub(r'\\s+', ' ', name.strip().lower())\n",
    ):
        if source:
            (workspace / "src" / fixture.source_relpath).write_text(source)
        host, _ = run_pytest(workspace)
        result = executor.run(workspace / "src", fixture.root / "tests", target="tests")
        assert result.passed == host == (source is not None)


def test_direct_container_stdout_is_bounded_and_container_is_removed(
    executor: DockerExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    source = (
        "import os\nfd = os.open('/proc/1/fd/1', os.O_WRONLY)\n"
        "os.write(fd, b'x' * 2_097_152)\nos.close(fd)\n"
    )
    src, tests = _probe(tmp_path, source)
    names = []
    command = executor._command

    def observed_command(name: str) -> list[str]:
        names.append(name)
        args = command(name)
        assert "--log-driver=none" in args
        return args

    monkeypatch.setattr(executor, "_command", observed_command)
    result = executor.run(src, tests, target="tests")
    assert result.status == "test_failure", result
    assert result.failure_reason == "output_limit", result
    assert len(result.output_tail) <= 2000
    assert result.image_id and result.platform
    assert len(names) == 1
    remaining = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={names[0]}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert remaining.stdout == ""


def test_corrupt_source_handoff_is_infrastructure_failure(
    executor: DockerExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench.real.coding_agent import executor as executor_module

    src, tests = _probe(tmp_path, "value = 'complete source'\n")
    stage = executor_module._stage_source

    def corrupt(source: Path, destination: Path) -> str:
        digest = stage(source, destination)
        (destination / "src" / "probe.py").write_text("from __future__ import\n")
        return digest

    monkeypatch.setattr(executor_module, "_stage_source", corrupt)
    result = executor.run(src, tests, target="tests")
    assert result.status == "infrastructure_error", result
    assert "Source digest mismatch: probe.py" in result.output_tail
    assert result.source_digest


@pytest.mark.parametrize(
    "fixture_id,reference_name",
    [
        ("csv_dedupe", "_dedupe_reference.py"),
        ("streaming_watermark_dedupe", "_watermark_dedupe_reference.py"),
    ],
)
def test_selected_fixture_cannot_import_trusted_reference(
    executor: DockerExecutor, tmp_path: Path, fixture_id: str, reference_name: str
) -> None:
    from bench.real.coding_agent.fixtures import snapshot_fixture

    fixture = fixtures_by_id(fixture_id)[0]
    workspace = snapshot_fixture(fixture, tmp_path)
    reference = (REFERENCES_ROOT / reference_name).read_text()
    assertions = (
        "\nimport importlib.util\nfrom pathlib import Path\n"
        f"assert importlib.util.find_spec({Path(reference_name).stem!r}) is None\n"
        f"assert not Path('/input/src/{reference_name}').exists()\n"
        f"assert not Path('/workspace/src/{reference_name}').exists()\n"
    )
    (workspace / "src" / fixture.source_relpath).write_text(reference + assertions)
    result = executor.run(workspace / "src", fixture.root / "tests_public", target="tests_public")
    assert result.status == "passed", result


def test_host_cancellation_removes_running_container_before_source_cleanup(
    executor: DockerExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import signal
    import subprocess

    from bench.real.coding_agent import executor as executor_module

    src, tests = _probe(tmp_path, "import time\ntime.sleep(20)\n")
    names = []
    running_states = []
    staged_paths = []
    remove = executor._remove
    stage = executor_module._stage_source

    def record_stage(source: Path, destination: Path) -> str:
        staged_paths.append(destination)
        return stage(source, destination)

    def remove_after_cancel(name: str) -> None:
        assert staged_paths[0].is_dir()
        names.append(name)
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
        )
        running_states.append((state.returncode, state.stdout.strip()))
        remove(name)

    def cancel(signum, frame) -> None:
        raise KeyboardInterrupt("bounded cancellation probe")

    monkeypatch.setattr(executor_module, "_stage_source", record_stage)
    monkeypatch.setattr(executor, "_remove", remove_after_cancel)
    previous = signal.signal(signal.SIGALRM, cancel)
    signal.setitimer(signal.ITIMER_REAL, 3)
    try:
        with pytest.raises(KeyboardInterrupt, match="bounded cancellation probe"):
            executor.run(src, tests, target="tests")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert len(names) == 1
    assert running_states == [(0, "true")]
    remaining = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={names[0]}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert remaining.stdout == ""
    assert not staged_paths[0].exists()
