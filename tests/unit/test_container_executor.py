"""Container lifecycle checks without a Docker dependency."""

import subprocess
from pathlib import Path

import pytest


def test_container_cleanup_failure_is_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench.real.coding_agent import executor as executor_module
    from bench.real.coding_agent.executor import DockerExecutor

    executor = DockerExecutor()
    executor._ready = True
    executor.image_id = "sha256:test-image"
    executor.platform = "linux/arm64"

    monkeypatch.setattr(
        executor_module,
        "_capture",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout=b'{"status":"passed","output_tail":""}'
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1, stderr="daemon unavailable"),
    )
    result = executor.run(tmp_path, tmp_path, target="tests")
    assert result.status == "infrastructure_error"
    assert "docker rm --force counterfact-" in result.output_tail
    assert result.image_id == "sha256:test-image"
    assert result.platform == "linux/arm64"
    import shutil

    retained = Path(result.output_tail.rsplit("Staged source retained at ", 1)[1])
    assert retained.is_dir()
    shutil.rmtree(retained)


def test_stream_capture_stops_at_output_limit_without_a_capture_file() -> None:
    import sys

    from bench.real.coding_agent.executor import _capture, _OutputLimitExceeded

    with pytest.raises(_OutputLimitExceeded, match="Verifier output exceeded"):
        _capture([sys.executable, "-c", "import os; os.write(1, b'x' * 2_097_152)"], timeout_s=5)


def test_stream_capture_timeout_kills_and_reaps_child() -> None:
    import sys

    from bench.real.coding_agent.executor import _capture

    with pytest.raises(subprocess.TimeoutExpired):
        _capture([sys.executable, "-c", "while True: pass"], timeout_s=1)


def test_stream_capture_preserves_small_verifier_response() -> None:
    import sys

    from bench.real.coding_agent.executor import _capture

    result = _capture([sys.executable, "-c", "print('verifier result')"], timeout_s=5)
    assert result.returncode == 0
    assert result.stdout == b"verifier result\n"


def test_source_staging_preserves_actual_bytes_and_distinct_versions(tmp_path: Path) -> None:
    import hashlib
    import json

    from bench.real.coding_agent.executor import _stage_source

    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    path = source / "nested" / "module.py"
    contents = [b"short = True\n", "value = 'caf\u00e9'\n".encode() * 100]
    digests = []
    for index, content in enumerate(contents):
        path.write_bytes(content)
        snapshot = tmp_path / f"snapshot-{index}"
        snapshot.mkdir()
        digests.append(_stage_source(source, snapshot))
        encoded = (snapshot / "manifest.json").read_bytes()
        manifest = json.loads(encoded)
        assert manifest == {"nested/module.py": hashlib.sha256(content).hexdigest()}
        assert digests[-1] == hashlib.sha256(encoded).hexdigest()
        assert (snapshot / "src" / "nested" / "module.py").read_bytes() == content
    assert digests[0] != digests[1]
    assert (tmp_path / "snapshot-0" / "src" / "nested" / "module.py").read_bytes() == contents[0]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cancellation_always_removes_container_before_staging_cleanup(
    cleanup_fails: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from bench.real.coding_agent import executor as executor_module
    from bench.real.coding_agent.executor import DockerExecutor, ExecutorUnavailable

    executor = DockerExecutor()
    executor._ready = True
    executor.image_id = "sha256:test-image"
    executor.platform = "linux/arm64"
    cancellation = KeyboardInterrupt("cancelled during capture")
    staged_paths = []
    calls = []
    stage = executor_module._stage_source

    def record_stage(source: Path, destination: Path) -> str:
        staged_paths.append(destination)
        return stage(source, destination)

    def interrupt(*args, **kwargs):
        raise cancellation

    def remove(name: str) -> None:
        assert staged_paths[0].is_dir()
        calls.append(name)
        if cleanup_fails:
            raise ExecutorUnavailable("docker rm failed")

    monkeypatch.setattr(executor_module, "_stage_source", record_stage)
    monkeypatch.setattr(executor_module, "_capture", interrupt)
    monkeypatch.setattr(executor, "_remove", remove)
    with pytest.raises(KeyboardInterrupt) as interrupted:
        executor.run(tmp_path, tmp_path, target="tests")
    assert interrupted.value is cancellation
    assert len(calls) == 1
    if cleanup_fails:
        assert staged_paths[0].is_dir()
        assert any(
            "docker rm failed" in note and str(staged_paths[0]) in note
            for note in interrupted.value.__notes__
        )
        shutil.rmtree(staged_paths[0])
    else:
        assert not staged_paths[0].exists()
