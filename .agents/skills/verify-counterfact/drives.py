"""The verify-counterfact drive table: each drive runs the real CLI in a temp HOME."""

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VENV_BIN = ROOT / ".venv" / "bin"
CLI = VENV_BIN / "counterfact"
SMOKE_CORPUS = ROOT / "bench" / "real" / "smoke_mixed_outcome"
FOCAL = SMOKE_CORPUS / "real-streaming_watermark_dedupe-000000.json"
CLAUDE_FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "claude_agent_sdk" / "minimal.jsonl"
APPROVAL = ROOT / ".counterfact" / "approved"
SCRATCH = Path(tempfile.mkdtemp(prefix="verify-counterfact-"))
SECRET_NAME = re.compile("KEY|TOKEN|SECRET|PASSWORD")
SHARED_PATHS = [
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".agents/skills/verify-counterfact/*",
]


class DriveFailure(Exception):
    pass


@dataclass(frozen=True)
class Drive:
    proves: str
    paths: list[str]
    run: Callable[[Run], str]
    timeout: float


class Run:
    def __init__(self, run_dir: Path, name: str) -> None:
        self.run_dir = run_dir
        self.name = name
        self.home = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=SCRATCH))
        self.env = {
            "PATH": f"{VENV_BIN}:/usr/bin:/bin",
            "HOME": str(self.home),
            "TZ": "UTC",
            "XDG_CONFIG_HOME": str(self.home / "config"),
            "XDG_CACHE_HOME": str(self.home / "cache"),
            "XDG_DATA_HOME": str(self.home / "data"),
            "NO_COLOR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.step = 0

    def cli(self, *args: str, expect_exit: int = 0) -> subprocess.CompletedProcess[str]:
        argv = [str(CLI), *args]
        proc = subprocess.run(
            argv,
            cwd=self.home,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            start_new_session=True,
        )
        self.step += 1
        env = {k: "<redacted>" if SECRET_NAME.search(k) else v for k, v in self.env.items()}
        record = {
            "argv": argv,
            "env": env,
            "exit": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        (self.run_dir / f"{self.name}.{self.step}.json").write_text(json.dumps(record, indent=2))
        if proc.returncode != expect_exit:
            tail = (proc.stderr or proc.stdout).strip()[-300:]
            raise DriveFailure(
                f"`counterfact {' '.join(args[:2])}` expected exit {expect_exit}, "
                f"got {proc.returncode}: {tail}"
            )
        return proc

    def cli_json(self, *args: str) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(self.cli(*args).stdout)
        return payload


def check(actual: object, expected: object, what: str) -> None:
    if actual != expected:
        raise DriveFailure(f"{what}: expected {expected!r}, got {actual!r}")


EXPECTED_TOP = {
    "decision_id": "d-000000-model-1",
    "chosen_action": "large",
    "identifiability": "identified",
}
# The control passes only when the drive fails on exactly this planted expectation.
CONTROL_EXPECTED = {**EXPECTED_TOP, "identifiability": "unidentified"}
CONTROL_FAILURE = f"top diagnosed decision: expected {CONTROL_EXPECTED!r}, got "


def drive_diagnose(run: Run, expected: dict[str, str] | None = None) -> str:
    report = run.cli_json(
        "diagnose", str(FOCAL), "--runs-dir", str(SMOKE_CORPUS), "--top-k", "3", "--json"
    )
    check(report["corpus_size"], 120, "smoke corpus size")
    top = report["entries"][0]
    check({k: top[k] for k in EXPECTED_TOP}, expected or EXPECTED_TOP, "top diagnosed decision")
    html = run.home / "explain.html"
    run.cli("explain", str(FOCAL), "--runs-dir", str(SMOKE_CORPUS), "--output", str(html))
    text = html.read_text() if html.exists() else ""
    if (
        "counterfact explain — real-streaming_watermark_dedupe-000000" not in text
        or "d-000000-model-1" not in text
    ):
        raise DriveFailure("explain HTML is missing its title or the model_call decision card")
    shutil.copy(html, run.run_dir / "explain.html")
    return "diagnose ranks d-000000-model-1 identified on 120 runs; explain writes its HTML report"


def drive_intervene(run: Run) -> str:
    common = [str(FOCAL), "--runs-dir", str(SMOKE_CORPUS), "--json"]
    model = run.cli_json(
        "intervene", *common, "--decision-id", "d-000000-model-1", "--set", "model_choice=small"
    )
    check(model["identifiability"], "identified", "model_choice=small identifiability")
    delta = model["outcome_delta"]
    if not (delta["ci_low"] <= delta["point"] <= delta["ci_high"] < 0):
        raise DriveFailure(
            f"model_choice=small delta should be negative with a CI around it, got {delta}"
        )
    retry = run.cli_json(
        "intervene", *common, "--decision-id", "d-000000-retry", "--set", "retry_policy=no_retry"
    )
    check(
        (retry["identifiability"], retry["outcome_delta"]),
        ("unidentified", None),
        "retry_policy=no_retry result",
    )
    check(retry["next_step"]["action"], "broaden_arm_support", "unsupported arm next step")
    return (
        f"model edit is identified (delta {delta['point']:+.3f}); "
        "an unsupported retry arm stays unidentified with no number"
    )


def drive_ingest(run: Run) -> str:
    corpus = run.home / "corpus"
    run.cli("ingest", "claude-agent-sdk", str(CLAUDE_FIXTURE), "--output-dir", str(corpus))
    check(
        sorted(p.name for p in corpus.iterdir() if not p.name.startswith(".")),
        ["ingest-receipt.json", "sess-fail-002.json", "sess-pass-001.json"],
        "ingested files",
    )
    readiness = run.cli("analyze", "corpus", str(corpus), expect_exit=1).stdout
    if "support_ready: False" not in readiness:
        raise DriveFailure("analyze on a 2-trace corpus did not report support_ready: False")
    report = run.cli_json(
        "diagnose", str(corpus / "sess-fail-002.json"), "--runs-dir", str(corpus), "--json"
    )
    check(
        (report["run_id"], report["outcome"], report["corpus_size"]),
        ("sess-fail-002", "fail", 2),
        "diagnose on ingested corpus",
    )
    out = run.home / "runs.parquet"
    exported = run.cli(
        "export-runs", str(corpus), "--to", "runrecord-parquet", "--output", str(out)
    ).stdout
    if "rows: 2" not in exported or not out.exists():
        raise DriveFailure(f"export-runs did not write 2 rows: {exported.strip()[-200:]}")
    return (
        "Claude SDK fixture ingests to 2 runs, analyze refuses readiness (exit 1), "
        "diagnose reads the failed run, and export-runs writes 2 parquet rows"
    )


DRIVES = {
    "diagnose": Drive(
        "diagnose and explain on the smoke corpus rank the identified model_call decision",
        ["src/counterfact/**", "bench/real/smoke_mixed_outcome/**"],
        drive_diagnose,
        30,
    ),
    "intervene": Drive(
        "intervene: model_choice is identified and negative; an unsupported arm gets no number",
        [
            "src/counterfact/intervene/**",
            "src/counterfact/analysis/**",
            "src/counterfact/dag/**",
            "src/counterfact/stats.py",
            "src/counterfact/cli/**",
        ],
        drive_intervene,
        30,
    ),
    "ingest": Drive(
        "ingest claude-agent-sdk, analyze (exit 1, not ready), diagnose, and export-runs work",
        [
            "src/counterfact/adapters/**",
            "src/counterfact/ingest.py",
            "src/counterfact/corpus_io.py",
            "src/counterfact/runrecord_export.py",
            "src/counterfact/corpus_analyzer/**",
            "src/counterfact/cli/**",
            "tests/fixtures/adapters/**",
        ],
        drive_ingest,
        30,
    ),
}
SMOKE = "diagnose"
