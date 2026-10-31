"""Only named owner files import process, network, or paid-model modules."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY_SKILL = ".agents/skills/verify-counterfact"
OWNERS = {
    "bench/real/coding_agent/executor.py",
    "bench/real/coding_agent/fixtures.py",
    "bench/real/coding_agent/llm.py",
    "bench/real/container/verify.py",
    "scripts/check_container.py",
    "scripts/check_wheel.py",
    "scripts/wheel_smoke.py",
    f"{VERIFY_SKILL}/drives.py",
    f"{VERIFY_SKILL}/verify",
}
BANNED_MODULES = {
    # Process calls
    "subprocess",
    "pty",
    "asyncio.subprocess",
    # Network clients
    "socket",
    "http.client",
    "urllib.request",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    # Paid model and cloud SDKs
    "litellm",
    "openai",
    "anthropic",
    "boto3",
    "botocore",
}
BANNED_ATTRIBUTES = {
    "os": {"system", "popen", "posix_spawn", "posix_spawnp"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell", "subprocess"},
}
BANNED_PREFIXES = {"os": ("exec", "spawn")}


def _banned_module(name: str) -> bool:
    return any(name == banned or name.startswith(f"{banned}.") for banned in BANNED_MODULES)


def _banned_attribute(module: str, attribute: str) -> bool:
    return attribute in BANNED_ATTRIBUTES.get(module, set()) or attribute.startswith(
        BANNED_PREFIXES.get(module, ("\0",))
    )


def banned_uses(source: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hits += [(node.lineno, a.name) for a in node.names if _banned_module(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if _banned_module(node.module):
                hits.append((node.lineno, node.module))
            hits += [
                (node.lineno, f"{node.module}.{a.name}")
                for a in node.names
                if _banned_attribute(node.module, a.name)
                or _banned_module(f"{node.module}.{a.name}")
            ]
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and _banned_attribute(node.value.id, node.attr)
        ):
            hits.append((node.lineno, f"{node.value.id}.{node.attr}"))
    return sorted(hits)


def _checked_files() -> list[Path]:
    globs = ["src/**/*.py", "bench/**/*.py", "scripts/*.py", f"{VERIFY_SKILL}/*.py"]
    return sorted({*(p for g in globs for p in ROOT.glob(g)), ROOT / VERIFY_SKILL / "verify"})


def test_only_owner_files_import_process_network_or_paid_modules() -> None:
    hits = []
    for path in _checked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in OWNERS:
            continue
        hits += [f"{relative}:{line} {name}" for line, name in banned_uses(path.read_text())]
    assert not hits, "Move these calls into an owner file listed in OWNERS: " + ", ".join(hits)


def test_detector_finds_each_import_form() -> None:
    source = (
        "import subprocess\n"
        "import urllib.request\n"
        "from os import system, execvp\n"
        "from asyncio import create_subprocess_exec\n"
        "from urllib import request\n"
        "import os\n"
        "os.spawnlp\n"
        "import litellm.utils\n"
    )
    assert [line for line, _ in banned_uses(source)] == [1, 2, 3, 3, 4, 5, 7, 8]
    assert banned_uses("import os\nimport json\nos.path.join\nfrom urllib.parse import quote") == []
