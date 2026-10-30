---
name: verify-counterfact
description: "Drives the real counterfact CLI (diagnose, explain, intervene, ingest, analyze, export-runs) against committed corpora in temp dirs, with no provider calls. Use to run, launch, drive, or verify counterfact, or to prove a change to it works."
---

# Verify counterfact

`verify` runs the installed `.venv/bin/counterfact` console script the way a user does. It uses the committed `bench/real/smoke_mixed_outcome` corpus and the Claude Agent SDK adapter fixture. Every output goes to a temp dir with a temp HOME. The child environment holds no API keys, and no drive reaches `bench real`, Docker, or a provider.

## Run

From the repo root, after `make install`:

- `make verify` after a change. It runs the drives whose paths match files changed against the merge base with `main`, plus uncommitted and untracked files. With no match it runs the smoke drive.
- `make verify ARGS=all` before handing over larger work. `make slow` (the full gate) also runs `all` and `control` at the end.
- `make verify ARGS=list` shows each drive, what it proves, and the paths it covers.
- `make verify ARGS="diagnose intervene"` runs named drives.
- `make verify ARGS=control` re-drives the smoke drive against a wrong identifiability. It exits 0 only when it observes FAIL.

You can also call `.agents/skills/verify-counterfact/verify <args>` directly.

## Reading results

Each drive prints `PASS|FAIL <drive> <seconds>s <detail>`. A FAIL detail names the expected and the actual value, or the command's exit code and the tail of its output. The run ends with `evidence: .verify/runs/<utc>-<sha7>[-dirty]/`, which holds `preflight.log`, one `<drive>.<step>.json` per command (argv, env, exit, stdout, stderr), `explain.html`, and `summary.json`. The helper does not retry, so it never reports `FLAKY`. A FAIL is a FAIL.

Preflight refuses to run when the locked environment does not resolve, when `counterfact` imports from outside this checkout, when the installed version differs from `pyproject.toml`, or when an `*_API_KEY` would reach a drive. Any drive fails if `.counterfact/approved` changes while it runs.

## Limits

- Real-agent trace generation (`counterfact bench real`) makes paid provider calls. It stays behind Adam's approval and outside `verify`. See `bench/real/README.md`.
- The Docker verifier boundary is proven by `make check-container`, not by these drives.
- Live tracers (`ClaudeAgentTracer`, `CounterfactSpanProcessor`) need a running SDK and are covered by unit tests only.

## Gotchas

- `counterfact analyze corpus` exits 1 when a corpus is not support-ready. That is a verdict, not a crash. The ingest drive expects exit 1 on its 2-trace corpus.
- Corpus writers leave `.counterfact-write.lock` in the output dir on purpose: `flock` needs a stable file. Ignore dotfiles when you compare a corpus listing.
- `intervene` needs `--decision-id` or `--step`. The smoke trace has no temperature decision, so the docs' `temperature=0.7` example does not apply to it.
