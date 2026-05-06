# Trace Forensics Gallery

This gallery is synthetic, illustrative trace-forensics material. It is here to show how `counterfact diagnose` and `counterfact compare` feel on common per-run questions. It is not benchmark evidence, not a leaderboard, and not a population-level model ranking.

Regenerate the committed fixtures:

```bash
uv run python examples/trace-forensics/make_gallery.py
```

## Wrong Model Choice

```bash
uv run counterfact diagnose examples/trace-forensics/runs/syn-000000.json \
  --runs-dir examples/trace-forensics/runs \
  --decision-type model_call \
  --html /tmp/counterfact-wrong-model.html
```

## Bad Tool Choice

```bash
uv run counterfact diagnose examples/trace-forensics/runs/syn-000000.json \
  --runs-dir examples/trace-forensics/runs \
  --decision-type tool_call \
  --html /tmp/counterfact-bad-tool.html
```

## Missed Retry

```bash
uv run counterfact diagnose examples/trace-forensics/runs/syn-000000.json \
  --runs-dir examples/trace-forensics/runs \
  --decision-type retry \
  --html /tmp/counterfact-missed-retry.html
```

## Stopped Too Early

```bash
uv run counterfact compare examples/trace-forensics/stopped-early/pass.json \
  examples/trace-forensics/stopped-early/fail.json
```

## Unsupported Counterfactual

This corpus has mixed outcomes but only one observed model arm, so the right answer is support guidance rather than a fake effect.

```bash
uv run counterfact diagnose examples/trace-forensics/single-arm-model/single-arm-000000.json \
  --runs-dir examples/trace-forensics/single-arm-model \
  --decision-type model_call
```

## Single-Class Support Refusal

This uses the committed degenerate anchor corpus. It should remain `unidentified` with `broaden_arm_support` guidance.

```bash
uv run counterfact diagnose bench/real/single_class_refusal/real-csv_dedupe-000000.json \
  --runs-dir bench/real/single_class_refusal
```

## Pass/Fail Trace Diff

```bash
uv run counterfact compare examples/trace-forensics/runs/syn-000003.json \
  examples/trace-forensics/runs/syn-000000.json \
  --runs-dir examples/trace-forensics/runs \
  --focal right
```
