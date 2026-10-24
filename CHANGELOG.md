# Changelog

This project follows [Semantic Versioning](https://semver.org/). Until 1.0, a minor release can change the public API.

## [0.1.0] - Unreleased

First public release.

### Added

- `intervene`: one-decision counterfactual contrasts with an identifiability label (`identified`, `bounded`, `unidentified`) that is checked against the logged propensities and support.
- Cross-fitted AIPW estimates with influence-function 95% CIs, plus g-formula and Hajek IPW cross-checks.
- Manski no-assumption bounds when both arms occur but the log does not prove how they were assigned.
- `evaluate_policy` and `counterfact evaluate-policy`: sequential doubly robust evaluation of deterministic policies over several decisions, with ICE and SNIS cross-checks and an effective sample size.
- `counterfact bench suite`: eight single-decision structural causal models and a sequential agent with exact true effects, scored over many seeds.
- `diagnose`, `explain`, `compare`, and `analyze` commands, with a self-contained HTML report.
- Adapters and live tracers for the Claude Agent SDK and the OpenAI Agents SDK, and a generic JSONL ingester.
- A real-agent harness with a routed-trial design that produces its own ground truth. It is behind an approval gate and a budget cap.
- A technical report with figures: [docs/report.md](docs/report.md).

### Requirements

- Python 3.13 or later.
