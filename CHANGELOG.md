# Changelog

All notable changes to EquiML are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 1.1.0-dev

### Added
- **Fairness policy-as-code.** New `fairness.yaml` schema (RFC 0001),
  `equiml/policy.py` evaluator, `--policy` flag on `equiml audit`, and a
  new `equiml verify <audit.json> --policy <fairness.yaml>` subcommand
  for re-checking historical artefacts without retraining.
- Documented exit codes: `0` success, `2` data error, `3` policy gate
  breached, `4` policy schema error.
- `examples/fairness.yaml` sample policy for the Adult-census demo.
- `pyyaml` runtime dependency.
- `docs/rfcs/0001-policy-as-code.md` design doc.

## [Unreleased] — 1.0.1

### Fixed
- `predict_proba` on the fairness-mitigated model no longer averages
  predictor probabilities (which is incorrect for `ExponentiatedGradient`).
  It now raises `NotImplementedError` with a clear message.
- `cross_validate` no longer silently strips the fairness constraint when
  called on a mitigator. It raises `NotImplementedError`.
- CLI now audits each `--sensitive` feature individually instead of only
  the first. Per-feature metrics live under `metrics["per_sensitive"]`.
- Substring collision: a feature named `race` no longer matches
  `racing_score` during sensitive-column detection. Encoded columns are
  matched by exact prefix.

### Changed
- Bias-assessment banner replaced with a numeric reading and a caveat
  about jurisdictional thresholds. No more LOW/MODERATE/SIGNIFICANT/HIGH
  bands.
- `equiml/data.py` slimmed: text/image/ARFF/SMOTE/PCA/poly-feature paths
  removed to match the README's scope (tabular classification only).
- README: added comparison section vs fairlearn/aif360/Aequitas,
  governance section (EU AI Act, ISO/IEC 42001, NIST AI RMF),
  "what this does not do" section.

### Added
- `_meta` block in audit JSON output (versions, seed, args).
- `examples/adult_census_audit.py` — runnable end-to-end demo.
- `SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`.
- Dependabot config for weekly pip and monthly action updates.
- New tests: CLI smoke, multi-sensitive contract, substring-collision
  regression, `predict_proba` contract, `cross_validate` contract,
  one adversarial case (perfect proxy).

## [1.0.0] — 2026-04-19

### Changed
- Polish for public release: added plotly dependency, updated LICENSE
  year, tightened README scope.

## [0.2.0] — 2026-04 (pre-release refactor)

### Changed
- Restructured EquiML from a broad ML framework into a focused CLI
  audit tool. Removed scope creep around training pipelines, model
  registries, and non-tabular modalities.

## Roadmap

The next two quarters, in rough priority order. Items below `1.2.0`
came out of a stakeholder review of the README Quickstart and are
ranked by how often they were raised across nine reviewer lenses.

- **1.0.x** — bug-fix releases as issues come in.
- **1.1.0** — intersectional fairness analysis (audit at the
  cross-product of two protected attributes).
- **1.1.x** candidates (raised by 3+ reviewer lenses):
  - `--offline` mode that refuses any network call and a bundled small
    sample CSV so the Quickstart is runnable without egress.
  - Dataset SHA-256, model SHA-256, and `git_sha` in the `_meta` block
    so the JSON is auditable evidence, not just a metric snapshot.
  - Published JSON schema (`equiml/schema/audit.v1.json`) with a
    `--schema-version` flag for downstream contract tests.
- **1.2.0** — distribution-shift handling (drift-aware audit, not just
  i.i.d.). Recommend `folktables` ACSIncome over Adult for new work.
- **1.3.0** candidates:
  - `equiml diff` for cross-quarter audit comparison (point at two
    JSON files, render a diff report).
  - `--demo` flag so the Quickstart works from `pip install equiml`
    alone, without cloning the repo.
- **Parked** — MCP server wrapper exposing `audit_dataset` as a tool.
