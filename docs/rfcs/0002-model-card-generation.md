# RFC 0002 — Auto Model-Card Generation

| Field | Value |
|---|---|
| Status | Implemented (1.1.0-dev) |
| Author | Michael Kupermann |
| Date | 2026-04-26 |
| Reviewers | (open for comment) |

## Context

A fairness audit JSON is evidence. It is not a publishable artefact. The thing that ends up on Hugging Face Hub, in a repo's `MODEL_CARD.md`, on a regulator's desk, or in a procurement onboarding pack is a **model card** — a structured document that combines the audit numbers with author-supplied context (intended use, training data, ethical considerations, licence).

Today, every team that uses EquiML hand-writes the model card by copy-pasting numbers from the JSON. That is the gap we close here.

A model card has two halves the audit cannot author for itself:
- **What only the audit knows:** metrics, sensitive features, library versions, random seed, gate results.
- **What only the human knows:** the model's name, what it is for, who can use it, what training data fed it, what risks the team has thought about.

This RFC ships a generator that combines both. The audit half is automatic; the human half is supplied via an optional config YAML. If no config is supplied, the card emits structured TODO placeholders rather than ghostwriting prose the human cannot defend.

## Decision

EquiML 1.1 ships:

1. A new module `equiml/card.py` with `build_card_from_audit()` and `render_markdown()`.
2. A new Jinja template `equiml/model_card_template.md.j2`.
3. A new CLI subcommand `equiml card audit.json --output MODEL_CARD.md` with optional `--policy` and `--config` flags.
4. A sample author-config at `examples/model_card_config.yaml`.

The output is a single markdown file with **Hugging Face-compatible YAML frontmatter** (so it can be uploaded to HF Hub as-is) and a body that follows the Mitchell et al. (2019) section structure: model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, compliance gates, ethical considerations, caveats, and a footer naming the EquiML version that produced it.

## Format choice

### Hugging Face Hub markdown + YAML frontmatter

The de facto standard for shareable models. Renders in HF Hub, in GitHub README previews, and in any markdown viewer.

```markdown
---
language: en
license: mit
tags: [fairness-audited, equiml-1.1.0]
metrics: [accuracy, demographic_parity_difference]
model-index:
  - name: <model name>
    results:
      - task: { type: tabular-classification }
        metrics:
          - { type: accuracy, value: 0.817 }
          - { type: demographic_parity_difference, value: 0.041 }
---

# <model name>

<body>
```

Pro: ubiquitous reach, zero-friction publishing.
Con: HF's own schema is loosely specified; we follow the conventions but do not promise hub-aware features that change between releases.

### Alternatives considered

- **Google Model Card Toolkit (protobuf + JSON + HTML).** Rigorous, almost unused outside Google. Adds a heavy dep for a marginal audience.
- **PDF.** Defeats the purpose. Cannot diff, cannot version, cannot edit.
- **Pure JSON.** Machine-readable but reviewers want prose; we already have JSON in the audit artefact.

**Verdict:** markdown with YAML frontmatter.

## What the generator auto-populates

From `audit.json`:
- `_meta.equiml_version`, `python_version`, `sklearn_version`, `fairlearn_version`
- `_meta.algorithm`, `random_seed`, `dataset_path`, `target`, `sensitive_features`
- All performance metrics (baseline + fair model)
- All fairness metrics (baseline + fair model)
- Per-sensitive-feature metric blocks under `metrics["per_sensitive"]`

From `--policy fairness.yaml` (optional):
- The full policy result table (gate, threshold, observed, severity, status)
- Policy metadata: reviewer, next review date, jurisdiction, legal basis

From the EquiML codebase itself:
- The standard caveats about i.i.d. assumptions, jurisdictional variability, and the conformity-assessment boundary.

## What the human supplies (optional `--config`)

```yaml
model_name: "Credit-decisioning model v3.2"
model_version: "3.2.0"
model_type: "Logistic regression for binary classification"
description: "Predicts whether a credit application meets risk policy."
license: "Proprietary — internal use only"

intended_use:
  primary_use: "Loan approval decisioning for retail credit"
  primary_users: "Loan officers; second-line model risk reviewers"
  out_of_scope:
    - "Fraud detection"
    - "Income estimation"

training_data:
  source: "Internal applications, 2023-2025"
  preprocessing: "Standard scaling on numerical features"
  notes: "Approximately 500k rows after dedup"

ethical_considerations:
  - "Disparate impact across demographic groups was monitored."
  - "Decisions must be reviewable by a human loan officer."

caveats:
  - "Trained on historical data; not validated under economic regime change."

contact: "ml-platform@example.com"
```

If no config is supplied, the generator emits each section with a `TODO(maintainer): ...` placeholder so the card is structurally complete but visibly incomplete. We do not ghostwrite intended-use or ethical-considerations text; those reflect a human's judgement about the model, not the audit's.

## CLI

```bash
# Minimal: card from audit JSON, with TODO placeholders for human fields
equiml card audit.json --output MODEL_CARD.md

# Full: with a policy and an author config
equiml card audit.json \
    --policy fairness.yaml \
    --config model_card_config.yaml \
    --output MODEL_CARD.md
```

Exit codes:
- `0` success
- `2` data error (audit JSON missing or unparseable; config YAML unparseable)
- `4` policy schema error (when `--policy` is passed and invalid)

`equiml card` does **not** evaluate the policy in the sense of failing the run on gate breach — the card surfaces the gate result for the reader. To gate CI on the policy, use `equiml audit --policy ...` or `equiml verify --policy ...`.

## Layout of the rendered card

```markdown
---
<YAML frontmatter>
---

# <Model name>

<description>

## Model details
- Model type, version
- Algorithm, library versions, seed
- Audit date, EquiML version

## Intended use
- Primary use, primary users, out-of-scope (or TODO)

## Factors
- Sensitive features audited

## Metrics

### Performance (baseline / fair model)
<table>

### Fairness (lower = fairer)
<table>

### Per sensitive feature
<one subsection per feature>

## Evaluation data
- Dataset path, target, sensitive features, row count

## Training data
- From config (or TODO)

## Quantitative analyses
- Pointer to audit JSON and HTML report

## Compliance gates
- Policy: <path> (v<n>)
- Status: PASSED / FAILED
- Per-gate table
- Reviewer, next review (or "no policy supplied")

## Ethical considerations
- From config (or TODO)

## Caveats and recommendations
- From config + EquiML standard caveats

## Generated by
- EquiML version, audit JSON path, timestamp
```

## Consequences

### What gets easier
- One command produces a regulator-ready, hub-publishable artefact from a CI-produced audit JSON.
- Procurement onboarding gets a single document that names the model, its evaluation, and its open caveats.
- HF Hub model uploads can include the card as `README.md` directly.

### What gets harder
- The card encodes a snapshot. Re-audit, re-render. We do not yet support diffing two cards (RFC 0003 candidate).
- The author-config schema will likely grow; we keep version 1 minimal so growth does not break existing configs.

### Backward compatibility
Net new feature. No existing CLI surface changes.

## Future work (not in this RFC)

- `equiml card diff card-q1.md card-q2.md` for cross-quarter audit comparison.
- Card schema versioning so author configs survive minor schema changes.
- Pre-baked configs per regulatory regime (EU AI Act, NYC AEDT, EEOC).
- Cosign / Sigstore signing of the rendered card with the audit JSON hash embedded.
- A linter that flags TODO placeholders before the card is shipped.
