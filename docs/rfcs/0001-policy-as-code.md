# RFC 0001 — Fairness Policy-as-Code

| Field | Value |
|---|---|
| Status | Implemented (1.1.0-dev) |
| Author | Michael Kupermann |
| Date | 2026-04-26 |
| Reviewers | (open for comment) |

## Context

The 1.0.x audit is a reporter: run it, read the JSON, decide. That works for an analyst on a laptop. It does not work for:

- An ML platform engineer who wants the audit to **fail a CI job** when a model regresses on fairness.
- A VP of Data who wants every team in a 40-person org to share the **same fairness contract** and have it enforced automatically.
- A procurement reviewer who wants the contract to be a **structured artefact** they can attach to a vendor onboarding ticket.
- A compliance officer who wants the same numbers their colleagues use to be the same numbers a regulator would see.

All four want the same primitive: a declarative file that says *"this model passes if these conditions hold,"* checked by the same code in CI, in vendor review, and in compliance evidence. None of them want a Python script.

This RFC adds that primitive: `fairness.yaml`.

## Decision

EquiML 1.1 ships:

1. A **YAML schema** for fairness policies (`fairness.yaml`).
2. A **policy evaluator** in `equiml/policy.py` that runs a parsed policy against an audit metrics dict and returns a structured result with violations.
3. **Two CLI surfaces**:
   - `equiml audit data.csv --policy fairness.yaml ...` — runs the audit, then evaluates the policy. Exit code reflects the policy result.
   - `equiml verify audit.json --policy fairness.yaml` — evaluates the policy against a previously-run audit JSON. No model retraining. Useful for re-checking historical artefacts after the policy tightens.
4. **Documented exit codes**:
   - `0` success (no error-severity violations)
   - `2` data error (missing file, unreadable CSV) — pre-existing
   - `3` policy gate breached (one or more `error`-severity gates failed)
   - `4` policy schema mismatch (the YAML is malformed or references unknown fields)

Warning-severity violations print to stderr and the run continues. Error-severity violations exit with `3`.

## Schema

### Top level

```yaml
version: 1
target: <column-name>            # required
sensitive:                        # required, list
  - <column-name>
algorithm: logistic_regression    # optional, defaults to logistic_regression
gates: { ... }                    # required
metadata: { ... }                 # optional, free-form
```

`version` is the schema version. `1` is the only currently-supported value. Future versions break-change this RFC.

### Gates block

Two shapes are supported, intentionally:

**Flat shape** (one threshold per metric, applied to the primary sensitive feature):

```yaml
gates:
  demographic_parity_difference:
    max: 0.10
    severity: error
  equalized_odds_difference:
    max: 0.15
    severity: warning
```

**Per-sensitive shape** (different thresholds per protected attribute):

```yaml
gates:
  per_sensitive:
    gender:
      demographic_parity_difference:
        max: 0.05         # tighter for gender
        severity: error
    race:
      demographic_parity_difference:
        max: 0.10
        severity: error
```

Both shapes can co-exist. Flat-shape gates apply to the primary (top-level) metrics. Per-sensitive-shape gates apply to entries under `metrics["per_sensitive"][<feature>]` in the audit JSON.

### Per-metric block

```yaml
<metric_name>:
  max: <float>           # threshold; absolute value of the metric must be <= max
  min: <float>           # optional: metric must be >= min (for ratios like disparate_impact)
  severity: error | warning   # default: error
```

`max` and `min` are independent — both can be specified. Violation occurs if either bound is breached.

### Supported metric names (v1)

- `demographic_parity_difference`
- `equalized_odds_difference`
- `equal_opportunity_difference`
- `disparate_impact` (ratio; typical lower bound is the four-fifths rule, `min: 0.8`)
- `accuracy`, `f1_score`, `precision`, `recall`, `roc_auc` (performance gates)

Unknown metric names emit a schema error (exit 4) — no silent typo tolerance.

### Metadata

Free-form, preserved verbatim into the policy result. Convention:

```yaml
metadata:
  model_owner: "ml-platform@example.com"
  reviewer: "model-risk@example.com"
  next_review: "2026-07-01"
  jurisdiction: "EU"
  legal_basis: "EU AI Act Art. 15"
```

## CLI integration

### `equiml audit ... --policy fairness.yaml`

```bash
equiml audit data.csv --target income --sensitive gender race \
    --policy fairness.yaml \
    --output audit.json --report audit.html
```

Run order:
1. Load + validate policy. On schema error: exit 4.
2. Run the audit as before.
3. Evaluate policy against the resulting metrics.
4. Print a `POLICY RESULT` block to console.
5. Exit 3 if any error-severity gate failed; otherwise exit 0.

If `--policy` is omitted, behaviour is unchanged from 1.0.x.

### `equiml verify audit.json --policy fairness.yaml`

```bash
equiml verify audit.json --policy fairness.yaml
```

Loads the JSON produced by a prior `equiml audit` run, evaluates the policy, prints the result, exits with the appropriate code. No model retraining, no dataset access. Useful for:

- Re-checking historical audit JSONs after tightening a policy.
- Running the gate in a job that does not need the original CSV.
- Compliance review: regulator hands over an artefact, you check it against your published policy.

## Result format

The policy evaluator returns a `PolicyResult`:

```python
@dataclass
class PolicyResult:
    passed: bool                          # all error-severity gates passed
    violations: list[GateViolation]       # one per failed gate (any severity)
    metadata: dict                        # echoed from policy
    policy_version: int

@dataclass
class GateViolation:
    metric: str
    severity: str                         # "error" | "warning"
    bound: str                            # "max" | "min"
    threshold: float
    observed: float
    sensitive_feature: str | None         # None for flat-shape; name for per-sensitive
    message: str                          # one-line, human-readable
```

Console output (one possible rendering):

```text
POLICY RESULT
  Policy: fairness.yaml (v1)
  Status: FAILED (1 error, 1 warning)

  ERROR  demographic_parity_difference (gender)
         observed 0.082, max 0.05 — exceeds threshold by 0.032

  WARN   equalized_odds_difference
         observed 0.21, max 0.15 — exceeds threshold by 0.06

  Reviewer: model-risk@example.com
  Next review: 2026-07-01
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. Either no `--policy` was passed, or all error-severity gates passed. |
| 2 | Data error (file not found, unreadable CSV). Pre-existing. |
| 3 | Policy gate breached. At least one error-severity gate failed. |
| 4 | Policy schema error. The YAML is malformed or references unknown fields/metrics. |

Codes 1, 5–127 are reserved.

## Alternatives considered

### TOML instead of YAML

Pro: stdlib in Python 3.11+ via `tomllib`. Familiar to Python devs (pyproject.toml).
Con: less idiomatic for CI policy files. The DevOps convention (`.github/workflows/*.yml`, `.gitlab-ci.yml`, `codeowners`) is YAML. Policy-as-code tools across the Python ecosystem (Great Expectations, Deepchecks) use YAML.

**Verdict:** YAML.

### JSON instead of YAML

Pro: stdlib, no parser dependency.
Con: humans hate writing JSON. Comments are not allowed. This file will be hand-edited by ML platform engineers; YAML wins on ergonomics.

**Verdict:** YAML.

### Python file instead of YAML

Pro: full programmatic power. Could express things YAML can't.
Con: defeats the whole point. A non-Python procurement reviewer cannot read a Python policy. CI enforcement of arbitrary code is a security problem.

**Verdict:** YAML.

### Pydantic for validation

Pro: well-known library. Friendly error messages out of the box.
Con: heavy dependency for one feature. Pydantic v1/v2 split is still painful. Hand-rolling validation for a small schema is straightforward and avoids the pin.

**Verdict:** hand-rolled validation. PyYAML is the only new dependency.

### Combined audit-and-policy-in-one-yaml

Idea: have the policy YAML also declare the audit run config (target, sensitive features, algorithm).
Pro: single source of truth, less CLI typing.
Con: confuses the boundary. The policy is what passes/fails. The audit config is what produces the metrics. Coupling them means you can't re-evaluate a historical audit against a tightened policy.

**Verdict:** keep them separate. The policy YAML names the `target` and `sensitive` columns it expects, but does not run the audit.

## Consequences

### What gets easier
- CI integration becomes a one-liner: `equiml audit ... --policy fairness.yaml`. Build fails on regression.
- VP-Data adoption story: one shared `fairness.yaml` per team, version-controlled like a Dockerfile.
- Compliance evidence: the policy file is part of the audit pack — regulators can see exactly what conditions the model was certified against.

### What gets harder
- Policies drift from reality. A policy from 2024 against a model trained on 2026 data may pass for the wrong reasons. Future RFC: policy expiry / next-review enforcement.
- The flat / per-sensitive shape duality is a small cognitive cost; the schema accepts both because real teams need both.

### Backward compatibility
Fully backward-compatible. Without `--policy`, every 1.0.x audit invocation produces identical results.

## Tested with

- Adult census demo (`examples/fairness.yaml`).
- Substring-collision regression fixture (verifies per-sensitive gates use exact matching, not substring).
- Policy schema-error exit code (verifies exit 4 on malformed YAML).
- Policy gate-breach exit code (verifies exit 3 on threshold violation).

## Future work (not in this RFC)

- `next_review` enforcement: refuse to evaluate after the date.
- Cosign / Sigstore signing of `_meta` + policy hash.
- Policy templates per regulatory regime (EU AI Act preset, NYC AEDT preset).
- `equiml policy diff` to compare two policy files and produce a regulatory-readable change log.
- Importing thresholds from the four-fifths rule (`min disparate_impact: 0.8`) as a built-in preset.
