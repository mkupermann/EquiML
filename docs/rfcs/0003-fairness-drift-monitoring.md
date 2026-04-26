# RFC 0003 — Fairness Drift Monitoring

| Field | Value |
|---|---|
| Status | Implemented (1.1.0-dev) |
| Author | Michael Kupermann |
| Date | 2026-04-26 |
| Reviewers | (open for comment) |

## Context

`equiml audit` and `equiml verify` are point-in-time tools. They evaluate one snapshot of a model against one snapshot of data. Production deployment is not a snapshot — it is a stream. Fairness in deployment can move because:

- Input data drifts (population composition shifts).
- The decision boundary stays put while the world moves under it.
- A new feature is added and silently correlates with a protected attribute.
- A retraining cycle adjusts the constraint trade-off.
- Ground-truth labels arrive late, so apparent fairness updates after weeks.

The existing `BiasMonitor` (1.0.x) handles the easiest case: a single batch of predictions, one threshold check, no time component, no statistical test. That covers the laptop-experiment use case. It does not cover production.

This RFC adds `FairnessDriftMonitor`: a time-aware monitor that records batches over time, computes windowed metrics, detects drift between baseline and current windows, and enforces a `fairness.yaml` policy continuously rather than once.

## Decision

EquiML 1.1 ships:

1. A new module `equiml/drift.py` with `FairnessDriftMonitor` and a `Batch` dataclass.
2. JSONL persistence: each batch is one line, append-only, recoverable across restarts.
3. Drift detection via the **Population Stability Index** (PSI) on per-group selection-rate distributions.
4. Policy integration: the monitor evaluates a `fairness.yaml` against the latest window's aggregated metrics.
5. A new CLI subcommand `equiml monitor` with three operations: `record` / `check` / `report`.
6. The existing `BiasMonitor` stays as-is for backward compatibility.

`FairnessDriftMonitor` is the recommended replacement for `BiasMonitor` for any production use. The existing tests and exports of `BiasMonitor` continue to work.

## Data model

### Batch

A single observation in the monitor's stream:

```python
@dataclass
class Batch:
    timestamp: datetime           # UTC
    n_samples: int
    metrics: dict[str, float]     # demographic_parity_difference, etc.
    per_sensitive: dict[str, dict[str, float]]  # by-feature metric blocks
    group_rates: dict[str, dict[str, float]]    # per feature, per group, selection rate
    metadata: dict                 # free-form (model_version, env, etc.)
```

`metrics` and `per_sensitive` follow the audit JSON shape so the same code path renders both audit results and monitor batches.

`group_rates` is new and load-bearing: drift detection runs on group selection rates, not on aggregate metrics. A model whose DP-difference is stable while every group's selection rate shifts in concert is *still drifting* — the metric just hides it.

### Persistence — JSONL

```jsonl
{"timestamp": "2026-04-01T08:00:00Z", "n_samples": 1000, "metrics": {"demographic_parity_difference": 0.04}, ...}
{"timestamp": "2026-04-02T08:00:00Z", "n_samples": 1023, "metrics": {"demographic_parity_difference": 0.05}, ...}
```

JSONL was picked because:
- Append-only writes are atomic at the line level on POSIX.
- A scheduled job that records hourly never has to read existing state.
- Tooling (`jq`, `tail -f`, `grep`) just works.
- No SQLite migration story to maintain.

## Drift detection

### Population Stability Index (PSI)

PSI is the canonical drift metric in credit risk and operations research. It compares a baseline distribution against a current distribution:

```
PSI = sum( (current_p - baseline_p) * ln(current_p / baseline_p) )
```

Conventional bands:
- `PSI < 0.10` — no drift
- `0.10 ≤ PSI < 0.25` — moderate drift, investigate
- `PSI ≥ 0.25` — material drift, action

EquiML applies PSI to **group selection rates per sensitive feature**. For each sensitive feature `f`:
1. Aggregate group rates across all batches in the baseline window (weighted by `n_samples`).
2. Aggregate group rates across all batches in the current window.
3. Compute PSI on the resulting two distributions over groups.

The output is one PSI value per sensitive feature, plus an "overall" PSI as the max across features.

### Why not KS or chi²

- **KS** is for continuous distributions. Group selection rates are categorical-grouped scalars; KS does not naturally apply.
- **Chi²** assumes independent samples and known group counts. The monitor receives aggregated rates, not raw counts.
- PSI is robust, intuitive, and what every credit-risk practitioner has used for thirty years. We borrow the convention.

KS and chi² will land in 1.2 alongside support for raw-prediction streaming (where they apply).

## API

```python
from equiml.drift import FairnessDriftMonitor

monitor = FairnessDriftMonitor(
    sensitive_features=["sex", "race"],
    state_path="monitor_state.jsonl",
)

# Append one batch (loads existing state on construction)
monitor.record(
    predictions=preds,                  # array of 0/1
    sensitive_features=sf_df,           # DataFrame with sex, race columns
    true_labels=labels,                 # optional; enables EO/EOpp metrics
    metadata={"model_version": "3.2.0"},
)

# Window selection
baseline = monitor.window(days=30, until=baseline_end_date)
current = monitor.window(days=7)

# Drift detection
drift = monitor.detect_drift(current_window=current, baseline_window=baseline)
# drift.psi_per_feature: {"sex": 0.04, "race": 0.18}
# drift.psi_overall: 0.18
# drift.severity: "moderate"  # "none" | "moderate" | "material"

# Policy enforcement on the latest window
from equiml.policy import load_policy
policy = load_policy("fairness.yaml")
result = monitor.evaluate_against_policy(policy, window=current)
# Same PolicyResult shape as `equiml verify`.
```

## CLI

```bash
# Record a batch from a CSV (predictions + sensitive columns + optional labels)
equiml monitor record \
    --state monitor_state.jsonl \
    --batch predictions.csv \
    --predictions-col prediction \
    --sensitive sex,race \
    --labels-col actual_label \
    --metadata model_version=3.2.0

# Check the latest 7-day window against a 30-day baseline + a fairness.yaml
equiml monitor check \
    --state monitor_state.jsonl \
    --baseline-days 30 \
    --current-days 7 \
    --policy fairness.yaml \
    --psi-threshold 0.10

# Render a markdown drift report
equiml monitor report \
    --state monitor_state.jsonl \
    --baseline-days 30 \
    --current-days 7 \
    --output drift_report.md
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. No policy breach, no material drift. |
| 2 | Data error (state file missing, batch CSV malformed). |
| 3 | Policy gate breached on the current window. |
| 4 | Policy schema error. |
| 5 | Drift threshold breached (PSI ≥ threshold) without a policy breach. |

A single invocation can emit at most one exit code; if both a policy gate fails AND drift is breached, exit 3 wins (policy breach is the load-bearing fact for compliance reporting; drift is the diagnostic).

## What this monitor does not do (yet)

- **Delayed labels.** Production decisions arrive without immediate ground truth. Today, batches without `true_labels` get partial metric coverage (DP works; EO and EOpp do not). Future work: a `backfill_labels` operation that retroactively attaches labels to recent batches and recomputes EO/EOpp.
- **Streaming raw predictions.** The monitor takes batches, not single observations. Genuinely streaming inference (one prediction at a time) needs a different abstraction (Bloom-filter-ish reservoir sampling). 1.3 candidate.
- **Alerting transport.** The monitor emits a result; it does not page anyone. CI integration covers most use cases (`equiml monitor check` exits non-zero, the CI job alerts via its own channel). PagerDuty/Slack/email transports are out of scope for the EquiML project.
- **Plot output.** PSI numbers and DP-over-time series are JSON; rendering them as charts is downstream tooling. The Markdown drift report includes the numbers; the HTML report is 1.2 candidate.

## Alternatives considered

### Existing `BiasMonitor` extension

Pro: one less class.
Con: would break the existing API. `BiasMonitor.monitor_predictions()` returns a per-call record dict; `FairnessDriftMonitor.record()` returns a `Batch` object and persists to JSONL. The shapes are incompatible.

**Verdict:** new class. Keep `BiasMonitor` for backward compat.

### SQLite for state

Pro: queryable, indexable.
Con: another binary in the repo for tests, schema-migration burden, harder to inspect with command-line tools.

**Verdict:** JSONL. SQLite is a future option behind the same interface if state gets large.

### KS / chi² as the primary drift test

Pro: more statistical power for some failure modes.
Con: assumes raw samples, which the monitor does not have. PSI works on aggregated rates.

**Verdict:** PSI for v1. KS/chi² when raw-prediction streaming lands.

### Continuous policy enforcement only (no drift)

Pro: simpler, leverages existing `evaluate_policy`.
Con: misses drift that does not breach a policy gate. A model whose DP-difference holds at 0.08 but whose underlying group rates have moved from 40/30 to 60/50 is drifting without breaching any policy.

**Verdict:** both. Drift detection is independent of the policy, complements it.

## Consequences

### What gets easier
- Production deployment monitoring becomes one cron job: `equiml monitor record` hourly + `equiml monitor check` daily, with the latter exiting non-zero on regression.
- A "fairness postmortem" after an incident becomes a report rendered from the JSONL state, showing exactly when drift began.
- The policy file (`fairness.yaml`) becomes the same artefact at training time, deployment time, and in a regulator's pack — enforced by the same code in three places.

### What gets harder
- The state file becomes a thing teams have to manage (back up, archive, retention). We document this in the RFC and the README; we do not solve it for them.
- PSI's "moderate / material" bands are conventional, not universal. Some domains will want different thresholds. The CLI's `--psi-threshold` flag exposes this; we do not invent jurisdiction-specific defaults.

### Backward compatibility
Net new feature. `BiasMonitor` is unchanged. `equiml.__init__` exports both classes.

## Future work (not in this RFC)

- KS / chi² drift tests on raw prediction streams.
- Delayed-label backfill.
- HTML drift report with plotly time-series.
- SQLite state backend behind the same interface.
- Cosign signing of state-file segments for tamper-evident drift evidence.
