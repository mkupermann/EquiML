# Fairness drift report

Generated: 2026-04-26 19:04:38 UTC
State file: `examples/monitor_state.jsonl`

## Windows

| Window | Days | Batches | Total samples |
|---|---|---|---|
| Baseline | 30 | 23 | 22779 |
| Current | 7 | 6 | 5911 |

## Drift

- **Overall severity:** MATERIAL
- **Overall PSI:** `0.2972`

PSI bands: `< 0.10` no drift · `0.10–0.25` moderate · `≥ 0.25` material.

| Sensitive feature | PSI | Reading |
|---|---|---|
| `sex` | 0.2972 | material |
| `race` | 0.0000 | none |




## Group selection rates over time

### Baseline window

**sex**

- `female`: 0.4534
- `male`: 0.5575


**race**

- `white`: 0.5088
- `nonwhite`: 0.5140




### Current window

**sex**

- `female`: 0.2104
- `male`: 0.8495


**race**

- `white`: 0.5607
- `nonwhite`: 0.5600




## Aggregated fairness metrics

| Metric | Baseline | Current | Δ |
|---|---|---|---|
| `demographic_parity_difference` | 0.1041 | 0.6391 | +0.5350 |


## Policy enforcement (current window)

- **Policy:** `examples/fairness.yaml` (schema v1)
- **Status:** FAILED (2 error(s), 1 warning(s))

| Metric | Scope | Bound | Threshold | Observed | Severity |
|---|---|---|---|---|---|
| `demographic_parity_difference` | (top-level) | max | 0.1000 | 0.6391 | error |
| `equalized_odds_difference` | (top-level) | presence | n/a | n/a | warning |
| `demographic_parity_difference` | sex | max | 0.0500 | 0.6391 | error |


## Caveats

- PSI is computed on group selection rates aggregated across batches in the window. It detects shifts in *who is being selected*, not in label distribution.
- Equalized-odds and equal-opportunity metrics require ground-truth labels; batches without labels report demographic parity only.
- A single PSI breach is a signal, not a verdict. Investigate before retraining.
