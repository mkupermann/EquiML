"""Fairness policy-as-code: YAML schema, loader, and gate evaluator.

See docs/rfcs/0001-policy-as-code.md for the design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


SUPPORTED_METRICS = {
    "demographic_parity_difference",
    "equalized_odds_difference",
    "equal_opportunity_difference",
    "disparate_impact",
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "roc_auc",
}

SCHEMA_VERSION = 1


class PolicyError(Exception):
    """Schema-level error in a policy file. Maps to CLI exit code 4."""


@dataclass
class GateThreshold:
    metric: str
    max: float | None = None
    min: float | None = None
    severity: str = "error"  # "error" or "warning"

    def __post_init__(self) -> None:
        if self.metric not in SUPPORTED_METRICS:
            raise PolicyError(
                f"Unknown metric '{self.metric}'. Supported metrics: "
                f"{sorted(SUPPORTED_METRICS)}"
            )
        if self.max is None and self.min is None:
            raise PolicyError(
                f"Gate for '{self.metric}' must specify at least one of "
                f"'max' or 'min'."
            )
        if self.severity not in {"error", "warning"}:
            raise PolicyError(
                f"Gate for '{self.metric}' has invalid severity "
                f"'{self.severity}'. Use 'error' or 'warning'."
            )


@dataclass
class Policy:
    version: int
    target: str
    sensitive: list[str]
    algorithm: str
    flat_gates: list[GateThreshold] = field(default_factory=list)
    per_sensitive_gates: dict[str, list[GateThreshold]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateViolation:
    metric: str
    severity: str
    bound: str  # "max" or "min"
    threshold: float
    observed: float
    sensitive_feature: str | None
    message: str


@dataclass
class PolicyResult:
    passed: bool
    violations: list[GateViolation]
    metadata: dict[str, Any]
    policy_version: int

    @property
    def errors(self) -> list[GateViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[GateViolation]:
        return [v for v in self.violations if v.severity == "warning"]


def load_policy(path: str | Path) -> Policy:
    """Parse and validate a policy YAML file. Raises PolicyError on schema problems."""
    path = Path(path)
    if not path.exists():
        raise PolicyError(f"Policy file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise PolicyError(f"Policy file is not valid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise PolicyError("Policy file must be a YAML mapping at the top level.")

    return _build_policy(raw)


def _build_policy(raw: dict[str, Any]) -> Policy:
    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise PolicyError(
            f"Unsupported policy version: {version!r}. This release supports "
            f"version {SCHEMA_VERSION}."
        )

    target = raw.get("target")
    if not isinstance(target, str) or not target:
        raise PolicyError("Policy must have a non-empty 'target' string.")

    sensitive = raw.get("sensitive")
    if not isinstance(sensitive, list) or not sensitive or not all(isinstance(s, str) for s in sensitive):
        raise PolicyError("Policy must have a non-empty 'sensitive' list of column names.")

    algorithm = raw.get("algorithm", "logistic_regression")
    if not isinstance(algorithm, str):
        raise PolicyError(f"'algorithm' must be a string, got {type(algorithm).__name__}.")

    gates = raw.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise PolicyError("Policy must have a non-empty 'gates' mapping.")

    flat_gates: list[GateThreshold] = []
    per_sensitive_gates: dict[str, list[GateThreshold]] = {}

    for key, value in gates.items():
        if key == "per_sensitive":
            if not isinstance(value, dict):
                raise PolicyError("'gates.per_sensitive' must be a mapping of feature → gates.")
            for feature, feature_gates in value.items():
                if feature not in sensitive:
                    raise PolicyError(
                        f"Per-sensitive gate references '{feature}', which is not "
                        f"in the policy's 'sensitive' list ({sensitive})."
                    )
                if not isinstance(feature_gates, dict):
                    raise PolicyError(
                        f"'gates.per_sensitive.{feature}' must be a mapping of "
                        f"metric → threshold."
                    )
                per_sensitive_gates[feature] = _parse_gate_block(feature_gates)
        else:
            if not isinstance(value, dict):
                raise PolicyError(
                    f"Gate '{key}' must be a mapping with at least 'max' or 'min'."
                )
            flat_gates.append(_build_gate(key, value))

    if not flat_gates and not per_sensitive_gates:
        raise PolicyError("Policy 'gates' mapping has no actual gates defined.")

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PolicyError("'metadata' must be a mapping if provided.")

    return Policy(
        version=version,
        target=target,
        sensitive=list(sensitive),
        algorithm=algorithm,
        flat_gates=flat_gates,
        per_sensitive_gates=per_sensitive_gates,
        metadata=metadata,
    )


def _parse_gate_block(block: dict[str, Any]) -> list[GateThreshold]:
    return [_build_gate(metric, spec) for metric, spec in block.items()]


def _build_gate(metric: str, spec: dict[str, Any]) -> GateThreshold:
    if not isinstance(spec, dict):
        raise PolicyError(
            f"Gate '{metric}' must be a mapping, got {type(spec).__name__}."
        )
    max_val = spec.get("max")
    min_val = spec.get("min")
    severity = spec.get("severity", "error")
    if max_val is not None and not isinstance(max_val, (int, float)):
        raise PolicyError(f"Gate '{metric}': 'max' must be a number, got {max_val!r}.")
    if min_val is not None and not isinstance(min_val, (int, float)):
        raise PolicyError(f"Gate '{metric}': 'min' must be a number, got {min_val!r}.")
    return GateThreshold(
        metric=metric,
        max=float(max_val) if max_val is not None else None,
        min=float(min_val) if min_val is not None else None,
        severity=severity,
    )


def evaluate_policy(metrics: dict[str, Any], policy: Policy) -> PolicyResult:
    """Run the policy against an audit metrics dict.

    `metrics` follows the shape produced by EquiMLEvaluation.evaluate(): a flat
    dict with metric names at the top level, plus an optional `per_sensitive`
    sub-dict keyed by feature name (added by cmd_audit when multiple sensitive
    features are passed).
    """
    violations: list[GateViolation] = []

    for gate in policy.flat_gates:
        violations.extend(_check_gate(gate, metrics, sensitive_feature=None))

    per_sensitive = metrics.get("per_sensitive", {})
    for feature, feature_gates in policy.per_sensitive_gates.items():
        feature_metrics = per_sensitive.get(feature)
        if feature_metrics is None:
            violations.append(GateViolation(
                metric="(missing)",
                severity="error",
                bound="presence",
                threshold=float("nan"),
                observed=float("nan"),
                sensitive_feature=feature,
                message=(
                    f"Policy expected per-sensitive metrics for '{feature}', but "
                    f"the audit did not produce any. Was '{feature}' in the "
                    f"--sensitive args?"
                ),
            ))
            continue
        for gate in feature_gates:
            violations.extend(_check_gate(gate, feature_metrics, sensitive_feature=feature))

    has_error = any(v.severity == "error" for v in violations)
    return PolicyResult(
        passed=not has_error,
        violations=violations,
        metadata=policy.metadata,
        policy_version=policy.version,
    )


def _check_gate(
    gate: GateThreshold,
    metrics: dict[str, Any],
    sensitive_feature: str | None,
) -> list[GateViolation]:
    if gate.metric not in metrics:
        return [GateViolation(
            metric=gate.metric,
            severity=gate.severity,
            bound="presence",
            threshold=float("nan"),
            observed=float("nan"),
            sensitive_feature=sensitive_feature,
            message=(
                f"Policy expects metric '{gate.metric}' but the audit did not "
                f"compute it. Check that the model produced the metric or relax the policy."
            ),
        )]

    raw_value = metrics[gate.metric]
    try:
        observed = float(raw_value)
    except (TypeError, ValueError):
        return [GateViolation(
            metric=gate.metric,
            severity=gate.severity,
            bound="presence",
            threshold=float("nan"),
            observed=float("nan"),
            sensitive_feature=sensitive_feature,
            message=(
                f"Metric '{gate.metric}' has non-numeric value {raw_value!r}; "
                f"cannot evaluate gate."
            ),
        )]

    # For difference-style metrics, fairness is "lower abs is fairer", so we
    # compare against abs(observed). For ratio / performance metrics, the raw
    # signed value is what matters.
    is_difference_metric = gate.metric.endswith("_difference")
    compare_value = abs(observed) if is_difference_metric else observed

    out: list[GateViolation] = []
    if gate.max is not None and compare_value > gate.max:
        out.append(GateViolation(
            metric=gate.metric,
            severity=gate.severity,
            bound="max",
            threshold=gate.max,
            observed=observed,
            sensitive_feature=sensitive_feature,
            message=(
                f"observed {observed:.4f}, max {gate.max:.4f} — exceeds "
                f"threshold by {compare_value - gate.max:.4f}"
            ),
        ))
    if gate.min is not None and compare_value < gate.min:
        out.append(GateViolation(
            metric=gate.metric,
            severity=gate.severity,
            bound="min",
            threshold=gate.min,
            observed=observed,
            sensitive_feature=sensitive_feature,
            message=(
                f"observed {observed:.4f}, min {gate.min:.4f} — falls short "
                f"by {gate.min - compare_value:.4f}"
            ),
        ))
    return out


def format_result(result: PolicyResult, policy_path: str | Path | None = None) -> str:
    """Render a PolicyResult as a console-friendly text block."""
    lines: list[str] = []
    lines.append("POLICY RESULT")
    if policy_path is not None:
        lines.append(f"  Policy: {policy_path} (v{result.policy_version})")
    n_err = len(result.errors)
    n_warn = len(result.warnings)
    if result.passed and n_warn == 0:
        lines.append("  Status: PASSED")
    elif result.passed and n_warn > 0:
        lines.append(f"  Status: PASSED with {n_warn} warning(s)")
    else:
        lines.append(f"  Status: FAILED ({n_err} error, {n_warn} warning)")

    for v in result.violations:
        prefix = "ERROR" if v.severity == "error" else "WARN "
        scope = f" ({v.sensitive_feature})" if v.sensitive_feature else ""
        lines.append(f"  {prefix}  {v.metric}{scope}")
        lines.append(f"         {v.message}")

    if result.metadata:
        lines.append("")
        for key in ("reviewer", "next_review", "model_owner", "jurisdiction", "legal_basis"):
            if key in result.metadata:
                lines.append(f"  {key.replace('_', ' ').title()}: {result.metadata[key]}")

    return "\n".join(lines)
