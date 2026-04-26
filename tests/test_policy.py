"""Tests for fairness policy-as-code (RFC 0001)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from equiml.policy import (
    GateThreshold,
    Policy,
    PolicyError,
    SUPPORTED_METRICS,
    evaluate_policy,
    format_result,
    load_policy,
)


# --- Schema / loader tests --------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_load_minimal_valid_policy(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  demographic_parity_difference: { max: 0.1 }
""")
    policy = load_policy(path)
    assert policy.version == 1
    assert policy.target == "income"
    assert policy.sensitive == ["gender"]
    assert policy.algorithm == "logistic_regression"  # default
    assert len(policy.flat_gates) == 1
    assert policy.flat_gates[0].metric == "demographic_parity_difference"
    assert policy.flat_gates[0].max == 0.1
    assert policy.flat_gates[0].severity == "error"  # default


def test_load_policy_with_per_sensitive_gates(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 1
target: income
sensitive: [gender, race]
gates:
  per_sensitive:
    gender:
      demographic_parity_difference: { max: 0.05 }
    race:
      demographic_parity_difference: { max: 0.10, severity: warning }
""")
    policy = load_policy(path)
    assert "gender" in policy.per_sensitive_gates
    assert "race" in policy.per_sensitive_gates
    assert policy.per_sensitive_gates["race"][0].severity == "warning"


def test_unknown_metric_raises(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  totally_made_up_metric: { max: 0.5 }
""")
    with pytest.raises(PolicyError, match="Unknown metric"):
        load_policy(path)


def test_unsupported_version_raises(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 99
target: income
sensitive: [gender]
gates:
  demographic_parity_difference: { max: 0.1 }
""")
    with pytest.raises(PolicyError, match="Unsupported policy version"):
        load_policy(path)


def test_missing_required_field_raises(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 1
sensitive: [gender]
gates:
  demographic_parity_difference: { max: 0.1 }
""")
    with pytest.raises(PolicyError, match="non-empty 'target'"):
        load_policy(path)


def test_per_sensitive_references_unknown_feature_raises(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  per_sensitive:
    race:
      demographic_parity_difference: { max: 0.05 }
""")
    with pytest.raises(PolicyError, match="not in the policy's 'sensitive' list"):
        load_policy(path)


def test_invalid_severity_raises(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  demographic_parity_difference: { max: 0.1, severity: blocking }
""")
    with pytest.raises(PolicyError, match="invalid severity"):
        load_policy(path)


def test_gate_with_neither_max_nor_min_raises(tmp_path):
    path = _write(tmp_path, "p.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  demographic_parity_difference: { severity: error }
""")
    with pytest.raises(PolicyError, match="must specify at least one of"):
        load_policy(path)


def test_missing_file_raises():
    with pytest.raises(PolicyError, match="not found"):
        load_policy("/nonexistent/path/policy.yaml")


def test_supported_metrics_cover_audit_outputs():
    # Every key the README documents should be in SUPPORTED_METRICS.
    expected = {
        "demographic_parity_difference",
        "equalized_odds_difference",
        "equal_opportunity_difference",
        "disparate_impact",
        "accuracy",
        "f1_score",
    }
    assert expected.issubset(SUPPORTED_METRICS)


# --- Evaluator tests --------------------------------------------------------

def _policy(flat=None, per_sensitive=None):
    return Policy(
        version=1,
        target="income",
        sensitive=["gender", "race"],
        algorithm="logistic_regression",
        flat_gates=flat or [],
        per_sensitive_gates=per_sensitive or {},
    )


def test_evaluate_passes_when_metric_below_max():
    policy = _policy(flat=[GateThreshold("demographic_parity_difference", max=0.1)])
    metrics = {"demographic_parity_difference": 0.05}
    result = evaluate_policy(metrics, policy)
    assert result.passed is True
    assert result.violations == []


def test_evaluate_uses_abs_for_difference_metrics():
    # A signed -0.12 still violates a max=0.1 gate on a *_difference metric.
    policy = _policy(flat=[GateThreshold("demographic_parity_difference", max=0.1)])
    metrics = {"demographic_parity_difference": -0.12}
    result = evaluate_policy(metrics, policy)
    assert result.passed is False
    assert len(result.errors) == 1
    assert result.errors[0].observed == -0.12


def test_evaluate_does_not_use_abs_for_disparate_impact():
    # disparate_impact is a ratio, not a difference. abs() would be wrong.
    policy = _policy(flat=[GateThreshold("disparate_impact", min=0.8)])
    metrics = {"disparate_impact": 0.6}
    result = evaluate_policy(metrics, policy)
    assert result.passed is False
    assert result.errors[0].bound == "min"


def test_evaluate_severity_warning_does_not_fail():
    policy = _policy(flat=[
        GateThreshold("demographic_parity_difference", max=0.1, severity="warning"),
    ])
    metrics = {"demographic_parity_difference": 0.5}
    result = evaluate_policy(metrics, policy)
    # 1 warning, 0 errors → passed=True.
    assert result.passed is True
    assert len(result.warnings) == 1
    assert result.warnings[0].severity == "warning"


def test_evaluate_per_sensitive_gates():
    policy = _policy(per_sensitive={
        "gender": [GateThreshold("demographic_parity_difference", max=0.05)],
        "race": [GateThreshold("demographic_parity_difference", max=0.10)],
    })
    metrics = {
        "per_sensitive": {
            "gender": {"demographic_parity_difference": 0.08},  # breach
            "race": {"demographic_parity_difference": 0.05},    # pass
        },
    }
    result = evaluate_policy(metrics, policy)
    assert result.passed is False
    assert len(result.errors) == 1
    assert result.errors[0].sensitive_feature == "gender"


def test_evaluate_missing_per_sensitive_metric_emits_violation():
    policy = _policy(per_sensitive={
        "gender": [GateThreshold("demographic_parity_difference", max=0.05)],
    })
    metrics = {"per_sensitive": {}}  # gender block missing
    result = evaluate_policy(metrics, policy)
    assert result.passed is False
    assert any(
        v.sensitive_feature == "gender" and "did not produce" in v.message
        for v in result.violations
    )


def test_format_result_includes_metadata():
    policy = Policy(
        version=1,
        target="income",
        sensitive=["gender"],
        algorithm="logistic_regression",
        flat_gates=[GateThreshold("demographic_parity_difference", max=0.1)],
        metadata={"reviewer": "risk@example.com", "next_review": "2026-07-01"},
    )
    metrics = {"demographic_parity_difference": 0.5}
    result = evaluate_policy(metrics, policy)
    rendered = format_result(result, policy_path="fairness.yaml")
    assert "FAILED" in rendered
    assert "demographic_parity_difference" in rendered
    assert "risk@example.com" in rendered
    assert "2026-07-01" in rendered


# --- CLI integration tests via subprocess ----------------------------------

@pytest.fixture
def fixture_audit_json(tmp_path):
    """A minimal audit JSON in the shape `equiml audit --output` produces."""
    payload = {
        "baseline": {
            "demographic_parity_difference": 0.20,
            "equalized_odds_difference": 0.30,
            "per_sensitive": {
                "gender": {"demographic_parity_difference": 0.20},
            },
        },
        "fair": {
            "demographic_parity_difference": 0.04,
            "equalized_odds_difference": 0.28,
            "per_sensitive": {
                "gender": {"demographic_parity_difference": 0.04},
            },
        },
        "_meta": {"equiml_version": "1.1.0", "sensitive_features": ["gender"]},
    }
    p = tmp_path / "audit.json"
    p.write_text(json.dumps(payload))
    return p


def test_cli_verify_passes_when_fair_metrics_under_threshold(tmp_path, fixture_audit_json):
    policy = _write(tmp_path, "fairness.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  demographic_parity_difference: { max: 0.10 }
""")
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "verify",
         str(fixture_audit_json), "--policy", str(policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_cli_verify_returns_3_on_gate_breach(tmp_path, fixture_audit_json):
    policy = _write(tmp_path, "fairness.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  demographic_parity_difference: { max: 0.01 }
""")
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "verify",
         str(fixture_audit_json), "--policy", str(policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "FAILED" in result.stdout


def test_cli_verify_returns_4_on_schema_error(tmp_path, fixture_audit_json):
    policy = _write(tmp_path, "fairness.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  not_a_real_metric: { max: 0.1 }
""")
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "verify",
         str(fixture_audit_json), "--policy", str(policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "schema error" in result.stderr.lower()


def test_cli_verify_returns_2_when_audit_json_missing(tmp_path):
    policy = _write(tmp_path, "fairness.yaml", """
version: 1
target: income
sensitive: [gender]
gates:
  demographic_parity_difference: { max: 0.1 }
""")
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "verify",
         str(tmp_path / "missing.json"), "--policy", str(policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
