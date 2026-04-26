"""Tests for model-card generation (RFC 0002)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from equiml.card import (
    CardConfig,
    build_card_from_audit,
    load_card_config,
    render_markdown,
)
from equiml.policy import load_policy


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


@pytest.fixture
def fixture_audit_json(tmp_path: Path) -> Path:
    payload = {
        "baseline": {
            "accuracy": 0.817,
            "f1_score": 0.807,
            "roc_auc": 0.879,
            "demographic_parity_difference": 0.205,
            "equalized_odds_difference": 0.338,
            "equal_opportunity_difference": 0.30,
            "per_sensitive": {
                "sex": {
                    "demographic_parity_difference": 0.21,
                    "equalized_odds_difference": 0.34,
                    "accuracy": 0.81, "f1_score": 0.80,
                },
                "race": {
                    "demographic_parity_difference": 0.10,
                    "equalized_odds_difference": 0.18,
                    "accuracy": 0.82, "f1_score": 0.80,
                },
            },
        },
        "fair": {
            "accuracy": 0.816,
            "f1_score": 0.803,
            "roc_auc": None,
            "demographic_parity_difference": 0.041,
            "equalized_odds_difference": 0.285,
            "per_sensitive": {
                "sex": {
                    "demographic_parity_difference": 0.04,
                    "equalized_odds_difference": 0.28,
                    "accuracy": 0.81, "f1_score": 0.80,
                },
                "race": {
                    "demographic_parity_difference": 0.05,
                    "equalized_odds_difference": 0.19,
                    "accuracy": 0.82, "f1_score": 0.80,
                },
            },
        },
        "_meta": {
            "equiml_version": "1.1.0-dev",
            "python_version": "3.13.12",
            "sklearn_version": "1.5.0",
            "fairlearn_version": "0.10.0",
            "random_seed": 42,
            "dataset_path": "examples/adult.csv",
            "target": "income",
            "sensitive_features": ["sex", "race"],
            "algorithm": "logistic_regression",
        },
    }
    p = tmp_path / "audit.json"
    p.write_text(json.dumps(payload))
    return p


# --- Config loader ---------------------------------------------------------

def test_load_card_config_with_full_yaml(tmp_path):
    path = _write(tmp_path, "card.yaml", """
model_name: "Test model"
model_version: "1.2.3"
model_type: "Logistic regression"
description: "A test"
license: "MIT"
contact: "test@example.com"
intended_use:
  primary_use: "Testing"
  primary_users: "Devs"
  out_of_scope: ["Real use"]
training_data:
  source: "Synthetic"
  preprocessing: "None"
  notes: "100 rows"
ethical_considerations: ["No real subjects"]
caveats: ["Demo only"]
""")
    config = load_card_config(path)
    assert config.model_name == "Test model"
    assert config.intended_use["primary_use"] == "Testing"
    assert config.training_data["notes"] == "100 rows"
    assert config.caveats == ["Demo only"]


def test_load_card_config_missing_fields_use_defaults(tmp_path):
    path = _write(tmp_path, "card.yaml", "model_name: Just a name\n")
    config = load_card_config(path)
    assert config.model_name == "Just a name"
    # Other fields fall back to TODO placeholders.
    assert "TODO" in config.description
    assert "TODO" in config.intended_use["primary_use"]


def test_load_card_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_card_config(tmp_path / "absent.yaml")


# --- Rendering -------------------------------------------------------------

def test_render_emits_yaml_frontmatter(fixture_audit_json):
    payload = json.loads(fixture_audit_json.read_text())
    md = render_markdown(
        audit_payload=payload,
        audit_json_path=fixture_audit_json,
        config=CardConfig(model_name="My model", license="MIT"),
    )
    # Frontmatter is the first --- block.
    match = re.match(r"---\n(.+?)\n---\n", md, flags=re.DOTALL)
    assert match, "card must start with YAML frontmatter"
    front = yaml.safe_load(match.group(1))
    assert front["license"] == "MIT"
    assert "fairness-audited" in front["tags"]
    metric_types = [m["type"] for m in front["model-index"][0]["results"][0]["metrics"]]
    assert "accuracy" in metric_types
    assert "demographic_parity_difference" in metric_types


def test_render_includes_per_sensitive_subsections(fixture_audit_json):
    payload = json.loads(fixture_audit_json.read_text())
    md = render_markdown(
        audit_payload=payload,
        audit_json_path=fixture_audit_json,
        config=CardConfig(),
    )
    assert "#### `sex`" in md
    assert "#### `race`" in md


def test_render_includes_policy_block_when_passed(tmp_path, fixture_audit_json):
    payload = json.loads(fixture_audit_json.read_text())
    policy_path = _write(tmp_path, "fairness.yaml", """
version: 1
target: income
sensitive: [sex, race]
gates:
  demographic_parity_difference: { max: 0.10 }
metadata:
  reviewer: "risk@example.com"
  next_review: "2026-07-01"
""")
    policy = load_policy(policy_path)
    md = render_markdown(
        audit_payload=payload,
        audit_json_path=fixture_audit_json,
        config=CardConfig(),
        policy=policy,
        policy_path=policy_path,
    )
    assert "Compliance gates" in md
    assert "PASSED" in md
    assert "risk@example.com" in md


def test_render_marks_card_as_no_policy_when_omitted(fixture_audit_json):
    payload = json.loads(fixture_audit_json.read_text())
    md = render_markdown(
        audit_payload=payload,
        audit_json_path=fixture_audit_json,
        config=CardConfig(),
    )
    assert "No fairness policy was supplied" in md


def test_render_raises_on_missing_baseline_or_fair(fixture_audit_json):
    with pytest.raises(ValueError, match="baseline"):
        render_markdown(
            audit_payload={"_meta": {}},
            audit_json_path=fixture_audit_json,
            config=CardConfig(),
        )


def test_render_includes_todo_for_missing_config_fields(fixture_audit_json):
    payload = json.loads(fixture_audit_json.read_text())
    md = render_markdown(
        audit_payload=payload,
        audit_json_path=fixture_audit_json,
        config=CardConfig(),  # all defaults
    )
    assert "TODO(maintainer)" in md


def test_build_card_writes_file(tmp_path, fixture_audit_json):
    out = tmp_path / "MODEL_CARD.md"
    rendered = build_card_from_audit(
        audit_json_path=fixture_audit_json,
        output_path=out,
        config=CardConfig(model_name="Demo"),
    )
    assert out.exists()
    assert "Demo" in rendered
    assert out.read_text() == rendered


# --- CLI integration -------------------------------------------------------

def test_cli_card_with_no_config_produces_card(tmp_path, fixture_audit_json):
    out = tmp_path / "MODEL_CARD.md"
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "card",
         str(fixture_audit_json), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.exists()
    text = out.read_text()
    assert text.startswith("---\n")
    assert "TODO(maintainer)" in text  # no config supplied


def test_cli_card_with_policy_includes_gates(tmp_path, fixture_audit_json):
    policy = _write(tmp_path, "fairness.yaml", """
version: 1
target: income
sensitive: [sex, race]
gates:
  demographic_parity_difference: { max: 0.10 }
""")
    out = tmp_path / "MODEL_CARD.md"
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "card",
         str(fixture_audit_json),
         "--output", str(out),
         "--policy", str(policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    text = out.read_text()
    assert "Compliance gates" in text
    assert "PASSED" in text


def test_cli_card_with_config_uses_supplied_fields(tmp_path, fixture_audit_json):
    config = _write(tmp_path, "card.yaml", """
model_name: "My credit model"
license: "Proprietary"
description: "Predicts whether a loan should be approved."
""")
    out = tmp_path / "MODEL_CARD.md"
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "card",
         str(fixture_audit_json),
         "--output", str(out),
         "--config", str(config)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    text = out.read_text()
    assert "My credit model" in text
    assert "Proprietary" in text
    assert "Predicts whether a loan should be approved" in text


def test_cli_card_returns_2_when_audit_missing(tmp_path):
    out = tmp_path / "MODEL_CARD.md"
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "card",
         str(tmp_path / "missing.json"),
         "--output", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2


def test_cli_card_returns_4_on_policy_schema_error(tmp_path, fixture_audit_json):
    bad_policy = _write(tmp_path, "fairness.yaml", """
version: 1
target: income
sensitive: [sex]
gates:
  not_a_real_metric: { max: 0.1 }
""")
    out = tmp_path / "MODEL_CARD.md"
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "card",
         str(fixture_audit_json),
         "--output", str(out),
         "--policy", str(bad_policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 4
