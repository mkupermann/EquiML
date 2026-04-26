"""Auto model-card generation from an EquiML audit JSON.

See docs/rfcs/0002-model-card-generation.md for the design.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .policy import Policy, PolicyResult, evaluate_policy

logger = logging.getLogger(__name__)


_TEMPLATE_FILENAME = "model_card_template.md.j2"

_TODO_PLACEHOLDER = "TODO(maintainer): fill in"


@dataclass
class CardConfig:
    """Author-supplied context the audit cannot know."""

    model_name: str = "TODO(maintainer): name your model"
    model_version: str = _TODO_PLACEHOLDER
    model_type: str = _TODO_PLACEHOLDER
    description: str = (
        "TODO(maintainer): one paragraph in your own voice. What does this "
        "model do? Why does it exist? What outcome does it influence?"
    )
    license: str = "unspecified"
    contact: str = _TODO_PLACEHOLDER
    intended_use: dict = field(default_factory=lambda: {
        "primary_use": _TODO_PLACEHOLDER,
        "primary_users": _TODO_PLACEHOLDER,
        "out_of_scope": [_TODO_PLACEHOLDER],
    })
    training_data: dict = field(default_factory=lambda: {
        "source": _TODO_PLACEHOLDER,
        "preprocessing": _TODO_PLACEHOLDER,
        "notes": _TODO_PLACEHOLDER,
    })
    ethical_considerations: list = field(default_factory=lambda: [
        _TODO_PLACEHOLDER + ": list the ethical risks you have considered.",
    ])
    caveats: list = field(default_factory=lambda: [
        _TODO_PLACEHOLDER + ": list deployment caveats specific to this model.",
    ])


def load_card_config(path: str | Path) -> CardConfig:
    """Parse a card config YAML into a CardConfig. Missing fields fall back to defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Card config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Card config at {path} must be a YAML mapping.")

    defaults = CardConfig()
    return CardConfig(
        model_name=raw.get("model_name", defaults.model_name),
        model_version=raw.get("model_version", defaults.model_version),
        model_type=raw.get("model_type", defaults.model_type),
        description=raw.get("description", defaults.description),
        license=raw.get("license", defaults.license),
        contact=raw.get("contact", defaults.contact),
        intended_use={**defaults.intended_use, **(raw.get("intended_use") or {})},
        training_data={**defaults.training_data, **(raw.get("training_data") or {})},
        ethical_considerations=raw.get("ethical_considerations") or defaults.ethical_considerations,
        caveats=raw.get("caveats") or defaults.caveats,
    )


def _safe_metric(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_metrics_view(metrics: dict[str, Any]) -> dict[str, Any]:
    """Project a metrics dict to the view the template expects."""
    return {
        "accuracy": _safe_metric(metrics, "accuracy") or 0.0,
        "f1_score": _safe_metric(metrics, "f1_score") or 0.0,
        "roc_auc": _safe_metric(metrics, "roc_auc"),
        "demographic_parity_difference": _safe_metric(metrics, "demographic_parity_difference") or 0.0,
        "equalized_odds_difference": _safe_metric(metrics, "equalized_odds_difference") or 0.0,
        "equal_opportunity_difference": _safe_metric(metrics, "equal_opportunity_difference"),
    }


def _build_per_sensitive_view(
    baseline: dict[str, Any],
    fair: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    base_per = baseline.get("per_sensitive", {}) or {}
    fair_per = fair.get("per_sensitive", {}) or {}
    out: dict[str, dict[str, Any]] = {}
    for feature in base_per:
        out[feature] = {
            "baseline": _build_metrics_view(base_per[feature]),
            "fair": _build_metrics_view(fair_per.get(feature, {})),
        }
    return out


def render_markdown(
    audit_payload: dict[str, Any],
    audit_json_path: str | Path,
    config: CardConfig,
    policy: Policy | None = None,
    policy_path: str | Path | None = None,
    html_report_path: str | Path | None = None,
    template_dir: str | Path | None = None,
) -> str:
    """Render a Hugging Face-compatible markdown model card."""
    baseline = audit_payload.get("baseline", {})
    fair = audit_payload.get("fair", {})
    meta = audit_payload.get("_meta", {})

    if not baseline or not fair:
        raise ValueError(
            "Audit payload must contain 'baseline' and 'fair' blocks. "
            "Was this produced by `equiml audit --output ...`?"
        )

    baseline_view = _build_metrics_view(baseline)
    fair_view = _build_metrics_view(fair)
    per_sensitive_view = _build_per_sensitive_view(baseline, fair)

    policy_result: PolicyResult | None = None
    policy_metadata: dict[str, Any] = {}
    if policy is not None:
        policy_result = evaluate_policy(fair, policy)
        policy_metadata = policy.metadata

    # Build frontmatter as a YAML string. Hand-rolling indentation in Jinja
    # is fragile; yaml.safe_dump produces correct output every time.
    frontmatter_metrics = [
        "accuracy",
        "f1_score",
        "demographic_parity_difference",
        "equalized_odds_difference",
    ]
    if baseline_view["roc_auc"] is not None:
        frontmatter_metrics.insert(2, "roc_auc")

    model_index_metrics = [
        {"type": "accuracy", "value": round(fair_view["accuracy"], 4)},
        {"type": "f1_score", "value": round(fair_view["f1_score"], 4)},
        {"type": "demographic_parity_difference",
         "value": round(fair_view["demographic_parity_difference"], 4)},
        {"type": "equalized_odds_difference",
         "value": round(fair_view["equalized_odds_difference"], 4)},
    ]

    frontmatter_dict = {
        "language": "en",
        "license": config.license,
        "tags": ["fairness-audited", f"equiml-{meta.get('equiml_version', 'unknown')}"],
        "metrics": frontmatter_metrics,
        "model-index": [{
            "name": config.model_name,
            "results": [{
                "task": {"type": "tabular-classification"},
                "dataset": {
                    "type": "tabular",
                    "name": os.path.basename(meta.get("dataset_path", "")) or "unspecified",
                },
                "metrics": model_index_metrics,
            }],
        }],
    }
    frontmatter_yaml = yaml.safe_dump(frontmatter_dict, sort_keys=False, default_flow_style=False).rstrip()

    if template_dir is None:
        template_dir = Path(__file__).parent
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(disabled_extensions=("md", "j2")),
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATE_FILENAME)

    return template.render(
        # Author-supplied
        model_name=config.model_name,
        model_version=config.model_version,
        model_type=config.model_type,
        description=config.description,
        license=config.license,
        contact=config.contact,
        intended_use=config.intended_use,
        training_data=config.training_data,
        ethical_considerations=config.ethical_considerations,
        caveats=config.caveats,
        # Audit-derived
        meta=meta,
        baseline=baseline_view,
        fair=fair_view,
        per_sensitive=per_sensitive_view,
        equiml_version=meta.get("equiml_version", "unknown"),
        audit_json_path=str(audit_json_path),
        html_report_path=str(html_report_path) if html_report_path else None,
        # Frontmatter
        frontmatter_yaml=frontmatter_yaml,
        # Policy
        policy_result=policy_result,
        policy_metadata=policy_metadata,
        policy_path=str(policy_path) if policy_path else None,
        # Run info
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def build_card_from_audit(
    audit_json_path: str | Path,
    output_path: str | Path,
    policy: Policy | None = None,
    policy_path: str | Path | None = None,
    config: CardConfig | None = None,
    html_report_path: str | Path | None = None,
) -> str:
    """Generate and write a model card from an audit JSON. Returns the rendered markdown."""
    audit_json_path = Path(audit_json_path)
    if not audit_json_path.exists():
        raise FileNotFoundError(f"Audit JSON not found: {audit_json_path}")

    with open(audit_json_path) as f:
        payload = json.load(f)

    if config is None:
        config = CardConfig()

    rendered = render_markdown(
        audit_payload=payload,
        audit_json_path=audit_json_path,
        config=config,
        policy=policy,
        policy_path=policy_path,
        html_report_path=html_report_path,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    return rendered
