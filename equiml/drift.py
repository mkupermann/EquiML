"""Fairness drift monitoring.

See docs/rfcs/0003-fairness-drift-monitoring.md.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)

from .policy import Policy, PolicyResult, evaluate_policy

logger = logging.getLogger(__name__)


# PSI thresholds — credit-risk convention.
PSI_NO_DRIFT = 0.10
PSI_MATERIAL_DRIFT = 0.25


@dataclass
class Batch:
    """One observation in a FairnessDriftMonitor stream."""
    timestamp: datetime
    n_samples: int
    metrics: dict[str, float]
    per_sensitive: dict[str, dict[str, float]]
    group_rates: dict[str, dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat()
        return d

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "Batch":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            n_samples=int(data["n_samples"]),
            metrics=dict(data.get("metrics", {})),
            per_sensitive={k: dict(v) for k, v in data.get("per_sensitive", {}).items()},
            group_rates={k: dict(v) for k, v in data.get("group_rates", {}).items()},
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DriftResult:
    psi_per_feature: dict[str, float]
    psi_overall: float
    severity: str  # "none" | "moderate" | "material"
    baseline_n: int
    current_n: int
    notes: list[str] = field(default_factory=list)


class FairnessDriftMonitor:
    """Time-aware fairness monitor for production deployment streams."""

    def __init__(
        self,
        sensitive_features: list[str],
        state_path: str | Path | None = None,
    ):
        self.sensitive_features = list(sensitive_features)
        self.state_path = Path(state_path) if state_path else None
        self.batches: list[Batch] = []
        if self.state_path and self.state_path.exists():
            self._load()

    # ----- record / persist -------------------------------------------------

    def record(
        self,
        predictions: np.ndarray | pd.Series,
        sensitive_features: pd.DataFrame,
        true_labels: np.ndarray | pd.Series | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Batch:
        """Compute metrics for a batch and append to the monitor."""
        ts = timestamp or datetime.now(timezone.utc)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_numpy()
        if isinstance(true_labels, pd.Series):
            true_labels = true_labels.to_numpy()

        if not isinstance(sensitive_features, pd.DataFrame):
            raise TypeError("sensitive_features must be a pandas DataFrame.")
        for sf in self.sensitive_features:
            if sf not in sensitive_features.columns:
                raise ValueError(
                    f"Sensitive feature '{sf}' not in DataFrame columns "
                    f"{list(sensitive_features.columns)}."
                )

        n = len(predictions)
        # Top-level metrics computed against the primary (first) sensitive feature.
        primary = sensitive_features[self.sensitive_features[0]]
        primary_arr = primary.to_numpy()
        top_metrics = self._compute_metrics(predictions, primary_arr, true_labels)

        # Per-feature metrics.
        per_sf: dict[str, dict[str, float]] = {}
        group_rates: dict[str, dict[str, float]] = {}
        for sf in self.sensitive_features:
            sf_values = sensitive_features[sf].to_numpy()
            per_sf[sf] = self._compute_metrics(predictions, sf_values, true_labels)
            group_rates[sf] = self._compute_group_rates(predictions, sf_values)

        batch = Batch(
            timestamp=ts,
            n_samples=n,
            metrics=top_metrics,
            per_sensitive=per_sf,
            group_rates=group_rates,
            metadata=metadata or {},
        )
        self.batches.append(batch)
        if self.state_path:
            self._append_to_state(batch)
        return batch

    @staticmethod
    def _compute_metrics(
        predictions: np.ndarray,
        sensitive: np.ndarray,
        true_labels: np.ndarray | None,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        # DP works without labels.
        try:
            out["demographic_parity_difference"] = float(
                demographic_parity_difference(
                    np.zeros_like(predictions) if true_labels is None else true_labels,
                    predictions,
                    sensitive_features=sensitive,
                )
            )
        except Exception as e:
            logger.debug(f"demographic_parity_difference failed: {e}")
        if true_labels is not None:
            try:
                out["equalized_odds_difference"] = float(
                    equalized_odds_difference(
                        true_labels, predictions, sensitive_features=sensitive,
                    )
                )
            except Exception as e:
                logger.debug(f"equalized_odds_difference failed: {e}")
            try:
                out["equal_opportunity_difference"] = float(
                    equal_opportunity_difference(
                        true_labels, predictions, sensitive_features=sensitive,
                    )
                )
            except Exception as e:
                logger.debug(f"equal_opportunity_difference failed: {e}")
        return out

    @staticmethod
    def _compute_group_rates(
        predictions: np.ndarray,
        sensitive: np.ndarray,
    ) -> dict[str, float]:
        rates: dict[str, float] = {}
        groups = pd.unique(sensitive)
        for g in groups:
            mask = sensitive == g
            if mask.sum() > 0:
                rates[str(g)] = float(np.mean(predictions[mask]))
        return rates

    # ----- persistence ------------------------------------------------------

    def _append_to_state(self, batch: Batch) -> None:
        assert self.state_path is not None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "a") as f:
            f.write(json.dumps(batch.to_jsonable(), default=str))
            f.write("\n")

    def _load(self) -> None:
        assert self.state_path is not None
        self.batches = []
        with open(self.state_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.batches.append(Batch.from_jsonable(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed monitor line: {e}")

    # ----- windows ----------------------------------------------------------

    def window(
        self,
        days: int | None = None,
        n_batches: int | None = None,
        until: datetime | None = None,
    ) -> list[Batch]:
        """Return batches in a time window or by count."""
        if days is None and n_batches is None:
            return list(self.batches)
        end = until or datetime.now(timezone.utc)
        if days is not None:
            start = end - timedelta(days=days)
            return [b for b in self.batches if start <= b.timestamp <= end]
        # n_batches
        return list(self.batches[-n_batches:])

    @staticmethod
    def aggregate_window(batches: Iterable[Batch]) -> dict[str, Any]:
        """Aggregate a window of batches into a single metric block.

        Top-level metrics are computed as a sample-weighted average across
        batches. Group rates are aggregated per (feature, group) the same way.
        """
        batches = list(batches)
        if not batches:
            return {"n_samples": 0, "metrics": {}, "per_sensitive": {}, "group_rates": {}}

        total_n = sum(b.n_samples for b in batches)
        if total_n == 0:
            return {"n_samples": 0, "metrics": {}, "per_sensitive": {}, "group_rates": {}}

        # Weighted average top-level metrics.
        metric_keys = {k for b in batches for k in b.metrics}
        agg_metrics: dict[str, float] = {}
        for k in metric_keys:
            num = sum(b.metrics.get(k, 0.0) * b.n_samples for b in batches if k in b.metrics)
            denom = sum(b.n_samples for b in batches if k in b.metrics)
            if denom > 0:
                agg_metrics[k] = num / denom

        # Per-sensitive aggregation.
        per_sf: dict[str, dict[str, float]] = {}
        sf_set = {sf for b in batches for sf in b.per_sensitive}
        for sf in sf_set:
            sub_keys = {k for b in batches for k in b.per_sensitive.get(sf, {})}
            per_sf[sf] = {}
            for k in sub_keys:
                num = sum(
                    b.per_sensitive[sf].get(k, 0.0) * b.n_samples
                    for b in batches if sf in b.per_sensitive and k in b.per_sensitive[sf]
                )
                denom = sum(
                    b.n_samples for b in batches
                    if sf in b.per_sensitive and k in b.per_sensitive[sf]
                )
                if denom > 0:
                    per_sf[sf][k] = num / denom

        # Group-rate aggregation.
        group_rates: dict[str, dict[str, float]] = {}
        for sf in sf_set:
            group_rates[sf] = {}
            groups_seen = {g for b in batches for g in b.group_rates.get(sf, {})}
            for g in groups_seen:
                num = sum(
                    b.group_rates[sf].get(g, 0.0) * b.n_samples
                    for b in batches if sf in b.group_rates and g in b.group_rates[sf]
                )
                denom = sum(
                    b.n_samples for b in batches
                    if sf in b.group_rates and g in b.group_rates[sf]
                )
                if denom > 0:
                    group_rates[sf][g] = num / denom

        return {
            "n_samples": total_n,
            "metrics": agg_metrics,
            "per_sensitive": per_sf,
            "group_rates": group_rates,
        }

    # ----- drift detection --------------------------------------------------

    def detect_drift(
        self,
        current_window: list[Batch],
        baseline_window: list[Batch],
    ) -> DriftResult:
        """Compute PSI on group selection rates between baseline and current windows.

        PSI is computed per sensitive feature; the "overall" PSI is the max
        across features. PSI < 0.10 = no drift; 0.10–0.25 = moderate;
        >= 0.25 = material.
        """
        notes: list[str] = []
        if not baseline_window:
            notes.append("baseline window empty — drift cannot be computed")
            return DriftResult(
                psi_per_feature={},
                psi_overall=0.0,
                severity="none",
                baseline_n=0,
                current_n=sum(b.n_samples for b in current_window),
                notes=notes,
            )
        if not current_window:
            notes.append("current window empty — drift cannot be computed")
            return DriftResult(
                psi_per_feature={},
                psi_overall=0.0,
                severity="none",
                baseline_n=sum(b.n_samples for b in baseline_window),
                current_n=0,
                notes=notes,
            )

        baseline_agg = self.aggregate_window(baseline_window)
        current_agg = self.aggregate_window(current_window)

        psi_per_feature: dict[str, float] = {}
        for sf in self.sensitive_features:
            base_rates = baseline_agg["group_rates"].get(sf, {})
            curr_rates = current_agg["group_rates"].get(sf, {})
            if not base_rates or not curr_rates:
                notes.append(f"feature '{sf}' missing rates in one window")
                continue
            psi_per_feature[sf] = _psi(base_rates, curr_rates)

        psi_overall = max(psi_per_feature.values(), default=0.0)
        if psi_overall >= PSI_MATERIAL_DRIFT:
            severity = "material"
        elif psi_overall >= PSI_NO_DRIFT:
            severity = "moderate"
        else:
            severity = "none"

        return DriftResult(
            psi_per_feature=psi_per_feature,
            psi_overall=psi_overall,
            severity=severity,
            baseline_n=baseline_agg["n_samples"],
            current_n=current_agg["n_samples"],
            notes=notes,
        )

    # ----- policy integration -----------------------------------------------

    def evaluate_against_policy(
        self,
        policy: Policy,
        window: list[Batch] | None = None,
    ) -> PolicyResult:
        """Evaluate a fairness policy against the (aggregated) window's metrics."""
        if window is None:
            window = self.batches
        agg = self.aggregate_window(window)
        # PolicyResult expects `metrics` with top-level keys plus a `per_sensitive` block.
        metrics_view: dict[str, Any] = dict(agg["metrics"])
        metrics_view["per_sensitive"] = agg["per_sensitive"]
        return evaluate_policy(metrics_view, policy)


def render_drift_report(
    monitor: FairnessDriftMonitor,
    baseline_window: list[Batch],
    current_window: list[Batch],
    drift: DriftResult,
    policy_result: PolicyResult | None = None,
    policy_path: str | Path | None = None,
    baseline_days: int | None = None,
    current_days: int | None = None,
    template_dir: str | Path | None = None,
) -> str:
    """Render a markdown drift report."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    if template_dir is None:
        template_dir = Path(__file__).parent
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(disabled_extensions=("md", "j2")),
        keep_trailing_newline=True,
    )
    template = env.get_template("drift_report_template.md.j2")

    baseline_agg = monitor.aggregate_window(baseline_window)
    current_agg = monitor.aggregate_window(current_window)
    metric_keys = sorted(set(baseline_agg["metrics"]) | set(current_agg["metrics"]))

    return template.render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        state_path=str(monitor.state_path) if monitor.state_path else "(in-memory)",
        baseline_days=baseline_days if baseline_days is not None else "all",
        current_days=current_days if current_days is not None else "all",
        baseline_n_batches=len(baseline_window),
        current_n_batches=len(current_window),
        baseline_n_samples=baseline_agg["n_samples"],
        current_n_samples=current_agg["n_samples"],
        baseline_group_rates=baseline_agg["group_rates"],
        current_group_rates=current_agg["group_rates"],
        baseline_metrics=baseline_agg["metrics"],
        current_metrics=current_agg["metrics"],
        metric_keys=metric_keys,
        drift=drift,
        policy_result=policy_result,
        policy_path=str(policy_path) if policy_path else None,
    )


def _psi(base: dict[str, float], current: dict[str, float], eps: float = 1e-6) -> float:
    """Population Stability Index between two distributions over the same keys.

    Both inputs are normalised so they sum to 1 (defensive: callers may pass
    selection-rate vectors that don't sum to 1; we treat them as relative
    frequencies and renormalise).
    """
    keys = set(base.keys()) | set(current.keys())
    base_vals = np.array([max(base.get(k, 0.0), eps) for k in keys], dtype=float)
    curr_vals = np.array([max(current.get(k, 0.0), eps) for k in keys], dtype=float)
    base_p = base_vals / base_vals.sum()
    curr_p = curr_vals / curr_vals.sum()
    return float(np.sum((curr_p - base_p) * np.log(curr_p / base_p)))
