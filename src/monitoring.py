import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import warnings

logger = logging.getLogger(__name__)

class BiasMonitor:
    """
    Monitor bias and fairness metrics over time for deployed models.
    """

    def __init__(self, sensitive_features: List[str]):
        """
        Initialize bias monitor.

        Args:
            sensitive_features: List of sensitive feature names to monitor
        """
        self.sensitive_features = sensitive_features
        self.monitoring_history = []
        self.alerts = []

    def monitor_predictions(self, predictions: np.ndarray, sensitive_features: pd.DataFrame,
                          true_labels: Optional[np.ndarray] = None,
                          threshold: float = 0.1) -> Dict[str, Any]:
        """
        Monitor predictions for bias and fairness violations.

        Args:
            predictions: Model predictions
            sensitive_features: Sensitive feature values
            true_labels: Optional true labels for accuracy monitoring
            threshold: Bias threshold for alerts

        Returns:
            Monitoring results dictionary
        """
        try:
            timestamp = datetime.now()

            # Calculate fairness metrics
            metrics = self._calculate_fairness_metrics(predictions, sensitive_features, true_labels)

            # Check for bias violations
            violations = self._check_bias_violations(metrics, threshold)

            # Generate alerts if needed
            if violations:
                alert = self._generate_alert(violations, timestamp)
                self.alerts.append(alert)
                logger.warning(f"Bias violation detected: {violations}")

            # Store monitoring record
            record = {
                'timestamp': timestamp,
                'metrics': metrics,
                'violations': violations,
                'prediction_count': len(predictions)
            }
            self.monitoring_history.append(record)

            return record

        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            raise

    def _calculate_fairness_metrics(self, predictions: np.ndarray,
                                  sensitive_features: pd.DataFrame,
                                  true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate fairness metrics for monitoring."""
        metrics = {}

        for feature in self.sensitive_features:
            if feature not in sensitive_features.columns:
                continue

            # Calculate demographic parity
            groups = sensitive_features[feature].unique()
            if len(groups) >= 2:
                group_rates = {}
                for group in groups:
                    group_mask = sensitive_features[feature] == group
                    if group_mask.sum() > 0:
                        positive_rate = np.mean(predictions[group_mask])
                        group_rates[str(group)] = positive_rate

                # Calculate demographic parity difference
                if len(group_rates) >= 2:
                    rates = list(group_rates.values())
                    dp_diff = max(rates) - min(rates)
                    metrics[f'{feature}_demographic_parity_diff'] = dp_diff

                # Calculate accuracy by group if true labels available
                if true_labels is not None:
                    for group in groups:
                        group_mask = sensitive_features[feature] == group
                        if group_mask.sum() > 0:
                            group_acc = np.mean(predictions[group_mask] == true_labels[group_mask])
                            metrics[f'{feature}_{group}_accuracy'] = group_acc

        return metrics

    def _check_bias_violations(self, metrics: Dict[str, float], threshold: float) -> List[str]:
        """Check for bias violations based on threshold."""
        violations = []

        for metric_name, value in metrics.items():
            if 'demographic_parity_diff' in metric_name and value > threshold:
                violations.append(f"Demographic parity violation: {metric_name} = {value:.3f}")
            elif 'accuracy' in metric_name:
                # Check for accuracy disparities between groups
                feature = metric_name.split('_')[0]
                accuracy_metrics = {k: v for k, v in metrics.items()
                                  if k.startswith(feature) and 'accuracy' in k}
                if len(accuracy_metrics) >= 2:
                    acc_values = list(accuracy_metrics.values())
                    acc_diff = max(acc_values) - min(acc_values)
                    if acc_diff > threshold:
                        violations.append(f"Accuracy disparity: {feature} difference = {acc_diff:.3f}")

        return violations

    def _generate_alert(self, violations: List[str], timestamp: datetime) -> Dict[str, Any]:
        """Generate bias alert."""
        return {
            'timestamp': timestamp,
            'severity': 'HIGH' if len(violations) > 2 else 'MEDIUM',
            'violations': violations,
            'message': f"Bias monitoring detected {len(violations)} violation(s)"
        }

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring history."""
        if not self.monitoring_history:
            return {'status': 'No monitoring data available'}

        recent_records = self.monitoring_history[-10:]  # Last 10 records

        summary = {
            'total_monitoring_sessions': len(self.monitoring_history),
            'recent_violations': sum(1 for r in recent_records if r['violations']),
            'active_alerts': len([a for a in self.alerts if a['severity'] == 'HIGH']),
            'last_monitoring': self.monitoring_history[-1]['timestamp'],
            'monitored_features': self.sensitive_features
        }

        return summary

    def export_monitoring_log(self, filepath: str) -> None:
        """Export monitoring history to JSON file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            export_data = []
            for record in self.monitoring_history:
                export_record = record.copy()
                export_record['timestamp'] = record['timestamp'].isoformat()
                export_data.append(export_record)

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Monitoring log exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export monitoring log: {str(e)}")
            raise


class DriftDetector:
    """
    Detect data drift in model inputs and performance.
    """

    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector with reference data.

        Args:
            reference_data: Reference dataset (usually training data)
        """
        self.reference_data = reference_data
        self.reference_stats = self._calculate_data_statistics(reference_data)

    def detect_drift(self, new_data: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift in new data compared to reference.

        Args:
            new_data: New data to check for drift
            threshold: Statistical significance threshold

        Returns:
            Drift detection results
        """
        try:
            new_stats = self._calculate_data_statistics(new_data)

            drift_results = {
                'timestamp': datetime.now(),
                'drift_detected': False,
                'feature_drifts': {},
                'overall_drift_score': 0.0
            }

            drift_scores = []

            for feature in self.reference_data.columns:
                if feature in new_data.columns:
                    drift_score = self._calculate_feature_drift(
                        feature, self.reference_stats, new_stats
                    )
                    drift_results['feature_drifts'][feature] = drift_score
                    drift_scores.append(drift_score)

                    if drift_score > threshold:
                        drift_results['drift_detected'] = True
                        logger.warning(f"Drift detected in feature {feature}: score={drift_score:.4f}")

            if drift_scores:
                drift_results['overall_drift_score'] = np.mean(drift_scores)

            return drift_results

        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            raise

    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for drift detection."""
        stats = {}

        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Numerical features
                stats[column] = {
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'median': data[column].median()
                }
            else:
                # Categorical features
                value_counts = data[column].value_counts(normalize=True)
                stats[column] = value_counts.to_dict()

        return stats

    def _calculate_feature_drift(self, feature: str, ref_stats: Dict, new_stats: Dict) -> float:
        """Calculate drift score for a single feature."""
        try:
            ref_feature_stats = ref_stats.get(feature, {})
            new_feature_stats = new_stats.get(feature, {})

            if not ref_feature_stats or not new_feature_stats:
                return 0.0

            if isinstance(ref_feature_stats, dict) and 'mean' in ref_feature_stats:
                # Numerical feature drift using normalized difference
                ref_mean = ref_feature_stats['mean']
                new_mean = new_feature_stats['mean']
                ref_std = ref_feature_stats['std']

                if ref_std > 0:
                    drift_score = abs(new_mean - ref_mean) / ref_std
                else:
                    drift_score = abs(new_mean - ref_mean)

                return min(drift_score, 1.0)  # Cap at 1.0

            else:
                # Categorical feature drift using Jensen-Shannon divergence
                ref_dist = ref_feature_stats
                new_dist = new_feature_stats

                # Calculate JS divergence (simplified)
                all_categories = set(ref_dist.keys()) | set(new_dist.keys())
                ref_probs = [ref_dist.get(cat, 0) for cat in all_categories]
                new_probs = [new_dist.get(cat, 0) for cat in all_categories]

                # Simple distance measure
                drift_score = sum(abs(r - n) for r, n in zip(ref_probs, new_probs)) / 2
                return drift_score

        except Exception as e:
            logger.warning(f"Failed to calculate drift for feature {feature}: {e}")
            return 0.0