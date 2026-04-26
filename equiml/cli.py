"""
equiml CLI - Quick fairness audits for ML datasets.

Usage:
    equiml audit data.csv --target income --sensitive gender race
    equiml audit data.csv --target income --sensitive gender --report report.html
"""

import argparse
import json
import sys
import os
import logging

import pandas as pd
import numpy as np

from .data import Data
from .model import Model
from .evaluation import EquiMLEvaluation
from .reporting import generate_html_report

logger = logging.getLogger(__name__)


def _load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV, JSON, Excel, or Parquet."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path}. Use CSV, JSON, Excel, or Parquet.")


def cmd_audit(args: argparse.Namespace) -> None:
    """Run a fairness audit on a dataset."""
    dataset_path = args.dataset
    target = args.target
    sensitive_cols = args.sensitive
    algorithm = args.algorithm
    output_json = args.output
    output_html = args.report

    if not os.path.exists(dataset_path):
        print(f"Error: file '{dataset_path}' not found.")
        sys.exit(1)

    print(f"Loading {dataset_path}...")
    df = _load_dataset(dataset_path)
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    # Validate columns
    missing = [c for c in [target] + sensitive_cols if c not in df.columns]
    if missing:
        print(f"Error: columns not found in dataset: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Identify feature types (exclude target, but keep sensitive for encoding)
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numerical = [c for c in numerical if c != target]
    categorical = [c for c in categorical if c != target]

    # Prepare data
    data = Data(dataset_path=dataset_path, sensitive_features=sensitive_cols)
    data.df = df
    data.preprocess(
        target_column=target,
        numerical_features=numerical if numerical else None,
        categorical_features=categorical if categorical else None,
    )
    data.split_data(test_size=0.2)

    # Find sensitive feature columns after encoding (exact-prefix match
    # so e.g. "race" does not match "racing_score").
    def _encoded_cols_for(sf: str) -> list:
        return [
            col for col in data.X_train.columns
            if col == sf or col.startswith(f"{sf}_")
        ]

    # Map each requested sensitive feature to a single representative
    # encoded column (first one). Skip features with no matching column.
    per_feature_encoded: dict = {}
    for sf in sensitive_cols:
        cols = _encoded_cols_for(sf)
        if cols:
            per_feature_encoded[sf] = cols[0]

    # The first sensitive feature is "primary": used for the constrained
    # training and for the top-level metrics dict (kept for backward
    # compat with _print_results and the HTML report).
    primary_sf = sensitive_cols[0] if sensitive_cols else None
    primary_col = per_feature_encoded.get(primary_sf) if primary_sf else None

    sensitive_train = data.X_train[primary_col] if primary_col else None
    sensitive_test = data.X_test[primary_col] if primary_col else None

    # Train model
    print(f"Training {algorithm} model...")
    model = Model(algorithm=algorithm)
    model.train(data.X_train, data.y_train, sensitive_features=sensitive_train)

    # Also train a fair variant for comparison.
    # NOTE: ExponentiatedGradient accepts a single sensitive vector, so
    # the constrained training uses ONLY the primary (first) sensitive
    # feature. Per-feature audit metrics below are evaluated against this
    # same model.
    print("Training fair model (demographic parity)...")
    fair_model = Model(algorithm=algorithm, fairness_constraint="demographic_parity")
    fair_model.train(data.X_train, data.y_train, sensitive_features=sensitive_train)

    # Evaluate both against the primary sensitive feature
    evaluation = EquiMLEvaluation()

    predictions = model.predict(data.X_test)
    metrics = evaluation.evaluate(
        model, data.X_test, data.y_test,
        y_pred=predictions,
        sensitive_features=sensitive_test,
    )

    fair_predictions = fair_model.predict(data.X_test)
    fair_metrics = evaluation.evaluate(
        fair_model, data.X_test, data.y_test,
        y_pred=fair_predictions,
        sensitive_features=sensitive_test,
    )

    # Per-sensitive-feature evaluation: compute metrics against each
    # sensitive feature individually so multi-sensitive audits are not
    # silently reduced to the first one.
    per_sensitive_baseline: dict = {}
    per_sensitive_fair: dict = {}
    for sf, encoded_col in per_feature_encoded.items():
        sf_test = data.X_test[encoded_col]
        per_sensitive_baseline[sf] = evaluation.evaluate(
            model, data.X_test, data.y_test,
            y_pred=predictions,
            sensitive_features=sf_test,
        )
        per_sensitive_fair[sf] = evaluation.evaluate(
            fair_model, data.X_test, data.y_test,
            y_pred=fair_predictions,
            sensitive_features=sf_test,
        )
    metrics["per_sensitive"] = per_sensitive_baseline
    fair_metrics["per_sensitive"] = per_sensitive_fair

    # Print results
    _print_results(metrics, fair_metrics, sensitive_cols)

    # Save JSON output
    if output_json:
        serializable = _make_serializable({"baseline": metrics, "fair": fair_metrics})
        with open(output_json, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_json}")

    # Generate HTML report
    if output_html:
        generate_html_report(metrics, output_path=output_html,
                           template_path=os.path.join(os.path.dirname(__file__), "report_template.html"))
        print(f"HTML report saved to {output_html}")


def _print_results(metrics: dict, fair_metrics: dict, sensitive_cols: list) -> None:
    """Print audit results to console."""
    print("\n" + "=" * 60)
    print("  FAIRNESS AUDIT RESULTS")
    print("=" * 60)

    # Performance
    print("\n  PERFORMANCE (Baseline / Fair Model)")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.1%}  /  {fair_metrics.get('accuracy', 0):.1%}")
    print(f"  F1-Score:  {metrics.get('f1_score', 0):.1%}  /  {fair_metrics.get('f1_score', 0):.1%}")
    if "roc_auc" in metrics:
        print(f"  ROC AUC:   {metrics.get('roc_auc', 0):.3f}  /  {fair_metrics.get('roc_auc', 0):.3f}")

    # Fairness
    dp_base = abs(metrics.get("demographic_parity_difference", 0))
    dp_fair = abs(fair_metrics.get("demographic_parity_difference", 0))
    eo_base = abs(metrics.get("equalized_odds_difference", 0))
    eo_fair = abs(fair_metrics.get("equalized_odds_difference", 0))

    print(f"\n  FAIRNESS (lower = fairer)")
    print(f"  Demographic Parity:  {dp_base:.3f}  /  {dp_fair:.3f}")
    print(f"  Equalized Odds:      {eo_base:.3f}  /  {eo_fair:.3f}")

    # Assessment
    max_bias = max(dp_base, eo_base)
    print(f"\n  ASSESSMENT")
    if max_bias <= 0.05:
        print("  LOW BIAS - Model appears fair across groups")
    elif max_bias <= 0.1:
        print("  MODERATE BIAS - Consider applying fairness constraints")
    elif max_bias <= 0.2:
        print("  SIGNIFICANT BIAS - Fairness mitigation recommended")
    else:
        print("  HIGH BIAS - Fairness mitigation strongly recommended")

    # Improvement
    if dp_fair < dp_base:
        improvement = (1 - dp_fair / dp_base) * 100 if dp_base > 0 else 0
        print(f"\n  Fair model reduces demographic parity gap by {improvement:.0f}%")
    print()


def _make_serializable(obj):
    """Make metrics dict JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(
        prog="equiml",
        description="Quick fairness audits for ML datasets",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Run fairness audit on a dataset")
    audit_parser.add_argument("dataset", help="Path to dataset (CSV, JSON, Excel, Parquet)")
    audit_parser.add_argument("--target", "-t", required=True, help="Target column name")
    audit_parser.add_argument("--sensitive", "-s", nargs="+", required=True, help="Sensitive feature columns")
    audit_parser.add_argument("--algorithm", "-a", default="logistic_regression",
                            choices=["logistic_regression", "random_forest", "ensemble"],
                            help="ML algorithm (default: logistic_regression)")
    audit_parser.add_argument("--output", "-o", help="Save detailed results as JSON")
    audit_parser.add_argument("--report", "-r", help="Generate HTML report")
    audit_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "audit":
        import warnings
        if args.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.ERROR)
            warnings.filterwarnings("ignore")
        cmd_audit(args)


if __name__ == "__main__":
    main()
