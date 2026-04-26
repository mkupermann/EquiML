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
from .policy import (
    PolicyError,
    evaluate_policy,
    format_result,
    load_policy,
)
from .card import CardConfig, build_card_from_audit, load_card_config

logger = logging.getLogger(__name__)

# Documented exit codes (see docs/rfcs/0001-policy-as-code.md)
EXIT_SUCCESS = 0
EXIT_DATA_ERROR = 2
EXIT_POLICY_GATE_BREACHED = 3
EXIT_POLICY_SCHEMA_ERROR = 4


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


def cmd_audit(args: argparse.Namespace) -> int:
    """Run a fairness audit on a dataset. Returns CLI exit code."""
    dataset_path = args.dataset
    target = args.target
    sensitive_cols = args.sensitive
    algorithm = args.algorithm
    output_json = args.output
    output_html = args.report
    policy_path = getattr(args, "policy", None)

    # Load policy first so a malformed policy fails fast (before training).
    policy = None
    if policy_path:
        try:
            policy = load_policy(policy_path)
        except PolicyError as e:
            print(f"Policy schema error: {e}", file=sys.stderr)
            return EXIT_POLICY_SCHEMA_ERROR

    if not os.path.exists(dataset_path):
        print(f"Error: file '{dataset_path}' not found.")
        return EXIT_DATA_ERROR

    print(f"Loading {dataset_path}...")
    df = _load_dataset(dataset_path)
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    # Validate columns
    missing = [c for c in [target] + sensitive_cols if c not in df.columns]
    if missing:
        print(f"Error: columns not found in dataset: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return EXIT_DATA_ERROR

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
        import platform
        import sklearn
        import fairlearn
        from . import __version__ as equiml_version
        serializable["_meta"] = {
            "equiml_version": equiml_version,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "fairlearn_version": fairlearn.__version__,
            "random_seed": 42,
            "dataset_path": dataset_path,
            "target": target,
            "sensitive_features": sensitive_cols,
            "algorithm": algorithm,
        }
        with open(output_json, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_json}")

    # Generate HTML report
    if output_html:
        generate_html_report(
            metrics,
            output_path=output_html,
            template_path=os.path.join(os.path.dirname(__file__), "report_template.html"),
            fair_metrics=fair_metrics,
        )
        print(f"HTML report saved to {output_html}")

    # Policy evaluation (fairness.yaml)
    if policy is not None:
        # Evaluate the policy against the FAIR model's metrics — the
        # policy gates the deployable model, not the unconstrained baseline.
        result = evaluate_policy(fair_metrics, policy)
        print()
        print(format_result(result, policy_path=policy_path))
        if not result.passed:
            return EXIT_POLICY_GATE_BREACHED

    return EXIT_SUCCESS


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
    print(f"\n  Maximum group disparity: {max_bias:.3f}")
    print("  Note: thresholds for 'acceptable' bias are domain- and")
    print("  jurisdiction-specific. This number is a starting point for")
    print("  review, not a regulatory verdict.")

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


def cmd_card(args: argparse.Namespace) -> int:
    """Generate a model card (markdown) from an audit JSON."""
    audit_path = args.audit_json
    output_path = args.output
    policy_path = getattr(args, "policy", None)
    config_path = getattr(args, "config", None)

    if not os.path.exists(audit_path):
        print(f"Error: audit JSON file '{audit_path}' not found.", file=sys.stderr)
        return EXIT_DATA_ERROR

    policy = None
    if policy_path:
        try:
            policy = load_policy(policy_path)
        except PolicyError as e:
            print(f"Policy schema error: {e}", file=sys.stderr)
            return EXIT_POLICY_SCHEMA_ERROR

    config = CardConfig()
    if config_path:
        try:
            config = load_card_config(config_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Card config error: {e}", file=sys.stderr)
            return EXIT_DATA_ERROR

    try:
        build_card_from_audit(
            audit_json_path=audit_path,
            output_path=output_path,
            policy=policy,
            policy_path=policy_path,
            config=config,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Could not build model card: {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    print(f"Model card written to {output_path}")
    return EXIT_SUCCESS


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a previously-saved audit JSON against a policy file."""
    audit_path = args.audit_json
    policy_path = args.policy

    if not os.path.exists(audit_path):
        print(f"Error: audit JSON file '{audit_path}' not found.", file=sys.stderr)
        return EXIT_DATA_ERROR

    try:
        policy = load_policy(policy_path)
    except PolicyError as e:
        print(f"Policy schema error: {e}", file=sys.stderr)
        return EXIT_POLICY_SCHEMA_ERROR

    try:
        with open(audit_path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Audit JSON is malformed: {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    # The audit JSON is shaped as {"baseline": ..., "fair": ..., "_meta": ...}.
    # Policy gates apply to the deployable (fair) model.
    fair_metrics = payload.get("fair")
    if not isinstance(fair_metrics, dict):
        print(
            "Audit JSON has no 'fair' block; was this produced by `equiml audit`?",
            file=sys.stderr,
        )
        return EXIT_DATA_ERROR

    result = evaluate_policy(fair_metrics, policy)
    print(format_result(result, policy_path=policy_path))

    return EXIT_SUCCESS if result.passed else EXIT_POLICY_GATE_BREACHED


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
    audit_parser.add_argument(
        "--policy", "-p",
        help="Path to fairness.yaml policy file. Audit fails (exit 3) on gate breach; exit 4 on schema error.",
    )
    audit_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a previously-saved audit JSON against a fairness.yaml policy",
    )
    verify_parser.add_argument("audit_json", help="Path to audit JSON produced by `equiml audit --output ...`")
    verify_parser.add_argument(
        "--policy", "-p", required=True,
        help="Path to fairness.yaml policy file",
    )
    verify_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # card command
    card_parser = subparsers.add_parser(
        "card",
        help="Generate a Hugging Face-compatible model card (markdown) from an audit JSON",
    )
    card_parser.add_argument("audit_json", help="Path to audit JSON produced by `equiml audit --output ...`")
    card_parser.add_argument(
        "--output", "-o", required=True,
        help="Path to write the model card (e.g. MODEL_CARD.md)",
    )
    card_parser.add_argument(
        "--policy", "-p",
        help="Optional fairness.yaml policy; gate results render in the card",
    )
    card_parser.add_argument(
        "--config", "-c",
        help="Optional author config YAML (model name, intended use, etc.). "
             "See examples/model_card_config.yaml.",
    )
    card_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(EXIT_SUCCESS)

    import warnings
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
        warnings.filterwarnings("ignore")

    if args.command == "audit":
        sys.exit(cmd_audit(args))
    elif args.command == "verify":
        sys.exit(cmd_verify(args))
    elif args.command == "card":
        sys.exit(cmd_card(args))


if __name__ == "__main__":
    main()
