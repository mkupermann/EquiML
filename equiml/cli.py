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
from datetime import timedelta
from pathlib import Path

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
from .drift import (
    FairnessDriftMonitor,
    PSI_NO_DRIFT,
    render_drift_report,
)

logger = logging.getLogger(__name__)

# Documented exit codes (see docs/rfcs/0001-policy-as-code.md, 0003-fairness-drift-monitoring.md)
EXIT_SUCCESS = 0
EXIT_DATA_ERROR = 2
EXIT_POLICY_GATE_BREACHED = 3
EXIT_POLICY_SCHEMA_ERROR = 4
EXIT_DRIFT_BREACHED = 5


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


def _parse_metadata_pairs(items: list[str] | None) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--metadata expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def cmd_monitor_record(args: argparse.Namespace) -> int:
    """Append one batch (from a CSV) to a monitor's JSONL state."""
    state_path = args.state
    batch_path = args.batch
    predictions_col = args.predictions_col
    sensitive = args.sensitive
    labels_col = getattr(args, "labels_col", None)
    metadata_pairs = getattr(args, "metadata", None) or []

    if not os.path.exists(batch_path):
        print(f"Error: batch file '{batch_path}' not found.", file=sys.stderr)
        return EXIT_DATA_ERROR

    try:
        df = pd.read_csv(batch_path)
    except Exception as e:
        print(f"Could not read batch CSV: {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    sensitive_cols = [s.strip() for s in sensitive.split(",") if s.strip()]
    missing = [c for c in [predictions_col] + sensitive_cols if c not in df.columns]
    if labels_col and labels_col not in df.columns:
        missing.append(labels_col)
    if missing:
        print(f"Error: columns not found in batch CSV: {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        return EXIT_DATA_ERROR

    try:
        metadata = _parse_metadata_pairs(metadata_pairs)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    monitor = FairnessDriftMonitor(
        sensitive_features=sensitive_cols,
        state_path=state_path,
    )
    batch = monitor.record(
        predictions=df[predictions_col].to_numpy(),
        sensitive_features=df[sensitive_cols],
        true_labels=df[labels_col].to_numpy() if labels_col else None,
        metadata=metadata,
    )
    print(f"Recorded batch at {batch.timestamp.isoformat()} ({batch.n_samples} samples) → {state_path}")
    return EXIT_SUCCESS


def cmd_monitor_check(args: argparse.Namespace) -> int:
    """Evaluate the latest window against a baseline (drift) and an optional policy."""
    state_path = args.state
    baseline_days = args.baseline_days
    current_days = args.current_days
    psi_threshold = args.psi_threshold
    policy_path = getattr(args, "policy", None)
    sensitive = args.sensitive

    if not os.path.exists(state_path):
        print(f"Error: state file '{state_path}' not found.", file=sys.stderr)
        return EXIT_DATA_ERROR

    sensitive_cols = [s.strip() for s in sensitive.split(",") if s.strip()]
    monitor = FairnessDriftMonitor(
        sensitive_features=sensitive_cols,
        state_path=state_path,
    )

    if not monitor.batches:
        print("Error: monitor state contains no batches.", file=sys.stderr)
        return EXIT_DATA_ERROR

    latest = max(b.timestamp for b in monitor.batches)
    baseline_end = latest - timedelta(days=current_days)
    current_window = monitor.window(days=current_days)
    baseline_window = monitor.window(days=baseline_days, until=baseline_end)

    drift = monitor.detect_drift(
        current_window=current_window,
        baseline_window=baseline_window,
    )

    print("DRIFT RESULT")
    print(f"  Severity: {drift.severity.upper()}")
    print(f"  PSI overall: {drift.psi_overall:.4f}  (threshold {psi_threshold:.4f})")
    for sf, psi in drift.psi_per_feature.items():
        print(f"    {sf}: {psi:.4f}")
    print(f"  Baseline: {drift.baseline_n} samples · Current: {drift.current_n} samples")
    for note in drift.notes:
        print(f"  Note: {note}")

    drift_breached = drift.psi_overall >= psi_threshold

    policy_breached = False
    if policy_path:
        try:
            policy = load_policy(policy_path)
        except PolicyError as e:
            print(f"Policy schema error: {e}", file=sys.stderr)
            return EXIT_POLICY_SCHEMA_ERROR
        result = monitor.evaluate_against_policy(policy, window=current_window)
        print()
        print(format_result(result, policy_path=policy_path))
        if not result.passed:
            policy_breached = True

    if policy_breached:
        return EXIT_POLICY_GATE_BREACHED
    if drift_breached:
        return EXIT_DRIFT_BREACHED
    return EXIT_SUCCESS


def cmd_monitor_report(args: argparse.Namespace) -> int:
    """Render a markdown drift report from the monitor state."""
    state_path = args.state
    baseline_days = args.baseline_days
    current_days = args.current_days
    output_path = args.output
    policy_path = getattr(args, "policy", None)
    sensitive = args.sensitive

    if not os.path.exists(state_path):
        print(f"Error: state file '{state_path}' not found.", file=sys.stderr)
        return EXIT_DATA_ERROR

    sensitive_cols = [s.strip() for s in sensitive.split(",") if s.strip()]
    monitor = FairnessDriftMonitor(
        sensitive_features=sensitive_cols,
        state_path=state_path,
    )
    if not monitor.batches:
        print("Error: monitor state contains no batches.", file=sys.stderr)
        return EXIT_DATA_ERROR

    latest = max(b.timestamp for b in monitor.batches)
    baseline_end = latest - timedelta(days=current_days)
    current_window = monitor.window(days=current_days)
    baseline_window = monitor.window(days=baseline_days, until=baseline_end)
    drift = monitor.detect_drift(
        current_window=current_window,
        baseline_window=baseline_window,
    )

    policy_result = None
    if policy_path:
        try:
            policy = load_policy(policy_path)
        except PolicyError as e:
            print(f"Policy schema error: {e}", file=sys.stderr)
            return EXIT_POLICY_SCHEMA_ERROR
        policy_result = monitor.evaluate_against_policy(policy, window=current_window)

    md = render_drift_report(
        monitor=monitor,
        baseline_window=baseline_window,
        current_window=current_window,
        drift=drift,
        policy_result=policy_result,
        policy_path=policy_path,
        baseline_days=baseline_days,
        current_days=current_days,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(md)
    print(f"Drift report written to {output_path}")
    return EXIT_SUCCESS


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

    # monitor command (with sub-subcommands record/check/report)
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Production fairness drift monitoring (record/check/report)",
    )
    monitor_subs = monitor_parser.add_subparsers(dest="monitor_op", required=True)

    rec = monitor_subs.add_parser("record", help="Append a batch to a monitor state file")
    rec.add_argument("--state", required=True, help="Path to JSONL state file (will be created if missing)")
    rec.add_argument("--batch", required=True, help="CSV with predictions + sensitive columns")
    rec.add_argument("--predictions-col", required=True, help="Column name with the model's predictions (0/1)")
    rec.add_argument("--sensitive", required=True, help="Comma-separated sensitive column names")
    rec.add_argument("--labels-col", help="Column name with ground-truth labels (optional)")
    rec.add_argument("--metadata", action="append", help="KEY=VALUE pairs to attach to the batch (repeatable)")

    chk = monitor_subs.add_parser("check", help="Evaluate latest window for drift and policy compliance")
    chk.add_argument("--state", required=True, help="Path to JSONL state file")
    chk.add_argument("--sensitive", required=True, help="Comma-separated sensitive column names")
    chk.add_argument("--baseline-days", type=int, default=30)
    chk.add_argument("--current-days", type=int, default=7)
    chk.add_argument("--psi-threshold", type=float, default=PSI_NO_DRIFT)
    chk.add_argument("--policy", help="Optional fairness.yaml to enforce on the current window")

    rep = monitor_subs.add_parser("report", help="Render a markdown drift report")
    rep.add_argument("--state", required=True, help="Path to JSONL state file")
    rep.add_argument("--sensitive", required=True, help="Comma-separated sensitive column names")
    rep.add_argument("--output", required=True, help="Path to write the markdown report")
    rep.add_argument("--baseline-days", type=int, default=30)
    rep.add_argument("--current-days", type=int, default=7)
    rep.add_argument("--policy", help="Optional fairness.yaml to embed in the report")

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
    elif args.command == "monitor":
        if args.monitor_op == "record":
            sys.exit(cmd_monitor_record(args))
        elif args.monitor_op == "check":
            sys.exit(cmd_monitor_check(args))
        elif args.monitor_op == "report":
            sys.exit(cmd_monitor_report(args))


if __name__ == "__main__":
    main()
