#!/usr/bin/env python3
"""
Quick Bias Check Script
Command-line tool for instant bias analysis
"""

import argparse
import pandas as pd
import sys
import os

# Add EquiML to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data import Data
from src.model import Model
from src.evaluation import EquiMLEvaluation

def quick_bias_analysis(dataset_path, target_col, sensitive_cols):
    """Perform quick bias analysis and return results"""

    print(f" Analyzing {dataset_path} for bias...")

    # Load data
    data = Data(dataset_path=dataset_path, sensitive_features=sensitive_cols)
    data.load_data()

    print(f" Dataset: {data.df.shape[0]} rows, {data.df.shape[1]} columns")

    # Quick preprocessing
    data.preprocess(
        target_column=target_col,
        numerical_features=data.df.select_dtypes(include=['number']).columns.tolist(),
        categorical_features=data.df.select_dtypes(include=['object']).columns.tolist()
    )

    data.split_data(test_size=0.2)

    # Quick model training
    model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')

    # Get features for training
    sensitive_feature_columns = [col for col in data.X_train.columns
                                if any(sf in col for sf in sensitive_cols)]

    if sensitive_feature_columns:
        sensitive_features_train = data.X_train[sensitive_feature_columns[0]]
        X_train = data.X_train.drop(columns=sensitive_feature_columns)
        sensitive_features_test = data.X_test[sensitive_feature_columns[0]]
        X_test = data.X_test.drop(columns=sensitive_feature_columns)
    else:
        print("  Warning: No sensitive feature columns found after preprocessing")
        X_train, X_test = data.X_train, data.X_test
        sensitive_features_train = sensitive_features_test = None

    # Train model
    model.train(X_train, data.y_train, sensitive_features=sensitive_features_train)

    # Evaluate
    evaluation = EquiMLEvaluation()
    predictions = model.predict(X_test)
    metrics = evaluation.evaluate(model, X_test, data.y_test,
                                y_pred=predictions,
                                sensitive_features=sensitive_features_test)

    return metrics

def print_results(metrics):
    """Print bias analysis results in a user-friendly format"""

    print("\n" + "="*60)
    print(" BIAS ANALYSIS RESULTS")
    print("="*60)

    # Performance metrics
    print(f"\n PERFORMANCE:")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
    print(f"   F1-Score: {metrics.get('f1_score', 0):.1%}")

    # Fairness metrics
    print(f"\n  FAIRNESS:")
    dp_diff = metrics.get('demographic_parity_difference', 0)
    eo_diff = metrics.get('equalized_odds_difference', 0)

    print(f"   Demographic Parity: {abs(dp_diff):.1%}")
    print(f"   Equalized Odds: {abs(eo_diff):.1%}")

    # Overall assessment
    max_bias = max(abs(dp_diff), abs(eo_diff))

    print(f"\n OVERALL ASSESSMENT:")

    if max_bias <= 0.1:
        print("    EXCELLENT - Low bias detected")
        print("   🏆 Eligible for EquiML GOLD certification")
    elif max_bias <= 0.2:
        print("    MODERATE - Some bias detected")
        print("   🥈 Eligible for EquiML SILVER certification")
    elif max_bias <= 0.3:
        print("   🟠 CONCERNING - Significant bias detected")
        print("   🥉 Eligible for EquiML BRONZE certification")
    else:
        print("    CRITICAL - High bias detected")
        print("    Immediate action required")

    print(f"\n RECOMMENDATIONS:")
    if max_bias > 0.2:
        print("   1. Apply bias mitigation techniques")
        print("   2. Use fairness-constrained training")
        print("   3. Implement real-time monitoring")
        print("   4. Consider data augmentation")
    elif max_bias > 0.1:
        print("   1. Monitor for bias drift")
        print("   2. Consider fairness constraints")
        print("   3. Validate on diverse test sets")
    else:
        print("   1. Maintain current fairness practices")
        print("   2. Set up monitoring for production")
        print("   3. Share your success story!")

    print(f"\n🔗 NEXT STEPS:")
    print("   • Get detailed analysis: https://equiml.ai")
    print("   • Read guides: docs/guides/")
    print("   • Join community: github.com/mkupermann/EquiML")

def main():
    parser = argparse.ArgumentParser(
        description="Quick bias analysis for any dataset",
        epilog="Example: python quick_bias_check.py data.csv --target income --sensitive sex race"
    )

    parser.add_argument('dataset', help='Path to CSV dataset')
    parser.add_argument('--target', '-t', required=True,
                       help='Target column name (what you\'re predicting)')
    parser.add_argument('--sensitive', '-s', nargs='+', required=True,
                       help='Sensitive feature column names (space-separated)')
    parser.add_argument('--output', '-o', help='Output file for detailed results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed metrics')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.dataset):
        print(f" Error: Dataset file '{args.dataset}' not found")
        sys.exit(1)

    try:
        # Run analysis
        metrics = quick_bias_analysis(args.dataset, args.target, args.sensitive)

        # Print results
        print_results(metrics)

        # Save detailed results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: {args.output}")

        # Verbose output
        if args.verbose:
            print(f"\n DETAILED METRICS:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")

    except Exception as e:
        print(f" Error during analysis: {str(e)}")
        print(f" Try: python quick_bias_check.py --help")
        sys.exit(1)

if __name__ == "__main__":
    main()