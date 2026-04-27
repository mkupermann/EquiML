"""Demo: a 10-round feedback-loop simulation reproducing the textbook failure.

Creates a synthetic biased dataset where group A is historically over-selected,
runs a threshold-based decision rule with selection-bias feedback for 10
rounds, and prints how the demographic-parity gap evolves.

The point of this script is to make the textbook claim concrete: when only
selected applicants' outcomes feed the next training set, a moderate initial
bias compounds. Run it, read the trajectory, and the abstract argument from
Ensign et al. 2018 lands as a number on a console.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equiml.model import Model
from equiml.simulation import (
    DecisionRule,
    FeedbackLoopSimulator,
    FeedbackRule,
)


def make_biased_dataset(n: int = 1500, seed: int = 42):
    """Synthetic credit-decisioning data with group A favoured by 0.6 logits.

    Two features feed the outcome; the sensitive group adds an additive
    boost on top of them. The boost mimics historical patterns where one
    group's records produce more positive outcomes than the features alone
    would predict — the canonical lending-data shape.
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], n, p=[0.5, 0.5])
    feat1 = rng.normal(0, 1, n)
    feat2 = rng.normal(0, 1, n)
    boost = np.where(group == "A", 0.6, -0.2)
    logits = 0.6 * feat1 + 0.3 * feat2 + boost
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)

    X = pd.DataFrame(
        {
            "feat1": feat1,
            "feat2": feat2,
            # Encode the sensitive feature so the model can use it. In a
            # real audit the team would decide whether to drop this; for
            # the simulation we want the model to do what production
            # systems do, which is exactly to learn from it.
            "group_A": (group == "A").astype(int),
        }
    )
    return X, pd.Series(y, name="approved"), pd.Series(group, name="group")


def split_train_test(X, y, s, test_frac: float = 0.4, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return (
        X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True),
        s.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        y.iloc[te].reset_index(drop=True),
        s.iloc[te].reset_index(drop=True),
    )


def main() -> None:
    seed = 13
    print("Building biased synthetic credit-decisioning dataset...")
    X, y, sensitive = make_biased_dataset(n=1500, seed=seed)
    Xtr, ytr, str_, Xte, yte, ste = split_train_test(X, y, sensitive, seed=seed)
    print(f"  train: {len(Xtr)} rows, test: {len(Xte)} rows")
    print(f"  groups in train: {sorted(str_.unique())}")

    print("\nRunning 10-round feedback-loop simulation:")
    print("  decision rule:  threshold > 0.65 (strict approval bar)")
    print("  feedback rule:  selection_bias (only approved applicants' outcomes observed)")

    sim = FeedbackLoopSimulator(
        model_factory=lambda: Model(algorithm="logistic_regression"),
        sensitive_features=["group"],
        decision_rule=DecisionRule.THRESHOLD,
        feedback_rule=FeedbackRule.SELECTION_BIAS,
        n_rounds=10,
        random_state=seed,
        threshold=0.65,
    )
    results = sim.run(Xtr, ytr, Xte, yte, str_, ste)

    print("\nFairness trajectory (demographic parity gap, lower = fairer):")
    print(f"  round 0: DP={abs(results[0].metrics['demographic_parity_difference']):.3f}, "
          f"acc={results[0].metrics['accuracy']:.3f}, train_n={results[0].n_train}")
    print(f"  round 5: DP={abs(results[5].metrics['demographic_parity_difference']):.3f}, "
          f"acc={results[5].metrics['accuracy']:.3f}, train_n={results[5].n_train}")
    print(f"  round 9: DP={abs(results[9].metrics['demographic_parity_difference']):.3f}, "
          f"acc={results[9].metrics['accuracy']:.3f}, train_n={results[9].n_train}")

    summary = sim.summary(results)
    print("\nSummary:")
    print(f"  {summary['drift_headline']}")
    print(f"  DP mean across rounds:    {summary['dp_mean']:.3f} +/- {summary['dp_std']:.3f}")
    print(f"  Accuracy mean:            {summary['accuracy_mean']:.3f}")

    print("\nPer-group selection rates (round 0 vs round 9):")
    for g in sorted(set(results[0].selection_rate_per_group)):
        r0 = results[0].selection_rate_per_group.get(g, float("nan"))
        r9 = results[9].selection_rate_per_group.get(g, float("nan"))
        print(f"  group {g}: {r0:.3f} -> {r9:.3f}")

    # Load-bearing assertion: this script is also a smoke test that the
    # simulator reproduces the textbook failure mode. If the assertion
    # ever fires, the demo's claim is false and the script must be fixed
    # (or the simulator regressed).
    dp0 = abs(results[0].metrics["demographic_parity_difference"])
    dp9 = abs(results[9].metrics["demographic_parity_difference"])
    assert dp9 > dp0, (
        f"DP gap should be worse at round 9 than at round 0 (got "
        f"dp0={dp0:.3f}, dp9={dp9:.3f}). The simulator's premise is "
        f"that selection-bias feedback compounds historical bias."
    )
    print("\nOK: round 9 DP gap is worse than round 0 — feedback loop reproduced.")


if __name__ == "__main__":
    main()
