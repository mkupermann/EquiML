"""Counterfactual fairness demo (RFC 0004).

Builds a synthetic hiring dataset where:
- ``gender`` is the protected attribute,
- ``salary_history`` is engineered to be a near-perfect proxy of gender,
- ``years_experience`` and ``test_score`` are non-proxy signals.

Trains a logistic regression on every feature (including the protected
attribute), then runs the naive counterfactual audit and the proxy-feature
ranking. Prints the flip rate and the top-3 proxies.

Run:
    python examples/counterfactual_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from equiml.counterfactual import (
    compute_counterfactual_audit,
    compute_proxy_features,
)


def build_dataset(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Synthetic hiring dataset with a near-perfect salary_history proxy.

    The label y is generated as a function of ``salary_history`` (which is
    itself a near-copy of gender) and ``test_score``. Gender is included in
    the feature matrix the model trains on, but the *causal* hiring signal
    in this synthetic world flows through salary_history. That is the
    pattern proxy detection is designed to catch: legacy bias laundered
    through a feature that looks legitimate.
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice([0, 1], size=n)  # 0 = female, 1 = male

    # salary_history: a near-copy of gender on the same scale, then we
    # inject some structural variation so it is not literally identical.
    salary_history = gender.astype(float) + rng.normal(0, 0.05, n)

    # years_experience: independent of gender.
    years_experience = rng.integers(0, 25, n).astype(float)

    # test_score: independent of gender, mildly predictive of the label.
    test_score = rng.normal(70, 15, n)

    # Hiring label: depends primarily on salary_history (the proxy
    # channel), and on test_score. Gender does NOT enter the label
    # directly — but because salary_history is a near-copy of gender, the
    # bias still flows through.
    logit = 3.5 * (salary_history - 0.5) + 0.04 * (test_score - 70)
    p = 1.0 / (1.0 + np.exp(-logit))
    hired = (rng.uniform(0, 1, n) < p).astype(int)

    return pd.DataFrame({
        "gender": gender,
        "salary_history": salary_history,
        "years_experience": years_experience,
        "test_score": test_score,
        "hired": hired,
    })


def main() -> int:
    df = build_dataset()
    feature_cols = ["gender", "salary_history", "years_experience", "test_score"]
    X = df[feature_cols]
    y = df["hired"].to_numpy()

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    print("Counterfactual audit")
    print("-" * 50)
    result = compute_counterfactual_audit(model, X_test, "gender")
    print(f"  n_samples:               {result.n_samples}")
    print(f"  flip_rate:               {result.flip_rate:.3f}")
    if result.mean_prediction_shift is not None:
        print(f"  mean_prediction_shift:   {result.mean_prediction_shift:.3f}")
    else:
        print(f"  mean_prediction_shift:   (n/a — no predict_proba)")
    for note in result.notes:
        print(f"  note: {note}")
    print()

    print("Top-3 proxy features")
    print("-" * 50)
    proxies = compute_proxy_features(
        model, X_test, "gender",
        candidate_features=["salary_history", "years_experience", "test_score"],
        top_k=3,
    )
    for rank, p in enumerate(proxies, start=1):
        print(
            f"  {rank}. {p.feature_name:<20} "
            f"with={p.flip_rate_with:.3f}  without={p.flip_rate_without:.3f}  "
            f"strength={p.proxy_strength:+.3f}"
        )
    print()

    # Sanity assertion: salary_history is the strongest proxy.
    top_name = proxies[0].feature_name
    assert top_name == "salary_history", (
        f"Expected salary_history to be the top proxy; got {top_name!r}"
    )
    print("OK: salary_history ranks first as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
