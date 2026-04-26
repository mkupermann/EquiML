"""Demonstrate group-uncertainty aware fairness intervals.

Shows three simulators from `equiml.uncertainty`:

1. Bootstrap on the complete-cases-only subset: how tight is DP if we
   pretend missingness does not exist and just throw away rows with NaN
   in `gender`?
2. Multiple imputation + bootstrap on the full dataset: what does DP
   look like when we acknowledge that 25% of `gender` is missing and
   draw imputations from the empirical distribution?
3. Noise model: what if the gender labels we have are 90% accurate?

Run:
    python examples/uncertainty_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from equiml.uncertainty import (
    DP,
    bootstrap_fairness_intervals,
    impute_and_bootstrap,
    noise_model_intervals,
)


# --- 1. Build a synthetic dataset with biased outcomes ---------------------


def make_synthetic(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 65, size=n)
    hours = rng.integers(20, 60, size=n)
    gender = rng.choice(["F", "M"], size=n)
    # Higher hours → higher income; men get a +0.15 boost on the prob.
    base = 0.05 * (hours - 30) / 10 + 0.02 * (age - 40) / 10
    boost = np.where(gender == "M", 0.15, -0.05)
    p = 1 / (1 + np.exp(-(base + boost)))
    income = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {"age": age, "hours_per_week": hours, "gender": gender, "income": income}
    )


def inject_missing_gender(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["gender"] = out["gender"].astype(object)
    mask = rng.random(len(out)) < fraction
    out.loc[mask, "gender"] = np.nan
    return out


# --- 2. Train a logistic regression model ----------------------------------


def train_lr(df: pd.DataFrame) -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """Train on age + hours only (gender NOT a feature) so the model is
    a realistic 'maybe-fair' baseline rather than overtly using gender.
    """
    X = df[["age", "hours_per_week"]].to_numpy(dtype=float)
    y = df["income"].to_numpy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.4, random_state=11
    )
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=200).fit(scaler.transform(X_train), y_train)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    return model, test_df, pd.Series(model.predict(scaler.transform(X_test)))


# --- 3. Run the three simulators -------------------------------------------


def main() -> None:
    full = make_synthetic(n=2000, seed=7)
    full_with_nan = inject_missing_gender(full, fraction=0.25, seed=13)

    # Train on the full data (missingness in gender is not a model input).
    model, test_df, y_pred = train_lr(full_with_nan)
    y_true = test_df["income"]
    sf_obs = test_df["gender"]

    print("=" * 60)
    print("  GROUP-UNCERTAINTY AWARE FAIRNESS INTERVALS")
    print("=" * 60)
    print(f"  Test rows: {len(test_df)}")
    print(f"  Missing gender on test: {sf_obs.isna().mean():.1%}")
    print()

    # --- 3a. complete-cases bootstrap --------------------------------------
    cc_mask = ~sf_obs.isna()
    cc_res = bootstrap_fairness_intervals(
        y_true=y_true[cc_mask].to_numpy(),
        y_pred=y_pred[cc_mask].to_numpy(),
        sensitive_features=sf_obs[cc_mask].to_numpy(),
        n_bootstrap=1000,
        confidence=0.95,
        random_state=42,
    )
    print("  [1] BOOTSTRAP — complete cases only (drop NaN gender)")
    print(f"      DP = {cc_res[DP]}  (n_bootstrap={cc_res.n_bootstrap})")
    print()

    # --- 3b. imputation + bootstrap ----------------------------------------
    feats = test_df[["age", "hours_per_week"]]
    mi_res = impute_and_bootstrap(
        y_true=y_true.to_numpy(),
        y_pred=y_pred.to_numpy(),
        sensitive_features_with_nan=sf_obs.to_numpy(),
        features=feats,
        n_imputations=20,
        n_bootstrap_per_impute=200,
        confidence=0.95,
        random_state=42,
    )
    print("  [2] MULTIPLE IMPUTATION + BOOTSTRAP — full dataset")
    print(f"      DP = {mi_res[DP]}  (n_imputations={mi_res.n_imputations}, "
          f"missing={mi_res.missing_fraction:.1%})")
    for note in mi_res.notes:
        print(f"      note: {note}")
    print()

    # --- 3c. noise model --------------------------------------------------
    # Suppose self-reported gender is 90% accurate, 10% mis-recorded.
    noise = pd.DataFrame(
        [[0.9, 0.1], [0.1, 0.9]], index=["F", "M"], columns=["F", "M"],
    )
    noise_res = noise_model_intervals(
        y_true=y_true[cc_mask].to_numpy(),
        y_pred=y_pred[cc_mask].to_numpy(),
        sensitive_observed=sf_obs[cc_mask].to_numpy(),
        error_rate_matrix=noise,
        n_simulations=1000,
        confidence=0.95,
        random_state=42,
    )
    print("  [3] NOISE MODEL — assume 10% label noise on observed gender")
    print(f"      DP = {noise_res[DP]}")
    print()

    # --- 4. Headline assertion -------------------------------------------
    # Imputation reasoning: MI uses the full N rows AND adds variance from
    # imputation. Compared to complete-cases (~75% N), it can either widen
    # (imputation noise dominates) or stay similar (sample-size gain
    # offsets imputation noise). The point we want to make to the user is
    # that the MI interval is *materially different* from the complete-
    # cases interval — i.e. imputation matters, ignoring it is wrong.
    width_ratio = mi_res[DP].width() / cc_res[DP].width() if cc_res[DP].width() > 0 else 1.0
    delta_lower = abs(mi_res[DP].lower - cc_res[DP].lower)
    delta_upper = abs(mi_res[DP].upper - cc_res[DP].upper)

    print("  ASSERTIONS")
    print(f"    MI / complete-cases width ratio: {width_ratio:.2f}")
    print(f"    Lower bound shift: {delta_lower:.4f}")
    print(f"    Upper bound shift: {delta_upper:.4f}")

    # Material difference: at least one bound shifts by >= 0.01, OR the
    # widths differ by more than 10%.
    materially_different = (
        delta_lower >= 0.01
        or delta_upper >= 0.01
        or abs(width_ratio - 1.0) >= 0.10
    )
    assert materially_different, (
        "Multiple imputation interval was indistinguishable from the "
        "complete-cases interval; this demo is not exercising the MI path."
    )
    print("    PASS — MI interval is materially different from complete-cases.")
    print()
    print("  Take-away: reporting only the complete-cases point estimate")
    print("  hides both the missingness uncertainty (interval [2]) and")
    print("  the label-noise uncertainty (interval [3]).")


if __name__ == "__main__":
    main()
