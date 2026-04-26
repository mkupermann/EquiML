import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_validate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import logging
from typing import Optional

logger = logging.getLogger(__name__)

FAIRNESS_CONSTRAINTS = {
    "demographic_parity": DemographicParity,
    "equalized_odds": EqualizedOdds,
}


class Model:
    """Train ML models with optional fairness constraints."""

    def __init__(self, algorithm: str = "logistic_regression", fairness_constraint: Optional[str] = None):
        self.algorithm = algorithm
        self.fairness_constraint = fairness_constraint
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.algorithm == "logistic_regression":
            return LogisticRegression(solver="liblinear", random_state=42, C=1.0, penalty="l2")
        elif self.algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42,
            )
        elif self.algorithm == "ensemble":
            return VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(solver="liblinear", random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                ],
                voting="soft",
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Use: logistic_regression, random_forest, ensemble")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              sensitive_features: Optional[pd.Series] = None,
              sample_weight: Optional[pd.Series] = None) -> None:
        """Train the model, optionally with fairness constraints."""
        if X_train is None or X_train.empty:
            raise ValueError("Training features cannot be empty.")
        if y_train is None or y_train.empty:
            raise ValueError("Training targets cannot be empty.")

        if self.fairness_constraint and sensitive_features is not None:
            if self.fairness_constraint not in FAIRNESS_CONSTRAINTS:
                raise ValueError(f"Unsupported constraint: {self.fairness_constraint}. Use: {list(FAIRNESS_CONSTRAINTS)}")
            constraint = FAIRNESS_CONSTRAINTS[self.fairness_constraint]()
            mitigator = ExponentiatedGradient(self.model, constraints=constraint)
            mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
            self.model = mitigator
            logger.info(f"Trained {self.algorithm} with {self.fairness_constraint}")
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            logger.info(f"Trained {self.algorithm}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if X is None or X.empty:
            raise ValueError("Input data cannot be empty.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "predictors_"):
            # ExponentiatedGradient is a randomised classifier: a correct
            # predict_proba would sample predictors according to their
            # weights, not average them. Averaging is mathematically wrong.
            raise NotImplementedError(
                "predict_proba is not well-defined for the ExponentiatedGradient "
                "mitigator; use predict() and report fairness metrics from hard "
                "predictions only"
            )
        else:
            raise AttributeError("Model does not support predict_proba.")

    def cross_validate(self, X, y, sensitive_features=None, cv=5) -> dict:
        """Cross-validate the model."""
        scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
        estimator = self.model.estimator if hasattr(self.model, "estimator") else self.model
        return cross_validate(estimator, X, y, cv=cv, scoring=scoring)
