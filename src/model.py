import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import logging
from typing import Optional
import optuna

logger = logging.getLogger(__name__)

class Model:
    """
    A class to handle model training, prediction, and fairness mitigation using Fairlearn
    in the EquiML framework.
    
    Attributes:
        algorithm (str): The ML algorithm to use ('logistic_regression', 'random_forest').
        fairness_constraint (str, optional): The fairness constraint to apply ('demographic_parity', 'equalized_odds').
        model: The trained ML model or Fairlearn mitigator.
    """
    
    def __init__(self, algorithm: str = 'logistic_regression', fairness_constraint: Optional[str] = None):
        """
        Initializes the Model object.

        Args:
            algorithm (str): The machine learning algorithm to use.
            fairness_constraint (str, optional): The fairness constraint to apply.
        """
        self.algorithm = algorithm
        self.fairness_constraint = fairness_constraint
        self.model = self._initialize_model()
        self.constraint_map = {
            'demographic_parity': DemographicParity,
            'equalized_odds': EqualizedOdds
        }

    def _initialize_model(self):
        """Initializes the underlying machine learning model."""
        if self.algorithm == 'logistic_regression':
            return LogisticRegression(solver='liblinear', random_state=42)
        elif self.algorithm == 'random_forest':
            return RandomForestClassifier(random_state=42)
        elif self.algorithm == 'xgboost':
            return xgb.XGBClassifier(random_state=42)
        elif self.algorithm == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sensitive_features: Optional[pd.Series] = None, sample_weight: Optional[pd.Series] = None) -> None:
        """
        Trains the model, applying fairness constraints if specified.

        Args:
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training target data.
            sensitive_features (pd.Series, optional): Sensitive features for fairness mitigation.
            sample_weight (pd.Series, optional): Sample weights for training.

        Raises:
            ValueError: If input data is invalid or fairness constraint is unsupported.
            RuntimeError: If model training fails.
        """
        # Input validation
        if X_train is None or X_train.empty:
            raise ValueError("Training features (X_train) cannot be empty.")
        if y_train is None or y_train.empty:
            raise ValueError("Training targets (y_train) cannot be empty.")
        if len(X_train) != len(y_train):
            raise ValueError("Training features and targets must have the same length.")

        try:
            if self.fairness_constraint and sensitive_features is not None:
                if self.fairness_constraint not in self.constraint_map:
                    raise ValueError(f"Unsupported fairness constraint: {self.fairness_constraint}")

                if len(sensitive_features) != len(X_train):
                    raise ValueError("Sensitive features must have the same length as training data.")

                if sample_weight is not None:
                    logger.warning("Sample weights are not supported with ExponentiatedGradient. Ignoring them.")

                constraint = self.constraint_map[self.fairness_constraint]()
                mitigator = ExponentiatedGradient(self.model, constraints=constraint)
                mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
                self.model = mitigator
                logger.info(f"Trained {self.algorithm} with {self.fairness_constraint} constraint.")
            else:
                if sample_weight is not None and len(sample_weight) != len(X_train):
                    raise ValueError("Sample weights must have the same length as training data.")

                self.model.fit(X_train, y_train, sample_weight=sample_weight)
                if sample_weight is not None:
                    logger.info(f"Trained {self.algorithm} with sample weights.")
                else:
                    logger.info(f"Trained {self.algorithm} without fairness constraints.")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise RuntimeError(f"Failed to train model: {str(e)}") from e

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            X (pd.DataFrame): Data to make predictions on.

        Returns:
            np.ndarray: Predicted labels.

        Raises:
            ValueError: If input data is invalid.
            RuntimeError: If model is not trained or prediction fails.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        if X is None or X.empty:
            raise ValueError("Input data (X) cannot be empty.")

        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}") from e

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes probability predictions on new data.

        Args:
            X (pd.DataFrame): Data to make predictions on.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'predictors_'):  # For ExponentiatedGradient
            probas = [p.predict_proba(X) for p in self.model.predictors_]
            return np.mean(probas, axis=0)
        else:
            raise AttributeError("Model does not have a predict_proba method.")

    def tune_hyperparameters(self, X_train, y_train, n_trials=50):
        """
        Tunes hyperparameters for the model using Optuna.

        Args:
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training target data.
            n_trials (int): The number of optimization trials.

        Returns:
            dict: The best hyperparameters found.
        """
        def objective(trial):
            if self.algorithm == 'logistic_regression':
                C = trial.suggest_float('C', 1e-5, 1e2, log=True)
                solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
                model = LogisticRegression(C=C, solver=solver, random_state=42)
            elif self.algorithm == 'random_forest':
                n_estimators = trial.suggest_int('n_estimators', 10, 1000)
                max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif self.algorithm == 'xgboost':
                param = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                }
                model = xgb.XGBClassifier(**param, random_state=42)
            elif self.algorithm == 'lightgbm':
                param = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
                }
                model = lgb.LGBMClassifier(**param, random_state=42)

            # Using cross_val_score for robust evaluation
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3).mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Set the model to the best estimator
        self.model.set_params(**study.best_params)

        logger.info(f"Best hyperparameters for {self.algorithm}: {study.best_params}")
        return study.best_params

    def cross_validate(self, X, y, sensitive_features=None, cv=5):
        """
        Performs cross-validation for the model.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            sensitive_features (pd.Series, optional): Sensitive features for fairness-aware cross-validation.
            cv (int): The number of cross-validation folds.

        Returns:
            dict: A dictionary of cross-validation scores.
        """
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        estimator = self.model
        if hasattr(self.model, 'estimator'):
            estimator = self.model.estimator

        return cross_validate(estimator, X, y, cv=cv, scoring=scoring)
