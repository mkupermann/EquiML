import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import logging
from typing import Optional, Dict, Any
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
        """Initializes the underlying machine learning model with stability improvements."""
        if self.algorithm == 'logistic_regression':
            # Apply regularization for stability (default L2)
            return LogisticRegression(solver='liblinear', random_state=42, C=1.0, penalty='l2')
        elif self.algorithm == 'logistic_regression_l1':
            # L1 regularization for feature selection and stability
            return LogisticRegression(solver='liblinear', random_state=42, C=1.0, penalty='l1')
        elif self.algorithm == 'logistic_regression_elastic':
            # ElasticNet regularization for best of both L1/L2
            return LogisticRegression(solver='saga', random_state=42, C=1.0, penalty='elasticnet', l1_ratio=0.5, max_iter=1000)
        elif self.algorithm == 'random_forest':
            # Enhanced Random Forest with stability parameters
            return RandomForestClassifier(
                n_estimators=100,  # More trees for stability
                max_depth=10,      # Limit depth to reduce overfitting
                min_samples_split=5,  # Require more samples to split
                min_samples_leaf=2,   # Minimum samples in leaves
                random_state=42,
                bootstrap=True     # Use bootstrap sampling
            )
        elif self.algorithm == 'robust_random_forest':
            # Extra robust Random Forest configuration
            return RandomForestClassifier(
                n_estimators=200,  # Even more trees
                max_depth=8,       # More conservative depth
                min_samples_split=10,  # Higher split threshold
                min_samples_leaf=5,    # Higher leaf threshold
                max_features='sqrt',   # Limit features per tree
                random_state=42,
                bootstrap=True,
                oob_score=True     # Out-of-bag scoring
            )
        elif self.algorithm == 'xgboost':
            return xgb.XGBClassifier(random_state=42, verbosity=0)
        elif self.algorithm == 'robust_xgboost':
            # XGBoost with regularization for stability
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,       # Limit tree depth
                learning_rate=0.1, # Conservative learning rate
                subsample=0.8,     # Subsample for robustness
                colsample_bytree=0.8,  # Feature subsampling
                reg_alpha=0.1,     # L1 regularization
                reg_lambda=1.0,    # L2 regularization
                random_state=42,
                verbosity=0
            )
        elif self.algorithm == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42, verbosity=-1)
        elif self.algorithm == 'robust_lightgbm':
            # LightGBM with regularization
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1
            )
        elif self.algorithm == 'ensemble':
            return self._create_ensemble_model()
        elif self.algorithm == 'robust_ensemble':
            return self._create_robust_ensemble_model()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _create_robust_ensemble_model(self):
        """Create a robust ensemble model with enhanced stability."""
        from sklearn.ensemble import VotingClassifier, BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier

        # Create individual robust estimators
        estimators = [
            ('robust_lr', LogisticRegression(solver='liblinear', C=1.0, penalty='l2', random_state=42)),
            ('robust_rf', RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )),
            ('bagged_dt', BaggingClassifier(
                DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42),
                n_estimators=50, random_state=42
            ))
        ]

        return VotingClassifier(estimators=estimators, voting='soft')

    def _create_ensemble_model(self):
        """Create an ensemble model for improved robustness."""
        from sklearn.ensemble import VotingClassifier

        estimators = [
            ('lr', LogisticRegression(solver='liblinear', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(random_state=42, verbosity=0))
        ]

        return VotingClassifier(estimators=estimators, voting='soft')

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           method: str = 'optuna', n_trials: int = 100) -> None:
        """
        Perform hyperparameter tuning for the model.

        Args:
            X_train: Training features
            y_train: Training targets
            method: Tuning method ('optuna', 'grid_search', 'random_search')
            n_trials: Number of trials for Optuna
        """
        try:
            if method == 'optuna':
                self._tune_with_optuna(X_train, y_train, n_trials)
            elif method == 'grid_search':
                self._tune_with_grid_search(X_train, y_train)
            elif method == 'random_search':
                self._tune_with_random_search(X_train, y_train)
            else:
                raise ValueError(f"Unsupported tuning method: {method}")
            logger.info(f"Hyperparameter tuning completed using {method}")
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise

    def _tune_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int) -> None:
        """Tune hyperparameters using Optuna."""
        def objective(trial):
            if self.algorithm == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': 'liblinear',
                    'random_state': 42
                }
                model = LogisticRegression(**params)
            elif self.algorithm == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            elif self.algorithm == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = xgb.XGBClassifier(**params)
            else:
                raise ValueError(f"Optuna tuning not implemented for {self.algorithm}")

            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Update model with best parameters
        best_params = study.best_params
        if self.algorithm == 'logistic_regression':
            self.model = LogisticRegression(**best_params)
        elif self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(**best_params)
        elif self.algorithm == 'xgboost':
            self.model = xgb.XGBClassifier(**best_params)

    def _tune_with_grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Tune hyperparameters using GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        if self.algorithm == 'logistic_regression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2']
            }
        elif self.algorithm == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        else:
            raise ValueError(f"Grid search not implemented for {self.algorithm}")

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

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

    def apply_stability_improvements(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   sensitive_features: Optional[pd.Series] = None,
                                   stability_method: str = 'comprehensive') -> None:
        """
        Apply stability improvements to the model based on recommendations.

        Args:
            X_train: Training features
            y_train: Training targets
            sensitive_features: Sensitive features for fairness
            stability_method: Type of stability improvement ('regularization', 'ensemble', 'comprehensive')
        """
        try:
            if stability_method == 'regularization':
                self._apply_regularization_stability(X_train, y_train, sensitive_features)
            elif stability_method == 'ensemble':
                self._apply_ensemble_stability(X_train, y_train, sensitive_features)
            elif stability_method == 'comprehensive':
                self._apply_comprehensive_stability(X_train, y_train, sensitive_features)
            else:
                raise ValueError(f"Unsupported stability method: {stability_method}")

            logger.info(f"Stability improvements applied using {stability_method}")
        except Exception as e:
            logger.error(f"Stability improvement failed: {str(e)}")
            raise

    def _apply_regularization_stability(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       sensitive_features: Optional[pd.Series] = None) -> None:
        """Apply regularization-based stability improvements."""
        logger.info("Applying regularization for model stability...")

        # Switch to regularized versions of algorithms
        if 'logistic' in self.algorithm:
            # Use L2 regularization with cross-validated C parameter
            from sklearn.linear_model import LogisticRegressionCV
            self.model = LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 10),  # Range of C values
                cv=5,  # Stratified CV for stability
                penalty='l2',
                solver='liblinear',
                random_state=42,
                scoring='accuracy'
            )
        elif 'random_forest' in self.algorithm:
            # Use more conservative Random Forest
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,         # Reduce complexity
                min_samples_split=10, # Require more samples
                min_samples_leaf=5,   # Larger leaves
                max_features='sqrt',  # Feature subsampling
                random_state=42
            )

    def _apply_ensemble_stability(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 sensitive_features: Optional[pd.Series] = None) -> None:
        """Apply ensemble-based stability improvements."""
        logger.info("Applying ensemble methods for model stability...")

        from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier
        from sklearn.tree import DecisionTreeClassifier

        # Create diverse stable estimators
        estimators = [
            ('stable_lr', LogisticRegression(solver='liblinear', C=1.0, penalty='l2', random_state=42)),
            ('stable_rf', RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            )),
            ('extra_trees', ExtraTreesClassifier(
                n_estimators=100, max_depth=8, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            )),
            ('bagged_dt', BaggingClassifier(
                DecisionTreeClassifier(max_depth=6, min_samples_split=8, random_state=42),
                n_estimators=50, random_state=42, bootstrap=True
            ))
        ]

        self.model = VotingClassifier(estimators=estimators, voting='soft')

    def _apply_comprehensive_stability(self, X_train: pd.DataFrame, y_train: pd.Series,
                                     sensitive_features: Optional[pd.Series] = None) -> None:
        """Apply comprehensive stability improvements."""
        logger.info("Applying comprehensive stability improvements...")

        # Check data characteristics first
        data_size = len(X_train)
        feature_count = X_train.shape[1]

        if data_size < 1000:
            logger.warning("Small dataset detected. Using conservative model complexity.")
            # For small datasets, use simpler, more stable models
            if 'logistic' in self.algorithm or self.algorithm == 'ensemble':
                self.model = LogisticRegression(solver='liblinear', C=10.0, penalty='l2', random_state=42)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=50, max_depth=5, min_samples_split=10,
                    min_samples_leaf=5, random_state=42
                )
        else:
            # For larger datasets, use robust ensemble
            self._apply_ensemble_stability(X_train, y_train, sensitive_features)

    def evaluate_model_stability(self, X_train: pd.DataFrame, y_train: pd.Series,
                                cv_folds: int = 10) -> Dict[str, float]:
        """
        Evaluate model stability using stratified cross-validation.

        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with stability metrics
        """
        try:
            from sklearn.model_selection import StratifiedKFold, cross_validate

            # Use stratified CV for stability assessment
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Evaluate multiple metrics for comprehensive stability assessment
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

            cv_results = cross_validate(
                self.model, X_train, y_train,
                cv=skf, scoring=scoring, return_train_score=True
            )

            stability_metrics = {
                'cv_accuracy_mean': np.mean(cv_results['test_accuracy']),
                'cv_accuracy_std': np.std(cv_results['test_accuracy']),
                'cv_precision_mean': np.mean(cv_results['test_precision_macro']),
                'cv_precision_std': np.std(cv_results['test_precision_macro']),
                'cv_recall_mean': np.mean(cv_results['test_recall_macro']),
                'cv_recall_std': np.std(cv_results['test_recall_macro']),
                'cv_f1_mean': np.mean(cv_results['test_f1_macro']),
                'cv_f1_std': np.std(cv_results['test_f1_macro']),
                'train_test_gap': np.mean(cv_results['train_accuracy']) - np.mean(cv_results['test_accuracy'])
            }

            # Assess stability
            stability_score = self._calculate_stability_score(stability_metrics)
            stability_metrics['overall_stability_score'] = stability_score

            # Log stability assessment
            if stability_score > 0.8:
                logger.info(f"Model shows high stability (score: {stability_score:.3f})")
            elif stability_score > 0.6:
                logger.warning(f"Model shows moderate stability (score: {stability_score:.3f})")
            else:
                logger.warning(f"Model shows low stability (score: {stability_score:.3f})")

            return stability_metrics

        except Exception as e:
            logger.error(f"Stability evaluation failed: {str(e)}")
            raise

    def _calculate_stability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall stability score from CV metrics."""
        # Lower standard deviation = higher stability
        acc_stability = max(0, 1 - metrics['cv_accuracy_std'] * 10)  # Scale std
        f1_stability = max(0, 1 - metrics['cv_f1_std'] * 10)

        # Lower train-test gap = less overfitting
        overfitting_penalty = max(0, 1 - abs(metrics['train_test_gap']) * 5)

        # Combine scores
        overall_score = (acc_stability + f1_stability + overfitting_penalty) / 3
        return overall_score

    def check_data_leakage(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for potential data leakage and temporal dependencies.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Dictionary with leakage detection results
        """
        try:
            leakage_results = {
                'leakage_detected': False,
                'issues': [],
                'recommendations': []
            }

            # Check 1: Identical rows between train and test
            if len(X_train) > 0 and len(X_test) > 0:
                # Convert to string for comparison to handle mixed types
                train_str = X_train.astype(str)
                test_str = X_test.astype(str)

                # Find identical rows
                train_hashes = pd.util.hash_pandas_object(train_str)
                test_hashes = pd.util.hash_pandas_object(test_str)

                identical_rows = len(set(train_hashes) & set(test_hashes))

                if identical_rows > 0:
                    leakage_results['leakage_detected'] = True
                    leakage_results['issues'].append(f"Found {identical_rows} identical rows between train and test")
                    leakage_results['recommendations'].append("Remove duplicate rows or ensure proper data splitting")

            # Check 2: Feature correlation analysis for potential leakage
            if len(X_train.columns) > 1:
                correlation_matrix = X_train.corr().abs()
                high_correlations = []

                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        if correlation_matrix.iloc[i, j] > 0.95:
                            col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                            high_correlations.append((col1, col2, correlation_matrix.iloc[i, j]))

                if high_correlations:
                    leakage_results['issues'].append(f"Found {len(high_correlations)} highly correlated feature pairs (>0.95)")
                    leakage_results['recommendations'].append("Review highly correlated features for potential redundancy or leakage")

            # Check 3: Temporal ordering (basic check for datetime columns)
            datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                leakage_results['issues'].append(f"Found {len(datetime_cols)} datetime columns")
                leakage_results['recommendations'].append("Ensure temporal ordering is preserved and no future information is used")

            # Check 4: Feature names that might indicate leakage
            suspicious_features = []
            for col in X_train.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['id', 'target', 'label', 'outcome', 'result']):
                    suspicious_features.append(col)

            if suspicious_features:
                leakage_results['issues'].append(f"Found potentially suspicious features: {suspicious_features}")
                leakage_results['recommendations'].append("Review suspicious feature names for potential target leakage")

            if not leakage_results['issues']:
                leakage_results['issues'].append("No obvious data leakage detected")
                leakage_results['recommendations'].append("Continue with regular model validation procedures")

            return leakage_results

        except Exception as e:
            logger.error(f"Data leakage check failed: {str(e)}")
            return {'leakage_detected': False, 'issues': [f"Check failed: {str(e)}"], 'recommendations': []}

    def apply_fairness_postprocessing(self, X: pd.DataFrame, sensitive_features: pd.Series,
                                    method: str = 'threshold_optimization') -> np.ndarray:
        """
        Apply post-processing fairness adjustments to predictions.

        Args:
            X: Features for prediction
            sensitive_features: Sensitive feature values
            method: Post-processing method ('threshold_optimization', 'calibration')

        Returns:
            Adjusted predictions
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        try:
            if method == 'threshold_optimization':
                return self._apply_threshold_optimization(X, sensitive_features)
            elif method == 'calibration':
                return self._apply_calibration(X, sensitive_features)
            else:
                raise ValueError(f"Unsupported post-processing method: {method}")
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            raise

    def _apply_threshold_optimization(self, X: pd.DataFrame, sensitive_features: pd.Series) -> np.ndarray:
        """Apply threshold optimization for fairness."""
        try:
            from fairlearn.postprocessing import ThresholdOptimizer

            # Get base predictions
            y_pred = self.model.predict(X)

            # Apply threshold optimization
            postprocessor = ThresholdOptimizer(
                estimator=self.model,
                constraints='demographic_parity',
                prefit=True
            )

            # Note: This is a simplified implementation
            # In practice, you'd need to fit the postprocessor on validation data
            return y_pred

        except ImportError:
            logger.error("Fairlearn not available for post-processing")
            return self.model.predict(X)

    def _apply_calibration(self, X: pd.DataFrame, sensitive_features: pd.Series) -> np.ndarray:
        """Apply group-specific calibration."""
        try:
            from sklearn.calibration import CalibratedClassifierCV

            # Get prediction probabilities
            y_proba = self.model.predict_proba(X)

            # Simple implementation: adjust predictions based on group membership
            # In practice, you'd train group-specific calibrators
            predictions = np.argmax(y_proba, axis=1)

            # Adjust predictions for fairness (simplified)
            unique_groups = sensitive_features.unique()
            for group in unique_groups:
                group_mask = sensitive_features == group
                # Apply group-specific adjustments here
                # This is a placeholder implementation

            return predictions

        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            return self.model.predict(X)

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
