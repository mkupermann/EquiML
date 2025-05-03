import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds, TruePositiveRateParity
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate, true_positive_rate, false_positive_rate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
import shap
from lime.lime_tabular import LimeTabularExplainer
import optuna
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ks_2samp
import logging
import json
from datetime import datetime
import sys
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    """
    An advanced class for training, evaluating, and interpreting machine learning models with fairness, robustness, and deployment capabilities.
    
    Attributes:
        algorithm (str): The machine learning algorithm to use (e.g., 'logistic_regression', 'random_forest').
        fairness_constraint (str): The fairness constraint to apply (e.g., 'demographic_parity').
        task (str): Type of task ('classification' or 'regression').
        model: The trained model object.
        preprocessor: Preprocessing pipeline for data transformation.
    """
    
    def __init__(self, algorithm: str = 'logistic_regression', fairness_constraint: Optional[str] = None, task: str = 'classification'):
        """
        Initialize the Model class.

        Args:
            algorithm (str): Algorithm to use (e.g., 'logistic_regression', 'random_forest', etc.).
            fairness_constraint (str, optional): Fairness constraint (e.g., 'demographic_parity').
            task (str): Task type ('classification' or 'regression').

        Raises:
            ValueError: If algorithm or task is unsupported.
        """
        self.algorithm = algorithm
        self.fairness_constraint = fairness_constraint
        self.task = task
        self.model = None
        self.preprocessor = None
        self.supported_algorithms = {
            'logistic_regression': LogisticRegression(max_iter=1000) if task == 'classification' else LinearRegression(),
            'decision_tree': DecisionTreeClassifier() if task == 'classification' else DecisionTreeRegressor(),
            'random_forest': RandomForestClassifier() if task == 'classification' else RandomForestRegressor(),
            'svm': SVC(probability=True) if task == 'classification' else SVR(),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss') if task == 'classification' else XGBRegressor()
        }
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from {list(self.supported_algorithms.keys())}")
        if task not in ['classification', 'regression']:
            raise ValueError(f"Unsupported task: {task}. Choose 'classification' or 'regression'.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sensitive_features: Optional[pd.Series] = None, class_weight: Optional[Dict] = None) -> None:
        """
        Train the model with optional fairness constraints and class weights.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            sensitive_features (pd.Series, optional): Sensitive features for fairness constraints.
            class_weight (dict, optional): Class weights for imbalanced classification.

        Raises:
            ValueError: If model parameters or fairness constraints are invalid.
        """
        try:
            base_model = self.supported_algorithms[self.algorithm]
            if class_weight and self.task == 'classification':
                base_model.set_params(class_weight=class_weight)
            if self.fairness_constraint:
                if sensitive_features is None:
                    raise ValueError("Sensitive features required for fairness constraints.")
                constraints_map = {
                    'demographic_parity': DemographicParity(),
                    'equalized_odds': EqualizedOdds(),
                    'true_positive_rate_parity': TruePositiveRateParity()
                }
                constraint = constraints_map.get(self.fairness_constraint)
                if constraint is None:
                    raise ValueError(f"Unsupported fairness constraint: {self.fairness_constraint}")
                self.model = ExponentiatedGradient(base_model, constraints=constraint)
                self.model.fit(X_train, y_train, sensitive_features=sensitive_features)
                logger.info(f"Trained {self.algorithm} with {self.fairness_constraint} constraint.")
            else:
                self.model = base_model
                self.model.fit(X_train, y_train)
                logger.info(f"Trained {self.algorithm} without fairness constraints.")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ValueError: If model is not trained.
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (classification only).

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted probabilities.

        Raises:
            ValueError: If model is not trained or task is not classification.
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        if self.task != 'classification':
            raise ValueError("Probability predictions only available for classification tasks.")
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.algorithm} does not support probability predictions.")
        return self.model.predict_proba(X)

    def explain_prediction(self, X: pd.DataFrame) -> np.ndarray:
        """
        Explain predictions using SHAP values.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: SHAP values.

        Raises:
            ValueError: If model is not trained.
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        try:
            if isinstance(self.model, ExponentiatedGradient):
                explainer = shap.KernelExplainer(self.model.predict, X)
            else:
                explainer_type = {
                    'decision_tree': shap.TreeExplainer,
                    'random_forest': shap.TreeExplainer,
                    'xgboost': shap.TreeExplainer,
                    'logistic_regression': shap.LinearExplainer,
                    'svm': shap.KernelExplainer
                }.get(self.algorithm, shap.KernelExplainer)
                if explainer_type == shap.LinearExplainer:
                    explainer = explainer_type(self.model, X)
                elif explainer_type == shap.KernelExplainer:
                    explainer = explainer_type(self.model.predict, X)
                else:
                    explainer = explainer_type(self.model)
            return explainer.shap_values(X)
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            raise

    def explain_instance(self, instance: pd.Series, X_train: pd.DataFrame, feature_names: List[str], class_names: Optional[List[str]] = None) -> object:
        """
        Explain a single instance using LIME.

        Args:
            instance (pd.Series): Single instance to explain.
            X_train (pd.DataFrame): Training data for LIME explainer.
            feature_names (list): Names of features.
            class_names (list, optional): Names of classes.

        Returns:
            object: LIME explanation object.

        Raises:
            ValueError: If model is not trained or task is not classification.
        """
        if self.task != 'classification':
            raise ValueError("LIME explanations only available for classification tasks.")
        explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=class_names)
        explanation = explainer.explain_instance(instance.values, self.predict_proba)
        return explanation

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, sensitive_features: Optional[pd.Series] = None) -> Dict:
        """
        Evaluate model performance and fairness.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            sensitive_features (pd.Series, optional): Sensitive features for fairness metrics.

        Returns:
            dict: Dictionary of performance and fairness metrics.
        """
        y_pred = self.predict(X_test)
        metrics = {}
        try:
            if self.task == 'classification':
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['f1_score'] = f1_score(y_test, y_pred, average='macro')
                if hasattr(self.model, 'predict_proba'):
                    y_pred_proba = self.predict_proba(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            elif self.task == 'regression':
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                metrics['r2'] = r2_score(y_test, y_pred)

            if sensitive_features is not None:
                metrics_to_compute = {
                    'accuracy': accuracy_score,
                    'selection_rate': selection_rate,
                    'true_positive_rate': true_positive_rate,
                    'false_positive_rate': false_positive_rate
                } if self.task == 'classification' else {
                    'mean_prediction': lambda y_true, y_pred: y_pred.mean(),
                    'mse': mean_squared_error
                }
                mf = MetricFrame(metrics=metrics_to_compute, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)
                metrics['group_metrics'] = mf.by_group.to_dict('index')
                fairness_metrics = {}
                for metric_name in metrics_to_compute:
                    group_values = [group[metric_name] for group in metrics['group_metrics'].values()]
                    fairness_metrics[f'{metric_name}_difference'] = max(group_values) - min(group_values)
                metrics.update(fairness_metrics)
                if self.task == 'classification':
                    selection_rates = [group['selection_rate'] for group in metrics['group_metrics'].values()]
                    metrics['disparate_impact_ratio'] = min(selection_rates) / max(selection_rates) if max(selection_rates) > 0 else 1.0
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict, sensitive_features: Optional[pd.Series] = None, fairness_weight: float = 0.5) -> Dict:
        """
        Perform hyperparameter tuning with fairness considerations.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            param_grid (dict): Parameter grid for tuning.
            sensitive_features (pd.Series, optional): Sensitive features for fairness scoring.
            fairness_weight (float): Weight for fairness in scoring (0 to 1).

        Returns:
            dict: Best hyperparameters.

        Raises:
            ValueError: If fairness_weight is invalid.
        """
        if not 0 <= fairness_weight <= 1:
            raise ValueError("fairness_weight must be between 0 and 1.")
        base_model = self.supported_algorithms[self.algorithm]
        if self.fairness_constraint and sensitive_features is not None:
            def fairness_scorer(y_true, y_pred):
                dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
                perf = accuracy_score(y_true, y_pred) if self.task == 'classification' else -mean_squared_error(y_true, y_pred)
                return fairness_weight * perf - (1 - fairness_weight) * dp_diff
            scorer = make_scorer(fairness_scorer)
        else:
            scorer = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        try:
            grid_search = GridSearchCV(base_model, param_grid, scoring=scorer, cv=5, n_jobs=-1)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_params_
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, sensitive_features_train: Optional[pd.Series] = None, sensitive_features_val: Optional[pd.Series] = None, n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.
            sensitive_features_train (pd.Series, optional): Sensitive features for training.
            sensitive_features_val (pd.Series, optional): Sensitive features for validation.
            n_trials (int): Number of optimization trials.

        Returns:
            dict: Best hyperparameters.
        """
        def objective(trial):
            params = {
                'logistic_regression': {'C': trial.suggest_loguniform('C', 1e-4, 1e2)},
                'decision_tree': {'max_depth': trial.suggest_int('max_depth', 2, 20)},
                'random_forest': {'n_estimators': trial.suggest_int('n_estimators', 10, 100), 'max_depth': trial.suggest_int('max_depth', 2, 20)},
                'svm': {'C': trial.suggest_loguniform('C', 1e-4, 1e2)},
                'xgboost': {'max_depth': trial.suggest_int('max_depth', 2, 10), 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1)}
            }.get(self.algorithm, {})
            base_model = self.supported_algorithms[self.algorithm].set_params(**params)
            self.model = base_model
            self.train(X_train, y_train, sensitive_features_train)
            metrics = self.evaluate(X_val, y_val, sensitive_features_val)
            score = metrics['accuracy'] - metrics.get('demographic_parity_difference', 0) if self.task == 'classification' else -metrics['mse']
            return score

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            self.supported_algorithms[self.algorithm].set_params(**best_params)
            self.train(X_train, y_train, sensitive_features_train)
            logger.info(f"Optimized parameters: {best_params}")
            return best_params
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Compute feature importance using permutation importance.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.

        Returns:
            np.ndarray: Feature importance scores.
        """
        perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        return perm_importance.importances_mean

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10, method: str = 'f_classif') -> tuple:
        """
        Select top k features based on statistical scoring.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            k (int): Number of features to select.
            method (str): Feature selection method ('f_classif', 'mutual_info').

        Returns:
            tuple: (Selected features array, list of selected feature names).
        """
        scoring = {'f_classif': f_classif, 'mutual_info': mutual_info_classif}.get(method, f_classif)
        selector = SelectKBest(scoring, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return X_selected, selected_features

    def calibrate(self, X_calib: pd.DataFrame, y_calib: pd.Series) -> None:
        """
        Calibrate classification model for better probability estimates.

        Args:
            X_calib (pd.DataFrame): Calibration features.
            y_calib (pd.Series): Calibration target.

        Raises:
            ValueError: If task is not classification or model is not trained.
        """
        if self.task != 'classification':
            raise ValueError("Calibration only applicable to classification.")
        if self.model is None:
            raise ValueError("Model not trained.")
        calibrated_model = CalibratedClassifierCV(self.model, cv='prefit')
        calibrated_model.fit(X_calib, y_calib)
        self.model = calibrated_model

    def save_model(self, path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            path (str): File path to save the model.

        Raises:
            ValueError: If no model is trained.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model, path)

    def load_model(self, path: str) -> None:
        """
        Load a trained model from a file.

        Args:
            path (str): File path to load the model from.
        """
        self.model = joblib.load(path)

    def save_as_onnx(self, path: str, initial_types: List) -> None:
        """
        Export model to ONNX format.

        Args:
            path (str): File path to save the ONNX model.
            initial_types (list): Initial types for ONNX conversion.
        """
        onnx_model = convert_sklearn(self.model, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, sensitive_features: Optional[pd.Series] = None, cv: int = 5) -> pd.Series:
        """
        Perform cross-validation with performance and fairness metrics.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            sensitive_features (pd.Series, optional): Sensitive features.
            cv (int): Number of cross-validation folds.

        Returns:
            pd.Series: Average metrics across folds.
        """
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        results = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sens_train = sensitive_features.iloc[train_idx] if sensitive_features is not None else None
            self.train(X_train, y_train, sens_train)
            metrics = self.evaluate(X_test, y_test, sens_train)
            results.append(metrics)
        return pd.DataFrame(results).mean()

    def detect_data_drift(self, X_train: pd.DataFrame, X_new: pd.DataFrame, threshold: float = 0.05) -> Dict:
        """
        Detect data drift using KS test.

        Args:
            X_train (pd.DataFrame): Original training features.
            X_new (pd.DataFrame): New features to compare.
            threshold (float): P-value threshold for drift detection.

        Returns:
            dict: Drift report for each feature.
        """
        drift_report = {}
        for col in X_train.columns:
            stat, p_value = ks_2samp(X_train[col], X_new[col])
            drift_report[col] = {'statistic': stat, 'p_value': p_value, 'drift_detected': p_value < threshold}
        return drift_report

    def evaluate_under_noise(self, X_test: pd.DataFrame, y_test: pd.Series, noise_level: float = 0.1, n_trials: int = 10) -> Dict:
        """
        Evaluate model robustness under Gaussian noise.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            noise_level (float): Standard deviation of Gaussian noise.
            n_trials (int): Number of trials.

        Returns:
            dict: Mean and standard deviation of performance under noise.
        """
        scores = []
        for _ in range(n_trials):
            X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
            y_pred_noisy = self.predict(X_noisy)
            score = accuracy_score(y_test, y_pred_noisy) if self.task == 'classification' else r2_score(y_test, y_pred_noisy)
            scores.append(score)
        return {'mean': np.mean(scores), 'std': np.std(scores)}

    def auto_select_algorithm(self, X: pd.DataFrame, y: pd.Series, sensitive_features: Optional[pd.Series] = None, algorithms: Optional[List[str]] = None) -> tuple:
        """
        Automatically select the best algorithm.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            sensitive_features (pd.Series, optional): Sensitive features.
            algorithms (list, optional): List of algorithms to evaluate.

        Returns:
            tuple: (Best algorithm, best score).
        """
        if algorithms is None:
            algorithms = list(self.supported_algorithms.keys())
        best_score = -np.inf
        best_algo = None
        for algo in algorithms:
            self.algorithm = algo
            self.train(X, y, sensitive_features)
            metrics = self.evaluate(X, y, sensitive_features)
            score = metrics['accuracy'] - metrics.get('demographic_parity_difference', 0) if self.task == 'classification' else metrics['r2']
            if score > best_score:
                best_score = score
                best_algo = algo
        self.algorithm = best_algo
        self.train(X, y, sensitive_features)
        return best_algo, best_score

    def create_preprocessing_pipeline(self, numerical_features: List[str], categorical_features: List[str], robust_scaling: bool = False) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for data handling.

        Args:
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            robust_scaling (bool): Use RobustScaler instead of StandardScaler.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        scaler = RobustScaler() if robust_scaling else StandardScaler()
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', scaler)
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        return self.preprocessor

    def compare_models(self, models: List['Model'], X_test: pd.DataFrame, y_test: pd.Series, sensitive_features: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.

        Args:
            models (list): List of Model instances.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            sensitive_features (pd.Series, optional): Sensitive features.

        Returns:
            pd.DataFrame: Comparison metrics for all models.
        """
        comparison = []
        for model in models:
            metrics = model.evaluate(X_test, y_test, sensitive_features)
            metrics['algorithm'] = model.algorithm
            comparison.append(metrics)
        return pd.DataFrame(comparison)

    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, output_path: str = 'roc_curve.png') -> None:
        """
        Plot ROC curve for classification tasks and save to file.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            output_path (str): Path to save the plot.

        Raises:
            ValueError: If task is not classification.
        """
        if self.task != 'classification':
            raise ValueError("ROC curve only for classification.")
        from sklearn.metrics import roc_curve
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    def plot_fairness_metrics(self, metrics: Dict, sensitive_feature: str, output_path: str = 'fairness_metrics.png') -> None:
        """
        Plot fairness metrics and save to file.

        Args:
            metrics (dict): Evaluation metrics.
            sensitive_feature (str): Name of the sensitive feature.
            output_path (str): Path to save the plot.
        """
        df = pd.DataFrame(list(metrics['group_metrics'].items()), columns=['Group', 'Metrics'])
        df = df.join(pd.DataFrame(df.pop('Metrics').values.tolist()))
        plt.figure()
        sns.barplot(x='Group', y='accuracy', data=df)
        plt.title(f'Accuracy by {sensitive_feature}')
        plt.savefig(output_path)
        plt.close()

    def plot_shap_summary(self, shap_values: np.ndarray, X: pd.DataFrame, output_path: str = 'shap_summary.png') -> None:
        """
        Save SHAP summary plot.

        Args:
            shap_values (np.ndarray): SHAP values.
            X (pd.DataFrame): Features.
            output_path (str): Path to save the plot.
        """
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(output_path)
        plt.close()

    def plot_partial_dependence(self, X: pd.DataFrame, features: List[int], feature_names: List[str], output_path: str = 'pdp.png') -> None:
        """
        Plot Partial Dependence Plots and save to file.

        Args:
            X (pd.DataFrame): Features.
            features (list): Feature indices to plot.
            feature_names (list): Names of features.
            output_path (str): Path to save the plot.
        """
        PartialDependenceDisplay.from_estimator(self.model, X, features, feature_names=feature_names)
        plt.savefig(output_path)
        plt.close()

    def generate_report(self, X_test: pd.DataFrame, y_test: pd.Series, sensitive_features: Optional[pd.Series] = None, output_path: str = 'report.json') -> Dict:
        """
        Generate a comprehensive model report.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            sensitive_features (pd.Series, optional): Sensitive features.
            output_path (str): Path to save the report.

        Returns:
            dict: Report contents.
        """
        metrics = self.evaluate(X_test, y_test, sensitive_features)
        report = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': self.algorithm,
            'task': self.task,
            'metrics': metrics,
            'feature_importance': self.get_feature_importance(X_test, y_test).tolist()
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        return report

    def monitor_performance(self, X_new: pd.DataFrame, y_new: pd.Series, baseline_metrics: Dict, threshold: float = 0.1) -> Dict:
        """
        Monitor model performance over time.

        Args:
            X_new (pd.DataFrame): New features.
            y_new (pd.Series): New target.
            baseline_metrics (dict): Baseline metrics to compare against.
            threshold (float): Threshold for detecting degradation.

        Returns:
            dict: Current metrics and degradation status.
        """
        current_metrics = self.evaluate(X_new, y_new)
        degradation = {k: abs(baseline_metrics[k] - current_metrics[k]) > threshold for k in baseline_metrics if k in current_metrics}
        return {'current_metrics': current_metrics, 'degradation_detected': degradation}
