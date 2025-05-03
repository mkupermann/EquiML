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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    """
    An advanced class for training, evaluating, and interpreting machine learning models with fairness, robustness, and deployment capabilities.
    
    Attributes:
        algorithm (str): The machine learning algorithm to use.
        fairness_constraint (str): The fairness constraint to apply.
        task (str): Type of task ('classification' or 'regression').
        model: The trained model object.
    """
    
    def __init__(self, algorithm='logistic_regression', fairness_constraint=None, task='classification'):
        """
        Initialize the Model class.
        
        Args:
            algorithm (str): Algorithm to use (e.g., 'logistic_regression', 'random_forest', etc.).
            fairness_constraint (str): Fairness constraint (e.g., 'demographic_parity').
            task (str): Task type ('classification' or 'regression').
        """
        self.algorithm = algorithm
        self.fairness_constraint = fairness_constraint
        self.task = task
        self.model = None
        self.supported_algorithms = {
            'logistic_regression': LogisticRegression(max_iter=1000) if task == 'classification' else LinearRegression(),
            'decision_tree': DecisionTreeClassifier() if task == 'classification' else DecisionTreeRegressor(),
            'random_forest': RandomForestClassifier() if task == 'classification' else RandomForestRegressor(),
            'svm': SVC(probability=True) if task == 'classification' else SVR(),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss') if task == 'classification' else XGBRegressor()
        }
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from {list(self.supported_algorithms.keys())}")
        self.preprocessor = None

    def train(self, X_train, y_train, sensitive_features=None, class_weight=None):
        """Train the model with optional fairness constraints and class weights."""
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
            constraint = constraints_map.get(self.fairness_constraint, None)
            if constraint is None:
                raise ValueError(f"Unsupported fairness constraint: {self.fairness_constraint}")
            self.model = ExponentiatedGradient(base_model, constraints=constraint)
            self.model.fit(X_train, y_train, sensitive_features=sensitive_features)
            logger.info(f"Trained {self.algorithm} with {self.fairness_constraint} constraint.")
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.algorithm} without fairness constraints.")

    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities (classification only)."""
        if self.model is None or self.task != 'classification':
            raise ValueError("Probability predictions not available.")
        return self.model.predict_proba(X)

    def explain_prediction(self, X):
        """Explain predictions using SHAP values."""
        if self.model is None:
            raise ValueError("Model not trained.")
        if isinstance(self.model, ExponentiatedGradient):
            explainer = shap.KernelExplainer(self.model.predict, X)
        else:
            explainer_type = {
                'decision_tree': shap.TreeExplainer,
                'random_forest': shap.TreeExplainer,
                'xgboost': shap.TreeExplainer,
                'logistic_regression': shap.LinearExplainer
            }.get(self.algorithm, shap.KernelExplainer)
            if explainer_type == shap.LinearExplainer:
                explainer = explainer_type(self.model, X)
            elif explainer_type == shap.KernelExplainer:
                explainer = explainer_type(self.model.predict, X)
            else:
                explainer = explainer_type(self.model)
        return explainer.shap_values(X)

    def explain_instance(self, instance, X_train, feature_names, class_names=None):
        """Explain a single instance using LIME."""
        explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=class_names)
        explanation = explainer.explain_instance(instance.values, self.predict_proba)
        return explanation

    def evaluate(self, X_test, y_test, sensitive_features=None):
        """Evaluate model performance and fairness."""
        y_pred = self.predict(X_test)
        metrics = {}
        if self.task == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='macro')
            if hasattr(self.model, 'predict_proba'):
                metrics['roc_auc'] = roc_auc_score(y_test, self.predict_proba(X_test), multi_class='ovr')
        elif self.task == 'regression':
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)

        if sensitive_features is not None:
            metrics_to_compute = {
                'accuracy': accuracy_score,
                'selection_rate': selection_rate,
                'true_positive_rate': true_positive_rate,
                'false_positive_rate': false_positive_rate
            } if self.task == 'classification' else {'mean_prediction': lambda y_true, y_pred: y_pred.mean()}
            mf = MetricFrame(metrics=metrics_to_compute, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)
            metrics['group_metrics'] = mf.by_group.to_dict('index')
            fairness_metrics = {}
            for metric_name in metrics_to_compute:
                group_values = [group[metric_name] for group in metrics['group_metrics'].values()]
                fairness_metrics[f'{metric_name}_difference'] = max(group_values) - min(group_values)
            metrics.update(fairness_metrics)
            if self.task == 'classification':
                selection_rates = [group['selection_rate'] for group in metrics['group_metrics'].values()]
                metrics['disparate_impact_ratio'] = min(selection_rates) / max(selection_rates) if len(selection_rates) > 1 else 1.0
        return metrics

    def tune_hyperparameters(self, X, y, param_grid, sensitive_features=None, fairness_weight=0.5):
        """Perform hyperparameter tuning with fairness considerations."""
        base_model = self.supported_algorithms[self.algorithm]
        if self.fairness_constraint:
            raise NotImplementedError("Tuning with fairness constraints not supported.")
        scorer = make_scorer(lambda y_true, y_pred: fairness_weight * accuracy_score(y_true, y_pred) - (1 - fairness_weight) * demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)) if sensitive_features else 'accuracy'
        grid_search = GridSearchCV(base_model, param_grid, scoring=scorer, cv=5)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, sensitive_features_train=None, sensitive_features_val=None, n_trials=50):
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = {'n_estimators': trial.suggest_int('n_estimators', 10, 100), 'max_depth': trial.suggest_int('max_depth', 2, 10)} if self.algorithm == 'random_forest' else {}
            base_model = self.supported_algorithms[self.algorithm].set_params(**params)
            self.train(X_train, y_train, sensitive_features_train)
            metrics = self.evaluate(X_val, y_val, sensitive_features_val)
            score = metrics['accuracy'] - metrics.get('demographic_parity_difference', 0) if self.task == 'classification' else -metrics['mse']
            return score
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.supported_algorithms[self.algorithm].set_params(**study.best_params)
        self.train(X_train, y_train, sensitive_features_train)
        return study.best_params

    def get_feature_importance(self, X, y):
        """Compute feature importance using permutation importance."""
        perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)
        return perm_importance.importances_mean

    def select_features(self, X, y, k=10):
        """Select top k features based on statistical scoring."""
        selector = SelectKBest(f_classif if self.task == 'classification' else f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return X_selected, selected_features

    def calibrate(self, X_calib, y_calib):
        """Calibrate classification model for better probability estimates."""
        if self.task != 'classification':
            raise ValueError("Calibration only applicable to classification.")
        calibrated_model = CalibratedClassifierCV(self.model, cv='prefit')
        calibrated_model.fit(X_calib, y_calib)
        self.model = calibrated_model

    def save_model(self, path):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model, path)

    def load_model(self, path):
        """Load a trained model from a file."""
        self.model = joblib.load(path)

    def save_as_onnx(self, path, initial_types):
        """Export model to ONNX format."""
        onnx_model = convert_sklearn(self.model, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def cross_validate(self, X, y, sensitive_features=None, cv=5):
        """Perform cross-validation with performance and fairness metrics."""
        kf = KFold(n_splits=cv)
        results = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sens_train = sensitive_features.iloc[train_idx] if sensitive_features is not None else None
            self.train(X_train, y_train, sens_train)
            metrics = self.evaluate(X_test, y_test, sens_train)
            results.append(metrics)
        return pd.DataFrame(results).mean()

    def detect_data_drift(self, X_train, X_new, threshold=0.05):
        """Detect data drift using KS test."""
        drift_report = {}
        for col in X_train.columns:
            stat, p_value = ks_2samp(X_train[col], X_new[col])
            drift_report[col] = {'statistic': stat, 'p_value': p_value, 'drift_detected': p_value < threshold}
        return drift_report

    def evaluate_under_noise(self, X_test, y_test, noise_level=0.1, n_trials=10):
        """Evaluate model robustness under Gaussian noise."""
        accuracies = []
        for _ in range(n_trials):
            X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
            y_pred_noisy = self.predict(X_noisy)
            acc = accuracy_score(y_test, y_pred_noisy) if self.task == 'classification' else r2_score(y_test, y_pred_noisy)
            accuracies.append(acc)
        return {'mean': np.mean(accuracies), 'std': np.std(accuracies)}

    def auto_select_algorithm(self, X, y, sensitive_features=None, algorithms=None):
        """Automatically select the best algorithm."""
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

    def create_preprocessing_pipeline(self, numerical_features, categorical_features):
        """Create a preprocessing pipeline for data handling."""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        return self.preprocessor

    def compare_models(self, models, X_test, y_test, sensitive_features=None):
        """Compare multiple models on the same test set."""
        comparison = []
        for model in models:
            metrics = model.evaluate(X_test, y_test, sensitive_features)
            metrics['algorithm'] = model.algorithm
            comparison.append(metrics)
        return pd.DataFrame(comparison)

    def plot_roc_curve(self, X_test, y_test):
        """Plot ROC curve for classification tasks."""
        if self.task != 'classification':
            raise ValueError("ROC curve only for classification.")
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig.show()

    def plot_fairness_metrics(self, metrics, sensitive_feature):
        """Plot fairness metrics interactively."""
        df = pd.DataFrame(list(metrics['group_metrics'].items()), columns=['Group', 'Metrics'])
        df = df.join(pd.DataFrame(df.pop('Metrics').values.tolist()))
        fig = px.bar(df, x='Group', y='accuracy', title=f'Accuracy by {sensitive_feature}')
        fig.show()

    def plot_shap_summary(self, shap_values, X):
        """Save SHAP summary plot."""
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig('shap_summary.png')

    def plot_partial_dependence(self, X, features, feature_names):
        """Plot Partial Dependence Plots."""
        PartialDependenceDisplay.from_estimator(self.model, X, features, feature_names=feature_names)
        plt.savefig('pdp.png')

    def generate_report(self, X_test, y_test, sensitive_features=None, output_path='report.json'):
        """Generate a comprehensive model report."""
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

    def monitor_performance(self, X_new, y_new, baseline_metrics, threshold=0.1):
        """Monitor model performance over time."""
        current_metrics = self.evaluate(X_new, y_new)
        degradation = {k: abs(baseline_metrics[k] - current_metrics[k]) > threshold for k in baseline_metrics if k in current_metrics}
        return {'current_metrics': current_metrics, 'degradation_detected': degradation}
