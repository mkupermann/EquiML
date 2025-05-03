import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.inspection import permutation_importance
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    """
    A comprehensive class for training, evaluating, and explaining machine learning models with fairness constraints.
    
    Attributes:
        algorithm (str): The machine learning algorithm to use.
        fairness_constraint (str): The fairness constraint to apply (e.g., 'demographic_parity', 'equalized_odds').
        model: The trained model object.
    """
    
    def __init__(self, algorithm='logistic_regression', fairness_constraint=None):
        """
        Initialize the Model class.
        
        Args:
            algorithm (str): The algorithm to use (e.g., 'logistic_regression', 'decision_tree', 'random_forest', 'svm', 'xgboost').
            fairness_constraint (str): The fairness constraint to apply (e.g., 'demographic_parity', 'equalized_odds').
        """
        self.algorithm = algorithm
        self.fairness_constraint = fairness_constraint
        self.model = None
        self.supported_algorithms = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'decision_tree': DecisionTreeClassifier(),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(probability=True),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from {list(self.supported_algorithms.keys())}")

    def train(self, X_train, y_train, sensitive_features=None):
        """
        Train the model with optional fairness constraints.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            sensitive_features (pd.DataFrame): Sensitive features for fairness constraints.
        """
        base_model = self.supported_algorithms[self.algorithm]
        if self.fairness_constraint:
            if sensitive_features is None:
                raise ValueError("Sensitive features are required for fairness constraints.")
            constraint = DemographicParity() if self.fairness_constraint == 'demographic_parity' else EqualizedOdds()
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
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def explain_prediction(self, X):
        """Explain predictions using SHAP values."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if isinstance(self.model, ExponentiatedGradient):
            explainer = shap.KernelExplainer(self.model.predict, X)
        else:
            if self.algorithm in ['decision_tree', 'random_forest', 'xgboost']:
                explainer = shap.TreeExplainer(self.model)
            elif self.algorithm == 'logistic_regression':
                explainer = shap.LinearExplainer(self.model, X)
            else:
                explainer = shap.KernelExplainer(self.model.predict, X)
        shap_values = explainer.shap_values(X)
        return shap_values

    def evaluate(self, X_test, y_test, sensitive_features=None):
        """Evaluate model performance and fairness."""
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, self.predict_proba(X_test)[:, 1])
        }
        if sensitive_features is not None:
            mf = MetricFrame(metrics={'accuracy': accuracy_score}, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)
            metrics['demographic_parity_difference'] = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
            metrics['equalized_odds_difference'] = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
            metrics['group_accuracy'] = mf.by_group['accuracy'].to_dict()
        return metrics

    def tune_hyperparameters(self, X, y, param_grid, sensitive_features=None, fairness_weight=0.5):
        """Perform hyperparameter tuning with fairness considerations."""
        base_model = self.supported_algorithms[self.algorithm]
        if self.fairness_constraint:
            raise NotImplementedError("Hyperparameter tuning with fairness constraints is not yet supported.")

        def fairness_scorer(y_true, y_pred, sensitive_features):
            return -demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)

        scorer = make_scorer(lambda y_true, y_pred: fairness_weight * accuracy_score(y_true, y_pred) + (1 - fairness_weight) * fairness_scorer(y_true, y_pred, sensitive_features))
        grid_search = GridSearchCV(base_model, param_grid, scoring=scorer, cv=5)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def get_feature_importance(self, X, y):
        """Compute feature importance using permutation importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)
        return perm_importance.importances_mean

    def save_model(self, path):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")
        joblib.dump(self.model, path)

    def load_model(self, path):
        """Load a trained model from a file."""
        self.model = joblib.load(path)

    def cross_validate(self, X, y, sensitive_features, cv=5):
        """Perform cross-validation with performance and fairness metrics."""
        kf = KFold(n_splits=cv)
        results = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sens_train, sens_test = sensitive_features.iloc[train_idx], sensitive_features.iloc[test_idx]
            self.train(X_train, y_train, sens_train)
            metrics = self.evaluate(X_test, y_test, sens_test)
            results.append(metrics)
        return pd.DataFrame(results).mean()

    def detect_data_drift(self, X_train, X_new, threshold=0.05):
        """Detect data drift using KS test."""
        drift_report = {}
        for col in X_train.columns:
            stat, p_value = ks_2samp(X_train[col], X_new[col])
            drift_report[col] = {'statistic': stat, 'p_value': p_value, 'drift': p_value < threshold}
        return drift_report

    def compare_models(self, models, X_test, y_test, sensitive_features=None):
        """Compare multiple models on the same test set."""
        comparison = []
        for model in models:
            model_metrics = model.evaluate(X_test, y_test, sensitive_features)
            comparison.append(model_metrics)
        return pd.DataFrame(comparison)

    def plot_roc_curve(self, X_test, y_test):
        """Plot ROC curve for the model."""
        from sklearn.metrics import roc_curve
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{self.algorithm} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def plot_fairness_metrics(self, metrics, sensitive_feature):
        """Plot fairness metrics across groups."""
        group_accuracy = metrics['group_accuracy']
        sns.barplot(x=list(group_accuracy.keys()), y=list(group_accuracy.values()))
        plt.title(f'Accuracy by {sensitive_feature}')
        plt.show()

    def plot_shap_summary(self, shap_values, X):
        """Plot SHAP summary for feature importance."""
        shap.summary_plot(shap_values, X)
