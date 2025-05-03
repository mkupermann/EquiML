import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, classification_report, mean_squared_error, r2_score, 
                             matthews_corrcoef, log_loss, balanced_accuracy_score)
from sklearn.calibration import calibration_curve
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
from scipy.stats import ks_2samp, ttest_ind, permutation_test
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
import shap
import lime.lime_tabular
from statsmodels.stats.contingency_tables import mcnemar

class EquiMLEvaluation:
    """Comprehensive evaluation module for the EquiML framework, covering performance, fairness, robustness, and interpretability."""
    
    def evaluate(self, model, X, y_true, y_pred=None, y_pred_proba=None, sensitive_features=None, cv=5):
        """
        Evaluate model performance, fairness, robustness, and interpretability metrics.
        
        Parameters:
        - model: Trained machine learning model
        - X: DataFrame or array of features
        - y_true: Array of true labels
        - y_pred: Array of predicted labels (optional if model provided)
        - y_pred_proba: Array of predicted probabilities (optional)
        - sensitive_features: DataFrame of sensitive attribute values (optional)
        - cv: Number of cross-validation folds
        
        Returns:
        - metrics: Dictionary containing all computed metrics
        """
        if y_pred is None and model is not None:
            y_pred = model.predict(X)
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1] if y_pred_proba is None else y_pred_proba

        metrics = {}
        
        # Standard performance metrics
        metrics.update(self.compute_performance_metrics(y_true, y_pred, y_pred_proba))
        
        # Calibration metrics
        if y_pred_proba is not None:
            metrics['calibration'] = self.compute_calibration_metrics(y_true, y_pred_proba)
        
        # Fairness metrics
        if sensitive_features is not None:
            metrics.update(self.compute_fairness_metrics(y_true, y_pred, sensitive_features))
        
        # Robustness metrics
        metrics['robustness'] = self.compute_robustness_metrics(model, X, y_true, cv=cv)
        
        # Interpretability metrics
        metrics['interpretability'] = self.compute_interpretability_metrics(model, X, y_true, y_pred)
        
        # Statistical tests
        metrics['statistical_tests'] = self.compute_statistical_tests(y_true, y_pred)
        
        return metrics
    
    def compute_performance_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Compute a wide range of performance metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        return metrics
    
    def compute_calibration_metrics(self, y_true, y_pred_proba):
        """Compute calibration metrics for probabilistic predictions."""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        return {
            'calibration_curve': {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()},
            'brier_score': mean_squared_error(prob_true, prob_pred)
        }
    
    def compute_fairness_metrics(self, y_true, y_pred, sensitive_features):
        """Compute fairness metrics across sensitive groups."""
        metrics = {
            'demographic_parity_difference': demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features),
            'equalized_odds_difference': equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features),
            'disparate_impact': self.compute_disparate_impact(y_true, y_pred, sensitive_features)
        }
        
        mf = MetricFrame(metrics={'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score},
                         y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
        metrics['group_metrics'] = mf.by_group.to_dict()
        return metrics
    
    def compute_disparate_impact(self, y_true, y_pred, sensitive_features):
        """Calculate Disparate Impact ratio for each sensitive feature."""
        di_ratios = {}
        for feature in sensitive_features.columns:
            sensitive = sensitive_features[feature]
            groups = sensitive.unique()
            rates = {}
            for group in groups:
                mask = (sensitive == group)
                rates[group] = selection_rate(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 0
            if len(rates) >= 2:
                min_rate = min(rates.values())
                max_rate = max(rates.values())
                di_ratios[feature] = min_rate / max_rate if max_rate > 0 else 0
        return di_ratios
    
    def compute_robustness_metrics(self, model, X, y_true, cv=5):
        """Compute robustness metrics including cross-validation and noise sensitivity."""
        cv_scores = cross_val_score(model, X, y_true, cv=cv, scoring='accuracy')
        return {
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
            'noise_sensitivity': self.compute_noise_sensitivity(model, X, y_true)
        }
    
    def compute_noise_sensitivity(self, model, X, y_true, noise_level=0.1):
        """Evaluate model sensitivity to input noise."""
        X_noisy = X + np.random.normal(0, noise_level, X.shape)
        y_pred_noisy = model.predict(X_noisy)
        return accuracy_score(y_true, y_pred_noisy)
    
    def compute_interpretability_metrics(self, model, X, y_true, y_pred):
        """Compute interpretability metrics like feature importance and SHAP values."""
        metrics = {
            'permutation_importance': self.compute_permutation_importance(model, X, y_true),
            'shap_values': self.compute_shap_values(model, X)
        }
        return metrics
    
    def compute_permutation_importance(self, model, X, y_true):
        """Compute feature importance using permutation importance."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        result = permutation_importance(model, X, y_true, n_repeats=10, random_state=42)
        return {f'feature_{i}': imp for i, imp in enumerate(result.importances_mean)}
    
    def compute_shap_values(self, model, X):
        """Compute SHAP values for interpretability."""
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        return shap_values.values.mean(axis=0).tolist()
    
    def compute_statistical_tests(self, y_true, y_pred):
        """Perform statistical tests to compare predictions and ground truth."""
        return {
            'ks_test': ks_2samp(y_true, y_pred).statistic,
            't_test': ttest_ind(y_true, y_pred).pvalue,
            'mcnemar_test': self.compute_mcnemar_test(y_true, y_pred)
        }
    
    def compute_mcnemar_test(self, y_true, y_pred):
        """Perform McNemar's test for paired nominal data."""
        table = confusion_matrix(y_true, y_pred)
        result = mcnemar(table, exact=True)
        return result.pvalue
    
    def plot_confusion_matrix(self, y_true, y_pred, filename='confusion_matrix.png'):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(filename)
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, filename='roc_curve.png'):
        """Plot and save ROC curve."""
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(filename)
        plt.close()
    
    def plot_calibration_curve(self, y_true, y_pred_proba, filename='calibration_curve.png'):
        """Plot and save calibration curve."""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.savefig(filename)
        plt.close()
    
    def plot_fairness_metrics(self, metrics, sensitive_feature, filename='fairness_metrics.png'):
        """Plot and save fairness metrics across groups."""
        group_accuracy = metrics['group_metrics']['accuracy'][sensitive_feature]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(group_accuracy.keys()), y=list(group_accuracy.values()))
        plt.title(f'Accuracy by {sensitive_feature}')
        plt.savefig(filename)
        plt.close()
    
    def plot_feature_importance(self, importance_dict, filename='feature_importance.png'):
        """Plot and save feature importance."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(importance_dict.values()), y=list(importance_dict.keys()))
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.savefig(filename)
        plt.close()
    
    def plot_shap_summary(self, shap_values, X, filename='shap_summary.png'):
        """Plot and save SHAP summary plot."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(filename)
        plt.close()
    
    def generate_evaluation_report(self, metrics, filename='evaluation_report.txt'):
        """Generate a text report of all evaluation metrics."""
        with open(filename, 'w') as f:
            f.write("EquiML Evaluation Report\n")
            f.write("="*50 + "\n")
            for category, values in metrics.items():
                f.write(f"\n{category.upper()}:\n")
                if isinstance(values, dict):
                    for key, val in values.items():
                        f.write(f"  {key}: {val}\n")
                else:
                    f.write(f"  {category}: {values}\n")

# Example usage
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    sensitive = pd.DataFrame({'gender': np.random.randint(0, 2, size=100)})
    
    # Train a model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Evaluate
    evaluator = EquiMLEvaluation()
    results = evaluator.evaluate(model, X, y, sensitive_features=sensitive)
    
    # Generate visualizations and report
    evaluator.plot_confusion_matrix(y, model.predict(X))
    evaluator.plot_roc_curve(y, model.predict_proba(X)[:, 1])
    evaluator.generate_evaluation_report(results)
