import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
from shap import summary_plot

def plot_group_metrics(metrics, metric_name, sensitive_feature, title=None, save_path=None):
    """
    Plot bar chart for group-wise metrics.
    
    Parameters:
    - metrics: Dictionary containing group-wise metrics.
    - metric_name: Name of the metric to plot (e.g., 'accuracy').
    - sensitive_feature: Name of the sensitive feature (e.g., 'gender').
    - title: Optional title for the plot.
    - save_path: Optional path to save the plot.
    """
    group_metrics = metrics['group_metrics'][sensitive_feature][metric_name]
    groups = list(group_metrics.keys())
    values = list(group_metrics.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=groups, y=values, palette='viridis')
    plt.title(title or f'{metric_name.capitalize()} by {sensitive_feature.capitalize()}')
    plt.xlabel(sensitive_feature.capitalize())
    plt.ylabel(metric_name.capitalize())
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """
    Plot heatmap for confusion matrix.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve_interactive(y_true, y_pred_proba, title='ROC Curve', save_path=None):
    """
    Plot interactive ROC curve using Plotly.
    
    Parameters:
    - y_true: True labels.
    - y_pred_proba: Predicted probabilities.
    - title: Title for the plot.
    - save_path: Optional path to save the plot as HTML.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    if save_path:
        fig.write_html(save_path)
    fig.show()

def plot_precision_recall_curve(y_true, y_pred_proba, title='Precision-Recall Curve', save_path=None):
    """
    Plot precision-recall curve.
    
    Parameters:
    - y_true: True labels.
    - y_pred_proba: Predicted probabilities.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Model')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, title='Calibration Curve', save_path=None):
    """
    Plot calibration curve for probability predictions.
    
    Parameters:
    - y_true: True labels.
    - y_pred_proba: Predicted probabilities.
    - n_bins: Number of bins for calibration.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.title(title)
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_summary(shap_values, X, feature_names, title='SHAP Summary Plot', save_path=None):
    """
    Plot SHAP summary plot for feature importance.
    
    Parameters:
    - shap_values: SHAP values from explainer.
    - X: DataFrame of features.
    - feature_names: List of feature names.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(importance_dict, title='Feature Importance', save_path=None):
    """
    Plot bar chart for feature importance.
    
    Parameters:
    - importance_dict: Dictionary of feature names and their importance scores.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features, palette='coolwarm')
    plt.title(title)
    plt.xlabel('Importance')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_interactive_fairness_dashboard(metrics, sensitive_features, save_path=None):
    """
    Create an interactive dashboard for fairness metrics using Plotly.
    
    Parameters:
    - metrics: Dictionary containing fairness metrics.
    - sensitive_features: List of sensitive feature names.
    - save_path: Optional path to save the dashboard as HTML.
    """
    fig = go.Figure()
    for feature in sensitive_features:
        group_metrics = metrics['group_metrics'][feature]
        df = pd.DataFrame.from_dict(group_metrics, orient='index')
        for metric in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df[metric], name=f'{feature} - {metric}'))
    fig.update_layout(barmode='group', title='Fairness Metrics by Group', xaxis_title='Group', yaxis_title='Metric Value')
    if save_path:
        fig.write_html(save_path)
    fig.show()

def plot_data_distribution(X, feature, sensitive_feature, title=None, save_path=None):
    """
    Plot distribution of a feature across sensitive groups.
    
    Parameters:
    - X: DataFrame of features.
    - feature: Feature to plot.
    - sensitive_feature: Sensitive feature to group by.
    - title: Optional title for the plot.
    - save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=X, x=feature, hue=sensitive_feature, multiple='stack', palette='Set2')
    plt.title(title or f'Distribution of {feature} by {sensitive_feature}')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_heatmap(X, title='Feature Correlation Heatmap', save_path=None):
    """
    Plot heatmap for feature correlations.
    
    Parameters:
    - X: DataFrame of features.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    corr = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning Curve', save_path=None):
    """
    Plot learning curve for model training.
    
    Parameters:
    - train_sizes: Array of training set sizes.
    - train_scores: Array of training scores.
    - test_scores: Array of test scores.
    - title: Title for the plot.
    - save_path: Optional path to save the plot.
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Train', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, label='Validation', marker='o')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()