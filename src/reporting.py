from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import os
from typing import Dict, Any, List

def generate_detailed_recommendations(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate detailed, actionable recommendations based on evaluation metrics.

    Args:
        metrics: Dictionary containing evaluation metrics

    Returns:
        List of recommendation dictionaries with category, priority, issue, and action
    """
    recommendations = []

    # Performance-based recommendations
    accuracy = metrics.get('accuracy', 0)
    f1_score = metrics.get('f1_score', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)

    if accuracy < 0.7:
        recommendations.append({
            'category': 'Performance',
            'priority': 'HIGH',
            'issue': f'Low model accuracy ({accuracy:.1%})',
            'action': 'IMMEDIATE ACTIONS:\n1. Collect more training data (target 2-5x current size)\n2. Try ensemble methods (Random Forest, XGBoost, LightGBM)\n3. Perform feature engineering: create polynomial features, interaction terms\n4. Use advanced preprocessing: standardization, outlier removal\n5. Implement hyperparameter tuning with GridSearchCV or Optuna\n6. Consider deep learning models if dataset is large (>10k samples)',
            'code_example': '''# Example hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    \'n_estimators\': [100, 200, 300],
    \'max_depth\': [10, 20, None],
    \'min_samples_split\': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring=\'accuracy\')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_'''
        })
    elif accuracy < 0.85:
        recommendations.append({
            'category': 'Performance',
            'priority': 'MEDIUM',
            'issue': f'Moderate model accuracy ({accuracy:.1%}) - room for improvement',
            'action': 'OPTIMIZATION ACTIONS:\n1. Fine-tune hyperparameters using Bayesian optimization\n2. Try stacking different algorithms\n3. Engineer domain-specific features\n4. Apply feature selection techniques (RFE, SelectKBest)\n5. Use cross-validation to ensure stable performance',
            'code_example': '''# Feature selection example
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_train, y_train)'''
        })

    if f1_score < 0.6:
        recommendations.append({
            'category': 'Performance',
            'priority': 'HIGH',
            'issue': f'Poor F1-score ({f1_score:.1%}) indicates class imbalance issues',
            'action': 'CLASS IMBALANCE SOLUTIONS:\n1. Apply SMOTE (Synthetic Minority Oversampling)\n2. Use class weights in model training\n3. Try ensemble methods designed for imbalanced data\n4. Collect more data for minority classes\n5. Use stratified sampling for train/test splits\n6. Consider cost-sensitive learning algorithms',
            'code_example': '''# SMOTE for handling imbalanced data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

# Class weights approach
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(\'balanced\',
                                   classes=np.unique(y_train),
                                   y=y_train)'''
        })

    # Fairness-based recommendations
    demographic_parity = metrics.get('demographic_parity_difference')
    equalized_odds = metrics.get('equalized_odds_difference')
    equal_opportunity = metrics.get('equal_opportunity_difference')

    if demographic_parity and abs(demographic_parity) > 0.1:
        severity = "CRITICAL" if abs(demographic_parity) > 0.2 else "HIGH"
        recommendations.append({
            'category': 'Fairness',
            'priority': severity,
            'issue': f'Significant demographic parity violation ({demographic_parity:.1%})',
            'action': 'BIAS MITIGATION STRATEGIES:\n1. PREPROCESSING: Apply fairness-aware data preprocessing\n   - Reweighing: Adjust sample weights by sensitive group\n   - Data augmentation for underrepresented groups\n   - Remove or transform biased features\n\n2. IN-PROCESSING: Use fairness-constrained algorithms\n   - Fairlearn\'s ExponentiatedGradient with DemographicParity\n   - Adversarial debiasing neural networks\n   - Fair representation learning\n\n3. POST-PROCESSING: Adjust predictions for fairness\n   - Threshold optimization per group\n   - Calibration across sensitive groups\n\n4. GOVERNANCE: Implement bias monitoring\n   - Set up automated fairness testing\n   - Regular bias audits\n   - Establish fairness metrics tracking',
            'code_example': '''# Fairlearn preprocessing
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Remove correlation with sensitive attribute
cr = CorrelationRemover(sensitive_feature_ids=[sensitive_column])
X_transformed = cr.fit_transform(X_train)

# Fair model training
constraint = DemographicParity()
mitigator = ExponentiatedGradient(LogisticRegression(), constraint)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)'''
        })

    if equalized_odds and abs(equalized_odds) > 0.1:
        recommendations.append({
            'category': 'Fairness',
            'priority': 'HIGH',
            'issue': f'Equalized odds violation ({equalized_odds:.1%}) - different error rates across groups',
            'action': 'EQUALIZED ODDS MITIGATION:\n1. Use EqualizedOdds constraint in training\n2. Post-processing threshold optimization\n3. Group-specific model calibration\n4. Implement fairness-aware ensemble methods\n5. Regular monitoring of group-specific performance',
            'code_example': '''# Equalized odds constraint
from fairlearn.reductions import EqualizedOdds
constraint = EqualizedOdds()
mitigator = ExponentiatedGradient(model, constraint)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)'''
        })

    # Robustness recommendations
    robustness = metrics.get('robustness', {})
    cv_std = robustness.get('cv_std', 0)

    if cv_std > 0.1:
        recommendations.append({
            'category': 'Robustness',
            'priority': 'MEDIUM',
            'issue': f'High performance variance ({cv_std:.1%}) indicates model instability',
            'action': 'STABILITY IMPROVEMENTS:\n1. Increase training data size\n2. Use more robust algorithms (Random Forest, XGBoost)\n3. Apply regularization (L1/L2 for linear models)\n4. Reduce model complexity\n5. Use stratified cross-validation\n6. Implement ensemble methods for stability\n7. Check for data leakage or temporal dependencies',
            'code_example': '''# Regularized model for stability
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1, penalty=\'l2\', random_state=42)

# Ensemble for robustness
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([
    (\'lr\', LogisticRegression()),
    (\'rf\', RandomForestClassifier()),
    (\'xgb\', XGBClassifier())
], voting=\'soft\')'''
        })

    # Data quality recommendations
    if len(recommendations) == 0:  # If no major issues, provide optimization recommendations
        recommendations.append({
            'category': 'Optimization',
            'priority': 'LOW',
            'issue': 'Model performance is satisfactory but can be optimized',
            'action': 'CONTINUOUS IMPROVEMENT:\n1. Implement A/B testing for model updates\n2. Set up model monitoring and drift detection\n3. Automate hyperparameter tuning pipelines\n4. Explore advanced feature engineering\n5. Consider model interpretability improvements\n6. Establish regular model retraining schedule',
            'code_example': '''# Model monitoring setup
import mlflow
mlflow.set_tracking_uri("your_tracking_server")
mlflow.sklearn.log_model(model, "model")
mlflow.log_metrics({
    "accuracy": accuracy,
    "f1_score": f1_score,
    "demographic_parity": demographic_parity
})'''
        })

    # Always add deployment recommendations
    recommendations.append({
        'category': 'Deployment',
        'priority': 'MEDIUM',
        'issue': 'Model deployment and monitoring considerations',
        'action': 'PRODUCTION READINESS:\n1. Set up model versioning and experiment tracking\n2. Implement automated testing pipeline\n3. Create model documentation and API documentation\n4. Establish monitoring dashboards for:\n   - Prediction accuracy\n   - Fairness metrics\n   - Data drift\n   - Model performance\n5. Plan for model updates and rollback procedures\n6. Implement logging for audit trails\n7. Set up alerts for performance degradation',
        'code_example': '''# Basic monitoring setup
import logging
logging.basicConfig(level=logging.INFO)

def predict_with_monitoring(model, X):
    predictions = model.predict(X)
    # Log predictions for monitoring
    logging.info(f"Prediction made: {len(predictions)} samples")
    return predictions'''
    })

    return recommendations

def generate_html_report(metrics, output_path='evaluation_report.html', template_path='src/report_template.html'):
    """
    Generates a comprehensive HTML report from evaluation metrics.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics.
        output_path (str): The path to save the HTML report.
        template_path (str): The path to the Jinja2 template file.
    """
    template_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    # Generate detailed recommendations
    recommendations = generate_detailed_recommendations(metrics)

    html_content = template.render(
        metrics=metrics,
        recommendations=recommendations,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    with open(output_path, 'w') as f:
        f.write(html_content)
