# EquiML: A Framework for Equitable and Responsible Machine Learning

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

EquiML is an open-source Python framework designed to empower developers to create machine learning models that are accurate, fair, transparent, and accountable. In a world where AI influences critical decisions—such as hiring, lending, and healthcare—EquiML integrates fairness, explainability, and ethical considerations into every stage of the machine learning lifecycle, making responsible AI accessible to all.

## Here’s how EquiML makes a difference

- **Promoting Fairness**...
  EquiML includes tools to detect and reduce biases in AI models. For example, it can enforce rules like demographic parity, ensuring that the model treats people from different groups—say, based on gender or race—equally.    This helps prevent AI from reinforcing unfair societal patterns, making outcomes more just for everyone.
- **Increasing Transparency**...
  The framework uses techniques like SHAP and LIME to explain how an AI model makes its decisions. Imagine a bank using AI to approve loans—if the model rejects someone, EquiML can show why in a way that’s clear to regular    people, not just tech experts. This builds trust and lets us hold AI accountable.
- **Ensuring Reliability**...
  EquiML has features to check if a model stays accurate over time (e.g., detecting data drift) or handles messy real-world data (e.g., noise sensitivity analysis). This means the AI remains dependable, even as conditions     change, which is critical for things like medical diagnoses.
- **Making AI Accessible**...
  I built EquiML with a Streamlit dashboard—a simple, interactive interface—so that non-technical users, like managers or policymakers, can use it too. This opens up AI oversight to more people, not just those with coding     skills.

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorial for Beginners](#tutorial-for-beginners)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Roadmap](#roadmap)

## Introduction
EquiML addresses the urgent need for ethical AI by providing a comprehensive toolkit that ensures machine learning models are both powerful and just. Unlike traditional tools that focus on isolated aspects of fairness or explainability, EquiML offers an end-to-end solution—from data preprocessing to model deployment. It is designed for data scientists, developers, and stakeholders aiming to comply with fairness regulations and build trust in AI systems.

The framework is still under development and welcomes contributions to evolve into a practical toolset for responsible AI.

## Key Features
EquiML provides a comprehensive set of responsible AI tools:

### **Core Capabilities**
- **Flexible Data Loading**: Load data from various formats including CSV, JSON, Excel, Parquet, and ARFF.
- **Advanced Preprocessing**: Rich preprocessing tools including outlier detection, feature engineering, and support for multi-modal data (text and images).
- **Comprehensive Bias Detection**: Identify and analyze biases across multiple dimensions with detailed reporting.

### **Bias Mitigation & Fairness**
- **Advanced Bias Mitigation**: Multiple preprocessing techniques including reweighing, correlation removal, and data augmentation.
- **Fair Model Training**: Train models with fairness constraints like demographic parity and equalized odds.
- **Post-Processing Fairness**: Threshold optimization and calibration techniques for fairness adjustment.
- **Class Imbalance Handling**: SMOTE, random sampling, and class weighting methods.

### **Enhanced Algorithm Support**
- **Traditional Algorithms**: Logistic Regression with L1/L2/ElasticNet regularization, Random Forest, XGBoost, LightGBM.
- **Robust Variants**: Enhanced algorithms with stability improvements (robust_random_forest, robust_xgboost, robust_ensemble).
- **Ensemble Methods**: Voting classifiers, bagging, and diverse estimator combinations for improved robustness.
- **Hyperparameter Tuning**: Automated optimization using Optuna, GridSearchCV, and RandomSearchCV.

### **Model Stability & Robustness**
- **Stability Improvements**: Comprehensive methods to reduce model variance and improve reliability.
- **Data Leakage Detection**: Automatic detection of potential data leakage and temporal dependencies.
- **Stratified Cross-Validation**: Enhanced validation techniques for stable performance assessment.
- **Model Complexity Reduction**: Automatic complexity adjustment based on dataset characteristics.

### **Monitoring & Governance**
- **Real-Time Bias Monitoring**: Continuous monitoring of model predictions for bias violations.
- **Data Drift Detection**: Statistical detection of data distribution changes over time.
- **Automated Alerting**: Priority-based alert system for bias violations and performance degradation.
- **Audit Trails**: Comprehensive logging and monitoring history export capabilities.

### **Evaluation & Reporting**
- **Comprehensive Evaluation**: Performance, fairness, robustness, and interpretability metrics.
- **Model Explainability**: SHAP and LIME integration for decision transparency.
- **Enhanced Reporting**: Detailed HTML reports with priority-coded recommendations and executable code examples.
- **Rich Visualizations**: Interactive plots with Plotly, comprehensive fairness visualizations.

### **User Experience**
- **Streamlit Dashboard**: Interactive interface for non-technical users.
- **Detailed Documentation**: Complete beginner's guides and LLM development tutorials.
- **Professional Configuration**: Linting, type checking, and development tool integration.
- **Extensibility**: Open-source architecture for easy customization and extension.

## Installation
Follow these steps to set up EquiML on your local machine.

### Prerequisites
- **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/).
- **Git**: Download from [git-scm.com](https://git-scm.com/downloads).
- **Virtual Environment** (recommended): Use `venv` or `conda`.

### Step-by-Step Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mkupermann/EquiML.git
   cd EquiML
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv equiml_env
   source equiml_env/bin/activate  # On Windows: equiml_env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install EquiML**:
   ```bash
   pip install .
   ```

   For development:
   ```bash
   pip install -e .
   ```

5. **Verify Installation**:
   ```python
   from src.data import Data
   print("EquiML is ready!")
   ```

## Quick Start
Here's an enhanced example using the Adult Income dataset with EquiML's latest capabilities:

### **Basic Fair Model Training**
```python
from src.data import Data
from src.model import Model
from src.evaluation import EquiMLEvaluation
from src.monitoring import BiasMonitor
import pandas as pd

# Load and preprocess data with bias mitigation
data = Data(dataset_path='tests/adult.csv', sensitive_features=['sex'])
data.load_data()
data.preprocess(
    target_column='income',
    numerical_features=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
    categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
)

# Apply enhanced bias mitigation and class imbalance handling
data.apply_bias_mitigation(method='reweighing')
data.handle_class_imbalance(method='class_weights')
data.split_data()

# Prepare features
sensitive_feature_column = [col for col in data.X_train.columns if col.startswith('sex_')][0]
sensitive_features_train = data.X_train[sensitive_feature_column]
X_train = data.X_train.drop(columns=[sensitive_feature_column])
sensitive_features_test = data.X_test[sensitive_feature_column]
X_test = data.X_test.drop(columns=[sensitive_feature_column])

# Train enhanced robust model with stability improvements
model = Model(algorithm='robust_random_forest', fairness_constraint='demographic_parity')
model.apply_stability_improvements(X_train, data.y_train, sensitive_features_train)
model.train(X_train, data.y_train, sensitive_features=sensitive_features_train)

# Comprehensive evaluation with monitoring
evaluation = EquiMLEvaluation()
predictions = model.predict(X_test)
metrics = evaluation.evaluate(model, X_test, data.y_test, y_pred=predictions, sensitive_features=sensitive_features_test)

# Set up real-time bias monitoring
monitor = BiasMonitor(sensitive_features=['sex'])
monitoring_result = monitor.monitor_predictions(
    predictions,
    pd.DataFrame({sensitive_feature_column: sensitive_features_test}),
    data.y_test.values
)

# Generate enhanced report with detailed recommendations
evaluation.generate_report(metrics, output_path='enhanced_evaluation_report.html', template_path='src/report_template.html')

print(f"Model Accuracy: {metrics['accuracy']:.1%}")
print(f"Bias Level: {abs(metrics.get('demographic_parity_difference', 0)):.1%}")
print(f"Bias Violations: {len(monitoring_result['violations'])}")
```

### **Advanced Usage with All Features**
```python
# For maximum robustness and fairness
model = Model(algorithm='robust_ensemble', fairness_constraint='equalized_odds')

# Tune hyperparameters automatically
model.tune_hyperparameters(method='optuna', n_trials=50)

# Apply comprehensive stability improvements
model.apply_stability_improvements(X_train, data.y_train, stability_method='comprehensive')

# Check for data quality issues
leakage_results = model.check_data_leakage(X_train, X_test)
stability_metrics = model.evaluate_model_stability(X_train, data.y_train)

# Deploy with post-processing fairness adjustments
fair_predictions = model.apply_fairness_postprocessing(X_test, sensitive_features_test)
```

This demonstrates EquiML's complete pipeline: enhanced preprocessing, robust training, comprehensive evaluation, real-time monitoring, and detailed actionable reporting.

## Comprehensive Tutorials
EquiML provides detailed guides for different learning paths:

### **For Complete Beginners**
**[Complete Beginner's Guide to EquiML](Complete_Beginners_Guide_to_EquiML.md)** - A comprehensive 1,000+ line guide covering:
- What EquiML does in simple terms with real-world examples
- Step-by-step installation for all operating systems
- Your first fair AI model with copy-paste ready code
- Understanding results and metrics
- Advanced features and real-world applications
- 4-week structured learning path
- Common problems and solutions

### **For LLM Development**
**[Complete Guide to Building Fair LLMs with EquiML](Complete_Guide_to_Building_Fair_LLMs_with_EquiML.md)** - A comprehensive guide for building responsible Large Language Models:
- Understanding LLMs vs traditional ML
- 6-phase fair LLM development framework
- Bias detection and mitigation for text generation
- Production-scale LLM training considerations
- Real-time monitoring for language models
- Complete code examples for fair LLM development

### **Original Tutorial**
**[Beginner's Tutorial](Beginners_Tutorial_for_Using_EquiML.md)** - The original tutorial covering basic EquiML usage.

## Project Structure
EquiML's enhanced architecture includes:

### **Core Modules (`src/` directory)**
- **`data.py`**: Enhanced data handling with bias mitigation, class imbalance handling, and multi-modal support.
- **`model.py`**: Advanced model training with robust algorithms, stability improvements, hyperparameter tuning, and fairness constraints.
- **`evaluation.py`**: Comprehensive evaluation including performance, fairness, robustness, and interpretability metrics.
- **`reporting.py`**: Enhanced HTML report generation with detailed recommendations and actionable code examples.
- **`monitoring.py`**: Real-time bias monitoring, data drift detection, and automated alerting systems.
- **`visualization.py`**: Rich visualizations for bias analysis, fairness metrics, and performance assessment.
- **`streamlit_app.py`**: Interactive dashboard for non-technical users.

### **Configuration Files**
- **`pyproject.toml`**: Modern Python project configuration with linting and type checking.
- **`.flake8`**: Code quality and style configuration.
- **`mypy.ini`**: Type checking configuration for improved code quality.

### **Documentation**
- **`Complete_Beginners_Guide_to_EquiML.md`**: Comprehensive guide for traditional ML with EquiML.
- **`Complete_Guide_to_Building_Fair_LLMs_with_EquiML.md`**: Complete guide for fair Large Language Model development.
- **`Beginners_Tutorial_for_Using_EquiML.md`**: Original tutorial for basic usage.

### **Testing & Examples**
- **`tests/`**: Unit tests, sample datasets (adult.csv), and test utilities.
- **Example reports**: Multiple HTML reports demonstrating different use cases and features.

## Contributing
We welcome contributions to make EquiML a leading tool for responsible AI. To contribute:
1. **Fork the Repository**: Create your own copy on GitHub.
2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/your-username/EquiML.git
   ```
3. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature
   ```
4. **Make Changes**: Follow PEP 8 standards and add tests.
5. **Submit a Pull Request**: Describe your changes clearly.

See our **[Contributing Guide](https://github.com/mkupermann/EquiML/blob/main/CONTRIBUTING.md)** for details.

## License
EquiML is released under the **[MIT License](https://github.com/mkupermann/EquiML/blob/main/LICENSE)**, allowing free use, modification, and distribution.

## Contact
- **GitHub Issues**: Report bugs or request features at [GitHub Issues](https://github.com/mkupermann/EquiML/issues).
- **Email**: Contact the maintainer at [mkupermann@kupermann.com](mailto:michael@kupermann.com).
- **Discussions**: Join the conversation at [GitHub Discussions](https://github.com/mkupermann/EquiML/discussions).

## Recent Enhancements & Roadmap

### **Recently Completed (Latest Version)**
- **Enhanced Bias Mitigation**: Implemented comprehensive preprocessing techniques (reweighing, correlation removal, data augmentation).
- **Advanced Algorithm Support**: Added robust algorithm variants with stability improvements and ensemble methods.
- **Real-Time Monitoring**: Built-in bias monitoring and data drift detection with automated alerting.
- **Stability Improvements**: Comprehensive model stability enhancements including regularization, complexity reduction, and cross-validation.
- **Enhanced Reporting**: Detailed HTML reports with priority-coded recommendations and executable code examples.
- **Class Imbalance Handling**: SMOTE, random sampling, and class weighting implementations.
- **Professional Configuration**: Added linting, type checking, and development tool integration.
- **Comprehensive Documentation**: Complete guides for both traditional ML and LLM development.

### **Current Development Focus**
- **Large Language Model Support**: Expanding fairness framework for LLM development and evaluation.
- **Advanced Fairness Metrics**: Additional fairness metrics and sophisticated bias detection algorithms.
- **Production Deployment Tools**: Enhanced deployment pipelines with comprehensive monitoring.
- **Community Integrations**: Integration with popular ML platforms and cloud services.
- **Performance Optimization**: Speed and memory optimizations for large-scale deployments.

### **Future Roadmap**
- **Federated Learning Support**: Fair ML across distributed datasets.
- **Automated Bias Remediation**: AI-powered bias detection and automatic correction.
- **Regulatory Compliance**: Built-in compliance checking for AI regulations (EU AI Act, etc.).
- **Advanced Interpretability**: Next-generation explainability techniques.
- **Community Platform**: Collaboration tools and shared fairness benchmarks.

Join us in building a future where AI is equitable and responsible!
