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
EquiML provides a robust set of tools:
- **Bias Detection and Mitigation**: Identify and reduce biases in datasets using advanced statistical methods.
- **Fair Model Training**: Train models with fairness constraints like demographic parity and equalized odds, leveraging libraries like Fairlearn.
- **Comprehensive Evaluation**: Assess model performance (accuracy, F1-score, ROC-AUC) and fairness metrics across demographic groups.
- **Model Explainability**: Use SHAP and LIME to understand feature contributions and model decisions.
- **Data Visualization**: Generate intuitive plots for bias, fairness, and performance metrics using Matplotlib, Seaborn, and Plotly.
- **User-Friendly Interface**: A Streamlit dashboard enables non-technical users to interact with the framework.
- **Extensibility**: Open-source architecture allows easy addition of new algorithms, metrics, or visualizations.
- **Robustness and Monitoring**: Tools for data drift detection, noise sensitivity analysis, and performance monitoring.
- **Deployment Readiness**: Export models to ONNX format for cross-platform compatibility.

## Supported Data Types
EquiML supports a variety of data types to facilitate equitable and responsible machine learning, including numerical, categorical, text, and image data. Below is an overview of how text and image data are processed:
Text Data
Text data is processed through the Data class in data.py. The framework:

Cleans text by converting to lowercase, removing punctuation, and eliminating stopwords using NLTK.
Transforms text into numerical features using TF-IDF vectorization (limited to 100 features) via scikit-learn's TfidfVectorizer.
Integrates these features into the dataset for model training.

Example Usage:
```python
from equiml import Data
import pandas as pd
```
### Sample dataset with a text column
```python
data = pd.DataFrame({'text_col': ['This is a sample text', 'Another text example']})
equiml_data =  = Data(data, text_features=['text_col'])
equiml_data.preprocess()
```
### Image Data
Image data is processed by loading images from file paths, resizing them to 32x32 pixels in grayscale, and flattening them into numerical arrays using OpenCV. These arrays are then integrated into the dataset.
Example Usage:

```python
from equiml import Data
import pandas as pd
```

### Sample dataset with image paths
```python
data = pd.DataFrame({'image_col': ['path/to/image1.jpg', 'path/to/image2.jpg']})
equiml_data = Data(data, image_features=['image_col'])
equiml_data.preprocess()
```

For detailed tutorials, see the Examples section.


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
   import equiml
   print("EquiML is ready!")
   ```

## Quick Start
Here’s a simple example using the Adult Income dataset to predict income fairly across gender:

```python
from equiml import Data, Model, Evaluation

# Load and preprocess data
data = Data(sensitive_features=['sex'])
data.load_data('adult.csv')
data.preprocess(target_column='income')

# Train a fair model
model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
model.train(data.X_train, data.y_train, sensitive_features=data.X_train[['sex_Female']])

# Evaluate fairness and performance
evaluation = Evaluation(model, data.X_test, data.y_test, sensitive_features=data.X_test[['sex_Female']])
metrics = evaluation.evaluate()
print(metrics)

# Visualize fairness metrics
evaluation.plot_fairness_metrics('selection_rate')
```

This code demonstrates loading data, training a fair model, evaluating performance, and visualizing fairness metrics.

## Tutorial for Beginners
New to machine learning or fairness? Our **[Beginner’s Tutorial](https://github.com/mkupermann/EquiML/blob/main/tests/Test%20and%20Try%20User_Guide.markdown)** guides you through using EquiML with the Adult Income dataset. It covers:
- Basics of machine learning and fairness.
- Setting up EquiML.
- Loading and preprocessing data.
- Training a fair model.
- Evaluating and visualizing results.

The tutorial is designed for absolute beginners, ensuring you understand each step.

## Project Structure
EquiML’s `src` directory contains the core modules:
- **`data.py`**: Handles data loading, preprocessing, bias detection, and mitigation.
- **`model.py`**: Manages model training, fairness constraints, explainability, and deployment.
- **`evaluation.py`**: Computes performance, fairness, robustness, and interpretability metrics.
- **`visualization.py`**: Provides tools for visualizing fairness and performance metrics.
- **`app.py`**: Implements a Streamlit dashboard for interactive use.

Additional directories:
- **`docs/`**: Contains tutorials and API documentation.
- **`tests/`**: Includes unit tests for ensuring reliability.

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
- **Email**: Contact the maintainer at [mkupermann@example.com](mailto:mkupermann@example.com).
- **Discussions**: Join the conversation at [GitHub Discussions](https://github.com/mkupermann/EquiML/discussions).

## Roadmap
EquiML is actively developing the following features:
- **Advanced Algorithms**: Support for neural networks and gradient boosting.
- **Enhanced Fairness Metrics**: Add calibration and disparate impact metrics.
- **Improved Data Handling**: Tools for missing data and imbalanced datasets.
- **Comprehensive Testing**: Expand unit tests for robustness.
- **Interactive Visualizations**: Enhance the Streamlit dashboard with dynamic plots.
- **Community Engagement**: Foster collaboration through forums and conferences.

Join us in building a future where AI is equitable and responsible!
