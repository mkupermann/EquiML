# EquiML User Guide: Testing and Getting Started

## Introduction
Welcome to EquiML, a framework designed to integrate fairness, transparency, and accountability into machine learning workflows. This guide provides a step-by-step process to test the framework and understand its core components: data handling, model training, evaluation, and visualization.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Core Components](#core-components)
   - [Data Handling](#data-handling)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
   - [Visualization](#visualization)
4. [Complete Test Script](#complete-test-script)
5. [Running the Test](#running-the-test)
6. [Troubleshooting](#troubleshooting)

## Prerequisites
Before you begin, ensure you have:
- Python 3.8 or higher installed
- The Adult Income dataset (`adult.csv`) available (download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) if needed)
- A working directory to store the dataset and output files

## Installation
To set up EquiML, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/mkupermann/EquiML.git
   cd EquiML
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the package:
   ```bash
   pip install .
   ```

## Core Components

### Data Handling
The `Data` class handles loading, preprocessing, and splitting your dataset.

#### Usage
- **Initialize**: Specify the dataset path and sensitive features.
- **Load**: Load the data into memory.
- **Preprocess**: Encode categorical variables and scale numerical features.
- **Split**: Divide into training and test sets.

#### Example
```python
from equiml.data import Data
data = Data(dataset_path='adult.csv', sensitive_features=['sex'])
data.load_data()
data.preprocess(target_column='income', numerical_features=['age'], categorical_features=['education'])
data.split_data(test_size=0.2, random_state=42)
```

### Model Training
The `Model` class trains a machine learning model with fairness constraints like demographic parity.

#### Usage
- **Initialize**: Choose an algorithm and fairness constraint.
- **Train**: Fit the model using training data and sensitive features.

#### Example
```python
from equiml.model import Model
model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
model.train(data.X_train, data.y_train, sensitive_features=data.X_train[['sex_Female']])
```

### Evaluation
The `Evaluation` class assesses model performance and fairness metrics.

#### Usage
- **Initialize**: Pass the trained model and test data.
- **Evaluate**: Compute metrics like accuracy and fairness scores.

#### Example
```python
from equiml.evaluation import Evaluation
evaluation = Evaluation(model, data.X_test, data.y_test, sensitive_features=data.X_test[['sex_Female']])
metrics = evaluation.evaluate()
```

### Visualization
The `visualization` module provides tools to plot fairness and performance metrics.

#### Usage
- **Group Metrics**: Plot metrics across sensitive groups.
- **Confusion Matrix**: Visualize prediction performance.

#### Example
```python
from equiml.visualization import plot_group_metrics, plot_confusion_matrix
plot_group_metrics(metrics, metric_names=['accuracy'], sensitive_feature='sex', save_path='group_metrics.png')
plot_confusion_matrix(data.y_test, model.predict(data.X_test), save_path='confusion_matrix.png')
```

## Complete Test Script
Here’s a full script to test EquiML’s functionality:

```python
import pandas as pd
from equiml.data import Data
from equiml.model import Model
from equiml.evaluation import Evaluation
from equiml.visualization import plot_group_metrics, plot_confusion_matrix

# Step 1: Initialize and load data
data = Data(dataset_path='adult.csv', sensitive_features=['sex'])
data.load_data()
data.preprocess(
    target_column='income',
    numerical_features=['age', 'hours-per-week'],
    categorical_features=['education', 'occupation']
)
data.split_data(test_size=0.2, random_state=42)

# Step 2: Train a model with fairness constraints
model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
model.train(data.X_train, data.y_train, sensitive_features=data.X_train[['sex_Female']])

# Step 3: Evaluate the model
evaluation = Evaluation(model, data.X_test, data.y_test, sensitive_features=data.X_test[['sex_Female']])
metrics = evaluation.evaluate()

# Step 4: Visualize the results
plot_group_metrics(
    metrics,
    metric_names=['accuracy', 'false_positive_rate'],
    sensitive_feature='sex',
    save_path='group_metrics.png'
)
plot_confusion_matrix(
    data.y_test,
    model.predict(data.X_test),
    save_path='confusion_matrix.png'
)

print("Framework test completed successfully. Check 'group_metrics.png' and 'confusion_matrix.png' for results.")
```

## Running the Test
1. Save the script as `test_equiml_framework.py`.
2. Place `adult.csv` in the same directory (or update the path in the script).
3. Run the script:
   ```bash
   python test_equiml_framework.py
   ```
4. Check the output:
   - Console message: "Framework test completed successfully..."
   - Files: `group_metrics.png` and `confusion_matrix.png`

## Troubleshooting
- **FileNotFoundError**: Ensure `adult.csv` is in the correct directory or update the path.
- **ModuleNotFoundError**: Verify EquiML and dependencies are installed (`pip install .`).
- **KeyError**: Check that column names in `preprocess` match your dataset.
- **No plots generated**: Ensure `matplotlib` is installed and save paths are writable.

This guide and script should help you confirm that EquiML works as intended. Happy testing!