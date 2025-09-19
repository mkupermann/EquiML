# Beginner’s Tutorial for Using EquiML: A Framework for Equitable Machine Learning

Welcome to this detailed tutorial on how to use EquiML, a framework designed to help you build machine learning models that are not only accurate but also fair, transparent, and accountable. This tutorial is crafted for absolute beginners, so we’ll go step-by-step, explaining everything from the basics of machine learning to running your first fair model using the EquiML framework from the GitHub repository at [https://github.com/mkupermann/EquiML](https://github.com/mkupermann/EquiML). Let’s dive in!

---

## 1. Introduction to Machine Learning and Fairness

### What is Machine Learning?
Machine learning (ML) is a way for computers to learn from data and make predictions or decisions without being explicitly told what to do. Imagine teaching a computer to recognize whether an email is spam by showing it lots of examples of spam and non-spam emails. Over time, it learns patterns and can predict on its own.

For example:
- **Input**: Emails with words like "free," "win," or "urgent."
- **Output**: "Spam" or "Not Spam."

This process involves feeding data into an algorithm, which then "trains" a model to make predictions.

### Why Fairness Matters
While ML is powerful, it can sometimes make unfair decisions if the data it learns from contains biases. For instance, if a model used to approve loans learns from historical data where certain groups were unfairly denied, it might continue that pattern, rejecting people based on race, gender, or other sensitive traits.

**Equitable and responsible machine learning** ensures that models treat everyone fairly. EquiML is a tool that helps you detect and reduce bias, making your models more just and trustworthy.

### What is EquiML?
EquiML is a Python framework that integrates fairness into the ML process. It provides tools to:
- **Detect bias** in your data or model.
- **Train fair models** that minimize unfair outcomes.
- **Evaluate fairness** alongside accuracy.
- **Explain predictions** so you understand why the model behaves the way it does.

In this tutorial, we’ll use EquiML to predict income levels fairly, ensuring our model doesn’t discriminate based on gender.

---

## 2. Setting Up Your Environment

Before we use EquiML, we need to set up our computer with the necessary tools. Don’t worry—we’ll walk through each step slowly.

### Prerequisites
- **Python**: EquiML is a Python package, so you need Python installed (version 3.8 or higher). Download it from [python.org](https://www.python.org/downloads/) if you don’t have it.
- **Git**: We’ll use Git to download the EquiML repository. If you don’t have Git, download it from [git-scm.com](https://git-scm.com/).
- **A Terminal**: You’ll run commands in a terminal (Command Prompt on Windows, Terminal on macOS/Linux).
- **A Text Editor**: Use something like VS Code, PyCharm, or even Notepad to write Python code.

### Step 2.1: Clone the EquiML Repository
The EquiML code lives on GitHub. "Cloning" means downloading it to your computer.

1. Open your terminal.
2. Run this command to download the repository:
   ```bash
   git clone https://github.com/mkupermann/EquiML.git
   ```
3. This creates a folder called `EquiML` on your computer. Move into it with:
   ```bash
   cd EquiML
   ```

### Step 2.2: Install Dependencies
EquiML relies on other Python packages (like `scikit-learn` for ML and `fairlearn` for fairness). These are listed in a file called `requirements.txt`.

1. In the terminal, while inside the `EquiML` folder, run:
   ```bash
   pip install -r requirements.txt
   ```
2. This downloads and installs all the packages EquiML needs. It might take a few minutes.

**Tip**: If you get errors (e.g., "pip is not recognized"), ensure Python is installed correctly and added to your system’s PATH. Restart your terminal and try again.

### Verify Installation
Let’s check if everything worked:
1. Open a Python interpreter by typing `python` in your terminal.
2. Type:
   ```python
   from src.data import Data
   print("EquiML is ready!")
   ```
3. If you see "EquiML is ready!" without errors, you’re set. Exit with `exit()`.

---

## 3. Understanding the Adult Income Dataset

We’ll use EquiML with the **Adult Income dataset**, a popular dataset for studying fairness. It contains information about people (like age, education, and gender) and predicts whether their income is above or below $50,000 per year. A small version of this dataset is included in the `tests` directory of this repository (`tests/adult.csv`).

### What’s in the Dataset?
- **Features**: Age, work class, education, occupation, sex, race, etc.
- **Target**: Income (>50K or <=50K).
- **Sensitive Feature**: We’ll focus on "sex" to ensure fairness across gender.

---

## 4. Using EquiML: A Step-by-Step Guide

Now, let’s use EquiML to build a fair model. We’ll predict income and ensure fairness with respect to gender.

### Step 4.1: Create a Python Script
1. In your `EquiML` folder, create a file called `tutorial.py` using your text editor.
2. We’ll add code to this file step-by-step.

### Step 4.2: Load and Preprocess Data
EquiML’s `Data` class helps us load and prepare the dataset.

Add this to `tutorial.py`:
```python
from src.data import Data
from src.model import Model
from src.evaluation import EquiMLEvaluation
import pandas as pd

# Create a Data object, specifying 'sex' as the sensitive feature
data = Data(dataset_path='tests/adult.csv', sensitive_features=['sex'])

# Load the dataset
data.load_data()

# Preprocess the data, setting 'income' as the target
data.preprocess(
    target_column='income',
    numerical_features=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
    categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
)
data.split_data()
```

#### What’s Happening?
- **Import**: We bring in EquiML’s main tools: `Data` for handling data, `Model` for training, and `Evaluation` for checking results.
- **Data Object**: We tell EquiML that "sex" is a sensitive feature we care about for fairness.
- **Load Data**: Reads `tests/adult.csv` into memory.
- **Preprocess**: Splits the data into features (X) and target (y), encodes categorical variables (e.g., "sex" becomes "sex_Male" and "sex_Female"), and splits it into training and testing sets.

### Step 4.3: Train a Fair Model
We’ll use logistic regression (a simple ML algorithm) and enforce "demographic parity," meaning the model’s predictions should be equal across gender groups.

Add this to `tutorial.py`:
```python
# The sensitive feature 'sex' is one-hot encoded. Let's find the column name.
sensitive_feature_column = [col for col in data.X_train.columns if col.startswith('sex_')][0]
sensitive_features_train = data.X_train[sensitive_feature_column]
X_train = data.X_train.drop(columns=[sensitive_feature_column])
sensitive_features_test = data.X_test[sensitive_feature_column]
X_test = data.X_test.drop(columns=[sensitive_feature_column])

# Create a Model object with logistic regression and demographic parity
model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')

# Train the model
model.train(X_train, data.y_train, sensitive_features=sensitive_features_train)
```

#### What’s Happening?
- **Model Object**: We choose logistic regression and set a fairness goal. Demographic parity ensures the percentage of people predicted to earn >50K is the same for males and females.
- **Train**: The model learns from the training data (`X_train`, `y_train`), adjusting for fairness based on the sensitive feature.

### Step 4.4: Evaluate the Model
Let’s see how well the model performs and whether it’s fair.

Add this to `tutorial.py`:
```python
# Create an Evaluation object
evaluation = EquiMLEvaluation()

# Get performance and fairness metrics
metrics = evaluation.evaluate(model, X_test, data.y_test, sensitive_features=sensitive_features_test)
print(metrics)
```

#### What’s Happening?
- **Evaluation Object**: We create an evaluation object.
- **Evaluate**: Calculates metrics like:
  - **Accuracy**: How often the model is correct.
  - **F1-Score**: Balances precision and recall.
  - **Demographic Parity Difference**: Measures fairness (closer to 0 means fairer).

When you run this, you’ll see a dictionary of metrics in your terminal.

### Step 4.5: Generate a Report
EquiML can generate a comprehensive HTML report with visualizations and recommendations.

Add this to `tutorial.py`:
```python
# Generate a report
evaluation.generate_report(metrics, output_path='evaluation_report.html', template_path='src/report_template.html')
```

#### What’s Happening?
- **generate_report**: Creates an HTML file with all the evaluation results, including plots for confusion matrix, ROC curve, and fairness metrics.

### Full Code
Your `tutorial.py` should now look like this:
```python
from src.data import Data
from src.model import Model
from src.evaluation import EquiMLEvaluation
import pandas as pd

# Load and preprocess data
data = Data(dataset_path='tests/adult.csv', sensitive_features=['sex'])
data.load_data()
data.preprocess(
    target_column='income',
    numerical_features=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
    categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
)
data.split_data()

# The sensitive feature 'sex' is one-hot encoded. Let's find the column name.
sensitive_feature_column = [col for col in data.X_train.columns if col.startswith('sex_')][0]
sensitive_features_train = data.X_train[sensitive_feature_column]
X_train = data.X_train.drop(columns=[sensitive_feature_column])
sensitive_features_test = data.X_test[sensitive_feature_column]
X_test = data.X_test.drop(columns=[sensitive_feature_column])

# Train a fair model
model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
model.train(X_train, data.y_train, sensitive_features=sensitive_features_train)

# Evaluate fairness and performance
evaluation = EquiMLEvaluation()
metrics = evaluation.evaluate(model, X_test, data.y_test, sensitive_features=sensitive_features_test)
print(metrics)

# Generate a report
evaluation.generate_report(metrics, output_path='evaluation_report.html', template_path='src/report_template.html')
```

### Run the Script
1. Save `tutorial.py`.
2. In your terminal, from the `EquiML` folder, run:
   ```bash
   python tutorial.py
   ```
3. Check the output: metrics in the terminal and an `evaluation_report.html` file in your folder.

---

## 5. Exploring Further: Additional Features and Customization

You’ve built your first fair model! EquiML offers more to explore:

### Different Algorithms
Try other algorithms instead of `logistic_regression`:
- `random_forest`
- `xgboost`
- `lightgbm`
Change the `algorithm` parameter in the `Model` call.

### Other Fairness Constraints
Instead of `demographic_parity`, try:
- **`equalized_odds`**: Ensures equal true positive and false positive rates across groups.
Example:
```python
model = Model(algorithm='logistic_regression', fairness_constraint='equalized_odds')
```

### Explain Predictions
EquiML integrates SHAP and LIME to explain model predictions. The `Evaluation` class automatically computes these metrics.

---

## Recap
You’ve learned:
1. What machine learning and fairness are.
2. How to set up EquiML by cloning the repo and installing it.
3. How to use the Adult Income dataset.
4. How to load data, train a fair model, evaluate it, and generate a report with EquiML.

This is just the beginning! Play with different datasets, algorithms, or fairness constraints to deepen your understanding. EquiML makes it easier to build ML models that are both powerful and fair—happy coding!
