import pandas as pd
from src.data import Data
from src.model import Model
from src.evaluation import EquiMLEvaluation

# Step 1: Initialize and load data
data = Data(dataset_path='adult.csv', sensitive_features=['sex'])
data.load_data()
data.preprocess(
    target_column='income',
    numerical_features=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
    categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
)
data.split_data(test_size=0.2, random_state=42)

# Step 2: Train a model with fairness constraints
model = Model(algorithm='logistic_regression', fairness_constraint='demographic_parity')
model.train(data.X_train, data.y_train, sensitive_features=data.X_train[['sex_Male']])

# Step 3: Evaluate the model
evaluation = EquiMLEvaluation()
metrics = evaluation.evaluate(model, data.X_test, data.y_test, sensitive_features=data.X_test[['sex_Male']])

# Step 4: Visualize the results
evaluation.plot_fairness_metrics(
    metrics,
    sensitive_feature='sex',
    output_path='group_metrics.png'
)
evaluation.plot_confusion_matrix(
    data.y_test,
    model.predict(data.X_test),
    output_path='confusion_matrix.png'
)

print("Framework test completed successfully. Check 'group_metrics.png' and 'confusion_matrix.png' for results.")