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