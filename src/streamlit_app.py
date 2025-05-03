import streamlit as st
import pandas as pd
from src.data import Data
from src.model import Model
from src.evaluation import Evaluation

st.title("EquiML: Fair Machine Learning Dashboard")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
    sensitive_features = st.multiselect("Select sensitive features", df.columns)
    target_column = st.selectbox("Select target column", df.columns)
    algorithm = st.selectbox("Select algorithm", ['logistic_regression', 'decision_tree', 'random_forest', 'svm'])
    fairness_constraint = st.checkbox("Apply demographic parity constraint")
    
    if st.button("Train and Evaluate"):
        data = Data(sensitive_features=sensitive_features)
        data.df = df
        data.preprocess(target_column=target_column)
        data.split_data(test_size=0.2, random_state=42)
        
        model = Model(algorithm=algorithm, fairness_constraint='demographic_parity' if fairness_constraint else None)
        model.train(data.X_train, data.y_train, sensitive_features=data.X_train[sensitive_features] if sensitive_features else None)
        
        evaluation = Evaluation(model, data.X_test, data.y_test, sensitive_features=data.X_test[sensitive_features] if sensitive_features else None)
        metrics = evaluation.evaluate()
        st.write("Evaluation Metrics:", metrics)
        
        if sensitive_features:
            sensitive_feature = sensitive_features[0]  # Use the first sensitive feature
            evaluation.plot_fairness_metrics('selection_rate', sensitive_feature=sensitive_feature, kind='bar', save_path='selection_rate.png')
        st.image('selection_rate.png')
