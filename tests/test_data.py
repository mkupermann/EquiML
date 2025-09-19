import pandas as pd
from src.data import Data
import os
import numpy as np

def test_load_data():
    """
    Tests the load_data method of the Data class with various file formats.
    """
    # Create the dummy files first
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [1.1, 2.2, 3.3],
        'c': ['foo', 'bar', 'baz'],
        'd': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
    })
    df.to_parquet('tests/data.parquet')
    df.to_csv('tests/data.csv', index=False)
    df.to_json('tests/data.json', orient='records')
    df.to_excel('tests/data.xlsx', index=False)
    arff_content = """
@RELATION data
@ATTRIBUTE a INTEGER
@ATTRIBUTE b REAL
@ATTRIBUTE c STRING
@ATTRIBUTE d STRING
@DATA
1,1.1,"foo","2024-01-01 00:00:00"
2,2.2,"bar","2024-01-02 00:00:00"
3,3.3,"baz","2024-01-03 00:00:00"
"""
    with open('tests/data.arff', 'w') as f:
        f.write(arff_content)

    data = Data()

    # Test loading CSV
    data.load_data('tests/data.csv')
    assert data.df is not None
    assert isinstance(data.df, pd.DataFrame)
    assert data.df.shape == (3, 4)
    
    # Test loading JSON
    data.load_data('tests/data.json')
    assert data.df is not None
    assert isinstance(data.df, pd.DataFrame)
    assert data.df.shape == (3, 4)
    
    # Test loading Excel
    data.load_data('tests/data.xlsx')
    assert data.df is not None
    assert isinstance(data.df, pd.DataFrame)
    assert data.df.shape == (3, 4)
    
    # Test loading Parquet
    data.load_data('tests/data.parquet')
    assert data.df is not None
    assert isinstance(data.df, pd.DataFrame)
    assert data.df.shape == (3, 4)

    # Test loading ARFF
    data.load_data('tests/data.arff')
    assert data.df is not None
    assert isinstance(data.df, pd.DataFrame)
    assert data.df.shape == (3, 4)
    assert data.df['a'][0] == 1.0
    assert data.df['c'][0] == "foo"

    # Clean up the created files
    os.remove('tests/data.csv')
    os.remove('tests/data.json')
    os.remove('tests/data.xlsx')
    os.remove('tests/data.parquet')
    os.remove('tests/data.arff')

def test_preprocess():
    """Tests the preprocess method of the Data class."""
    df = pd.DataFrame({
        'num1': [1, 2, 3, np.nan],
        'cat1': ['a', 'b', 'a', 'c'],
        'target': [0, 1, 0, 1]
    })
    data = Data()
    data.df = df
    data.preprocess(
        target_column='target',
        numerical_features=['num1'],
        categorical_features=['cat1']
    )
    assert data.X is not None
    assert data.y is not None
    assert data.X.shape == (4, 3)
    assert 'num1' in data.X.columns
    assert 'cat1_b' in data.X.columns
    assert not data.X.isnull().values.any()

def test_detect_outliers():
    """Tests the detect_outliers method of the Data class."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 100],
        'target': [0, 1, 0, 1]
    })
    data = Data()
    data.df = df
    data.preprocess(target_column='target', numerical_features=['feature1'])

    # Test flagging outliers
    outliers = data.detect_outliers(features=['feature1'], threshold=1.5, action='flag')
    assert outliers.shape == (4, 1)
    assert outliers['feature1'].sum() == 1 # one outlier

    # Test removing outliers
    data.detect_outliers(features=['feature1'], threshold=1.5, action='remove')
    assert data.X.shape == (3, 1)

    # Test capping outliers
    data.df = df # reset dataframe
    data.preprocess(target_column='target', numerical_features=['feature1'])
    data.detect_outliers(features=['feature1'], method='iqr', action='cap')
    assert data.X['feature1'].max() < 100

def test_mitigate_bias():
    """Tests the mitigate_bias method of the Data class."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'sensitive': ['A', 'A', 'B', 'B'],
        'target': [0, 1, 0, 1]
    })
    data = Data(sensitive_features=['sensitive'])
    data.df = df
    data.preprocess(target_column='target', categorical_features=['sensitive'])
    data.mitigate_bias(method='reweighing')
    assert data.sample_weights is not None
    assert len(data.sample_weights) == 4
    # In this balanced case, weights should be 1
    assert all(data.sample_weights == 1)

def test_feature_engineering():
    """Tests the feature_engineering method of the Data class."""
    df = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    data = Data()
    data.df = df
    data.preprocess(target_column='target', numerical_features=['num1', 'num2'])

    # Test polynomial features
    data.feature_engineering(polynomial_degree=2)
    assert 'num1^2' in data.X.columns
    assert 'num1 num2' in data.X.columns
    assert 'num2^2' in data.X.columns
    assert data.X.shape[1] == 6  # 1, num1, num2, num1^2, num1*num2, num2^2
