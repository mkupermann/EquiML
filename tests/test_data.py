import pandas as pd
from src.data import Data

def test_data_class():
    # Create a synthetic dataset
    data = {
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'sensitive': ['A', 'A', 'B', 'B'],
        'target': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Initialize Data object
    data_obj = Data(sensitive_features=['sensitive'])
    data_obj.df = df  # Directly set df for testing
    
    # Test preprocess
    data_obj.preprocess(target_column='target')
    assert data_obj.X.shape == (4, 3)  # feature1, feature2, sensitive_B after one-hot
    assert data_obj.y.name == 'target'
    
    # Test analyze_bias
    bias_report = data_obj.analyze_bias()
    assert 'sensitive' in bias_report
    assert bias_report['sensitive'] == 0.0  # Both groups have selection rate 0.5
    
    # Test mitigate_bias
    data_obj.mitigate_bias(method='reweighting')
    assert hasattr(data_obj, 'sample_weights')
    assert data_obj.sample_weights.sum() == 4
    assert all(data_obj.sample_weights == 1)  # Equal weights since groups are balanced
    
    # Test split_data
    data_obj.split_data(test_size=0.5, random_state=42)
    assert data_obj.X_train.shape == (2, 3)
    assert data_obj.X_test.shape == (2, 3)
    assert len(data_obj.sample_weights_train) == 2
    assert len(data_obj.sample_weights_test) == 2

# Run the test
test_data_class()