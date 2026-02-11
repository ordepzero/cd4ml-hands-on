import joblib
from wickedhot import OneHotEncoder
from cd4ml.ml_model import MLModel
from cd4ml.feature_set import FeatureSetBase
from sklearn.tree import DecisionTreeRegressor

# Define a minimal FeatureSet at module level for pickling
class DummyFeatureSet(FeatureSetBase):
    def __init__(self):
        super().__init__('id', 'target')
        self.params = {
            'extra_information_fields': [],
            'base_fields_numerical': ['num1'],
            'base_categorical_n_levels_dict': {},
            'derived_fields_numerical': [],
            'derived_categorical_n_levels_dict': {},
            'encoder_excluded_fields': [],
            'encoder_untransformed_fields': []
        }
    
    def features(self, row):
        return {'num1': row['num1']}

def generate_dummy_model(path):
    """
    Generates a minimal MLModel for testing purposes.
    """
    # Initialize components
    feature_set = DummyFeatureSet()
    # Correct initialization for OneHotEncoder: categorical as dict, numerical as list
    categorical = {}
    numerical = ['num1']
    encoder = OneHotEncoder(categorical, numerical)
    
    # Fit encoder with a dummy row using correct method
    dummy_features = [{'num1': 1.0}]
    encoder.load_from_data_stream(dummy_features)
    
    # Initialize MLModel
    model = MLModel(
        algorithm_name='decision_tree',
        algorithm_params={'max_depth': 1},
        feature_set=feature_set,
        encoder=encoder,
        random_seed=42
    )
    
    # Train model with dummy data
    dummy_data = [{'id': '1', 'target': 10.0, 'num1': 1.0}]
    model.train(dummy_data)
    
    # Save model
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"Dummy model generated at {path}")

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        model_path = Path(__file__).parent / "full_model.pkl"
        
    generate_dummy_model(model_path)
