import joblib
from wickedhot import OneHotEncoder
from cd4ml.ml_model import MLModel
from cd4ml.feature_set import FeatureSetBase
from sklearn.tree import DecisionTreeRegressor

# Define a minimal FeatureSet at module level for pickling
class DummyFeatureSet(FeatureSetBase):
    def __init__(self):
        super().__init__('sale_id', 'price')
        # Matching 'houses/default' schema for compatibility with the web app
        self.params = {
            'feature_set_name': 'default',
            'extra_information_fields': [],
            'base_categorical_n_levels_dict': {
                'zipcode': 50,
                'style': 50,
                'state': 50
            },
            'base_fields_numerical': [
                'lot_size_sf', 'beds', 'baths', 'year_built', 
                'kitchen_refurbished', 'square_feet', 'pool', 
                'parking', 'multi_family'
            ],
            'derived_categorical_n_levels_dict': {},
            'derived_fields_numerical': [
                'avg_price_in_zip', 'num_in_zip', 
                'avg_price_in_state', 'num_in_state'
            ],
            'encoder_excluded_fields': [],
            'encoder_untransformed_fields': ['zipcode']
        }
    
    def features(self, row):
        # Extract features matching the params
        feat = {}
        for k in self.params['base_fields_numerical'] + self.params['derived_fields_numerical']:
            feat[k] = float(row.get(k, 0.0))
        for k in self.params['base_categorical_n_levels_dict'].keys():
            feat[k] = str(row.get(k, 'unknown'))
        return feat

def generate_dummy_model(path):
    """
    Generates a minimal MLModel for testing purposes.
    """
    feature_set = DummyFeatureSet()
    
    # Initialize OneHotEncoder with house schema
    categorical = feature_set.params['base_categorical_n_levels_dict']
    numerical = feature_set.params['base_fields_numerical'] + feature_set.params['derived_fields_numerical']
    
    # zipcode is categorical but untransformed in houses schema
    encoder = OneHotEncoder(categorical, numerical, 
                            omit_cols=feature_set.params['encoder_untransformed_fields'])
    
    # Fit encoder with a dummy row
    dummy_row = feature_set.features({})
    encoder.load_from_data_stream([dummy_row])
    
    # Initialize MLModel
    model = MLModel(
        algorithm_name='decision_tree',
        algorithm_params={'max_depth': 1},
        feature_set=feature_set,
        encoder=encoder,
        random_seed=42
    )
    
    # Train model with dummy data
    dummy_data = [{
        'sale_id': '1', 
        'price': 300000.0,
        'zipcode': '12345',
        'style': 'colonial',
        'lot_size_sf': 5000.0,
        'beds': 3,
        'baths': 2,
        'year_built': 1990,
        'kitchen_refurbished': 0,
        'square_feet': 2000,
        'pool': 0,
        'parking': 1,
        'multi_family': 0
    }]
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
