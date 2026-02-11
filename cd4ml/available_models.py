from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier

regressor_classes = {
    'random_forest': RandomForestRegressor,
    'adaboost': AdaBoostRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'decision_tree': DecisionTreeRegressor,
    'ridge': Ridge,
    'lasso': Lasso,
}

classifier_classes = {
    'random_forest_classifier': RandomForestClassifier
}

algorithm_classes = regressor_classes.copy()
algorithm_classes.update(classifier_classes)

# parâmetros default para alguns algoritmos
default_params = {
    'random_forest': {'max_features': 'sqrt'},
    'random_forest_classifier': {'max_features': 'sqrt'},
}


def get_model_type(model_name):
    regressor = model_name in regressor_classes
    classifier = model_name in classifier_classes

    if regressor and classifier:
        raise ValueError(f"model_name {model_name} in BOTH regressors and classifiers, not allowed")

    if not regressor and not classifier:
        raise ValueError(f"Model {model_name} not recognized")

    return 'regressor' if regressor else 'classifier'


def get_algorithm_class(model_name):
    _ = get_model_type(model_name)
    return algorithm_classes[model_name]


def get_algorithm_instance(model_name, **kwargs):
    """Retorna uma instância do modelo com parâmetros default aplicados."""
    ModelClass = get_algorithm_class(model_name)
    params = default_params.get(model_name, {})
    params.update(kwargs)  # permite sobrescrever defaults
    return ModelClass(**params)
# Final comment to ensure newline
