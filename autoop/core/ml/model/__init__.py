"""
This module defines a collection of machine learning model classes
for both regression and classification tasks, along with a factory
function to retrieve a model instance by name.

Available Models:
    - REGRESSION_MODELS: A list of available regression models.
    - CLASSIFICATION_MODELS: A list of available classification models.

Functions:
    - get_model(model_name: str) -> Model: Returns a model instance
      based on the provided model name.
"""

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.decision_tree_regressor import (
    DecisionTreeRegressorModel,
)
from autoop.core.ml.model.regression.random_forest_regressor import (
    RandomForestRegressor,
)
from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeClassifierModel,
)
from autoop.core.ml.model.classification.random_forest_classifier import (
    RandomForestClassifierModel,
)
from autoop.core.ml.model.classification.k_nearest_neighbour import (
    KNNClassifier
)

REGRESSION_MODELS = [
    "decision_tree_regressor",
    "multiple_linear_regression",
    "random_forest_regressor",
]

CLASSIFICATION_MODELS = [
    "decision_tree_classifier",
    "k_nearest_neighbour",
    "random_forest_classifier",
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "decision_tree_regressor":
        return DecisionTreeRegressorModel(Model)
    if model_name == "multiple_linear_regression":
        return MultipleLinearRegression(Model)
    if model_name == "random_forest_regressor":
        return RandomForestRegressor(Model)
    if model_name == "decision_tree_classifier":
        return DecisionTreeClassifierModel(Model)
    if model_name == "random_forest_classifier":
        return RandomForestClassifierModel(Model)
    if model_name == "k_nearest_neighbour":
        return KNNClassifier(Model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
