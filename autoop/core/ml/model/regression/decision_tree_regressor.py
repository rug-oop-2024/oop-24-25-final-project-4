from sklearn.tree import DecisionTreeRegressor
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric
import numpy as np


class DecisionTreeRegressorModel(Model):
    """
    A wrapper class for the DecisionTreeRegressor model from scikit-learn.
    It inherits from the abstract `Model` class and implements the
    `fit`, `predict`, and `evaluate` methods.

    Attributes:
        model (DecisionTreeRegressor): The underlying scikit-learn DecisionTreeRegressor model instance.
    """

    model: DecisionTreeRegressor = None
    def __init__(self) -> None:
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 metric_name: str) -> float:
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)
