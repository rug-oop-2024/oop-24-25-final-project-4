from sklearn.ensemble import RandomForestClassifier
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric
from typing import Dict, Any


class RandomForestClassifierModel(Model):
    """
    A wrapper class for the RandomForestClassifier model from scikit-learn.
    It inherits from the abstract `Model` class and implements the
    `fit`, `predict`, and `evaluate` methods.

    Attributes:
        model (RandomForestClassifier): The underlying scikit-learn RandomForestClassifier model instance.
    """


    def __init__(self, parameters: dict[str, Any] = {}, n_estimators: int = 100) -> None:
        """
        Initialize the RandomForestClassifierModel with the specified number of estimators.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
        """
        super().__init__(type="classification", parameters=parameters)
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, **self.parameters)

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, n_estimators):
        if isinstance(n_estimators, int):
            self._n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the RandomForestClassifier on the provided data.

        Args:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target labels for training.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained RandomForestClassifier.

        Args:
            X (np.ndarray): Input features for making predictions.

        Returns:
            np.ndarray: Predicted labels for the input data.
        """
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric_name: str) -> float:
        """
        Evaluate the model's performance using the specified metric.

        Args:
            X (np.ndarray): Input features for testing.
            y (np.ndarray): True labels for testing.
            metric_name (str): The name of the metric to use for evaluation.

        Returns:
            float: The evaluation score based on the provided metric.
        """
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)
