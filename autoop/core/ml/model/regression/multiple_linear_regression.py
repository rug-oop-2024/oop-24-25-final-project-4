from sklearn.linear_model import LinearRegression
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric
from autoop.core.ml.artifact import Artifact
import numpy as np
from typing import Dict, Any


class MultipleLinearRegression(Model):
    """
    A wrapper class for the LinearRegression model from scikit-learn.
    It inherits from the abstract `Model` class and implements the
    `fit`, `predict`, and `evaluate` methods.

    Attributes:
        artifact (Artifact): The artifact associated with the model, containing model metadata and path.
        model (LinearRegression): The underlying scikit-learn LinearRegression model instance.
    """

    def __init__(self, parameters: dict[str, Any] = {}) -> None:
        """
        Initialize the MultipleLinearRegression model with the given artifact and LinearRegression instance.
        """
        super().__init__(type="regression", parameters=parameters)
        self.model = LinearRegression(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the MultipleLinearRegression model on the provided data.

        Args:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target values for training.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained MultipleLinearRegression model.

        Args:
            X (np.ndarray): Input features for making predictions.

        Returns:
            np.ndarray: Predicted values for the input data.
        """
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric_name: str) -> float:
        """
        Evaluate the model's performance using the specified metric.

        Args:
            X (np.ndarray): Input features for testing.
            y (np.ndarray): True values for testing.
            metric_name (str): The name of the metric to use for evaluation.

        Returns:
            float: The evaluation score based on the provided metric.
        """
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)
