from sklearn.ensemble import RandomForestRegressor
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric
import numpy as np


class RandomForestRegressorModel(Model):
    """
    A wrapper class for the RandomForestRegressor model from scikit-learn.
    It inherits from the abstract `Model` class and implements the
    `fit`, `predict`, and `evaluate` methods.

    Attributes:
        model (RandomForestRegressor): The underlying scikit-learn RandomForestRegressor model instance.
    """

    def __init__(self, n_estimators: int = 100) -> None:
        """
        Initialize the RandomForestRegressorModel with the specified number of estimators.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
        """
        super().__init__(type="regression")
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the RandomForestRegressor on the provided data.

        Args:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target values for training.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained RandomForestRegressor.

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
