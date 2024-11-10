from sklearn.tree import DecisionTreeClassifier
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import get_metric
from pydantic import PrivateAttr


class DecisionTreeClassifierModel(Model):
    """
    A wrapper class for the DecisionTreeClassifier model from scikit-learn.
    It inherits from the abstract `Model` class and implements the
    `fit`, `predict`, and `evaluate` methods.

    Attributes:
        artifact (Artifact): The artifact associated with the model, containing model metadata and path.
        model (DecisionTreeClassifier): The underlying scikit-learn DecisionTreeClassifier model instance.
    """

    def __init__(self):
        """
        Train the DecisionTreeClassifier on the provided data.

        Args:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target labels for training.
        """
        super().__init__(type="classification")
        self._model = DecisionTreeClassifier()


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the DecisionTreeClassifier on the provided data.

        Args:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target labels for training.
        """
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained DecisionTreeClassifier.

        Args:
            X (np.ndarray): Input features for making predictions.

        Returns:
            np.ndarray: Predicted labels for the input data.
        """
        return self._model.predict(X)

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
