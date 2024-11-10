from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric


class KNNClassifier(Model):
    """
    A wrapper class for the KNeighborsClassifier model from scikit-learn.
    It inherits from the abstract `Model` class and implements the
    `fit`, `predict`, and `evaluate` methods.

    Attributes:
        model (KNeighborsClassifier): The underlying scikit-learn KNeighborsClassifier model instance.
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        """
        Initialize the KNNClassifierModel with the specified number of neighbors.

        Args:
            n_neighbors (int, optional): The number of neighbors to use for classification. Defaults to 5.
        """
        super().__init__(type="classification")
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the KNeighborsClassifier on the provided data.

        Args:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target labels for training.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained KNeighborsClassifier.

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
