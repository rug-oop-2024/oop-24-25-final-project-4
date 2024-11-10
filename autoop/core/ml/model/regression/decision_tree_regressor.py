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
    model: DecisionTreeRegressor = DecisionTreeRegressor()

    def __init__(self) -> None:
        """
        Initializes a new instance of DecisionTreeRegressorModel, setting up
        the underlying DecisionTreeRegressor model from scikit-learn.
        """
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the DecisionTreeRegressor model to the training data.

        Args:
            X (np.ndarray): Training feature data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).

        Returns:
            None
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given feature data using the trained model.

        Args:
            X (np.ndarray): Feature data for which predictions are to be made,
                            of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric_name: str) -> float:
        """
        Evaluates the model's predictions against true values using a specified metric.

        Args:
            X (np.ndarray): Feature data for evaluation, of shape (n_samples, n_features).
            y (np.ndarray): True target values of shape (n_samples,).
            metric_name (str): The name of the evaluation metric to be used.

        Returns:
            float: The computed metric value representing model performance.
        """
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)
