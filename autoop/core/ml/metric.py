from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "mean_absolute_error",
    "r2_score",
]


def get_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Factory function to get each metric by name.

    Arguments:
        name (str): The name of the metric to retrieve.
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The computed metric value.

    Raises:
        ValueError: If the metric name is unknown.
    """
    if name == "mean_squared_error":
        return MeanSquaredError(y_true, y_pred)
    elif name == "accuracy":
        return Accuracy(y_true, y_pred)
    elif name == "precision":
        return Precision(y_true, y_pred)
    elif name == "recall":
        return Recall(y_true, y_pred)
    elif name == "mean_absolute_error":
        return MeanAbsoluteError(y_true, y_pred)
    elif name == "r2_score":
        return R2Score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric name: {name}")


class Metric(ABC):
    """
    Base abstract class for all metrics.

    This class provides a common interface for metric calculation.
    Each metric must implement the __call__ method to compute the metric.

    Methods:
        __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Compute the metric for the given true and predicted values.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the metric.

        Arguments:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed metric value.
        """
        pass


class MeanSquaredError(Metric):
    """
    Mean Squared Error (MSE) metric.

    Measures the average squared difference between predicted and true values.

    Methods:
        __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the Mean Squared Error between true and predicted values.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error.

        Arguments:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed Mean Squared Error.
        """
        return np.mean((y_true - y_pred) ** 2)


class Accuracy(Metric):
    """
    Accuracy metric.

    Measures the proportion of correctly predicted labels.

    Methods:
        __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the accuracy between true and predicted labels.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the accuracy.

        Arguments:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed accuracy.
        """
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0


class Precision(Metric):
    """
    Precision metric.

    Measures the proportion of true positives among the predicted positives.

    Methods:
        __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate precision between true and predicted labels.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision.

        Arguments:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed precision.
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives != 0 else 0.0


class Recall(Metric):
    """
    Recall metric.

    Measures the proportion of true positives among the actual positives.

    Methods:
        __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate recall between true and predicted labels.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall.

        Arguments:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed recall.
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives != 0 else 0.0


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error (MAE) metric.

    Measures the average absolute difference between predicted and true values.

    Methods:
        __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the Mean Absolute Error between true and predicted values.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Error.

        Arguments:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed Mean Absolute Error.
        """
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    """
    R-squared metric.

    Measures how well the regression predictions approximate real data.

    Methods:
        __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the R-squared value between true and predicted values.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R-squared value.

        Arguments:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed R-squared value.
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
