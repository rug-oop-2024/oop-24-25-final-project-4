from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "mean_absolute_error",
    "r2_score",
]


def get_metric(name: str):
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r2_score":
        return R2Score()
    else:
        raise ValueError(f"Unknown metric name: {name}")

class Metric(ABC):
    """Base class for all metrics.
    """
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class MeanSquaredError(Metric):
    """Mean Squared Error metric."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)


class Accuracy(Metric):
    """Accuracy metric."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the accuracy."""
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0


class Precision(Metric):
    """Precision metric.
    Measures the proportion of true positives among the predicted postives."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives != 0 else 0.0

class Recall(Metric):
    """Recall metric.
    Measures the proportion of true positives among the actual postives."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives != 0 else 0.0

class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric.
    Measures absolute difference between predicted and true values."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    """R-squared metric.
    Measures how well the regression predictions approximate real data"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
