from sklearn.ensemble import RandomForestClassifier
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric


class RandomForestClassifierModel(Model):
    model: RandomForestClassifier = None
    def __init__(self, n_estimators: int = 100) -> None:
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 metric_name: str) -> float:
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)
