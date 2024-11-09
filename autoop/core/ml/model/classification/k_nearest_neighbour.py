from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric


class KNNClassifier(Model):
    model: KNeighborsClassifier = None
    def __init__(self, n_neighbors: int = 5) -> None:
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 metric_name: str) -> float:
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)
