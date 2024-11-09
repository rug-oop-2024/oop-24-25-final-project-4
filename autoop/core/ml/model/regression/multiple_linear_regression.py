from sklearn.linear_model import LinearRegression
from autoop.core.ml.model import Model
from autoop.core.ml.metric import get_metric
import numpy as np

class MultipleLinearRegression(Model):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric_name: str) -> float:
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)
