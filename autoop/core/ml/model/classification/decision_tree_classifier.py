from sklearn.tree import DecisionTreeClassifier
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import get_metric


class DecisionTreeClassifierModel(Model):
    model: DecisionTreeClassifier = None
    def __init__(self, artifact: Artifact):
        super().__init__(artifact=artifact)
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric_name: str) -> float:
        y_pred = self.predict(X)
        metric = get_metric(metric_name)
        return metric(y, y_pred)