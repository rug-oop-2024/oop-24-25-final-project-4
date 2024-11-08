from sklearn.neighbors import KNeighborsClassifier
from typing import Any, Dict, Literal
import numpy as np
from autoop.core.ml.model.model import Model
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import METRICS, get_metric

class KNNModel(Model):
    def __init__(self, artifact: Artifact, parameters: Dict[str, Any] = None, model_type: Literal["classification"] = "classification"):
        super().__init__(artifact=artifact, parameters=parameters or {}, model_type=model_type)
        
        # Initialize the sklearn KNeighborsClassifier with parameters from self.parameters
        n_neighbors = self.parameters.get("n_neighbors", 5)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the KNN model on the provided data."""
        self.knn.fit(X, y)
        self.trained = True
        print("KNN model trained.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained KNN model."""
        if not self.trained:
            raise ValueError("Model must be trained before predicting.")
        return self.knn.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metrics: list = METRICS) -> Dict[str, float]:
        """Evaluate the KNN model on test data."""
        if not self.trained:
            raise ValueError("Model must be trained before evaluation.")
        
        y_pred = self.predict(X)
        scores = {}

        for metric_name in metrics:
            metric = get_metric(metric_name)
            scores[metric_name] = metric(y, y_pred)

        return scores

    def save(self) -> None:
        """Save model state to artifact path."""
        import joblib
        joblib.dump(self.knn, self.artifact.asset_path)
        print(f"KNN model saved to {self.artifact.asset_path}.")

    def load(self) -> None:
        """Load model state from artifact path."""
        import joblib
        self.knn = joblib.load(self.artifact.asset_path)
        self.trained = True
        print(f"KNN model loaded from {self.artifact.asset_path}.")
