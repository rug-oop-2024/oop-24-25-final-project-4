
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from autoop.core.ml.artifact import Artifact
import numpy as np
import joblib
from copy import deepcopy
from typing import Literal, Any, Dict


class Model(ABC, BaseModel):
    artifact: Artifact
    parameters: Dict[str, Any] = Field(default_factory=dict)
    trained: bool = False
    type: Literal["classification", "regression"] = "other"

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions based on input data."""
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        pass

    def save(self) -> None:
        """Save model state, parameters, and training details."""
        save_path = self.artifact.asset_path
        joblib.dump({"model": self.model, "parameters": self.parameters, "trained": self.trained}, save_path)
        print(f"Model saved at {save_path}.")

    def load(self) -> None:
        """Load model state, parameters, and training details."""
        load_path = self.artifact.asset_path
        data = joblib.load(load_path)
        self.model = data["model"]
        self.parameters = data["parameters"]
        self.trained = data["trained"]
        print(f"Model loaded from {load_path}.")

    def __deepcopy__(self,  memo: Dict[int, Any]) -> 'Model':
        """Provide a deep copy of the model."""
        return Model(deepcopy(self.artifact), deepcopy(self.parameters))

    @property
    def is_trained(self) -> bool:
        return self.trained
    
