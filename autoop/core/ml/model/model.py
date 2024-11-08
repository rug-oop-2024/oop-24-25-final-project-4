
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Any, Dict


class Model(ABC, BaseModel):
    artifact: Artifact
    parameters: Dict[str, Any] = Field(default_factory=dict)
    trained: bool = False
    type_of_model: Literal["classification", "regression"] = "other"

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
        """Save model state and parameters."""
        print(f"Saving model to {self.artifact.asset_path}.")

    def load(self) -> None:
        """Load model state and parameters."""
        print(f"Loading model from {self.artifact.asset_path}.")

    def __deepcopy__(self,  memo: Dict[int, Any]) -> 'Model':
        """Provide a deep copy of the model."""
        return Model(deepcopy(self.artifact), deepcopy(self.parameters))

    @property
    def is_trained(self) -> bool:
        return self.trained
    
