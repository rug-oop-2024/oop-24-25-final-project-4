from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from autoop.core.ml.artifact import Artifact
import numpy as np
import joblib
from copy import deepcopy
from typing import Literal, Any, Dict


class Model(ABC):
    """
    Abstract base class for machine learning models.
    """

    def __init__(self, artifact: Any = None, parameters: Dict[str, Any] = {}, trained: bool = False, type: str = "other"):
        self.artifact = artifact
        self.parameters = parameters
        self.trained = trained
        self.type = type
        self.model = None  # Default to None, subclass should define it.

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pass

    def save(self) -> None:
        """
        Save the model state, parameters, and training details to the specified artifact path.
        """
        save_path = self.artifact.asset_path
        joblib.dump(
            {
                "model": self.model,
                "parameters": self.parameters,
                "trained": self.trained,
            },
            save_path,
        )
        print(f"Model saved at {save_path}.")

    def load(self) -> None:
        """
        Load the model state, parameters, and training details from the specified artifact path.
        """
        load_path = self.artifact.asset_path
        data = joblib.load(load_path)
        self.model = data["model"]
        self.parameters = data["parameters"]
        self.trained = data["trained"]
        print(f"Model loaded from {load_path}.")

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Model":
        """
        Create a deep copy of the model, including its artifact and parameters.

        Args:
            memo (Dict[int, Any]): A memo dictionary to help with deep copy.

        Returns:
            Model: A new instance of the model that is a deep copy of the current one.
        """
        return Model(deepcopy(self.artifact), deepcopy(self.parameters))

    class Config:
        """
        Configuration class for the `Model` class, used to define additional
        settings for the `Model` class.

        Attributes:
            arbitrary_types_allowed (bool): If set to `True`, allows attributes to have types
                                            that are not built-in types or Pydantic models.
        """

        arbitrary_types_allowed = True

    @property
    def is_trained(self) -> bool:
        """
        Get the training status of the model.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        return self.trained
