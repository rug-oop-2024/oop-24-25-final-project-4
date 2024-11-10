from abc import ABC, abstractmethod
import numpy as np
import joblib
from copy import deepcopy
from typing import Any, Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Model(ABC):
    """
    Abstract base class for machine learning models,providing a standard
    interface for machine learning workflows, including training,
    prediction, evaluation, and persistence.

    Attributes:
        parameters (dict[str, Any]): Parameters for model configuration.
        trained (bool): Indicates if the model has been trained.
        type (str): Specifies the model type ('classification',
        'regression', or 'other').
        model (Union[DecisionTreeClassifier, KNeighborsClassifier,
        RandomForestClassifier,
            DecisionTreeRegressor, LinearRegression, RandomForestRegressor]):
            The underlying scikit-learn model instance.
    """

    def __init__(
        self,
        parameters: dict[str, Any] = None,
        trained: bool = False,
        type: str = "other",
    ) -> None:
        """
        Initializes the model with parameters, training status, and model type.

        Args:
            parameters (dict[str, Any], optional): Model parameters.
            trained (bool, optional): Initial training status.
            type (str, optional): Type of the model (e.g., "classification").
        """
        self.parameters = parameters if parameters is not None else {}
        self.trained = trained
        self.type = type
        self.model = None

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Get the model's configuration parameters.

        Returns:
            dict[str, Any]: The dictionary containing the model's parameters.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict[str, Any]) -> None:
        """
        Set the model's configuration parameters.

        Args:
            parameters (dict[str, Any]): A dictionary of
            model configuration parameters.
        """
        if isinstance(parameters, dict):
            self._parameters = parameters

    @property
    def trained(self) -> bool:
        """
        Get the model's training status.

        Returns:
            bool: True if the model has been trained, False otherwise.
        """
        return self._trained

    @trained.setter
    def trained(self, trained: bool) -> None:
        """
        Set the model's training status.

        Args:
            trained (bool): Boolean value indicating if the model is trained.
        """
        if isinstance(trained, bool):
            self._trained = trained

    @property
    def type(self) -> str:
        """
        Get the type of the model (classification, regression, or other).

        Returns:
            str: The type of the model,
            which can be "classification", "regression", or "other".
        """
        return self._type

    @type.setter
    def type(self, type: str) -> None:
        if isinstance(type, str) and type in ["classification",
                                              "regression", "other"]:
            self._type = type

    @property
    def model(
        self,
    ) -> (
        DecisionTreeClassifier | (
            KNeighborsClassifier) | (
                RandomForestClassifier) | (
                    DecisionTreeRegressor) | (
                        LinearRegression) | (
                            RandomForestRegressor)
    ):
        """The scikit-learn model instance."""
        return self._model

    @model.setter
    def model(
        self,
        model: (
            DecisionTreeClassifier | (
                KNeighborsClassifier) | (
                    RandomForestClassifier) | (
                        DecisionTreeRegressor) | (
                            LinearRegression) | (
                                RandomForestRegressor)
        ),
    ) -> None:
        if isinstance(
            model,
            (
                DecisionTreeClassifier,
                KNeighborsClassifier,
                RandomForestClassifier,
                DecisionTreeRegressor,
                LinearRegression,
                RandomForestRegressor,
            ),
        ):
            self._model = model

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model using the provided data.

        Args:
            X (np.ndarray): Feature matrix for training.
            y (np.ndarray): Target labels for training.

        This method should be implemented by subclasses
        to fit a model to the data.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix for making predictions.

        Returns:
            np.ndarray: Predicted values based on the input features.

        This method should be implemented by subclasses
        to return model predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the model performance using a specified metric.

        Args:
            X (np.ndarray): Feature matrix for evaluation.
            y (np.ndarray): True labels for comparison.

        Returns:
            Dict[str, float]: A dictionary containing
            evaluation metric(s) and their values.

        This method should be implemented by subclasses to
        compute performance metrics.
        """
        pass

    def save(self) -> None:
        """
        Save the model state, parameters,
        and training details to the specified artifact path.
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
        Load the model state, parameters,
        and training details from the specified artifact path.
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
            Model: A new instance of the model that is
            a deep copy of the current one.
        """
        return Model(deepcopy(self.artifact), deepcopy(self.parameters))

    class Config:
        """
        Configuration class for the `Model` class, used to define additional
        settings for the `Model` class.

        Attributes:
            arbitrary_types_allowed (bool):
            If set to `True`, allows attributes to have types
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
