import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class Feature:
    """
    A class representing a feature in a dataset. This class contains
    the feature's name, type, and information about whether it's categorical
    or continuous. It also includes methods for encoding categorical features
    and normalizing continuous features.

    Attributes:
        name (str): The name of the feature.
        type (str): The data type of the feature
        (e.g., 'numerical', 'categorical').
        is_categorical (bool): Whether the feature is categorical.
        is_continuous (bool): Whether the feature is continuous.
    """

    def __init__(
        self,
        name: str,
        type: str = "other",
        is_categorical: bool = False,
        is_continuous: bool = False,
    ) -> None:
        """
        Initialize a Feature instance with its name, type, and properties.

        Args:
            name (str): The name of the feature.
            type (str): The data type of the feature
            (e.g., 'numerical', 'categorical').
            is_categorical (bool): Whether the feature is categorical.
            Defaults to False.
            is_continuous (bool): Whether the feature is continuous.
            Defaults to False.
        """
        self.name = name
        self.type = type
        self.is_categorical = is_categorical
        self.is_continuous = is_continuous
        self._encoder = LabelEncoder() if is_categorical else None
        self._scaler = MinMaxScaler() if is_continuous else None

    @property
    def name(self) -> str:
        """
        Returns the name of the feature.

        Returns:
            str: The name of the feature.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        Sets the name of the feature.

        Args:
            name (str): The new name for the feature.
        """
        if isinstance(name, str):
            self._name = name

    @property
    def type(self) -> str:
        """
        Returns the data type of the feature.

        Returns:
            str: The data type of the feature.
        """
        return self._type

    @type.setter
    def type(self, type: str) -> None:
        if isinstance(type, str) and type in ["categorical",
                                              "numerical", "other"]:
            self._type = type

    @property
    def is_categorical(self) -> bool:
        """
        Returns whether the feature is categorical.

        Returns:
            bool: True if the feature is categorical, otherwise False.
        """
        return self._is_categorical

    @is_categorical.setter
    def is_categorical(self, is_categorical: bool) -> None:
        """
        Sets whether the feature is categorical.

        Args:
            is_categorical (bool): True if the feature is categorical,
            otherwise False.
        """
        if isinstance(is_categorical, bool):
            self._is_categorical = is_categorical

    @property
    def is_continuous(self) -> bool:
        """
        Returns whether the feature is continuous.

        Returns:
            bool: True if the feature is continuous, otherwise False.
        """
        return self._is_continuous

    @is_continuous.setter
    def is_continuous(self, is_continuous: bool) -> None:
        """
        Sets whether the feature is continuous.

        Args:
            is_continuous (bool): True if the feature is continuous,
            otherwise False.
        """
        if isinstance(is_continuous, bool):
            self._is_continuous = is_continuous

    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode categorical features into numerical values.
        Uses label encoding.

        Args:
            values (np.ndarray): The categorical values to encode.

        Returns:
            np.ndarray: Encoded numerical values for the categorical feature.
        """
        if self.is_categorical:
            if self._encoder is None:
                self._encoder = LabelEncoder()
            encoded_values = self._encoder.fit_transform(values)
            return encoded_values
        else:
            raise ValueError(
                """encode() can only be called
                             on categorical features"""
            )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize continuous features to a 0-1 range using MinMax scaling.

        Args:
            values (np.ndarray): The continuous values to normalize.

        Returns:
            np.ndarray: Normalized values for the continuous feature.
        """
        if self.is_continuous:
            if self._scaler is None:
                self._scaler = MinMaxScaler()
            normalized_values = self._scaler.fit_transform(
                values.reshape(-1, 1)
            ).flatten()
            return normalized_values
        else:
            raise ValueError(
                """normalize() can only be
                             called on continuous features"""
            )

    def __repr__(self) -> str:
        """
        Return a string representation of the Feature instance.

        This method provides a human-readable summary of the feature's
        attributes for debugging and logging purposes.

        Returns:
            str: A string representation of the Feature instance.
        """
        return (
            f"Feature(name={self.name}, data_type={self.type}, "
            f"is_categorical={self.is_categorical},"
            f"is_continuous={self.is_continuous})"
        )
