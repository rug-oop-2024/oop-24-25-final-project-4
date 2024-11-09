from pydantic import BaseModel, Field


class Feature(BaseModel):
    """
    A class representing a feature in a dataset. This class contains
    the feature's name, type, and information about whether it's categorical
    or continuous. It also includes methods for encoding categorical features
    and normalizing continuous features.

    Attributes:
        name (str): The name of the feature.
        type (str): The data type of the feature (e.g., 'numerical', 'categorical').
        is_categorical (bool): Whether the feature is categorical.
        is_continuous (bool): Whether the feature is continuous.
    """

    name: str = Field(..., description="The feature's name")
    type: str = Field(
        ...,
        description="""The feature's data type
                (e.g., 'numerical', 'categorical')""",
    )
    is_categorical: bool = Field(
        default=False, description="Whether the feature is categorical"
    )
    is_continuous: bool = Field(
        default=False, description="Whether the feature is continuous"
    )

    def encode(self) -> None:
        """
        Encode categorical features into numerical values.

        This method is meant to convert categorical feature values into
        numerical representations. The method should only act on features where `is_categorical`
        is set to True.

        Returns:
            None
        """
        if self.is_categorical:
            pass

    def normalize(self) -> None:
        """
        Normalize continuous features.

        This method should be used to normalize continuous features (e.g.,
        scaling them to a standard range or normal distribution). The method
        should only act on features where `is_continuous` is set to True.

        Returns:
            None
        """
        if self.is_continuous:
            pass

    def __repr__(self) -> str:
        """
        Return a string representation of the Feature instance.

        This method provides a human-readable summary of the feature's
        attributes for debugging and logging purposes.

        Returns:
            str: A string representation of the Feature instance.
        """
        return (
            f"Feature(name={self.name}, data_type={self.data_type}, "
            f"is_categorical={self.is_categorical},"
            f"is_continuous={self.is_continuous})"
        )
