from pydantic import BaseModel, Field


class Feature(BaseModel):
    name: str = Field(
        ..., description="The feature's name")
    type: str = Field(
        ..., description="""The feature's data type
                (e.g., 'numerical', 'categorical')""")
    is_categorical: bool = Field(
        default=False, description="Whether the feature is categorical")
    is_continuous: bool = Field(
        default=False, description="Whether the feature is continuous")

    def encode(self) -> None:
        """Encode categorical features into numerical values."""
        if self.is_categorical:
            pass

    def normalize(self) -> None:
        """Normalize continuous features."""
        if self.is_continuous:
            pass

    def __repr__(self) -> str:
        return (f"Feature(name={self.name}, data_type={self.data_type}, "
                f"is_categorical={self.is_categorical},
                is_continuous={self.is_continuous})")
